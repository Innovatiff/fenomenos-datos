#!/usr/bin/env python3
"""
FENÓMENOS DEL CARIBE — build_gmgsi.py (repo de datos)

MUNDO ENTERO: GMGSI, el mosaico global que NOAA compone cada hora con
todos los satélites geoestacionarios (GOES-Este, GOES-Oeste, Himawari,
Meteosat), desde AWS Open Data y sin claves. Canal infrarrojo de onda
larga (~4-8 km). De cada mosaico salen dos productos:

  world/ir/<epoch>.webp    ← nubes IR mundiales (cielo transparente)
  world/rain/<epoch>.webp  ← lluvia estimada mundial (paleta tipo radar)
  world/meta.json

En la app: capa base planetaria; sobre las Américas se montan el GOES
regional de 10 min y el radar MRMS.
"""

import datetime as dt
import json
import math
import os
import re
import sys
import tempfile
import urllib.request

import numpy as np

BUCKET = "noaa-gmgsi-pds"
PRODUCT = "GMGSI_LW"  # infrarrojo de onda larga

# Mercator llega hasta ~±85°; se cubre hasta ±80° (más allá los
# geoestacionarios ya no ven) y el borde se desvanece en vez de cortarse
WEST, EAST = -180.0, 180.0
SOUTH, NORTH = -80.0, 80.0
FADE_DEG = 4.0  # desvanecido del borde norte/sur
OUT_W = 3600

MAX_FRAMES = 6       # 6 horas de animación (el mosaico es horario)
MAX_NEW_PER_RUN = 4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "world")
IR_DIR = os.path.join(OUT_DIR, "ir")
RAIN_DIR = os.path.join(OUT_DIR, "rain")

# MISMAS rampas que el producto regional (build_goes.py) para que el mundo
# y el GOES de las Américas se vean igual de claros y empalmen sin salto.
# El realce por latitud (más adelante) "entibia" la temperatura aparente
# hacia los polos para que ni las superficies heladas ni la meseta
# antártica en noche polar se pinten como nube o tormenta.
IR_STOPS = [
    (300.0, 0, 0, 0, 0),
    (280.0, 200, 205, 215, 0),
    (270.0, 205, 210, 220, 70),
    (255.0, 225, 228, 235, 130),
    (240.0, 245, 246, 250, 185),
    (230.0, 255, 232, 120, 215),
    (220.0, 255, 170, 50, 230),
    (210.0, 235, 80, 60, 240),
    (200.0, 200, 50, 160, 248),
    (185.0, 150, 60, 230, 255),
]
RAIN_STOPS = [
    (240.0, 0, 0, 0, 0),
    (235.0, 80, 200, 120, 90),
    (228.0, 60, 190, 100, 150),
    (220.0, 250, 220, 70, 190),
    (212.0, 255, 160, 40, 215),
    (204.0, 240, 70, 50, 235),
    (196.0, 220, 40, 150, 245),
    (188.0, 255, 120, 230, 255),
]


def log(*a):
    print(*a, flush=True)


def http_get(url, timeout=240):
    req = urllib.request.Request(url, headers={"User-Agent": "fenomenos-datos/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_bucket(prefix):
    url = f"https://{BUCKET}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    try:
        xml = http_get(url, timeout=60).decode("utf-8", "replace")
    except Exception:
        return []
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def latest_keys():
    """últimos mosaicos horarios (clave, epoch), más recientes al final"""
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for back in range(MAX_FRAMES + 2, -1, -1):
        t = now - dt.timedelta(hours=back)
        prefix = f"{PRODUCT}/{t.year}/{t:%m}/{t:%d}/{t:%H}/"
        keys = list_bucket(prefix)
        if keys:
            keys.sort()
            epoch = int(t.replace(minute=0, second=0, microsecond=0).timestamp())
            out.append((keys[-1], epoch))
    return out


def colorize(bt, stops):
    h, w = bt.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ks = np.array([s[0] for s in stops])
    comp = np.array([[s[1], s[2], s[3], s[4]] for s in stops], dtype=np.float64)
    v = np.clip(bt, ks.min(), ks.max())
    for i in range(len(stops) - 1):
        k0, k1 = ks[i], ks[i + 1]
        mask = (v <= k0) & (v >= k1)
        if not mask.any():
            continue
        t = (k0 - v[mask]) / (k0 - k1)
        for c in range(4):
            out[..., c][mask] = np.round(
                comp[i, c] + (comp[i + 1, c] - comp[i, c]) * t
            ).astype(np.uint8)
    out[..., 3][~np.isfinite(bt)] = 0
    return out


def mercY(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def to_bt(ds):
    """El GMGSI trae la imagen como cuentas de 8 bits (blanco = frío) o,
    según la versión, temperatura de brillo directa. Se normaliza a K."""
    for name in ("data", "Data", "IR"):
        if name in ds:
            v = np.asarray(np.squeeze(ds[name].values), dtype=np.float64)
            break
    else:
        v = np.asarray(np.squeeze(ds[list(ds.data_vars)[0]].values), dtype=np.float64)
    vmax = np.nanmax(v)
    log(f"  variable rango: {np.nanmin(v):.1f}..{vmax:.1f}")
    if vmax <= 255.0 + 1e-6:
        # cuentas de mosaico: 0 (cálido) → 255 (tope frío)
        return 330.0 - v * (330.0 - 180.0) / 255.0
    return v  # ya es temperatura de brillo


def grids(ds, shape):
    """lat/lon del mosaico (1D o 2D) → vectores 1D por eje"""
    lat = None
    lon = None
    for name in ("lat", "latitude", "yc"):
        if name in ds.variables:
            lat = np.asarray(ds[name].values)
            break
    for name in ("lon", "longitude", "xc"):
        if name in ds.variables:
            lon = np.asarray(ds[name].values)
            break
    if lat is None or lon is None:
        raise RuntimeError(f"sin lat/lon; variables: {list(ds.variables)}")
    if lat.ndim == 2:
        lat = lat[:, lat.shape[1] // 2]
    if lon.ndim == 2:
        lon = lon[lon.shape[0] // 2, :]
    if lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
    return lat.astype(np.float64), lon.astype(np.float64)


def main():
    import xarray as xr
    from PIL import Image

    os.makedirs(IR_DIR, exist_ok=True)
    os.makedirs(RAIN_DIR, exist_ok=True)

    pairs = latest_keys()
    if not pairs:
        log("sin mosaicos GMGSI")
        sys.exit(0)

    have = {f[:-5] for f in os.listdir(IR_DIR) if f.endswith(".webp")}
    todo = [(k, e) for k, e in pairs[-MAX_FRAMES:] if str(e) not in have]
    todo = todo[-MAX_NEW_PER_RUN:]
    log(f"{len(todo)} mosaicos nuevos por procesar")

    lut = None
    for key, epoch in todo:
        log(f"  {key} → {epoch}")
        raw = http_get(f"https://{BUCKET}.s3.amazonaws.com/{key}")
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            ds = xr.open_dataset(tmp, engine="h5netcdf").load()
        except Exception:
            import xarray as xr2

            ds = xr2.open_dataset(tmp).load()
        try:
            bt = to_bt(ds)
            lat, lon = grids(ds, bt.shape)
            if lat[0] < lat[-1]:  # norte primero
                lat = lat[::-1]
                bt = bt[::-1, :]
            order = np.argsort(lon)
            lon = lon[order]
            bt = bt[:, order]

            if lut is None:
                # extensión efectiva: hasta donde el mosaico tenga filas
                n_eff = min(NORTH, float(lat.max()))
                s_eff = max(SOUTH, float(lat.min()))
                y0, y1 = mercY(n_eff), mercY(s_eff)
                out_h = int(round(OUT_W * (y0 - y1) / math.radians(EAST - WEST)))
                ys = np.linspace(y0, y1, out_h)
                lats_t = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)
                lons_t = np.linspace(WEST, EAST, OUT_W)
                iy = np.clip(
                    np.searchsorted(-lat, -lats_t), 0, lat.size - 1
                )
                ix = np.clip(np.searchsorted(lon, lons_t), 0, lon.size - 1)
                # realce dependiente de la latitud, en dos tramos: suave desde
                # ±40° (las cimas frías de latitudes medias no son convección
                # tropical) y MUY fuerte desde ±60° (hielo marino y meseta
                # antártica en noche polar bajan a 190-250 K sin ser nube).
                lat_adj = (
                    np.maximum(0.0, np.abs(lats_t) - 40.0) * 1.1
                    + np.maximum(0.0, np.abs(lats_t) - 60.0) * 4.0
                )[:, None]
                # borde desvanecido en vez de corte seco en ±80°
                fade = (
                    np.clip((n_eff - lats_t) / FADE_DEG, 0.0, 1.0)
                    * np.clip((lats_t - s_eff) / FADE_DEG, 0.0, 1.0)
                )[:, None]
                lut = (iy, ix, lat_adj, fade, n_eff, s_eff)
            iy, ix, lat_adj, fade, n_eff, s_eff = lut

            world_adj = bt[np.ix_(iy, ix)] + lat_adj

            ir = colorize(world_adj, IR_STOPS)
            ir[..., 3] = np.round(ir[..., 3] * fade).astype(np.uint8)
            Image.fromarray(ir, "RGBA").save(
                os.path.join(IR_DIR, f"{epoch}.webp"), "WEBP", quality=80, method=4
            )

            rain = colorize(world_adj, RAIN_STOPS)
            rain[..., 3] = np.round(rain[..., 3] * fade).astype(np.uint8)
            Image.fromarray(rain, "RGBA").save(
                os.path.join(RAIN_DIR, f"{epoch}.webp"), "WEBP", quality=80, method=4
            )
        finally:
            ds.close()
            os.remove(tmp)

    frames = sorted(
        int(f[:-5]) for f in os.listdir(IR_DIR) if re.fullmatch(r"\d+\.webp", f)
    )
    for old in frames[:-MAX_FRAMES]:
        for d in (IR_DIR, RAIN_DIR):
            p = os.path.join(d, f"{old}.webp")
            if os.path.exists(p):
                os.remove(p)
    frames = frames[-MAX_FRAMES:]

    # bbox efectivo: el del lut de esta corrida o el del meta anterior
    bbox = {"west": WEST, "south": SOUTH, "east": EAST, "north": NORTH}
    if lut is not None:
        bbox = {"west": WEST, "south": lut[5], "east": EAST, "north": lut[4]}
    else:
        meta_path = os.path.join(OUT_DIR, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    bbox = json.load(f).get("bbox", bbox)
            except Exception:
                pass

    meta = {
        "updated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "source": "NOAA GMGSI · mosaico geoestacionario global (IR, horario)",
        "bbox": bbox,
        "ir": [{"time": t, "file": f"ir/{t}.webp"} for t in frames],
        "rain": [
            {"time": t, "file": f"rain/{t}.webp"}
            for t in frames
            if os.path.exists(os.path.join(RAIN_DIR, f"{t}.webp"))
        ],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    log(f"world/meta.json con {len(frames)} mosaicos")


if __name__ == "__main__":
    main()
