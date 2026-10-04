#!/usr/bin/env python3
"""
FENÓMENOS DEL CARIBE — build_goes.py (repo de datos)

Satélite GOES-19 (GOES-East) desde AWS Open Data, sin claves:
descarga las imágenes más recientes del canal 13 (infrarrojo limpio,
10.3 µm — tormentas de día y de noche), las reproyecta de la vista
geoestacionaria a Web Mercator sobre la región del Caribe y las publica
como WebP con TRANSPARENCIA (el cielo despejado no tapa el mapa).

Salida:
  goes/meta.json            ← lista de fotogramas + bbox
  goes/frames/<epoch>.webp  ← últimos N fotogramas (~10 min entre sí)

Pensado para GitHub Actions cada 10 min en un repo PÚBLICO (minutos
ilimitados). El workflow re-crea la historia con un solo commit
(force-push huérfano) para que el repo no crezca.
"""

import datetime as dt
import json
import math
import os
import re
import sys
import urllib.request

import numpy as np

# ── Región (la misma de la app) y tamaño de salida ───────────────────────
WEST, EAST = -112.0, -52.0
SOUTH, NORTH = 4.0, 36.0
OUT_W = 2400  # px (~2.7 km/px; el canal 13 es de 2 km)

MAX_FRAMES = 12          # ~2 horas de animación
MAX_NEW_PER_RUN = 6      # tope de descargas por corrida (primer arranque)

BUCKETS = ["noaa-goes19", "noaa-goes16"]  # GOES-East actual y respaldo
PRODUCT = "ABI-L2-CMIPF"                   # Full Disk, cada 10 min
CHANNEL = "C13"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "goes")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
RAIN_OUT = os.path.join(ROOT, "rain")
RAIN_DIR = os.path.join(RAIN_OUT, "frames")


def log(*a):
    print(*a, flush=True)


def http_get(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "fenomenos-datos/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ── listado S3 público (sin boto3) ───────────────────────────────────────

def list_bucket(bucket, prefix):
    url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}"
    try:
        xml = http_get(url, timeout=60).decode("utf-8", "replace")
    except Exception:
        return []
    return re.findall(r"<Key>([^<]+)</Key>", xml)


def latest_files():
    """Claves de los últimos fotogramas C13 Full Disk (más recientes al final)."""
    now = dt.datetime.now(dt.timezone.utc)
    for bucket in BUCKETS:
        keys = []
        for back in range(3, -1, -1):  # 3 horas hacia atrás → ahora
            t = now - dt.timedelta(hours=back)
            prefix = f"{PRODUCT}/{t.year}/{t.timetuple().tm_yday:03d}/{t.hour:02d}/"
            keys += [k for k in list_bucket(bucket, prefix) if f"M6{CHANNEL}" in k]
        if keys:
            keys.sort()  # el nombre incluye la hora de escaneo
            return bucket, keys
    return None, []


def scan_epoch(key):
    """OR_..._s20261981500204_... → epoch UTC del inicio del escaneo"""
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})", key)
    y, doy, hh, mm = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    t = dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=doy - 1, hours=hh, minutes=mm
    )
    return int(t.timestamp())


# ── reproyección: Mercator (destino) → geoestacionaria (fuente) ──────────

def mercY(lat):
    r = math.radians(lat)
    return math.log(math.tan(math.pi / 4 + r / 2))


def build_target_lut(ds):
    """Para cada píxel de salida (Mercator sobre el bbox), el índice del
    píxel fuente en la rejilla fija del satélite. Se calcula una vez por
    corrida con las fórmulas estándar de la proyección geoestacionaria."""
    proj = ds["goes_imager_projection"]
    h = float(proj.attrs["perspective_point_height"])
    req = float(proj.attrs["semi_major_axis"])
    rpol = float(proj.attrs["semi_minor_axis"])
    lon0 = math.radians(float(proj.attrs["longitude_of_projection_origin"]))
    H = req + h

    y0, y1 = mercY(NORTH), mercY(SOUTH)
    out_h = int(round(OUT_W * (y0 - y1) / math.radians(EAST - WEST)))

    lons = np.linspace(WEST, EAST, OUT_W)
    ys = np.linspace(y0, y1, out_h)
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)

    lon_g, lat_g = np.meshgrid(np.radians(lons), np.radians(lats))

    # latitud geocéntrica y radio local (elipsoide GRS80 del producto)
    e2 = 1 - (rpol / req) ** 2
    lat_c = np.arctan((rpol / req) ** 2 * np.tan(lat_g))
    rc = rpol / np.sqrt(1 - e2 * np.cos(lat_c) ** 2)

    sx = H - rc * np.cos(lat_c) * np.cos(lon_g - lon0)
    sy = -rc * np.cos(lat_c) * np.sin(lon_g - lon0)
    sz = rc * np.sin(lat_c)

    # visibilidad desde el satélite
    visible = (H * (H - sx)) >= (sy**2 + sz**2 * (req / rpol) ** 2)

    rnorm = np.sqrt(sx**2 + sy**2 + sz**2)
    x_ang = np.arcsin(-sy / rnorm)
    y_ang = np.arctan(sz / sx)

    xv = ds["x"].values.astype(np.float64)
    yv = ds["y"].values.astype(np.float64)
    ix = np.round((x_ang - xv[0]) / (xv[1] - xv[0])).astype(np.int64)
    iy = np.round((y_ang - yv[0]) / (yv[1] - yv[0])).astype(np.int64)

    inside = visible & (ix >= 0) & (ix < xv.size) & (iy >= 0) & (iy < yv.size)
    ix = np.clip(ix, 0, xv.size - 1)
    iy = np.clip(iy, 0, yv.size - 1)

    # desvanecido del borde del recorte (~1.5°): la capa regional se funde
    # con el mosaico mundial que va debajo en vez de cortarse en seco
    FADE = 1.5
    fy = np.clip((NORTH - lats) / FADE, 0.0, 1.0) * np.clip(
        (lats - SOUTH) / FADE, 0.0, 1.0
    )
    fx = np.clip((lons - WEST) / FADE, 0.0, 1.0) * np.clip(
        (EAST - lons) / FADE, 0.0, 1.0
    )
    fade = fy[:, None] * fx[None, :]
    return iy, ix, inside, out_h, fade


# ── color: temperatura de brillo → RGBA (cielo despejado transparente) ───

# Lluvia estimada desde los topes fríos (paleta tipo radar). Es la técnica
# estándar de los visores globales donde no hay radar en tierra: cuanto más
# frío el tope de la nube convectiva, más intensa la lluvia probable.
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

IR_STOPS = [
    # (K, R, G, B, A) — de cálido/transparente a topes fríos violeta
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


def colorize(bt, stops=None):
    """bt: matriz de temperatura de brillo (K) → RGBA uint8"""
    stops = stops or IR_STOPS
    h, w = bt.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ks = np.array([s[0] for s in stops])
    comp = np.array([[s[1], s[2], s[3], s[4]] for s in stops], dtype=np.float64)
    v = np.clip(bt, ks.min(), ks.max())
    # ks es descendente: interpolamos por tramos
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


# ── proceso principal ────────────────────────────────────────────────────

def main():
    import xarray as xr
    from PIL import Image

    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(RAIN_DIR, exist_ok=True)

    bucket, keys = latest_files()
    if not keys:
        log("sin archivos GOES disponibles")
        sys.exit(1)
    log(f"bucket {bucket}: {len(keys)} escaneos en ventana")

    wanted = keys[-MAX_FRAMES:]
    have = {
        f
        for f in os.listdir(FRAMES_DIR)
        if f.endswith(".webp") and os.path.exists(os.path.join(RAIN_DIR, f))
    }
    todo = [k for k in wanted if f"{scan_epoch(k)}.webp" not in have]
    todo = todo[-MAX_NEW_PER_RUN:]
    log(f"{len(todo)} fotogramas nuevos por procesar")

    lut = None
    for key in todo:
        epoch = scan_epoch(key)
        log(f"  {key.split('/')[-1]} → {epoch}")
        raw = http_get(f"https://{bucket}.s3.amazonaws.com/{key}")
        tmp = os.path.join(FRAMES_DIR, "_tmp.nc")
        with open(tmp, "wb") as f:
            f.write(raw)
        ds = xr.open_dataset(tmp, engine="h5netcdf")
        try:
            if lut is None:
                lut = build_target_lut(ds)
            iy, ix, inside, out_h, fade = lut
            cmi = ds["CMI"].values  # temperatura de brillo (K) en C13
            bt = cmi[iy, ix]
            bt[~inside] = np.nan
            rgba = colorize(bt)
            rgba[..., 3] = np.round(rgba[..., 3] * fade).astype(np.uint8)
            Image.fromarray(rgba, "RGBA").save(
                os.path.join(FRAMES_DIR, f"{epoch}.webp"),
                "WEBP",
                quality=82,
                method=4,
            )
            rain = colorize(bt, RAIN_STOPS)
            rain[..., 3] = np.round(rain[..., 3] * fade).astype(np.uint8)
            Image.fromarray(rain, "RGBA").save(
                os.path.join(RAIN_DIR, f"{epoch}.webp"),
                "WEBP",
                quality=82,
                method=4,
            )
        finally:
            ds.close()
            os.remove(tmp)

    # conserva solo los últimos MAX_FRAMES
    frames = sorted(
        int(f[:-5]) for f in os.listdir(FRAMES_DIR) if re.fullmatch(r"\d+\.webp", f)
    )
    for old in frames[:-MAX_FRAMES]:
        os.remove(os.path.join(FRAMES_DIR, f"{old}.webp"))
        rp = os.path.join(RAIN_DIR, f"{old}.webp")
        if os.path.exists(rp):
            os.remove(rp)
    frames = frames[-MAX_FRAMES:]

    meta = {
        "updated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "source": f"NOAA {bucket} · ABI {CHANNEL} Full Disk",
        "bbox": {"west": WEST, "south": SOUTH, "east": EAST, "north": NORTH},
        "frames": [{"time": t, "file": f"frames/{t}.webp"} for t in frames],
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))

    rain_meta = dict(meta)
    rain_meta["source"] = "Lluvia estimada por satélite (GOES-19 C13)"
    rain_meta["frames"] = [
        {"time": t, "file": f"frames/{t}.webp"}
        for t in frames
        if os.path.exists(os.path.join(RAIN_DIR, f"{t}.webp"))
    ]
    with open(os.path.join(RAIN_OUT, "meta.json"), "w") as f:
        json.dump(rain_meta, f, separators=(",", ":"))
    log(f"meta.json con {len(frames)} fotogramas (IR + lluvia)")


if __name__ == "__main__":
    main()
