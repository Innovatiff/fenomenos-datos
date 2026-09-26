#!/usr/bin/env python3
"""
FENÓMENOS DEL CARIBE — build_mrms.py (repo de datos)

Radar REAL de alta resolución desde AWS Open Data (NOAA MRMS, ~1 km):
reflectividad compuesta con control de calidad para los dominios que
cubren nuestra región — CONUS (EE. UU. y Golfo) y CARIB (Puerto Rico e
Islas Vírgenes). Donde no hay radar en tierra (RD, Cuba, Centroamérica)
la app lo complementa con la lluvia estimada por satélite (build_goes).

Salida:
  radar/meta.json                    ← fotogramas + bboxes por dominio
  radar/frames/<dom>-<epoch>.webp    ← reflectividad coloreada, fondo
                                       transparente (< 5 dBZ no se pinta)

Cada corrida añade el escaneo más reciente por dominio (MRMS publica cada
~2 min; nuestra cadencia de 10 min toma una instantánea por corrida).
"""

import datetime as dt
import gzip
import json
import math
import os
import re
import sys
import tempfile
import urllib.request

import numpy as np

# región total de la app (para recortar)
WEST, EAST = -112.0, -52.0
SOUTH, NORTH = 4.0, 36.0

MAX_FRAMES = 7  # ~1 h de animación a paso de 10 min

BUCKET = "noaa-mrms-pds"
PRODUCT = "MergedReflectivityQCComposite_00.50"
DOMAINS = ["CONUS", "CARIB"]

# resolución de salida por dominio (px por grado ≈ nitidez)
PX_PER_DEG = {"CONUS": 60, "CARIB": 100}  # ~1.8 km y ~1.1 km

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "radar")
FRAMES_DIR = os.path.join(OUT_DIR, "frames")

# paleta clásica de reflectividad (dBZ) con fondo transparente
DBZ_STOPS = [
    (5.0, 60, 160, 180, 0),
    (10.0, 80, 190, 200, 110),
    (20.0, 60, 130, 240, 160),
    (28.0, 80, 200, 120, 185),
    (35.0, 250, 220, 70, 205),
    (42.0, 255, 160, 40, 225),
    (50.0, 240, 70, 50, 240),
    (58.0, 220, 40, 150, 250),
    (68.0, 255, 255, 255, 255),
]


def log(*a):
    print(*a, flush=True)


def http_get(url, timeout=180):
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


def latest_key(domain):
    """clave más reciente del producto en el dominio (hoy o ayer UTC)"""
    now = dt.datetime.now(dt.timezone.utc)
    for day in (now, now - dt.timedelta(days=1)):
        prefix = f"{domain}/{PRODUCT}/{day:%Y%m%d}/"
        keys = [k for k in list_bucket(prefix) if k.endswith(".grib2.gz")]
        if keys:
            keys.sort()
            return keys[-1]
    return None


def key_epoch(key):
    m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", key)
    t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc
    )
    return int(t.timestamp())


def mercY(lat):
    r = math.radians(lat)
    return math.log(math.tan(math.pi / 4 + r / 2))


def colorize_dbz(v):
    h, w = v.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    ks = np.array([s[0] for s in DBZ_STOPS])
    comp = np.array([[s[1], s[2], s[3], s[4]] for s in DBZ_STOPS], dtype=np.float64)
    x = np.clip(v, ks.min(), ks.max())
    for i in range(len(DBZ_STOPS) - 1):
        k0, k1 = ks[i], ks[i + 1]
        mask = (x >= k0) & (x <= k1) & np.isfinite(v)
        if not mask.any():
            continue
        t = (x[mask] - k0) / (k1 - k0)
        for c in range(4):
            out[..., c][mask] = np.round(
                comp[i, c] + (comp[i + 1, c] - comp[i, c]) * t
            ).astype(np.uint8)
    out[..., 3][~np.isfinite(v)] = 0
    out[..., 3][v < DBZ_STOPS[0][0]] = 0
    return out


def process_domain(domain):
    """→ (epoch, archivo, bbox) o None"""
    import xarray as xr
    from PIL import Image

    key = latest_key(domain)
    if not key:
        log(f"[{domain}] sin archivos")
        return None
    epoch = key_epoch(key)
    fname = f"{domain.lower()}-{epoch}.webp"
    fpath = os.path.join(FRAMES_DIR, fname)

    raw = gzip.decompress(http_get(f"https://{BUCKET}.s3.amazonaws.com/{key}"))
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(raw)
        tmp = f.name
    try:
        ds = xr.open_dataset(
            tmp, engine="cfgrib", backend_kwargs={"indexpath": ""}
        ).load()
        da = ds[list(ds.data_vars)[0]]
        lat = da["latitude"].values
        lon = da["longitude"].values
        if lon.max() > 180:
            lon = ((lon + 180) % 360) - 180
        vals = np.asarray(da.values, dtype=np.float32)
        vals[vals <= -99] = np.nan  # sin dato / fuera de alcance

        # recorte del dominio a nuestra región
        lat_desc = lat[0] > lat[-1]
        if not lat_desc:
            lat = lat[::-1]
            vals = vals[::-1, :]
        n = min(float(lat[0]), NORTH)
        s = max(float(lat[-1]), SOUTH)
        w = max(float(lon[0]), WEST)
        e = min(float(lon[-1]), EAST)
        if n <= s or e <= w:
            log(f"[{domain}] fuera de la región")
            return None

        # rejilla de salida en Mercator (para el estirado lineal del mapa)
        ppd = PX_PER_DEG[domain]
        out_w = max(64, int(round((e - w) * ppd)))
        y0, y1 = mercY(n), mercY(s)
        out_h = max(64, int(round(out_w * (y0 - y1) / math.radians(e - w))))

        ys = np.linspace(y0, y1, out_h)
        lats_t = np.degrees(2 * np.arctan(np.exp(ys)) - np.pi / 2)
        lons_t = np.linspace(w, e, out_w)

        dlat = (lat[0] - lat[-1]) / (lat.size - 1)
        dlon = (lon[-1] - lon[0]) / (lon.size - 1)
        iy = np.clip(np.round((lat[0] - lats_t) / dlat).astype(np.int64), 0, lat.size - 1)
        ix = np.clip(np.round((lons_t - lon[0]) / dlon).astype(np.int64), 0, lon.size - 1)
        crop = vals[np.ix_(iy, ix)]

        rgba = colorize_dbz(crop)
        Image.fromarray(rgba, "RGBA").save(fpath, "WEBP", quality=85, method=4)
        ds.close()
    finally:
        os.remove(tmp)

    log(f"[{domain}] {fname} ({out_w}x{out_h})")
    return {"epoch": epoch, "file": f"frames/{fname}", "bbox": {"west": w, "south": s, "east": e, "north": n}}


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    results = {}
    for domain in DOMAINS:
        try:
            r = process_domain(domain)
            if r:
                results[domain.lower()] = r
        except Exception as ex:
            log(f"[{domain}] falló: {ex}")

    if not results:
        log("sin datos MRMS en esta corrida")
        sys.exit(0)  # no rompe el workflow: el satélite se publica igual

    # meta acumulativo: conserva fotogramas previos hasta MAX_FRAMES
    meta_path = os.path.join(OUT_DIR, "meta.json")
    frames = []
    bboxes = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                old = json.load(f)
            frames = old.get("frames", [])
            bboxes = old.get("bboxes", {})
        except Exception:
            pass

    entry = {"time": max(r["epoch"] for r in results.values()), "files": {}}
    for dom, r in results.items():
        entry["files"][dom] = r["file"]
        bboxes[dom] = r["bbox"]
    if not frames or frames[-1].get("time") != entry["time"]:
        frames.append(entry)
    frames = frames[-MAX_FRAMES:]

    # poda de archivos que ya no están en la lista
    keep = {f for fr in frames for f in fr.get("files", {}).values()}
    for f in os.listdir(FRAMES_DIR):
        if f.endswith(".webp") and f"frames/{f}" not in keep:
            os.remove(os.path.join(FRAMES_DIR, f))

    meta = {
        "updated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "source": "NOAA MRMS · reflectividad compuesta QC (~1 km)",
        "bboxes": bboxes,
        "frames": frames,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    log(f"radar/meta.json con {len(frames)} fotogramas · dominios: {', '.join(results)}")


if __name__ == "__main__":
    main()
