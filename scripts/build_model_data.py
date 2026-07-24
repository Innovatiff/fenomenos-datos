#!/usr/bin/env python3
"""
FENÓMENOS DEL CARIBE — build_model_data.py

Procesa los datos ABIERTOS de los tres centros mundiales a JSONs ligeros
que Fenómenos App lee como archivos estáticos (cero APIs por usuario):

  · ECMWF  — IFS determinista + ENS (ensemble) · AWS Open Data / data.ecmwf.int
  · NOAA   — GFS determinista + GEFS (ensemble) · AWS Open Data (S3 público)
  · GEM    — GDPS determinista + GEPS (ensemble) · Datamart de ECCC (Canadá)

Salida (data/modelos/):
  meta.json               ← qué centros/corridas están disponibles
  {ecmwf|noaa|gem}/det.json   ← campos por período de 6 h: viento, ráfagas,
                                lluvia + u/v para las partículas animadas
  {ecmwf|noaa|gem}/prob.json  ← % de miembros del ensemble sobre el umbral

Pensado para GitHub Actions (ubuntu-latest) cada 6 horas. Cada centro es
independiente: si uno falla, los demás se publican igual.
"""

import datetime as dt
import json
import math
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request

import numpy as np

# ── Región y rejillas ────────────────────────────────────────────────────
# Caribe + México + Centroamérica + sur/este de EE. UU.
LAT_MIN, LAT_MAX = 4.0, 36.0
LON_MIN, LON_MAX = -112.0, -52.0
DET_SP = 0.25   # rejilla determinista (°) — la malla nativa de 0.25°
PROB_SP = 0.5   # rejilla de probabilidades (°)

HOURS_MAX = 96          # 4 días
PERIOD = 6              # horas por período
SNAP_STEP = 3           # se muestrea el modelo cada 3 h

# Umbrales de tiempo peligroso (los mismos de la app)
THR_WIND_MPH = 25.0
THR_GUST_MPH = 40.0
THR_RAIN_MM = 25.0

MS_TO_MPH = 2.236936

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "modelos")


def log(*a):
    print(*a, flush=True)


def grid_axes(sp):
    lats = np.arange(LAT_MAX, LAT_MIN - 1e-6, -sp)   # norte → sur
    lons = np.arange(LON_MIN, LON_MAX + 1e-6, sp)    # oeste → este
    return lats, lons


def grid_json(sp):
    lats, lons = grid_axes(sp)
    return {
        "lats": [round(float(x), 3) for x in lats],
        "lons": [round(float(x), 3) for x in lons],
        "sp": sp,
        "key": "static",
    }


def regrid(da, sp):
    """Remuestrea un DataArray (lat/lon) a nuestra rejilla por VECINO MÁS
    CERCANO con numpy puro (orden de la app: filas norte→sur, columnas
    oeste→este). Con fuentes de 0.25° es una copia exacta; con el resto el
    error es de medio píxel — y es ~100× más rápido que interpolar con
    scipy, que se comía el tiempo del job con 51 miembros del ensemble."""
    lats, lons = grid_axes(sp)
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    src_lat = np.asarray(da[lat_name].values, dtype=np.float64)
    src_lon = np.asarray(da[lon_name].values, dtype=np.float64)
    if src_lon.max() > 180:  # 0..360 → −180..180
        src_lon = ((src_lon + 180) % 360) - 180
    vals = np.asarray(da.values)
    if vals.ndim != 2:
        vals = np.squeeze(vals)
    lat_ord = np.argsort(src_lat)
    lon_ord = np.argsort(src_lon)
    src_lat = src_lat[lat_ord]
    src_lon = src_lon[lon_ord]
    vals = vals[np.ix_(lat_ord, lon_ord)]

    def nearest(src, targets):
        i = np.clip(np.searchsorted(src, targets), 1, src.size - 1)
        return np.where(
            np.abs(src[i] - targets) < np.abs(src[i - 1] - targets), i, i - 1
        )

    iy = nearest(src_lat, lats)
    ix = nearest(src_lon, lons)
    return vals[np.ix_(iy, ix)].astype(np.float64)


def flat(values_2d):
    return values_2d.reshape(-1)


def q1(x):
    """redondeo a 1 decimal apto para JSON (None si NaN)"""
    return None if x is None or not math.isfinite(x) else round(float(x), 1)


def period_times(run_dt):
    n = HOURS_MAX // PERIOD
    return [int((run_dt + dt.timedelta(hours=PERIOD * i)).timestamp()) for i in range(n)]


def http_get(url, timeout=120, retries=3):
    # los cortes de conexión son normales bajando cientos de GRIBs; se
    # reintenta salvo en errores HTTP definitivos (404 = el archivo no está)
    for intento in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fenomenos-app-data/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as ex:
            if ex.code < 500 or intento == retries:
                raise
        except Exception:
            if intento == retries:
                raise
        time.sleep(3 * (intento + 1))


def http_range(url, start, end, timeout=120):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "fenomenos-app-data/1.0",
            "Range": f"bytes={start}-{end}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def open_grib(raw_bytes, filter_keys=None):
    """Abre bytes GRIB2 con cfgrib y devuelve el dataset xarray."""
    import xarray as xr

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(raw_bytes)
        path = f.name
    kwargs = {"engine": "cfgrib", "backend_kwargs": {"indexpath": ""}}
    if filter_keys:
        kwargs["backend_kwargs"]["filter_by_keys"] = filter_keys
    try:
        return xr.open_dataset(path, **kwargs)
    finally:
        # cfgrib ya leyó a memoria en .load(); el llamador debe .load()
        pass


# ═════════════════════════ agregación común ══════════════════════════════

def periods_from_snapshots(snaps, mode):
    """snaps: dict paso_horas → matriz aplanada (o None).
    Devuelve lista por período de 6 h agregando los pasos de 3 h.
    mode 'max' usa los snapshots dentro del período (t+3, t+6);
    mode 'diff' espera ACUMULADOS desde el inicio de la corrida y devuelve
    la diferencia acumulado(t+6) − acumulado(t)."""
    out = []
    n = HOURS_MAX // PERIOD
    for i in range(n):
        t0, t1 = i * PERIOD, (i + 1) * PERIOD
        if mode == "max":
            parts = [snaps.get(h) for h in range(t0 + SNAP_STEP, t1 + 1, SNAP_STEP)]
            parts = [p for p in parts if p is not None]
            out.append(np.maximum.reduce(parts) if parts else None)
        else:  # diff de acumulados
            a, b = snaps.get(t0), snaps.get(t1)
            if a is None and t0 == 0:
                a = np.zeros_like(b) if b is not None else None
            out.append((b - a) if (a is not None and b is not None) else None)
    return out


def pack(series, npoints, dec=1):
    """[período][punto] → [punto][período] con redondeo, para el JSON.
    dec=0 guarda enteros (viento en mph no necesita decimales y el archivo
    pesa la mitad — importa con la rejilla nativa de 0.25°)."""
    n = len(series)
    cols = []
    for p in range(npoints):
        row = []
        for s in range(n):
            if series[s] is None:
                row.append(None)
                continue
            x = float(series[s][p])
            if not math.isfinite(x):
                row.append(None)
            elif dec == 0:
                row.append(int(round(x)))
            else:
                row.append(round(x, dec))
        cols.append(row)
    return cols


def prob_pack(member_series, threshold, npoints):
    """member_series: lista por miembro de [período][punto] (valores) →
    [punto][período] con % de miembros > umbral."""
    nper = HOURS_MAX // PERIOD
    out = []
    for p in range(npoints):
        row = []
        for s in range(nper):
            vals = [
                m[s][p]
                for m in member_series
                if m[s] is not None and math.isfinite(float(m[s][p]))
            ]
            row.append(round(100.0 * sum(1 for v in vals if v > threshold) / len(vals)) if vals else None)
        out.append(row)
    return out


# ═══════════ mapa mundial: imágenes pre-proyectadas del modelo ═══════════
# El campo global se pinta AQUÍ (no en el teléfono) como webp en proyección
# Mercator, con la misma paleta de la app: la capa se ve en todo el mundo,
# nítida y suave al acercarse, y al cliente solo le cuesta una imagen.

MAPA_LAT_N, MAPA_LAT_S = 74.0, -60.0  # todos los países habitados
MAPA_W = 2880                          # 2 px por celda de 0.25°

# paletas idénticas a EURO_VARS/EURO_PROB_STOPS de js/app.js (mph / mm / %)
MAPA_STOPS = {
    ("det", "wind"): [
        (8, (70, 150, 165, 0)), (12, (70, 160, 170, 120)),
        (18, (110, 190, 120, 150)), (25, (255, 224, 90, 185)),
        (32, (255, 176, 32, 205)), (40, (255, 110, 60, 220)),
        (50, (229, 60, 70, 232)), (62, (200, 60, 200, 242)),
    ],
    ("det", "gusts"): [
        (12, (70, 150, 165, 0)), (20, (70, 160, 170, 120)),
        (28, (110, 190, 120, 150)), (40, (255, 224, 90, 185)),
        (50, (255, 176, 32, 205)), (58, (255, 110, 60, 220)),
        (70, (229, 60, 70, 232)), (85, (200, 60, 200, 242)),
    ],
    ("det", "rain"): [
        (0.5, (90, 150, 255, 0)), (2, (90, 150, 255, 125)),
        (6, (70, 190, 240, 155)), (12, (90, 220, 150, 180)),
        (25, (255, 224, 90, 200)), (40, (255, 150, 40, 215)),
        (60, (229, 60, 70, 230)), (100, (200, 60, 200, 242)),
    ],
    # temperatura 2 m en °C (rampa clásica frío→calor, casi opaca: el campo
    # cubre todo el mundo y se lee con la opacidad global de la capa)
    ("det", "temp"): [
        (-40, (130, 60, 180, 215)), (-30, (90, 70, 200, 215)),
        (-20, (60, 110, 230, 215)), (-10, (70, 160, 240, 215)),
        (0, (90, 200, 220, 215)), (5, (80, 210, 160, 215)),
        (10, (110, 220, 110, 215)), (15, (180, 230, 90, 215)),
        (20, (235, 225, 80, 215)), (25, (250, 180, 60, 215)),
        (30, (250, 120, 50, 215)), (35, (235, 60, 45, 215)),
        (40, (180, 30, 60, 215)), (45, (120, 20, 60, 215)),
    ],
    ("prob", "wind"): [
        (0, (255, 224, 138, 0)), (5, (255, 224, 138, 45)),
        (15, (255, 224, 138, 150)), (30, (255, 176, 32, 185)),
        (50, (255, 122, 69, 205)), (70, (229, 72, 77, 222)),
        (90, (186, 60, 190, 235)), (100, (148, 40, 190, 245)),
    ],
}
MAPA_STOPS[("prob", "rain")] = MAPA_STOPS[("prob", "wind")]


def isobar_geojson(msl_hpa, src_lats, src_lons, out_path, step_hpa=4):
    """Isobaras del MSLP como GeoJSON (líneas con propiedad p en hPa).
    Se suaviza, se recorta a la banda del mapa y se submuestrea a 0.5°
    para que cada paso pese decenas de KB, no MB."""
    from scipy import ndimage
    from skimage import measure

    fld, alat, alon = _tc_norm(msl_hpa, src_lats, src_lons)
    fld = ndimage.uniform_filter(fld, size=5, mode=("nearest", "wrap"))
    sel = (alat >= MAPA_LAT_S) & (alat <= MAPA_LAT_N)
    fld = fld[sel][::2, ::2]
    la = alat[sel][::2]
    lo = alon[::2]
    la0, dla = float(la[0]), float(la[1] - la[0])
    lo0, dlo = float(lo[0]), float(lo[1] - lo[0])
    feats = []
    vmin = max(920.0, float(np.nanmin(fld)))
    vmax = min(1080.0, float(np.nanmax(fld)))
    lev0 = int(math.floor(vmin / step_hpa) * step_hpa)
    lev1 = int(math.ceil(vmax / step_hpa) * step_hpa)
    for lev in range(lev0, lev1 + 1, step_hpa):
        for c in measure.find_contours(fld, lev):
            if len(c) < 12:
                continue
            pts = c[::3]
            coords = [
                [round(lo0 + p[1] * dlo, 2), round(la0 + p[0] * dla, 2)] for p in pts
            ]
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"p": lev},
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )
    write_json(out_path, {"type": "FeatureCollection", "features": feats})
    return os.path.getsize(out_path)


def merc_y(lat_deg):
    return math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2))


def mapa_render(field, src_lats, src_lons, stops, out_path):
    """field 2D (lat×lon, global) → webp RGBA en Mercator, bilineal.
    src_lats/src_lons tal cual vienen del GRIB (se normalizan aquí)."""
    from PIL import Image

    lats = np.asarray(src_lats, dtype=np.float64)
    lons = np.asarray(src_lons, dtype=np.float64)
    vals = np.asarray(field, dtype=np.float32)
    # lon 0..360 → −180..180 (rotando columnas), y todo ascendente
    if lons.max() > 180:
        lons = ((lons + 180) % 360) - 180
    lo = np.argsort(lons)
    lons, vals = lons[lo], vals[:, lo]
    la = np.argsort(lats)
    lats, vals = lats[la], vals[la, :]

    y_n, y_s = merc_y(MAPA_LAT_N), merc_y(MAPA_LAT_S)
    H = int(round(MAPA_W * (y_n - y_s) / (2 * math.pi)))

    # filas de salida: uniformes en Mercator → latitud real → fila origen
    yy = y_n - (np.arange(H) + 0.5) * (y_n - y_s) / H
    lat_out = np.degrees(2 * np.arctan(np.exp(yy)) - math.pi / 2)
    fr = np.interp(lat_out, lats, np.arange(lats.size))
    r0 = np.clip(np.floor(fr).astype(np.int32), 0, lats.size - 2)
    tr = (fr - r0).astype(np.float32)[:, None]
    rows = vals[r0, :] * (1 - tr) + vals[r0 + 1, :] * tr

    # columnas: −180..180 uniforme
    lon_out = -180.0 + (np.arange(MAPA_W) + 0.5) * 360.0 / MAPA_W
    fc = np.interp(lon_out, lons, np.arange(lons.size))
    c0 = np.clip(np.floor(fc).astype(np.int32), 0, lons.size - 2)
    tc = (fc - c0).astype(np.float32)[None, :]
    grid = rows[:, c0] * (1 - tc) + rows[:, c0 + 1] * tc

    # color por tramos (np.interp por canal); NaN → transparente
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    rgba = np.empty((H, MAPA_W, 4), dtype=np.uint8)
    nan = ~np.isfinite(grid)
    g = np.nan_to_num(grid, nan=float(xs[0]))
    for ch in range(4):
        ys = np.array([s[1][ch] for s in stops], dtype=np.float32)
        rgba[:, :, ch] = np.interp(g, xs, ys).astype(np.uint8)
    rgba[:, :, 3][nan] = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(out_path, "WEBP", quality=82, method=4)
    return os.path.getsize(out_path)


def mapa_bbox():
    return {"west": -180.0, "south": MAPA_LAT_S, "east": 180.0, "north": MAPA_LAT_N}


# ═══════════ ciclones tropicales: detección en el ENS de ECMWF ════════════
# Detector estilo "genesis tracker" sobre las rejillas globales por
# miembro: mínimo CERRADO de presión + vorticidad ciclónica a 10 m +
# núcleo cálido en 850 hPa, enlazado en trayectorias de 6 en 6 horas.
# Los criterios exactos se publican en ciclones.json — nada de caja negra.

TC_LAT_MAX = 40.0        # banda de SEGUIMIENTO (un TC puede recurvar alto)
TC_GEN_LAT_MAX = 30.0    # banda de GÉNESIS (fuera: bajas extratropicales)
TC_MIN_DEPTH = 2.0       # hPa: centro bajo la media del entorno (14°)
TC_VORT_MIN = 3.0e-5     # s^-1: vorticidad relativa ciclónica a 10 m
TC_WARM_MIN = 1.0        # K: núcleo cálido en 850 hPa (1.0: descarta baroclínicas débiles)
TC_LINK_KM = 450.0       # enlace máximo entre pasos de 6 h
TC_MIN_WIND_KT = 20.0    # descarta ruido débil
TC_MIN_DUR_H = 24.0      # el sistema debe sostenerse ≥24 h en algún miembro
MS_TO_KT = 1.943844


try:
    from global_land_mask import globe as _glm  # noqa: F401
    _HAS_LAND_MASK = True
except Exception:
    _HAS_LAND_MASK = False


def tc_near_ocean(lat, lon, d=1.0):
    """génesis sobre (o junto a) el mar: mata bajas térmicas del desierto.
    Si el paquete de máscara no está, no filtra (y se anota en criteria)."""
    try:
        from global_land_mask import globe
    except Exception:
        return True
    for dy in (-d, 0.0, d):
        for dx in (-d, 0.0, d):
            la = max(-89.9, min(89.9, lat + dy))
            lo = ((lon + dx + 180) % 360) - 180
            if globe.is_ocean(la, lo):
                return True
    return False


def _tc_norm(field, lats, lons):
    """lat ascendente y lon −180..180 ascendente (mismo criterio del mapa)"""
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    vals = np.asarray(field, dtype=np.float32)
    if lons.max() > 180:
        lons = ((lons + 180) % 360) - 180
    lo = np.argsort(lons)
    la = np.argsort(lats)
    return vals[np.ix_(la, lo)], lats[la], lons[lo]


def tc_detect_step(msl_pa, u10, v10, t850, lats, lons):
    """Candidatos en UNA rejilla global de un miembro/paso.
    → lista de (lat, lon, p_hPa, viento_kt)."""
    from scipy import ndimage

    msl, alat, alon = _tc_norm(msl_pa, lats, lons)
    u, _, _ = _tc_norm(u10, lats, lons)
    v, _, _ = _tc_norm(v10, lats, lons)
    t8, _, _ = _tc_norm(t850, lats, lons)
    msl = msl / 100.0  # Pa → hPa
    modes = ("nearest", "wrap")  # la longitud da la vuelta al mundo

    LAT = alat[:, None] * np.ones((1, alon.size), dtype=np.float32)
    band = np.abs(LAT) <= TC_LAT_MAX

    sm = ndimage.uniform_filter(msl, size=3, mode=modes)
    is_min = (sm == ndimage.minimum_filter(sm, size=25, mode=modes)) & band
    depth = ndimage.uniform_filter(sm, size=57, mode=modes) - sm  # entorno 14°
    ok_depth = depth >= TC_MIN_DEPTH

    dy = 111.0e3 * 0.25
    dx = 111.0e3 * 0.25 * np.clip(np.cos(np.radians(alat)), 0.05, None)[:, None]
    dvdx = np.gradient(v, axis=1) / dx
    dudy = np.gradient(u, axis=0) / dy
    vort = ndimage.uniform_filter((dvdx - dudy).astype(np.float32), size=5, mode=modes)
    vort_cyc = vort * np.sign(LAT + 1e-9)
    ok_vort = ndimage.maximum_filter(vort_cyc, size=17, mode=modes) >= TC_VORT_MIN

    warm = ndimage.uniform_filter(t8, size=13, mode=modes) - ndimage.uniform_filter(
        t8, size=57, mode=modes
    )
    ok_warm = warm >= TC_WARM_MIN

    speed_kt = np.hypot(u, v).astype(np.float32) * MS_TO_KT
    wmax = ndimage.maximum_filter(speed_kt, size=13, mode=modes)

    mask = is_min & ok_depth & ok_vort & ok_warm
    ys, xs = np.nonzero(mask)
    cands = []
    for y, x in zip(ys, xs):
        cands.append((float(alat[y]), float(alon[x]), float(sm[y, x]), float(wmax[y, x])))
    # de-duplicado: dos centros a <3° son el mismo sistema (gana el más hondo)
    cands.sort(key=lambda c: c[2])
    out = []
    for c in cands:
        if all(abs(c[0] - o[0]) > 3 or (abs(c[1] - o[1]) % 360) > 3 for o in out):
            out.append(c)
    return [c for c in out if c[3] >= TC_MIN_WIND_KT]


def _hav_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def tc_link(cands_by_step, steps):
    """enlaza candidatos consecutivos en trayectorias por cercanía"""
    tracks = []
    active = []
    for h in steps:
        cands = cands_by_step.get(h, [])
        used = set()
        for tr in active[:]:
            best, bd = None, 1e12
            last = tr["pts"][-1]
            for i, c in enumerate(cands):
                if i in used:
                    continue
                d = _hav_km((last[1], last[2]), (c[0], c[1]))
                if d < bd:
                    bd, best = d, i
            if best is not None and bd <= TC_LINK_KM:
                c = cands[best]
                tr["pts"].append([h, round(c[0], 2), round(c[1], 2), round(c[2], 1), round(c[3])])
                used.add(best)
            else:
                active.remove(tr)
                tracks.append(tr)
        for i, c in enumerate(cands):
            if i not in used:
                active.append({"pts": [[h, round(c[0], 2), round(c[1], 2), round(c[2], 1), round(c[3])]]})
    tracks.extend(active)
    return [t for t in tracks if len(t["pts"]) >= 2]


def tc_basin(lat, lon):
    """cuenca RSMC aproximada; la frontera atl/epac sigue América Central"""
    if lat >= 0:
        if lon >= 100:
            return "wpac"
        if 30 <= lon < 100:
            return "nio"  # incluye mar Rojo y mar Arábigo
        if lon < -140:
            return "epac"  # Pacífico central: avisos de NOAA (CPHC)
        if lon < -40:
            if lat <= 8:
                lim = -77.0
            elif lat <= 17:
                lim = -77.0 - (lat - 8.0) * 2.0   # Panamá (8N,77W) → Tehuantepec (17N,95W)
            elif lat <= 20:
                lim = -95.0 - (lat - 17.0) * (5.0 / 3.0)
            else:
                lim = -100.0
            return "epac" if lon < lim else "atl"
        return "atl"
    if 20 <= lon < 90:
        return "sio"
    if 90 <= lon < 160:
        return "aus"
    if -70 <= lon < 20:
        return "satl"  # Atlántico sur: sin RSMC tropical (ciclones rarísimos)
    return "spac"


# ═══════════════════════════════ ECMWF ═══════════════════════════════════

def build_ecmwf(outdir, model=None, subdir="ecmwf", with_ens=True):
    """IFS determinista + ENS vía el cliente oficial de datos abiertos
    (usa los .index para bajar solo los campos pedidos). Con model=
    "aifs-single" produce el determinista del AIFS (la IA de ECMWF);
    su ensemble no está en datos abiertos, así que with_ens=False."""
    from ecmwf.opendata import Client

    steps = list(range(0, HOURS_MAX + 1, SNAP_STEP))
    steps6 = list(range(0, HOURS_MAX + 1, PERIOD))

    # el AIFS publica pasos de 6 h (sin t+3) y no trae ráfagas
    aifs = model == "aifs-single"
    det_steps = steps6 if aifs else steps
    # msl entra para la trayectoria HRES del rastreador de ciclones
    det_params = ["10u", "10v", "tp"] if aifs else ["10u", "10v", "10fg", "tp", "msl", "2t"]

    # bucket de AWS Open Data
    client = Client(source="aws", model=model) if model else Client(source="aws")
    lat_d, lon_d = grid_axes(DET_SP)
    npoints_d = len(lat_d) * len(lon_d)

    # ── determinista (HRES/oper) ──
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        det_path = f.name
    for intento in range(4):
        try:
            res = client.retrieve(
                type="fc", stream="oper", step=det_steps,
                param=det_params, target=det_path,
            )
            break
        except Exception as ex:
            if intento == 3:
                raise
            log(f"[{subdir}] det: reintento {intento + 1} ({type(ex).__name__})")
            time.sleep(10 * (intento + 1))
    run_dt = dt.datetime.combine(res.datetime.date(), dt.time(res.datetime.hour), dt.timezone.utc)
    log(f"[{subdir}] corrida {run_dt:%Y-%m-%d %Hz} descargada")

    import xarray as xr

    def series_for(short, sp, mode):
        ds = xr.open_dataset(
            det_path, engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": short}},
        ).load()
        var = list(ds.data_vars)[0]
        da = ds[var]
        snaps = {}
        step_dim = "step" if "step" in da.dims else None
        for k in range(da.sizes.get(step_dim, 1)):
            sl = da.isel({step_dim: k}) if step_dim else da
            h = int(sl["step"].values / np.timedelta64(1, "h")) if "step" in sl.coords else 0
            snaps[h] = flat(regrid(sl, sp))
        return periods_from_snapshots(snaps, mode)

    # u/v: para partículas usamos el snapshot de mitad de período (t+3)
    def uv_series(short):
        ds = xr.open_dataset(
            det_path, engine="cfgrib",
            backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": short}},
        ).load()
        da = ds[list(ds.data_vars)[0]]
        snaps = {}
        for k in range(da.sizes.get("step", 1)):
            sl = da.isel(step=k)
            h = int(sl["step"].values / np.timedelta64(1, "h"))
            snaps[h] = flat(regrid(sl, DET_SP))
        out = []
        for i in range(HOURS_MAX // PERIOD):
            mid = snaps.get(i * PERIOD + SNAP_STEP)
            out.append(mid if mid is not None else snaps.get(i * PERIOD))
        return out

    us, vs = uv_series("10u"), uv_series("10v")
    speed = [
        (np.hypot(us[s], vs[s]) * MS_TO_MPH) if us[s] is not None and vs[s] is not None else None
        for s in range(HOURS_MAX // PERIOD)
    ]
    gust = None
    if not aifs:
        try:
            gust = [g * MS_TO_MPH if g is not None else None for g in series_for("10fg", DET_SP, "max")]
        except Exception:
            pass
    if gust is None:
        log(f"[{subdir}] sin ráfagas del modelo; ráfagas ≈ viento × 1.5")
        gust = [s_ * 1.5 if s_ is not None else None for s_ in speed]
    rain = [r * 1000.0 if r is not None else None for r in series_for("tp", DET_SP, "diff")]

    det = {
        "grid": grid_json(DET_SP),
        "times": period_times(run_dt),
        "wind": pack(speed, npoints_d, 0),
        "gusts": pack(gust, npoints_d, 0),
        "rain": pack(rain, npoints_d),
        "u": pack([x * MS_TO_MPH if x is not None else None for x in us], npoints_d, 0),
        "v": pack([x * MS_TO_MPH if x is not None else None for x in vs], npoints_d, 0),
        "members": 1,
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "run": f"{run_dt:%Y%m%d%H}",
    }
    write_json(os.path.join(outdir, subdir, "det.json"), det)

    # ── mapa mundial (solo el IFS de ECMWF por ahora) ──
    mapa = None
    if subdir == "ecmwf":
        def snaps_global(short):
            ds = xr.open_dataset(
                det_path, engine="cfgrib",
                backend_kwargs={"indexpath": "", "filter_by_keys": {"shortName": short}},
            ).load()
            da = ds[list(ds.data_vars)[0]]
            g_lats = da["latitude"].values.copy()
            g_lons = da["longitude"].values.copy()
            out = {}
            if "step" in da.dims:
                for k in range(da.sizes["step"]):
                    sl = da.isel(step=k)
                    h = int(sl["step"].values / np.timedelta64(1, "h"))
                    out[h] = sl.values.astype(np.float32)
            else:
                out[0] = da.values.astype(np.float32)
            ds.close()
            return out, g_lats, g_lons

        try:
            gu_s, glats, glons = snaps_global("10u")
            gv_s, _, _ = snaps_global("10v")
            sw = {h: (np.hypot(gu_s[h], gv_s[h]) * MS_TO_MPH).astype(np.float32) for h in gu_s if h in gv_s}
            del gu_s, gv_s
            per_wind = periods_from_snapshots(sw, "max")
            del sw
            try:
                gg_s, _, _ = snaps_global("10fg")
                per_gust = periods_from_snapshots(
                    {h: v * MS_TO_MPH for h, v in gg_s.items()}, "max"
                )
                del gg_s
            except Exception:
                per_gust = [w * 1.5 if w is not None else None for w in per_wind]
            gr_s, _, _ = snaps_global("tp")
            per_rain = periods_from_snapshots(
                {h: v * 1000.0 for h, v in gr_s.items()}, "diff"
            )
            del gr_s
            try:
                gt_s, _, _ = snaps_global("2t")
                per_temp = periods_from_snapshots(
                    {h: v - 273.15 for h, v in gt_s.items()}, "max"
                )
                del gt_s
            except Exception:
                per_temp = None

            mapa = {
                "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
                "run": f"{run_dt:%Y%m%d%H}",
                "times": period_times(run_dt),
                "bbox": mapa_bbox(),
                "det": {},
                "prob": {},
                "members": 1,
            }
            total = 0
            series_by_var = [("wind", per_wind), ("gusts", per_gust), ("rain", per_rain)]
            if per_temp is not None:
                series_by_var.append(("temp", per_temp))
            for var, series in series_by_var:
                files = []
                for i, fld in enumerate(series):
                    if fld is None:
                        files.append(None)
                        continue
                    rel = f"img/det-{var}-{i:02d}.webp"
                    total += mapa_render(
                        fld, glats, glons, MAPA_STOPS[("det", var)],
                        os.path.join(outdir, subdir, rel),
                    )
                    files.append(rel)
                mapa["det"][var] = files
            log(f"[{subdir}] mapa mundial det: {total // 1024} KB")

            # ── isobaras del MSLP (del HRES ya bajado), una por período ──
            try:
                gm_s, _, _ = snaps_global("msl")
                iso_files = []
                iso_total = 0
                n_per = HOURS_MAX // PERIOD
                for i in range(n_per):
                    snap = gm_s.get(PERIOD * i)
                    if snap is None:
                        snap = gm_s.get(PERIOD * i + SNAP_STEP)
                    if snap is None:
                        iso_files.append(None)
                        continue
                    rel = f"img/iso-{i:02d}.json"
                    iso_total += isobar_geojson(
                        snap / 100.0, glats, glons,
                        os.path.join(outdir, subdir, rel),
                    )
                    iso_files.append(rel)
                del gm_s
                mapa["isobars"] = iso_files
                mapa["isobars_step_hpa"] = 4
                log(f"[{subdir}] isobaras: {iso_total // 1024} KB en {sum(1 for f in iso_files if f)} pasos")
            except Exception:
                log(f"[{subdir}] isobaras fallaron (el mapa sigue sin ellas):")
                traceback.print_exc()
        except Exception:
            log(f"[{subdir}] mapa mundial det falló:")
            traceback.print_exc()
            mapa = None

    if not with_ens:
        if mapa is not None:
            write_json(os.path.join(outdir, subdir, "mapa.json"), mapa)
        return {"det": True, "prob": False, "run": f"{run_dt:%Y%m%d%H}", "members": 1}

    # ── ensemble (ENS): 51 miembros. El bucket sirve ~0.6 MB/s POR
    # conexión y el ENS completo pesa ~2 GB: se baja POR PASO en paralelo
    # (6 conexiones ≈ 6× más rápido) y se procesa archivo por archivo. ──
    from concurrent.futures import ThreadPoolExecutor

    def ens_fetch(h):
        for intento in range(4):
            try:
                with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                    path = f.name
                client.retrieve(
                    type=["cf", "pf"], stream="enfo", step=[h],
                    param=["10u", "10v", "tp"], target=path,
                )
                return h, path
            except Exception as ex:
                log(f"[{subdir}] ENS paso {h}: reintento {intento + 1} ({type(ex).__name__})")
                time.sleep(8 * (intento + 1))
        log(f"[{subdir}] ENS paso {h}: abandonado tras 4 intentos")
        return None  # un paso perdido no tumba el ensemble entero

    with ThreadPoolExecutor(max_workers=6) as ex:
        step_files = [r for r in ex.map(ens_fetch, steps6) if r is not None]
    log(f"[{subdir}] ENS descargado ({len(step_files)}/{len(steps6)} pasos en paralelo)")
    lat_p, lon_p = grid_axes(PROB_SP)
    npoints_p = len(lat_p) * len(lon_p)

    def ens_members(short, mode, scale=1.0):
        """control (cf) + 50 perturbados (pf), leyendo cada paso-archivo"""
        per_member = {}  # clave → {paso_h: matriz aplanada}
        for h, path in step_files:
            for dtype in ("cf", "pf"):
                try:
                    ds = xr.open_dataset(
                        path, engine="cfgrib",
                        backend_kwargs={
                            "indexpath": "",
                            "filter_by_keys": {"shortName": short, "dataType": dtype},
                        },
                    ).load()
                except Exception:
                    continue
                if not list(ds.data_vars):
                    ds.close()
                    continue  # p. ej. tp no existe en el paso 0
                da = ds[list(ds.data_vars)[0]]
                if "number" in da.dims:
                    for m in range(da.sizes["number"]):
                        key = f"{dtype}{int(da['number'].values[m])}"
                        per_member.setdefault(key, {})[h] = (
                            flat(regrid(da.isel(number=m), PROB_SP)) * scale
                        )
                else:
                    per_member.setdefault(f"{dtype}0", {})[h] = (
                        flat(regrid(da, PROB_SP)) * scale
                    )
                ds.close()
        return [
            periods_from_snapshots(per_member[k], mode)
            for k in sorted(per_member)
        ]

    mu = ens_members("10u", "max", MS_TO_MPH)
    mv = ens_members("10v", "max", MS_TO_MPH)
    # velocidad por miembro a partir de u/v del FINAL del período (6-hourly)
    mspeed = []
    for i in range(len(mu)):
        mem = []
        for s in range(HOURS_MAX // PERIOD):
            if mu[i][s] is None or mv[i][s] is None:
                mem.append(None)
            else:
                mem.append(np.hypot(mu[i][s], mv[i][s]))
        mspeed.append(mem)
    mrain = ens_members("tp", "diff", 1000.0)

    prob = {
        "grid": grid_json(PROB_SP),
        "times": period_times(run_dt),
        "members": len(mspeed),
        "wind": prob_pack(mspeed, THR_WIND_MPH, npoints_p),
        "gusts": None,  # el ENS abierto no publica ráfagas
        "rain": prob_pack(mrain, THR_RAIN_MM, npoints_p),
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "run": f"{run_dt:%Y%m%d%H}",
    }
    write_json(os.path.join(outdir, subdir, "prob.json"), prob)

    # ── mapa mundial de probabilidades (segunda pasada sobre los mismos
    # archivos por paso, ya en disco: solo cuesta decodificar) ──
    if mapa is not None:
        def members_global(path, short):
            out = {}
            g_lats = g_lons = None
            for dtype in ("cf", "pf"):
                try:
                    ds = xr.open_dataset(
                        path, engine="cfgrib",
                        backend_kwargs={
                            "indexpath": "",
                            "filter_by_keys": {"shortName": short, "dataType": dtype},
                        },
                    ).load()
                except Exception:
                    continue
                if not list(ds.data_vars):
                    ds.close()
                    continue
                da = ds[list(ds.data_vars)[0]]
                g_lats = da["latitude"].values.copy()
                g_lons = da["longitude"].values.copy()
                if "number" in da.dims:
                    for m in range(da.sizes["number"]):
                        out[f"{dtype}{int(da['number'].values[m])}"] = (
                            da.isel(number=m).values.astype(np.float32)
                        )
                else:
                    out[f"{dtype}0"] = da.values.astype(np.float32)
                ds.close()
            return out, g_lats, g_lons

        try:
            nper = HOURS_MAX // PERIOD
            pw = [None] * nper
            pr = [None] * nper
            tp_prev = None
            glats2 = glons2 = None
            thr_ms = THR_WIND_MPH / MS_TO_MPH
            for h, path in sorted(step_files):
                i = h // PERIOD - 1
                mu2, la2, lo2 = members_global(path, "10u")
                mv2, _, _ = members_global(path, "10v")
                mtp, _, _ = members_global(path, "tp")
                if la2 is not None:
                    glats2, glons2 = la2, lo2
                if i >= 0 and mu2 and mv2:
                    ks = sorted(set(mu2) & set(mv2))
                    cnt = np.zeros(mu2[ks[0]].shape, dtype=np.uint16)
                    for k in ks:
                        cnt += np.hypot(mu2[k], mv2[k]) > thr_ms
                    pw[i] = (100.0 * cnt / len(ks)).astype(np.float32)
                if i >= 0 and mtp:
                    ks = sorted(set(mtp) & set(tp_prev)) if tp_prev else sorted(mtp)
                    if ks:
                        cnt = np.zeros(mtp[ks[0]].shape, dtype=np.uint16)
                        for k in ks:
                            prev = tp_prev[k] if tp_prev else 0.0
                            cnt += (mtp[k] - prev) * 1000.0 > THR_RAIN_MM
                        pr[i] = (100.0 * cnt / len(ks)).astype(np.float32)
                if mtp:
                    tp_prev = mtp
            total = 0
            for var, series in (("wind", pw), ("rain", pr)):
                files = []
                for i, fld in enumerate(series):
                    if fld is None or glats2 is None:
                        files.append(None)
                        continue
                    rel = f"img/prob-{var}-{i:02d}.webp"
                    total += mapa_render(
                        fld, glats2, glons2, MAPA_STOPS[("prob", var)],
                        os.path.join(outdir, subdir, rel),
                    )
                    files.append(rel)
                mapa["prob"][var] = files
            mapa["members"] = len(mspeed)
            log(f"[{subdir}] mapa mundial prob: {total // 1024} KB")
        except Exception:
            log(f"[{subdir}] mapa mundial prob falló:")
            traceback.print_exc()
        write_json(os.path.join(outdir, subdir, "mapa.json"), mapa)

    # ── CICLONES TROPICALES (todas las cuencas) — ENS + HRES ──
    # Un fallo aquí no tumba el centro: se registra y se sigue.
    if mapa is not None:
        try:
            def tc_fetch(job):
                kind, h, kw = job
                for intento in range(3):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                            path = f.name
                        client.retrieve(type=["cf", "pf"], stream="enfo", step=[h], target=path, **kw)
                        return (kind, h, path)
                    except Exception as ex2:
                        log(f"[{subdir}] tc {kind} paso {h}: reintento {intento + 1} ({type(ex2).__name__})")
                        time.sleep(6 * (intento + 1))
                return None

            jobs = [("msl", h, {"param": ["msl"]}) for h in steps6] + [
                ("t850", h, {"param": ["t"], "levtype": "pl", "levelist": [850]}) for h in steps6
            ]
            with ThreadPoolExecutor(max_workers=6) as ex:
                fetched = [r for r in ex.map(tc_fetch, jobs) if r]
            msl_by_h = {h: p for k, h, p in fetched if k == "msl"}
            t8_by_h = {h: p for k, h, p in fetched if k == "t850"}
            sf_by_h = dict(step_files)
            log(f"[{subdir}] tc: msl {len(msl_by_h)}/{len(steps6)} pasos · t850 {len(t8_by_h)}/{len(steps6)}")

            cand = {}  # miembro → {h: [candidatos]}
            for h in steps6:
                if h not in msl_by_h or h not in t8_by_h or h not in sf_by_h:
                    continue
                mu2, la2, lo2 = members_global(sf_by_h[h], "10u")
                mv2, _, _ = members_global(sf_by_h[h], "10v")
                mm2, _, _ = members_global(msl_by_h[h], "msl")
                mt2, _, _ = members_global(t8_by_h[h], "t")
                for k in sorted(set(mu2) & set(mv2) & set(mm2) & set(mt2)):
                    cand.setdefault(k, {})[h] = tc_detect_step(mm2[k], mu2[k], mv2[k], mt2[k], la2, lo2)
                del mu2, mv2, mm2, mt2

            ens_tracks = []
            for k, by in cand.items():
                for tr in tc_link(by, steps6):
                    tr["m"] = k
                    ens_tracks.append(tr)
            n_mem = len(cand)

            # trayectoria determinista (HRES): msl del det + t850 det aparte
            det_tracks = []
            try:
                with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
                    det_t850 = f.name
                client.retrieve(type="fc", stream="oper", step=steps6, param=["t"],
                                levtype="pl", levelist=[850], target=det_t850)
                dmsl, dlat, dlon = snaps_global("msl")
                du, _, _ = snaps_global("10u")
                dv, _, _ = snaps_global("10v")
                import xarray as _xr
                dst = _xr.open_dataset(det_t850, engine="cfgrib",
                                       backend_kwargs={"indexpath": ""}).load()
                da = dst[list(dst.data_vars)[0]]
                dt8 = {}
                for kk in range(da.sizes.get("step", 1)):
                    sl = da.isel(step=kk) if "step" in da.dims else da
                    hh2 = int(sl["step"].values / np.timedelta64(1, "h")) if "step" in sl.coords else 0
                    dt8[hh2] = sl.values.astype(np.float32)
                dst.close()
                dby = {}
                for h in steps6:
                    if h in dmsl and h in du and h in dv and h in dt8:
                        dby[h] = tc_detect_step(dmsl[h], du[h], dv[h], dt8[h], dlat, dlon)
                det_tracks = tc_link(dby, steps6)
            except Exception:
                log(f"[{subdir}] tc det falló (se sigue solo con el ENS):")
                traceback.print_exc()

            # ── sistemas: agrupar génesis cercanas (<600 km, ±12 h) ──
            sistemas = []
            def asigna(tr, es_det):
                p0 = tr["pts"][0]
                for s in sistemas:
                    if abs(p0[0] - s["h0"]) <= 12 and _hav_km((p0[1], p0[2]), (s["lat"], s["lon"])) <= 600:
                        s["tracks"].append((tr, es_det))
                        return
                sistemas.append({"h0": p0[0], "lat": p0[1], "lon": p0[2], "tracks": [(tr, es_det)]})
            for tr in det_tracks:
                asigna(tr, True)
            for tr in ens_tracks:
                asigna(tr, False)

            out_sis = []
            for i, s in enumerate(sistemas):
                mems = sorted({tr["m"] for tr, d in s["tracks"] if not d})
                has_det = any(d for _, d in s["tracks"])
                if len(mems) < 3 and not has_det:
                    continue  # ruido: menos de 3 escenarios y sin señal HRES
                # génesis tropical de verdad: banda ±30, sobre el mar y con
                # un sistema que se sostiene ≥24 h en al menos un miembro
                if abs(s["lat"]) > TC_GEN_LAT_MAX:
                    continue
                if not tc_near_ocean(s["lat"], s["lon"]):
                    continue
                dur = max(tr["pts"][-1][0] - tr["pts"][0][0] for tr, _ in s["tracks"])
                if dur < TC_MIN_DUR_H:
                    continue
                maxkt = sorted(max(p[4] for p in tr["pts"]) for tr, _ in s["tracks"])
                med_kt = maxkt[len(maxkt) // 2]
                finals = [tr["pts"][-1] for tr, d in s["tracks"] if not d]
                esc = None
                if len(finals) >= 6:
                    lons_f = sorted(p[2] for p in finals)
                    med = lons_f[len(lons_f) // 2]
                    oeste = sum(1 for p in finals if p[2] <= med)
                    esc = [{"rumbo": "oeste", "n": oeste}, {"rumbo": "este", "n": len(finals) - oeste}]
                sid = len(out_sis) + 1
                for tr, d in s["tracks"]:
                    tr["sys"] = sid
                out_sis.append({
                    "id": sid,
                    "basin": tc_basin(s["lat"], s["lon"]),
                    "genesis": [round(s["lat"], 1), round(s["lon"], 1)],
                    "members": len(mems),
                    "pct": round(100 * len(mems) / max(1, n_mem)),
                    "det": has_det,
                    "max_kt_med": int(med_kt),
                    "escenarios": esc,
                })

            keep_ids = {s["id"] for s in out_sis}
            ens_out = [
                {"m": tr["m"], "sys": tr["sys"], "pts": tr["pts"]}
                for tr in ens_tracks if tr.get("sys") in keep_ids
            ]
            det_out = [
                {"sys": tr.get("sys"), "pts": tr["pts"]}
                for tr in det_tracks if tr.get("sys") in keep_ids
            ]

            # ── probabilidad de impacto: % de miembros con centro ≥umbral
            # a <120 km de cada celda (0.5°), en todo el período ──
            strike = {"img34": None, "img64": None, "bbox": mapa_bbox()}
            slats = np.arange(90, -90.001, -0.5)
            slons = np.arange(-180, 180, 0.5)
            for thr, keyname in ((34, "img34"), (64, "img64")):
                acc = np.zeros((slats.size, slons.size), dtype=np.uint16)
                any_hit = False
                for k, by in cand.items():
                    grid = np.zeros_like(acc, dtype=bool)
                    hit = False
                    for tr in ens_tracks:
                        if tr["m"] != k or tr.get("sys") not in keep_ids:
                            continue
                        for p in tr["pts"]:
                            if p[4] < thr:
                                continue
                            hit = True
                            iy = int(round((90 - p[1]) / 0.5))
                            ix = int(round((p[2] + 180) / 0.5)) % slons.size
                            ry = 3  # ~120 km
                            rx = int(math.ceil(3 / max(0.15, math.cos(math.radians(p[1])))))
                            y0, y1 = max(0, iy - ry), min(slats.size, iy + ry + 1)
                            for xx in range(ix - rx, ix + rx + 1):
                                grid[y0:y1, xx % slons.size] = True
                    if hit:
                        any_hit = True
                        acc += grid
                if any_hit and n_mem:
                    pct = (100.0 * acc / n_mem).astype(np.float32)
                    rel = f"img/tc{thr}.webp"
                    mapa_render(pct, slats, slons, MAPA_STOPS[("prob", "wind")],
                                os.path.join(outdir, subdir, rel))
                    strike[keyname] = rel

            ciclones = {
                "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
                "run": f"{run_dt:%Y%m%d%H}",
                "base": int(run_dt.timestamp()),
                "members": n_mem,
                "criteria": {
                    "min_cerrado_hPa": TC_MIN_DEPTH,
                    "vorticidad_10m": TC_VORT_MIN,
                    "nucleo_calido_850_K": TC_WARM_MIN,
                    "enlace_km": TC_LINK_KM,
                    "banda_lat": TC_LAT_MAX,
                    "banda_genesis_lat": TC_GEN_LAT_MAX,
                    "genesis_oceano": _HAS_LAND_MASK,
                    "duracion_min_h": TC_MIN_DUR_H,
                    "min_miembros_sistema": 3,
                },
                "sistemas": out_sis,
                "ens": ens_out,
                "det": det_out,
                "strike": strike,
            }
            write_json(os.path.join(outdir, subdir, "ciclones.json"), ciclones)
            log(f"[{subdir}] ciclones: {len(out_sis)} sistemas · {len(ens_out)} trayectorias ENS · det {len(det_out)}")
        except Exception:
            log(f"[{subdir}] ciclones falló (no bloquea el centro):")
            traceback.print_exc()

    return {"det": True, "prob": True, "run": f"{run_dt:%Y%m%d%H}", "members": len(mspeed)}


# ═══════════════════════════════ NOAA ════════════════════════════════════

GFS_BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
GEFS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"


def latest_run(hours_back=6):
    """corrida sinóptica más reciente con margen de publicación"""
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_back)
    hh = (now.hour // 6) * 6
    return now.replace(hour=hh, minute=0, second=0, microsecond=0)


def idx_ranges(idx_text, wanted):
    """Del .idx de NOAA: rangos de bytes de los campos pedidos.
    wanted: lista de subcadenas tipo ':UGRD:10 m above ground:'."""
    lines = idx_text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if any(w in line for w in wanted):
            start = int(line.split(":")[1])
            end = ""
            for j in range(i + 1, len(lines)):
                nxt = int(lines[j].split(":")[1])
                if nxt > start:
                    end = nxt - 1
                    break
            out.append((start, end))
    return out


def fetch_noaa_fields(base_url, wanted):
    """Descarga solo los campos pedidos de un GRIB con su .idx."""
    idx = http_get(base_url + ".idx").decode("utf-8", "replace")
    chunks = []
    for start, end in idx_ranges(idx, wanted):
        rng_end = end if end != "" else start + 40_000_000
        chunks.append(http_range(base_url, start, rng_end))
    return b"".join(chunks)


NOAA_WANTED = [
    ":UGRD:10 m above ground:",
    ":VGRD:10 m above ground:",
    ":GUST:surface:",
    ":APCP:surface:",
]


def build_noaa(outdir):
    import xarray as xr

    run = latest_run(hours_back=6)
    ymd, hh = f"{run:%Y%m%d}", f"{run:%H}"
    steps = list(range(0, HOURS_MAX + 1, SNAP_STEP))

    # ── GFS determinista 0.25° ──
    raws = []
    for h in steps:
        url = f"{GFS_BUCKET}/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.0p25.f{h:03d}"
        raws.append((h, fetch_noaa_fields(url, NOAA_WANTED)))
    log(f"[noaa] GFS {ymd}{hh} · {len(raws)} pasos")

    lat_d, lon_d = grid_axes(DET_SP)
    npoints_d = len(lat_d) * len(lon_d)

    def decode(raw, filt):
        ds = open_grib(raw, filt).load()
        return flat(regrid(ds[list(ds.data_vars)[0]], DET_SP))

    su, sv, sg, sr = {}, {}, {}, {}
    for h, raw in raws:
        try:
            su[h] = decode(raw, {"shortName": "10u"})
            sv[h] = decode(raw, {"shortName": "10v"})
        except Exception:
            pass
        try:
            sg[h] = decode(raw, {"shortName": "gust"})
        except Exception:
            pass
        try:
            # APCP: acumulado del bucket (3/6 h según paso)
            sr[h] = decode(raw, {"shortName": "tp"})
        except Exception:
            pass

    nper = HOURS_MAX // PERIOD
    speed, gust, us, vs = [], [], [], []
    for i in range(nper):
        hs = [i * PERIOD + SNAP_STEP, (i + 1) * PERIOD]
        sp_parts = [np.hypot(su[h], sv[h]) * MS_TO_MPH for h in hs if h in su and h in sv]
        speed.append(np.maximum.reduce(sp_parts) if sp_parts else None)
        g_parts = [sg[h] * MS_TO_MPH for h in hs if h in sg]
        gust.append(np.maximum.reduce(g_parts) if g_parts else None)
        mid = i * PERIOD + SNAP_STEP
        us.append(su.get(mid) * MS_TO_MPH if mid in su else None)
        vs.append(sv.get(mid) * MS_TO_MPH if mid in sv else None)
    # lluvia: APCP de GFS viene en cubos que se reinician cada 6 h → el paso
    # múltiplo de 6 trae el acumulado 6-horario completo
    rain = [sr.get((i + 1) * PERIOD) for i in range(nper)]

    det = {
        "grid": grid_json(DET_SP),
        "times": period_times(run),
        "wind": pack(speed, npoints_d, 0),
        "gusts": pack(gust, npoints_d, 0),
        "rain": pack(rain, npoints_d),
        "u": pack(us, npoints_d, 0),
        "v": pack(vs, npoints_d, 0),
        "members": 1,
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "run": f"{ymd}{hh}",
    }
    write_json(os.path.join(outdir, "noaa", "det.json"), det)

    # ── GEFS 0.25° (pgrb2s): 30 perturbados + control, pasos de 6 h ──
    lat_p, lon_p = grid_axes(PROB_SP)
    npoints_p = len(lat_p) * len(lon_p)
    steps6 = list(range(0, HOURS_MAX + 1, PERIOD))
    members = ["gec00"] + [f"gep{m:02d}" for m in range(1, 31)]

    def decode_p(raw, filt):
        ds = open_grib(raw, filt).load()
        return flat(regrid(ds[list(ds.data_vars)[0]], PROB_SP))

    mspeed, mrain = [], []
    for mem in members:
        su2, sv2, sr2 = {}, {}, {}
        try:
            for h in steps6:
                url = (
                    f"{GEFS_BUCKET}/gefs.{ymd}/{hh}/atmos/pgrb2sp25/"
                    f"{mem}.t{hh}z.pgrb2s.0p25.f{h:03d}"
                )
                raw = fetch_noaa_fields(url, NOAA_WANTED)
                try:
                    su2[h] = decode_p(raw, {"shortName": "10u"})
                    sv2[h] = decode_p(raw, {"shortName": "10v"})
                except Exception:
                    pass
                try:
                    sr2[h] = decode_p(raw, {"shortName": "tp"})
                except Exception:
                    pass
        except Exception as e:
            log(f"[noaa] miembro {mem} falló: {e}")
            continue
        nsp, nrn = [], []
        for i in range(len(steps6) - 1):
            h = steps6[i + 1]
            nsp.append(np.hypot(su2[h], sv2[h]) * MS_TO_MPH if h in su2 and h in sv2 else None)
            nrn.append(sr2.get(h))
        mspeed.append(nsp)
        mrain.append(nrn)

    prob = {
        "grid": grid_json(PROB_SP),
        "times": period_times(run),
        "members": len(mspeed),
        "wind": prob_pack(mspeed, THR_WIND_MPH, npoints_p) if mspeed else None,
        "gusts": None,
        "rain": prob_pack(mrain, THR_RAIN_MM, npoints_p) if mrain else None,
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "run": f"{ymd}{hh}",
    }
    write_json(os.path.join(outdir, "noaa", "prob.json"), prob)
    return {"det": True, "prob": bool(mspeed), "run": f"{ymd}{hh}", "members": len(mspeed)}


# ═══════════════════════════════ GEM (Canadá) ════════════════════════════

DATAMART = "https://dd.weather.gc.ca"


def build_gem(outdir):
    import xarray as xr
    from concurrent.futures import ThreadPoolExecutor

    # GDPS y GEPS solo corren a las 00 y 12 UTC (a diferencia de GFS/IFS);
    # con ~6 h de margen la corrida ya está completa en el Datamart
    now = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=6)
    run = now.replace(hour=0 if now.hour < 12 else 12, minute=0, second=0, microsecond=0)
    ymd, hh = f"{run:%Y%m%d}", f"{run:%H}"
    lat_d, lon_d = grid_axes(DET_SP)
    npoints_d = len(lat_d) * len(lon_d)
    steps = list(range(0, HOURS_MAX + 1, SNAP_STEP))

    def gem_file(var, lvl, h):
        # GDPS 15 km, un archivo por variable y paso. Desde 2026 el Datamart
        # usa /YYYYMMDD/WXO-DD/model_gdps/15km/<hh>/<hhh>/ con nombres MSC
        return (
            f"{DATAMART}/{ymd}/WXO-DD/model_gdps/15km/{hh}/{h:03d}/"
            f"{ymd}T{hh}Z_MSC_GDPS_{var}_{lvl}_LatLon0.15_PT{h:03d}H.grib2"
        )

    def snaps_for(var, lvl, sp, scale=1.0):
        # el Datamart limita cada conexión: se baja en paralelo y se
        # decodifica en serie (cfgrib no es seguro entre hilos)
        def fetch(h):
            try:
                return h, http_get(gem_file(var, lvl, h))
            except Exception as ex:
                return h, ex

        out = {}
        first_err = None
        with ThreadPoolExecutor(max_workers=6) as pool:
            fetched = list(pool.map(fetch, steps))
        for h, raw in fetched:
            if isinstance(raw, Exception):
                if first_err is None:
                    first_err = raw
                continue
            try:
                ds = open_grib(raw).load()
                out[h] = flat(regrid(ds[list(ds.data_vars)[0]], sp)) * scale
            except Exception as ex:
                if first_err is None:
                    first_err = ex
        if not out and first_err is not None:
            log(f"[gem] {var}: {type(first_err).__name__}: {first_err}")
        return out

    su = snaps_for("WindU", "AGL-10m", DET_SP, MS_TO_MPH)
    sv = snaps_for("WindV", "AGL-10m", DET_SP, MS_TO_MPH)
    sg = snaps_for("WindGust", "AGL-10m", DET_SP, MS_TO_MPH)
    sr = snaps_for("Precip-Accum", "Sfc", DET_SP)  # acumulado desde inicio
    log(f"[gem] GDPS {ymd}{hh}: u{len(su)} v{len(sv)} gust{len(sg)} rain{len(sr)}")
    if not su or not sv:
        raise RuntimeError("GDPS sin campos de viento: no se publica un det vacío")

    nper = HOURS_MAX // PERIOD
    speed, gust, us, vs, rain = [], [], [], [], []
    for i in range(nper):
        hs = [i * PERIOD + SNAP_STEP, (i + 1) * PERIOD]
        sp_parts = [np.hypot(su[h], sv[h]) for h in hs if h in su and h in sv]
        speed.append(np.maximum.reduce(sp_parts) if sp_parts else None)
        g_parts = [sg[h] for h in hs if h in sg]
        gust.append(np.maximum.reduce(g_parts) if g_parts else None)
        mid = i * PERIOD + SNAP_STEP
        us.append(su.get(mid))
        vs.append(sv.get(mid))
        a, b = sr.get(i * PERIOD), sr.get((i + 1) * PERIOD)
        if a is None and i == 0 and b is not None:
            a = np.zeros_like(b)
        rain.append((b - a) if a is not None and b is not None else None)

    det = {
        "grid": grid_json(DET_SP),
        "times": period_times(run),
        "wind": pack(speed, npoints_d, 0),
        "gusts": pack(gust, npoints_d, 0),
        "rain": pack(rain, npoints_d),
        "u": pack(us, npoints_d, 0),
        "v": pack(vs, npoints_d, 0),
        "members": 1,
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "run": f"{ymd}{hh}",
    }
    write_json(os.path.join(outdir, "gem", "det.json"), det)

    # ── GEPS (ensemble, 0.5°): cada archivo trae TODOS los miembros ──
    lat_p, lon_p = grid_axes(PROB_SP)
    npoints_p = len(lat_p) * len(lon_p)
    steps6 = list(range(0, HOURS_MAX + 1, PERIOD))

    def geps_members(var, scale=1.0):
        """→ lista por miembro de dict paso→matriz"""

        def fetch(h):
            url = (
                f"{DATAMART}/{ymd}/WXO-DD/ensemble/geps/grib2/raw/{hh}/{h:03d}/"
                f"CMC_geps-raw_{var}_latlon0p5x0p5_{ymd}{hh}_P{h:03d}_allmbrs.grib2"
            )
            try:
                return h, http_get(url)
            except Exception:
                return h, None

        with ThreadPoolExecutor(max_workers=6) as pool:
            fetched = list(pool.map(fetch, steps6))
        per_member = {}
        # cada allmbrs mezcla control (cf) y perturbados (pf): cfgrib no los
        # abre juntos, hay que filtrar por dataType como en el ENS de ECMWF
        for h, raw in fetched:
            if raw is None:
                continue
            for dtype in ("cf", "pf"):
                try:
                    ds = open_grib(raw, {"dataType": dtype}).load()
                except Exception:
                    continue
                if not list(ds.data_vars):
                    ds.close()
                    continue
                da = ds[list(ds.data_vars)[0]]
                if "number" in da.dims:
                    for m in range(da.sizes["number"]):
                        key = f"{dtype}{int(da['number'].values[m])}"
                        per_member.setdefault(key, {})[h] = flat(regrid(da.isel(number=m), PROB_SP)) * scale
                else:
                    per_member.setdefault(dtype, {})[h] = flat(regrid(da, PROB_SP)) * scale
        return per_member

    gu = geps_members("UGRD_TGL_10m", MS_TO_MPH)
    gv = geps_members("VGRD_TGL_10m", MS_TO_MPH)
    gr = geps_members("APCP_SFC_0", 1.0)

    mspeed, mrain = [], []
    for m in sorted(set(gu) & set(gv)):
        mem = []
        for i in range(nper):
            h = (i + 1) * PERIOD
            mem.append(np.hypot(gu[m][h], gv[m][h]) if h in gu[m] and h in gv[m] else None)
        mspeed.append(mem)
    for m in sorted(gr):
        mem = []
        for i in range(nper):
            a, b = gr[m].get(i * PERIOD), gr[m].get((i + 1) * PERIOD)
            if a is None and i == 0 and b is not None:
                a = np.zeros_like(b)
            mem.append((b - a) if a is not None and b is not None else None)
        mrain.append(mem)

    prob = {
        "grid": grid_json(PROB_SP),
        "times": period_times(run),
        "members": len(mspeed),
        "wind": prob_pack(mspeed, THR_WIND_MPH, npoints_p) if mspeed else None,
        "gusts": None,
        "rain": prob_pack(mrain, THR_RAIN_MM, npoints_p) if mrain else None,
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "run": f"{ymd}{hh}",
    }
    write_json(os.path.join(outdir, "gem", "prob.json"), prob)
    return {"det": True, "prob": bool(mspeed), "run": f"{ymd}{hh}", "members": len(mspeed)}


# ═══════════════════════════════ salida ══════════════════════════════════

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    log(f"  → {path} ({os.path.getsize(path) // 1024} KB)")


def build_aifs(outdir):
    return build_ecmwf(outdir, model="aifs-single", subdir="aifs", with_ens=False)


BUILDERS = {
    "ecmwf": build_ecmwf,
    "noaa": build_noaa,
    "gem": build_gem,
    "aifs": build_aifs,
}


def merge(outdir):
    """junta los status.json de los trabajos paralelos en el meta final"""
    centers = {}
    for name in BUILDERS:
        path = os.path.join(outdir, name, "status.json")
        try:
            with open(path) as f:
                centers[name] = json.load(f)
        except Exception:
            centers[name] = {"det": False, "prob": False}
    meta = {
        "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "centers": centers,
    }
    write_json(os.path.join(outdir, "meta.json"), meta)
    ok = any(c.get("det") for c in centers.values())
    log(f"meta.json: " + ", ".join(f"{k}={'ok' if v.get('det') else 'no'}" for k, v in centers.items()))
    sys.exit(0 if ok else 1)


def main():
    outdir = os.path.abspath(OUT_DIR)
    args = sys.argv[1:]

    if args and args[0] == "--merge":
        merge(outdir)
        return

    # con argumento construye SOLO ese centro (trabajos paralelos del CI);
    # sin argumentos construye todos en serie (uso local)
    wanted = args or list(BUILDERS)
    centers = {}
    for name in wanted:
        try:
            centers[name] = BUILDERS[name](outdir)
            log(f"[{name}] OK")
        except Exception:
            log(f"[{name}] FALLÓ:")
            traceback.print_exc()
            centers[name] = {"det": False, "prob": False}
        write_json(os.path.join(outdir, name, "status.json"), centers[name])

    if not args:
        meta = {
            "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
            "centers": centers,
        }
        write_json(os.path.join(outdir, "meta.json"), meta)

    # en modo un-centro el job del CI queda verde y status.json cuenta la
    # verdad; el fallo de un centro no debe tumbar a los demás
    sys.exit(0)


if __name__ == "__main__":
    main()
