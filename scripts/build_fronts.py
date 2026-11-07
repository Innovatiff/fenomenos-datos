#!/usr/bin/env python3
"""
FENÓMENOS DEL CARIBE — build_fronts.py (repo de datos)

Frentes y centros de presión del análisis de superficie de NOAA (el mismo
que dibujan los meteorólogos del WPC cada 3 horas), desde el "boletín
codificado" oficial (ASUS02 KWBC / CODSUS): un texto plano con las
posiciones de cada frente frío/cálido/ocluido/estacionario, las vaguadas
y los centros de alta y baja presión. Aquí se convierte a JSON y la app
lo dibuja nativo sobre el mapa.

Salida: fronts/meta.json
  { updated, valid, source,
    highs: [{p, lat, lon}], lows: [{p, lat, lon}],
    fronts: [{type: cold|warm|stnry|ocfnt|trof, strength, points: [[lon,lat],...]}] }
"""

import datetime as dt
import json
import os
import re
import sys
import urllib.request

SOURCES = [
    "https://www.wpc.ncep.noaa.gov/discussions/codsus",
    "https://www.wpc.ncep.noaa.gov/discussions/codsus_hr",
]

FRONT_TYPES = {
    "COLD": "cold",
    "WARM": "warm",
    "STNRY": "stnry",
    "STATIONARY": "stnry",
    "OCFNT": "ocfnt",
    "TROF": "trof",
}
STRENGTHS = {"WK", "MDT", "STG"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "fronts")


def log(*a):
    print(*a, flush=True)


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "fenomenos-datos/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def decode_point(group):
    """grupo de dígitos del boletín → (lat, lon) en grados (oeste negativo).
    4-5 dígitos: grados enteros (lat 2 + lon 2-3).
    7 dígitos: décimas (lat 3 + lon 4) — boletín de alta resolución."""
    n = len(group)
    if n == 4:
        lat, lon = int(group[:2]), int(group[2:])
    elif n == 5:
        lat, lon = int(group[:2]), int(group[2:])
    elif n == 7:
        lat, lon = int(group[:3]) / 10.0, int(group[3:]) / 10.0
    else:
        return None
    lon = -lon
    if lon < -180:
        lon += 360.0
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (round(lat, 1), round(lon, 1))


def parse_valid(text, now):
    """VALID del boletín → epoch. El CODSUS del WPC usa MMDDHH
    (071715Z = 17 de julio, 15Z); otros boletines usan DDHHMM. Se prueban
    ambas lecturas y gana la más cercana al reloj; si ninguna es coherente
    (±30 h) se usa la hora actual."""
    m = re.search(r"VALID\s+(\d{2})(\d{2})(\d{2})Z", text)
    if not m:
        return int(now.timestamp())
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    candidates = []
    for year in (now.year, now.year - 1, now.year + 1):
        try:  # MMDDHH
            candidates.append(
                dt.datetime(year, a, b, c, 0, tzinfo=dt.timezone.utc)
            )
        except ValueError:
            pass
        try:  # DDHHMM
            candidates.append(
                dt.datetime(year, now.month, a, b, c, tzinfo=dt.timezone.utc)
            )
        except ValueError:
            pass
    best = None
    for t in candidates:
        if best is None or abs(t - now) < abs(best - now):
            best = t
    if best is None or abs((best - now).total_seconds()) > 30 * 3600:
        return int(now.timestamp())
    return int(best.timestamp())


def parse_bulletin(text):
    """→ (highs, lows, fronts). Las secciones pueden continuar en varias
    líneas; una línea que empieza con dígitos continúa la sección previa.
    En HIGHS/LOWS los tokens alternan estrictamente presión → posición."""
    highs, lows, fronts = [], [], []
    section = None  # {"centers": lista, "expect": "p"|"pos"} | frente | None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        tokens = line.split()
        head = tokens[0].upper()

        if head == "HIGHS":
            section = {"centers": highs, "expect": "p"}
            tokens = tokens[1:]
        elif head == "LOWS":
            section = {"centers": lows, "expect": "p"}
            tokens = tokens[1:]
        elif head in FRONT_TYPES:
            section = {"type": FRONT_TYPES[head], "strength": None, "points": []}
            fronts.append(section)
            tokens = tokens[1:]
            if tokens and tokens[0].upper() in STRENGTHS:
                section["strength"] = tokens[0].upper()
                tokens = tokens[1:]
        elif not re.fullmatch(r"\d+", head):
            section = None  # línea de otra cosa (encabezados, $$, etc.)
            continue

        if section is None:
            continue
        for tok in tokens:
            if not re.fullmatch(r"\d+", tok):
                continue
            if "centers" in section:
                if section["expect"] == "p":
                    if len(tok) in (3, 4) and 870 <= int(tok) <= 1090:
                        section["centers"].append({"p": int(tok), "lat": None, "lon": None})
                        section["expect"] = "pos"
                    # si no parece presión se ignora y se sigue esperando una
                else:
                    pt = decode_point(tok)
                    if pt and section["centers"]:
                        section["centers"][-1]["lat"], section["centers"][-1]["lon"] = pt
                    section["expect"] = "p"
            else:
                pt = decode_point(tok)
                if pt:
                    section["points"].append([pt[1], pt[0]])  # [lon, lat]
    highs = [h for h in highs if h["lat"] is not None]
    lows = [l for l in lows if l["lat"] is not None]

    # un token corrupto no debe dibujar una línea cruzando medio planeta:
    # se parte el frente donde haya un salto absurdo entre puntos seguidos
    split = []
    for fr in fronts:
        seg = []
        for pt in fr["points"]:
            if seg and (abs(pt[0] - seg[-1][0]) > 20 or abs(pt[1] - seg[-1][1]) > 15):
                if len(seg) >= 2:
                    split.append({**fr, "points": seg})
                seg = []
            seg.append(pt)
        if len(seg) >= 2:
            split.append({**fr, "points": seg})
    return highs, lows, split


def main():
    text = None
    src_used = None
    for url in SOURCES:
        try:
            text = http_get(url)
            src_used = url
            break
        except Exception as ex:
            log(f"fuente falló: {url} → {ex}")
    if not text:
        log("sin boletín de frentes")
        sys.exit(0)

    log(f"fuente: {src_used}")
    for ln in text.splitlines():
        if "VALID" in ln.upper():
            log(f"línea VALID: {ln.strip()!r}")
            break

    now = dt.datetime.now(dt.timezone.utc)
    valid = parse_valid(text, now)
    highs, lows, fronts = parse_bulletin(text)
    if not fronts and not highs and not lows:
        log("boletín sin contenido analizable")
        sys.exit(0)

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "updated": int(now.timestamp()),
        "valid": valid,
        "source": "NOAA WPC · análisis de superficie (boletín codificado)",
        "highs": highs,
        "lows": lows,
        "fronts": fronts,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, separators=(",", ":"))
    log(
        f"fronts/meta.json · {len(fronts)} frentes, {len(highs)} altas, "
        f"{len(lows)} bajas · válido {dt.datetime.fromtimestamp(valid, dt.timezone.utc):%d %H:%M}Z"
    )


if __name__ == "__main__":
    main()
