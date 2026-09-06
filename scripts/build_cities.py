#!/usr/bin/env python3
"""
FENÓMENOS DEL CARIBE — build_cities.py (repo de datos)

Índice de ciudades propio (GeoNames, CC-BY) para que el buscador de la
app no dependa de ninguna API externa: todas las ciudades del mundo con
≥15 000 habitantes + TODOS los lugares poblados del Caribe y
Centroamérica (ahí la audiencia necesita hasta el pueblo más pequeño).

Salida: cities/index.json
  { updated, count, source, cities: [[nombre, admin1, cc, lat, lon, pob], ...] }
  (ordenado por población descendente; el país se traduce en el navegador
   con Intl.DisplayNames, así el índice no carga nombres repetidos)
"""

import datetime as dt
import io
import json
import os
import sys
import urllib.request
import zipfile

BASE = "https://download.geonames.org/export/dump"

# Caribe + Centroamérica: de estos países entra TODO lugar poblado
REGION = """DO PR CU HT JM BS BB TT DM GD LC VC AG KN VG VI AI MS GP MQ
BL MF SX CW AW BQ TC KY BM GT BZ SV HN NI CR PA""".split()

MAX_CITIES = 80000

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "cities")


def log(*a):
    print(*a, flush=True)


def fetch(url, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "fenomenos-datos/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_zip_txt(name):
    """descarga <name>.zip y devuelve las líneas de <name>.txt"""
    raw = fetch(f"{BASE}/{name}.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(f"{name}.txt") as f:
            return io.TextIOWrapper(f, encoding="utf-8").read().splitlines()


def admin1_names():
    """'CC.A1' → nombre de la provincia/estado"""
    out = {}
    for line in fetch(f"{BASE}/admin1CodesASCII.txt").decode("utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def parse_rows(lines, admin1, min_pop=None):
    """líneas GeoNames → dict geonameid → entrada compacta"""
    out = {}
    for line in lines:
        p = line.split("\t")
        if len(p) < 15 or p[6] != "P":  # solo lugares poblados
            continue
        try:
            pop = int(p[14] or 0)
        except ValueError:
            pop = 0
        if min_pop is not None and pop < min_pop:
            continue
        try:
            lat = round(float(p[4]), 3)
            lon = round(float(p[5]), 3)
        except ValueError:
            continue
        cc = p[8]
        a1 = admin1.get(f"{cc}.{p[10]}", "") if p[10] else ""
        out[p[0]] = [p[1], a1, cc, lat, lon, pop]
    return out


def main():
    admin1 = admin1_names()
    log(f"admin1: {len(admin1)} regiones")

    cities = {}

    # mundo: ciudades de 15 000+ habitantes
    world = parse_rows(fetch_zip_txt("cities15000"), admin1)
    cities.update(world)
    log(f"mundo (≥15k hab): {len(world)}")

    # región: todos los lugares poblados de cada país
    for cc in REGION:
        try:
            rows = parse_rows(fetch_zip_txt(cc), admin1)
            cities.update(rows)
            log(f"  {cc}: {len(rows)} lugares")
        except Exception as ex:
            log(f"  {cc} falló: {ex}")

    if len(cities) < 10000:
        log("índice sospechosamente pequeño; no se publica")
        sys.exit(1)

    ordered = sorted(cities.values(), key=lambda c: -c[5])[:MAX_CITIES]

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "updated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
        "count": len(ordered),
        "source": "GeoNames (CC-BY 4.0) · cities15000 + Caribe/Centroamérica completo",
        "cities": ordered,
    }
    with open(os.path.join(OUT_DIR, "index.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(os.path.join(OUT_DIR, "index.json"))
    log(f"cities/index.json · {len(ordered)} lugares · {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
