#!/usr/bin/env python3
"""
FENÓMENOS — build_geodata.py (Fase 3: lugares globales, GENERADOS)

Nada de listas de países escritas a mano. Este robot genera:

  geo/countries.json   — TODOS los estados y territorios (Natural Earth
                         admin-0, 258 unidades): ISO a2/a3, nombres en
                         es/en/fr/pt/de/ar/zh/ru, husos IANA (GeoNames),
                         bbox (con cruce del antimeridiano), centroide,
                         zoom sugerido, unidades por defecto, capital,
                         población y URL del servicio meteorológico
                         oficial SOLO si respondió en vivo (si no: null —
                         jamás se adivina un enlace oficial).
  cities/idx/*.json    — índice de búsqueda COMPACTO por fragmentos
                         (prefijo de 2 caracteres normalizados → archivo),
                         construido de GeoNames cities1000 con nombres
                         alternativos (Kyiv/Kiev, München/Munchen,
                         東京/Tokyo/Tōkyō): insensible a diacríticos y
                         tolerante a transliteración, rápido con 170k+.
  cities/idx/popular.json — top de población para consultas de 1 letra.

Fuentes (todas sondeadas en vivo — corrida 30369449924):
  - Natural Earth ne_10m_admin_0_countries.geojson (dominio público)
  - GeoNames cities1000 / timeZones / countryInfo / admin1Codes (CC-BY)
"""

import datetime as dt
import io
import json
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

GN = "https://download.geonames.org/export/dump"
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(*a):
    print(*a, flush=True)


def fetch(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "fenomenos-datos/1.0 (github.com/Innovatiff/fenomenos-datos)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    return os.path.getsize(path)


# ═══════════════ normalización de búsqueda (espejo del cliente) ═══════════
# NFKD + quitar marcas combinantes + casefold + todo lo no alfanumérico
# (unicode) a espacio. La MISMA función vive en js/app.js — si cambias una,
# cambia la otra.
def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold()
    s = re.sub(r"[^0-9a-zßɐ-ʯͰ-῿⺀-꓏가-힯豈-﫿]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ═══════════════ países y territorios (Natural Earth) ═════════════════════

# Servicios meteorológicos oficiales CANDIDATOS (conocimiento curado).
# Solo entran a countries.json los que RESPONDEN en vivo (<400); el resto
# queda null. Nunca se inventa: si no está aquí, es null directamente.
MET_CANDIDATES = {
    "DO": "https://onamet.gob.do/", "US": "https://www.weather.gov/",
    "PR": "https://www.weather.gov/sju/", "VI": "https://www.weather.gov/sju/",
    "CU": "https://www.insmet.cu/", "MX": "https://smn.conagua.gob.mx/",
    "GT": "https://insivumeh.gob.gt/", "BZ": "https://www.nms.gov.bz/",
    "JM": "https://metservice.gov.jm/", "TT": "https://www.metoffice.gov.tt/",
    "CR": "https://www.imn.ac.cr/", "PA": "https://www.imhpa.gob.pa/",
    "NI": "https://www.ineter.gob.ni/", "CO": "https://www.ideam.gov.co/",
    "VE": "https://www.inameh.gob.ve/", "EC": "https://www.inamhi.gob.ec/",
    "PE": "https://www.senamhi.gob.pe/", "BO": "https://senamhi.gob.bo/",
    "CL": "https://www.meteochile.gob.cl/", "AR": "https://www.smn.gob.ar/",
    "UY": "https://www.inumet.gub.uy/", "PY": "https://www.meteorologia.gov.py/",
    "BR": "https://portal.inmet.gov.br/", "CA": "https://weather.gc.ca/",
    "ES": "https://www.aemet.es/", "PT": "https://www.ipma.pt/",
    "FR": "https://meteofrance.com/", "GB": "https://www.metoffice.gov.uk/",
    "IE": "https://www.met.ie/", "DE": "https://www.dwd.de/",
    "NL": "https://www.knmi.nl/", "BE": "https://www.meteo.be/",
    "CH": "https://www.meteoswiss.admin.ch/", "AT": "https://www.geosphere.at/",
    "IT": "https://www.meteoam.it/", "GR": "https://www.emy.gr/",
    "TR": "https://www.mgm.gov.tr/", "PL": "https://www.imgw.pl/",
    "CZ": "https://www.chmi.cz/", "NO": "https://www.met.no/",
    "SE": "https://www.smhi.se/", "FI": "https://www.ilmatieteenlaitos.fi/",
    "DK": "https://www.dmi.dk/", "IS": "https://www.vedur.is/",
    "RU": "https://meteoinfo.ru/", "IL": "https://ims.gov.il/",
    "SA": "https://ncm.gov.sa/", "PK": "https://www.pmd.gov.pk/",
    "IN": "https://mausam.imd.gov.in/", "BD": "https://www.bmd.gov.bd/",
    "LK": "https://www.meteo.gov.lk/", "TH": "https://www.tmd.go.th/",
    "VN": "https://nchmf.gov.vn/", "PH": "https://www.pagasa.dost.gov.ph/",
    "ID": "https://www.bmkg.go.id/", "MY": "https://www.met.gov.my/",
    "SG": "https://www.weather.gov.sg/", "JP": "https://www.jma.go.jp/",
    "KR": "https://www.kma.go.kr/", "CN": "https://www.cma.gov.cn/",
    "TW": "https://www.cwa.gov.tw/", "HK": "https://www.hko.gov.hk/",
    "AU": "http://www.bom.gov.au/", "NZ": "https://www.metservice.com/",
    "FJ": "https://www.met.gov.fj/", "ZA": "https://www.weathersa.co.za/",
    "NG": "https://nimet.gov.ng/", "KE": "https://meteo.go.ke/",
    "MA": "https://www.marocmeteo.ma/", "CW": "https://www.meteo.cw/",
    "AW": "https://www.meteo.aw/", "RE": "https://meteofrance.re/fr",
}

# unidades por defecto: °F donde es la norma; mph en EE. UU. y Reino Unido
FAHRENHEIT = {"US", "PR", "VI", "GU", "AS", "MP", "BS", "BZ", "KY", "PW", "FM", "MH", "LR"}
MPH = {"US", "PR", "VI", "GU", "AS", "MP", "GB"}


def met_verify(candidates):
    """GET real a cada candidato; solo sobreviven los <400."""
    ok = {}

    def check(cc, url):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; fenomenos-datos/1.0)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                if r.status < 400:
                    return cc, url, r.status
        except Exception as e:
            return cc, None, str(e)[:60]
        return cc, None, "?"

    with ThreadPoolExecutor(max_workers=8) as ex:
        for cc, url, st in ex.map(lambda kv: check(*kv), candidates.items()):
            if url:
                ok[cc] = url
            else:
                log(f"  met {cc}: candidato NO respondió ({st}) → null")
    return ok


def geom_bbox(geom):
    """bbox [w,s,e,n]; si el territorio cruza ±180 (Fiyi, Rusia), se calcula
    en 0..360 y el este puede quedar >180 (MapLibre lo acepta)."""
    lons, lats = [], []

    def walk(c):
        if isinstance(c[0], (int, float)):
            lons.append(c[0])
            lats.append(c[1])
        else:
            for x in c:
                walk(x)

    walk(geom["coordinates"])
    w, e = min(lons), max(lons)
    if e - w > 180:  # cruza el antimeridiano
        shifted = [x % 360 for x in lons]
        w, e = min(shifted), max(shifted)
        if w > 180:
            w -= 360
        if e > 180 and w < 0:
            pass  # oeste negativo + este >180: rango continuo válido
    return [round(w, 2), round(min(lats), 2), round(e, 2), round(max(lats), 2)]


def zoom_for(bbox):
    import math
    span = max(bbox[2] - bbox[0], (bbox[3] - bbox[1]) * 1.4, 0.2)
    return max(2, min(10, round(math.log2(360 / span)) + 1))


# unidades de NE sin ISO utilizable pero con código real conocido; el
# resto de los -99 son territorios disputados sin ISO y se saltan adrede
NE_ISO_FALLBACK = {"Tokelau": "TK", "Kosovo": "XK"}


def build_countries(outdir):
    log("— países: Natural Earth admin-0 …")
    gj = json.loads(fetch(NE))
    feats = gj["features"]
    log(f"  {len(feats)} unidades admin-0")
    # dependencias que NE funde en su soberano (p. ej. Tokelau dentro de
    # Nueva Zelanda): se rescatan de map_units si traen ISO propio
    try:
        gj2 = json.loads(fetch(NE.replace("admin_0_countries", "admin_0_map_units")))
        have = set()
        for f in feats:
            for k in ("ISO_A2_EH", "ISO_A2"):
                v = f["properties"].get(k) or ""
                if re.fullmatch(r"[A-Z]{2}", v):
                    have.add(v)
        extra = 0
        for f in gj2["features"]:
            p2 = f["properties"]
            a2x = p2.get("ISO_A2_EH") or p2.get("ISO_A2") or ""
            if re.fullmatch(r"[A-Z]{2}", a2x) and a2x not in have:
                feats.append(f)
                have.add(a2x)
                extra += 1
        log(f"  +{extra} unidades de map_units (dependencias con ISO propio)")
    except Exception as e:
        log(f"  map_units no disponible ({e}); se sigue solo con countries")

    tz_by_cc = {}
    for line in fetch(f"{GN}/timeZones.txt").decode("utf-8").splitlines()[1:]:
        p = line.split("\t")
        if len(p) >= 2 and p[0]:
            tz_by_cc.setdefault(p[0], []).append(p[1])

    cap_by_cc = {}
    for line in fetch(f"{GN}/countryInfo.txt").decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) > 5 and p[0]:
            cap_by_cc[p[0]] = p[5]

    log("— verificando en vivo los servicios meteorológicos candidatos …")
    met_ok = met_verify(MET_CANDIDATES)
    log(f"  {len(met_ok)}/{len(MET_CANDIDATES)} candidatos respondieron")

    out = []
    skipped = []
    for f in feats:
        p = f["properties"]
        a2 = p.get("ISO_A2_EH") or p.get("ISO_A2") or ""
        if not re.fullmatch(r"[A-Z]{2}", a2):
            a2 = p.get("ISO_A2") or ""
        if not re.fullmatch(r"[A-Z]{2}", a2):
            a2 = NE_ISO_FALLBACK.get(p.get("NAME"), "")
        if not re.fullmatch(r"[A-Z]{2}", a2):
            skipped.append(p.get("NAME"))
            continue
        a3 = p.get("ISO_A3_EH") or p.get("ISO_A3") or ""
        bbox = geom_bbox(f["geometry"])
        names = {}
        for lang in ("es", "en", "fr", "pt", "de", "ar", "zh", "ru"):
            names[lang] = p.get(f"NAME_{lang.upper()}") or p.get("NAME") or a2
        out.append(
            {
                "a2": a2.lower(),
                "a3": (a3 if re.fullmatch(r"[A-Z]{3}", a3) else "").lower(),
                "name": names,
                "type": p.get("TYPE") or "",
                "tz": tz_by_cc.get(a2, []),
                "bbox": bbox,
                "centroid": [round(p.get("LABEL_Y") or 0, 3), round(p.get("LABEL_X") or 0, 3)],
                "zoom": zoom_for(bbox),
                "units": {
                    "temp": "fahrenheit" if a2 in FAHRENHEIT else "celsius",
                    "wind": "mph" if a2 in MPH else "kmh",
                },
                "cap": cap_by_cc.get(a2),
                "pop": int(p.get("POP_EST") or 0),
                "met": met_ok.get(a2),
            }
        )
    # con unidades repetidas del mismo ISO (p. ej. dependencias que NE parte
    # en varias entradas), gana la de mayor población
    best = {}
    for c in out:
        k = c["a2"]
        if k not in best or c["pop"] > best[k]["pop"]:
            best[k] = c
    final = sorted(best.values(), key=lambda c: -c["pop"])
    size = write_json(
        os.path.join(outdir, "geo", "countries.json"),
        {
            "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
            "sources": "Natural Earth 10m admin-0 (dominio público) · GeoNames (CC-BY 4.0)",
            "count": len(final),
            "met_verificados": len(met_ok),
            "countries": final,
        },
    )
    log(f"  geo/countries.json: {len(final)} países/territorios · {size // 1024} KB · sin ISO: {skipped}")
    return final


# ═══════════════ índice de ciudades por fragmentos ════════════════════════

LATIN_RE = re.compile(r"^[0-9A-Za-zÀ-ɏ' .-]+$")
NONLATIN_RE = re.compile(r"[Ͱ-ϿЀ-ӿ֐-׿؀-ۿऀ-ॿ⺀-鿿가-힯]")


CODE_RE = re.compile(r"^[A-Z0-9]{2,4}$")  # IATA/ICAO/ISO: ruido, no nombres


def pick_alternates(name, ascii_name, alts_raw, latin_cap=2, native_cap=1):
    """alternativos útiles para buscar: latinos distintos (sin códigos de
    aeropuerto) + escritura nativa (Киев/東京)"""
    base = {norm(name), norm(ascii_name)}
    latin, native = [], []
    for a in alts_raw.split(","):
        a = a.strip()
        if not (2 <= len(a) <= 40) or CODE_RE.match(a):
            continue
        n = norm(a)
        if not n or n in base:
            continue
        if LATIN_RE.match(a):
            latin.append(a)
            base.add(n)
        elif NONLATIN_RE.search(a) and len(a) <= 20:
            native.append(a)
            base.add(n)
    latin.sort(key=len)
    native.sort(key=len)
    return latin[:latin_cap] + native[:native_cap]


def admin1_names():
    out = {}
    for line in fetch(f"{GN}/admin1CodesASCII.txt").decode("utf-8").splitlines():
        p = line.split("\t")
        if len(p) >= 2:
            out[p[0]] = p[1]
    return out


def build_cities_index(outdir):
    log("— ciudades: GeoNames cities1000 …")
    raw = fetch(f"{GN}/cities1000.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = io.TextIOWrapper(z.open("cities1000.txt"), encoding="utf-8").read().splitlines()
    adm1 = admin1_names()
    log(f"  {len(lines)} lugares · admin1 {len(adm1)}")

    shards = {}
    n_rows = 0
    popular = []

    def add(key, row):
        nonlocal n_rows
        b = key.encode("utf-8")
        if len(b) < 2:
            return
        pref = b[:2].hex()
        shards.setdefault(pref, []).append([key] + row)
        n_rows += 1

    for line in lines:
        c = line.split("\t")
        if len(c) < 18:
            continue
        name, ascii_name, alts = c[1], c[2], c[3]
        lat, lon = round(float(c[4]), 3), round(float(c[5]), 3)
        cc, a1, pop = c[8], c[10], int(c[14] or 0)
        # presupuesto por población: el índice cabe en el repo sin engordar
        # el clon del robot de 10 minutos (la 1a corrida pesó 135 MB; con
        # esto queda en ~15-20 MB)
        big = pop >= 15000
        admin = adm1.get(f"{cc}.{a1}", "") if big else ""
        row = [name, admin, cc, lat, lon, round(pop / 1000)]
        popular.append((pop, row))

        variants = [name]
        if norm(ascii_name) != norm(name):
            variants.append(ascii_name)
        if pop >= 50000:
            # las metrópolis (>=1M) llevan TODOS sus alternativos: son las
            # que se buscan entre idiomas (Kiev/Kyiv/Київ, Moscow/Москва) y
            # los cupos "más cortos primero" dejaban fuera justo esos
            lc, nc = (30, 15) if pop >= 1000000 else (4, 3) if pop >= 250000 else (2, 1)
            variants += pick_alternates(name, ascii_name, alts, latin_cap=lc, native_cap=nc)
        seen = set()
        for v in variants:
            n = norm(v)
            if not n or n in seen:
                continue
            seen.add(n)
            add(n, row)
        if big:
            for w in norm(name).split(" ")[1:4]:  # palabras no iniciales
                if len(w) >= 3 and not w.isdigit() and w not in seen:
                    seen.add(w)
                    add(w, row)

    log(f"  {n_rows} entradas de índice en {len(shards)} fragmentos")
    total = 0
    idxdir = os.path.join(outdir, "cities", "idx")
    if os.path.isdir(idxdir):
        for f in os.listdir(idxdir):
            os.remove(os.path.join(idxdir, f))
    for pref, rows in shards.items():
        rows.sort(key=lambda r: -r[6])
        total += write_json(os.path.join(idxdir, f"{pref}.json"), rows)
    popular.sort(key=lambda t: -t[0])
    total += write_json(os.path.join(idxdir, "popular.json"), [r for _, r in popular[:400]])
    write_json(
        os.path.join(idxdir, "meta.json"),
        {
            "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
            "count": len(lines),
            "rows": n_rows,
            "shards": len(shards),
            "scheme": "prefijo = 2 primeros bytes utf-8 de la clave normalizada",
            "source": "GeoNames cities1000 (CC-BY 4.0), con nombres alternativos",
        },
    )
    log(f"  cities/idx: {total // 1024 // 1024} MB en {len(shards)} fragmentos (+popular/meta)")


def enrich_with_cities(outdir, countries):
    """capital con coordenadas + top-5 ciudades por país (de cities1000);
    reescribe geo/countries.json ya enriquecido"""
    raw = fetch(f"{GN}/cities1000.zip")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        lines = io.TextIOWrapper(z.open("cities1000.txt"), encoding="utf-8").read().splitlines()
    by_cc = {}
    for line in lines:
        c = line.split("\t")
        if len(c) < 18:
            continue
        by_cc.setdefault(c[8], []).append(
            (int(c[14] or 0), c[1], round(float(c[4]), 3), round(float(c[5]), 3))
        )
    for c in countries:
        cc = c["a2"].upper()
        cities = sorted(by_cc.get(cc, []), reverse=True)
        c["top"] = [[n, la, lo, p] for p, n, la, lo in cities[:5]]
        cap = c.get("cap")
        c["capital"] = None
        if cap:
            capn = norm(cap)
            hit = next((x for x in cities if norm(x[1]) == capn), None)
            if hit is None:  # 'Washington, D.C.' vs 'Washington' etc.
                hit = next((x for x in cities if capn.startswith(norm(x[1])) or norm(x[1]).startswith(capn)), None)
            if hit:
                c["capital"] = [hit[1], hit[2], hit[3]]
        c.pop("cap", None)
    size = write_json(
        os.path.join(outdir, "geo", "countries.json"),
        {
            "generated": int(dt.datetime.now(dt.timezone.utc).timestamp()),
            "sources": "Natural Earth 10m admin-0 (dominio público) · GeoNames (CC-BY 4.0)",
            "count": len(countries),
            "countries": countries,
        },
    )
    con_cap = sum(1 for c in countries if c["capital"])
    log(f"  countries.json enriquecido: {con_cap}/{len(countries)} con capital resuelta · {size // 1024} KB")


def main():
    outdir = ROOT
    countries = build_countries(outdir)
    enrich_with_cities(outdir, countries)
    build_cities_index(outdir)
    log("GEODATOS OK")


if __name__ == "__main__":
    main()
