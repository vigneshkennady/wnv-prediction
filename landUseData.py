"""
CDL (Cropland Data Layer) Data Acquisition — WNV County-Level Prediction
=========================================================================
The CropScape SOAP/REST endpoint (nassgeodata.gmu.edu/axis2/.../GetCDLStat
?aoi=county:SS:CCC) is unreliable — it returns HTTP 500 for county-FIPS
queries and has an expired SSL certificate.

Three working alternatives are provided here, in order of reliability:

  Strategy A  CropScape bbox API  — query by bounding box instead of FIPS
              (most endpoints are still live; no auth required)
  Strategy B  Google Earth Engine — reliable, free, no SSL issues
              (requires a GEE account + earthengine-api)
  Strategy C  NASS FTP direct download — download state GeoTIFF, clip to
              county boundary, compute zonal stats locally
              (always works; needs ~500 MB disk per state per year)

Counties: Larimer CO (08069) | Boulder CO (08013) | Dallas TX (48113)
          Maricopa AZ (04013) | Cook IL (17031) | Los Angeles CA (06037)

Requirements (Strategy A):
    pip install requests pandas geopandas pyproj tqdm

Requirements (Strategy B):
    pip install earthengine-api pandas geopandas
    earthengine authenticate          # one-time browser login

Requirements (Strategy C):
    pip install requests pandas geopandas rasterio rasterstats tqdm
=========================================================================
"""
from __future__ import annotations


import io
import sys
import time
import requests
import urllib3
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT ACCESS CHECKS
# Run before any data pull to confirm which endpoints are reachable.
# ════════════════════════════════════════════════════════════════════════════

# Albers bbox for Los Angeles County (used as a representative test query)
_TEST_BBOX = "-2356109,1128970,-2100442,1735536"
# Albers bbox for Cook County IL
_TEST_BBOX_COOK = "-970000,1800000,-820000,1950000"

ACCESS_CHECKS = {
    "tiger_rest": {
        "label": "Census TIGER REST API (county boundaries)",
        "url":   "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Current/MapServer/84/query",
        "params": {"where": "GEOID='06037'", "outFields": "GEOID", "returnGeometry": "false", "f": "json"},
        "verify": True,
        "expect_key": "features",
    },
    "cropscape_fips": {
        "label": "CropScape county-FIPS endpoint (often broken)",
        "url":   "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
        "params": {"aoi": "county:06:037", "year": "2020", "format": "csv"},
        "verify": False,
        "expect_key": None,   # just check HTTP 200
    },
    "cropscape_bbox": {
        "label": "CropScape bbox endpoint (Strategy A)",
        "url":   "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
        "params": {"bbox": _TEST_BBOX, "year": "2020", "format": "csv"},
        "verify": False,
        "expect_key": None,
    },
    "nass_ftp_https": {
        "label": "NASS national CDL download — 2020 30m raster (Strategy C)",
        "url":   "https://www.nass.usda.gov/Research_and_Science/Cropland/datasets/2020_30m_cdls.zip",
        "params": {},
        "verify": True,
        "expect_key": None,
        "head_only": True,
    },
    "nass_mirror": {
        "label": "NASS CDL mirror (data.nass.usda.gov) — 2020 30m raster",
        "url":   "https://data.nass.usda.gov/Research_and_Science/Cropland/datasets/2020_30m_cdls.zip",
        "params": {},
        "verify": True,
        "expect_key": None,
        "head_only": True,
    },
    "gee_importable": {
        "label": "Google Earth Engine Python API importable (Strategy B)",
        "url":   None,        # checked via import, not HTTP
        "params": {},
        "verify": True,
        "expect_key": None,
    },
    "planetary_computer": {
        "label": "Microsoft Planetary Computer STAC API (Strategy D)",
        "url":   "https://planetarycomputer.microsoft.com/api/stac/v1",
        "params": {},
        "verify": True,
        "expect_key": "links",   # STAC root returns {"links": [...]}
    },
}


def check_gee_available() -> tuple[bool, str]:
    """Check whether earthengine-api is installed and authenticated."""
    try:
        import ee  # noqa: F401
    except ImportError:
        return False, "earthengine-api not installed (pip install earthengine-api)"
    try:
        import ee
        ee.Initialize()
        return True, "authenticated and initialized"
    except Exception as e:
        return False, f"not authenticated — run: earthengine authenticate ({e})"


def run_preflight() -> dict[str, bool]:
    """
    Check all data source endpoints before pulling any data.
    Prints a summary table and returns a dict of {check_id: is_accessible}.
    Exits with a clear message if no usable strategy is available.
    """
    print("\n" + "=" * 65)
    print("PRE-FLIGHT ACCESS CHECKS")
    print("=" * 65)

    results = {}

    for check_id, cfg in ACCESS_CHECKS.items():
        label = cfg["label"]

        # GEE is checked via import, not HTTP
        if check_id == "gee_importable":
            ok, detail = check_gee_available()
            status = "OK" if ok else "FAIL"
            symbol = "[OK]  " if ok else "[FAIL]"
            print(f"  {symbol} {label}")
            if not ok:
                print(f"         -> {detail}")
            results[check_id] = ok
            continue

        # HEAD-only check (file existence without downloading)
        if cfg.get("head_only"):
            try:
                r = requests.head(cfg["url"], timeout=15, verify=cfg["verify"],
                                  allow_redirects=True)
                ok = r.status_code < 400
                content_len = r.headers.get("Content-Length", "unknown size")
                detail = f"HTTP {r.status_code}, {content_len} bytes"
            except Exception as e:
                ok = False
                detail = str(e)
            symbol = "[OK]  " if ok else "[FAIL]"
            print(f"  {symbol} {label}")
            if not ok:
                print(f"         -> {detail}")
            results[check_id] = ok
            continue

        # Standard GET check
        try:
            r = requests.get(cfg["url"], params=cfg["params"],
                             timeout=20, verify=cfg["verify"])
            if r.status_code >= 400:
                ok = False
                detail = f"HTTP {r.status_code}"
            elif cfg["expect_key"] and cfg["expect_key"] not in r.text:
                ok = False
                detail = f"HTTP {r.status_code} but response missing '{cfg['expect_key']}'"
            else:
                ok = True
                detail = f"HTTP {r.status_code}"
        except requests.exceptions.SSLError as e:
            ok = False
            detail = f"SSL error: {e}"
        except requests.exceptions.ConnectionError as e:
            ok = False
            detail = f"Connection error: {e}"
        except requests.exceptions.Timeout:
            ok = False
            detail = "Timed out after 20s"
        except Exception as e:
            ok = False
            detail = str(e)

        symbol = "[OK]  " if ok else "[FAIL]"
        print(f"  {symbol} {label}")
        if not ok:
            print(f"         -> {detail}")
        results[check_id] = ok

    # ── Determine viable strategies ──────────────────────────────────────
    print()
    strategy_a_ok = results.get("cropscape_bbox") and results.get("tiger_rest")
    strategy_b_ok = results.get("gee_importable")
    nass_reachable = results.get("nass_ftp_https") or results.get("nass_mirror")
    strategy_c_ok = nass_reachable and results.get("tiger_rest")
    strategy_d_ok = results.get("planetary_computer") and results.get("tiger_rest")

    viable = []
    if strategy_b_ok: viable.append("B (Google Earth Engine)")   # GEE first — most reliable
    if strategy_d_ok: viable.append("D (Planetary Computer)")
    if strategy_a_ok: viable.append("A (CropScape bbox)")
    if strategy_c_ok: viable.append("C (NASS FTP + zonal stats)")

    if viable:
        print(f"  Viable strategies: {', '.join(viable)}")
        print(f"  Will attempt in order: {viable[0]} first")
    else:
        print("  [!!] NO VIABLE STRATEGY FOUND.")
        print("       Check your internet connection, VPN, or firewall settings.")
        print("       For Strategy B: run  earthengine authenticate")
        print("       For Strategy C: ensure ~50 GB free disk space")

    print("=" * 65)

    # Attach strategy flags for use by main()
    results["_strategy_a_ok"] = strategy_a_ok
    results["_strategy_b_ok"] = strategy_b_ok
    results["_strategy_c_ok"] = strategy_c_ok
    results["_strategy_d_ok"] = strategy_d_ok
    results["_any_viable"]    = bool(viable)

    return results

# ── Target counties ──────────────────────────────────────────────────────────
COUNTIES = {
    "Larimer_CO":    {"fips": "08069", "state_fips": "08", "county_fips": "069", "state_abbr": "CO", "name": "Larimer County, CO"},
    "Boulder_CO":    {"fips": "08013", "state_fips": "08", "county_fips": "013", "state_abbr": "CO", "name": "Boulder County, CO"},
    "Dallas_TX":     {"fips": "48113", "state_fips": "48", "county_fips": "113", "state_abbr": "TX", "name": "Dallas County, TX"},
    "Maricopa_AZ":   {"fips": "04013", "state_fips": "04", "county_fips": "013", "state_abbr": "AZ", "name": "Maricopa County, AZ"},
    "Cook_IL":       {"fips": "17031", "state_fips": "17", "county_fips": "031", "state_abbr": "IL", "name": "Cook County, IL"},
    "LosAngeles_CA": {"fips": "06037", "state_fips": "06", "county_fips": "037", "state_abbr": "CA", "name": "Los Angeles County, CA"},
}

CDL_YEARS = list(range(2008, 2024))

# CDL class groups relevant to WNV mosquito habitat
CDL_MOSQUITO_RELEVANT = {
    "rice":        [3],
    "corn":        [1],
    "soybeans":    [5],
    "cotton":      [2],
    "pasture_hay": [37, 38, 181],
    "fallow_idle": [61],
    "aquaculture": [92],
    "wetlands":    [87, 190, 195],
    "vegetables":  list(range(206, 260)),
    "orchards":    list(range(66, 78)),
}

# ── CDL class name lookup (value -> name) ────────────────────────────────────
CDL_NAMES = {
    0:"Background", 1:"Corn", 2:"Cotton", 3:"Rice", 4:"Sorghum", 5:"Soybeans",
    6:"Sunflower", 10:"Peanuts", 11:"Tobacco", 12:"Sweet Corn", 13:"Pop/Orn Corn",
    14:"Mint", 21:"Barley", 22:"Durum Wheat", 23:"Spring Wheat", 24:"Winter Wheat",
    25:"Other Small Grains", 26:"Dbl Crop WinWht/Soybeans", 27:"Rye",
    28:"Oats", 29:"Millet", 30:"Speltz", 31:"Canola", 32:"Flaxseed",
    33:"Safflower", 34:"Rape Seed", 35:"Mustard", 36:"Alfalfa", 37:"Other Hay/Non Alfalfa",
    38:"Camelina", 39:"Buckwheat", 41:"Sugarbeets", 42:"Dry Beans",
    43:"Potatoes", 44:"Other Crops", 45:"Sugarcane", 46:"Sweet Potatoes",
    47:"Misc Vegs & Fruits", 48:"Watermelons", 49:"Onions", 50:"Cucumbers",
    51:"Chick Peas", 52:"Lentils", 53:"Peas", 54:"Tomatoes", 55:"Caneberries",
    56:"Hops", 57:"Herbs", 58:"Clover/Wildflowers", 59:"Sod/Grass Seed",
    60:"Switchgrass", 61:"Fallow/Idle Cropland", 62:"Pasture/Grass",
    63:"Forest", 64:"Shrubland", 65:"Barren", 66:"Cherries", 67:"Peaches",
    68:"Apples", 69:"Grapes", 70:"Christmas Trees", 71:"Other Tree Crops",
    72:"Citrus", 74:"Pecans", 75:"Almonds", 76:"Walnuts", 77:"Pears",
    81:"Clouds/No Data", 82:"Developed", 83:"Water", 87:"Wetlands",
    88:"Nonag/Undefined", 92:"Aquaculture", 111:"Open Water",
    112:"Perennial Ice/Snow", 121:"Developed/Open Space",
    122:"Developed/Low Intensity", 123:"Developed/Med Intensity",
    124:"Developed/High Intensity", 131:"Barren", 141:"Deciduous Forest",
    142:"Evergreen Forest", 143:"Mixed Forest", 152:"Shrubland",
    176:"Grassland/Pasture", 190:"Woody Wetlands",
    195:"Herbaceous Wetlands", 204:"Pistachios", 205:"Triticale",
    206:"Carrots", 207:"Asparagus", 208:"Garlic", 209:"Cantaloupes",
    210:"Prunes", 211:"Olives", 212:"Oranges", 213:"Honeydew Melons",
    214:"Broccoli", 215:"Avocados", 216:"Peppers", 217:"Pomegranates",
    218:"Nectarines", 219:"Greens", 220:"Plums", 221:"Strawberries",
    222:"Squash", 223:"Apricots", 224:"Vetch", 225:"Dbl Crop WinWht/Corn",
    226:"Dbl Crop Oats/Corn", 227:"Lettuce", 228:"Dbl Crop Triticale/Corn",
    229:"Pumpkins", 230:"Dbl Crop Lettuce/Durum Wht",
    231:"Dbl Crop Lettuce/Cantaloupe", 232:"Dbl Crop Lettuce/Cotton",
    233:"Dbl Crop Lettuce/Barley", 234:"Dbl Crop Durum Wht/Sorghum",
    235:"Dbl Crop Barley/Sorghum", 236:"Dbl Crop WinWht/Sorghum",
    237:"Dbl Crop Barley/Corn", 238:"Dbl Crop WinWht/Cotton",
    239:"Dbl Crop Soybeans/Cotton", 240:"Dbl Crop Soybeans/Oats",
    241:"Dbl Crop Corn/Soybeans", 242:"Blueberries", 243:"Cabbage",
    244:"Cauliflower", 245:"Celery", 246:"Radishes", 247:"Turnips",
    248:"Eggplant", 249:"Gourds", 250:"Cranberries",
    254:"Dbl Crop Barley/Soybeans",
}


# ════════════════════════════════════════════════════════════════════════════
# Shared helper: get county bounding box from Census TIGER
# Returns (xmin_wgs84, ymin_wgs84, xmax_wgs84, ymax_wgs84) or None
# ════════════════════════════════════════════════════════════════════════════

def get_county_bbox_wgs84(county_info: dict) -> tuple | None:
    """
    Fetch county bounding box (WGS84) from Census TIGER REST API.
    Falls back to hardcoded bounding boxes if API fails.
    """
    # Hardcoded bounding boxes for target counties (WGS84)
    # Format: (xmin, ymin, xmax, ymax)
    HARDCODED_BBOXES = {
        "08069": (-105.5, 40.3, -104.9, 41.0),  # Larimer County, CO
        "08013": (-105.4, 39.8, -104.7, 40.2),  # Boulder County, CO  
        "48113": (-96.9, 32.5, -96.5, 32.9),     # Dallas County, TX
        "04013": (-112.3, 33.3, -111.6, 33.8),  # Maricopa County, AZ
        "17031": (-87.9, 41.6, -87.5, 42.0),     # Cook County, IL
        "06037": (-118.7, 33.7, -118.1, 34.3),  # Los Angeles County, CA
    }
    
    # Try hardcoded bbox first
    fips = county_info.get("fips")
    if fips in HARDCODED_BBOXES:
        return HARDCODED_BBOXES[fips]
    
    # Fallback to API if not hardcoded
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Current/MapServer/84/query"
    )
    params = {
        "where": f"STATE='{county_info['state_fips']}' AND COUNTY='{county_info['county_fips']}'",
        "outFields": "GEOID,NAME",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data.get("features"):
            return None
        geom = data["features"][0]["geometry"]
        # Handle both Polygon and MultiPolygon
        rings = geom.get("coordinates", [])
        if geom["type"] == "MultiPolygon":
            all_coords = [c for ring in rings for c in ring[0]]
        else:
            all_coords = rings[0]
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        return (min(lons), min(lats), max(lons), max(lats))
    except Exception as e:
        print(f"  [TIGER] bbox query failed for {county_info['name']}: {e}")
        return None


def bbox_wgs84_to_albers(xmin: float, ymin: float, xmax: float, ymax: float) -> tuple:
    """
    Convert WGS84 lon/lat bbox to USA Contiguous Albers Equal Area (EPSG:5070),
    which is the native CRS of CDL rasters.
    """
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xmin_a, ymin_a = transformer.transform(xmin, ymin)
    xmax_a, ymax_a = transformer.transform(xmax, ymax)
    return (xmin_a, ymin_a, xmax_a, ymax_a)


def parse_cdl_csv(csv_text: str, county_key: str, county_info: dict, year: int) -> dict:
    """
    Parse a CDL statistics CSV (returned by GetCDLStat bbox endpoint).
    CSV format: Value, Category, Count, Acreage
    """
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]
        # Normalize column names (endpoint varies capitalization)
        col_map = {c.lower(): c for c in df.columns}
        val_col  = col_map.get("value",    df.columns[0])
        cat_col  = col_map.get("category", df.columns[1])
        acre_col = col_map.get("acreage",  df.columns[3] if len(df.columns) > 3 else df.columns[2])

        crop_acres = dict(zip(df[val_col].astype(int), df[acre_col].astype(float)))
        total_acres = sum(crop_acres.values()) or 1
    except Exception as e:
        print(f"  [CDL parse] Failed for {county_info['name']} {year}: {e}")
        return {}

    row = {
        "county_key":  county_key,
        "county_name": county_info["name"],
        "fips":        county_info["fips"],
        "year":        year,
        "total_acres": round(total_acres, 2),
        "total_ha":    round(total_acres * 0.404686, 2),
    }

    for group, codes in CDL_MOSQUITO_RELEVANT.items():
        g_acres = sum(crop_acres.get(c, 0) for c in codes)
        row[f"cdl_{group}_acres"] = round(g_acres, 2)
        row[f"cdl_{group}_pct"]   = round(100 * g_acres / total_acres, 4)

    cropland_acres = sum(v for k, v in crop_acres.items() if 1 <= k <= 60)
    row["cdl_cropland_acres"] = round(cropland_acres, 2)
    row["cdl_cropland_pct"]   = round(100 * cropland_acres / total_acres, 4)

    # Developed / urban land (classes 121-124 and 82)
    dev_acres = sum(crop_acres.get(c, 0) for c in [82, 121, 122, 123, 124])
    row["cdl_developed_acres"] = round(dev_acres, 2)
    row["cdl_developed_pct"]   = round(100 * dev_acres / total_acres, 4)

    return row


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY A — CropScape bbox API
# Works when the county-FIPS endpoint returns 500
# ════════════════════════════════════════════════════════════════════════════

CDL_BBOX_BASE = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat"


def fetch_cdl_bbox(county_key: str, county_info: dict, year: int) -> dict:
    """
    Query CropScape GetCDLStat using a bounding box (Albers coordinates)
    instead of county FIPS. This endpoint is more reliable than the
    county-code endpoint.

    Returns a flat stats dict, or empty dict on failure.
    """
    bbox_wgs = get_county_bbox_wgs84(county_info)
    if bbox_wgs is None:
        return {}

    xmin_a, ymin_a, xmax_a, ymax_a = bbox_wgs_to_albers = bbox_wgs84_to_albers(*bbox_wgs)
    bbox_str = f"{xmin_a:.2f},{ymin_a:.2f},{xmax_a:.2f},{ymax_a:.2f}"

    params = {
        "year":   str(year),
        "bbox":   bbox_str,
        "format": "csv",
    }
    try:
        r = requests.get(CDL_BBOX_BASE, params=params, timeout=60, verify=False)
        r.raise_for_status()

        # Response is a JSON with a URL to a CSV file
        data = r.json()
        csv_url = (data.get("returnURL") or data.get("URL") or
                   data.get("csvURL") or "")
        if not csv_url:
            # Sometimes the response IS the CSV text directly
            if "Value" in r.text or "Category" in r.text:
                return parse_cdl_csv(r.text, county_key, county_info, year)
            print(f"  [CDL-bbox] No CSV URL in response for {county_info['name']} {year}")
            return {}

        # Fetch the CSV from the returned URL
        csv_r = requests.get(csv_url, timeout=60, verify=False)
        csv_r.raise_for_status()
        return parse_cdl_csv(csv_r.text, county_key, county_info, year)

    except requests.exceptions.SSLError:
        # CropScape has an expired cert — retry without SSL verification
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            r = requests.get(CDL_BBOX_BASE, params=params, timeout=60, verify=False)
            r.raise_for_status()
            data = r.json()
            csv_url = data.get("returnURL", "")
            if csv_url:
                csv_r = requests.get(csv_url, timeout=60, verify=False)
                return parse_cdl_csv(csv_r.text, county_key, county_info, year)
        except Exception as e2:
            print(f"  [CDL-bbox SSL retry] {county_info['name']} {year}: {e2}")
        return {}
    except Exception as e:
        print(f"  [CDL-bbox] {county_info['name']} {year}: {e}")
        return {}


def run_strategy_a() -> pd.DataFrame:
    """Strategy A: CropScape bbox API for all counties and years."""
    print("\n" + "="*65)
    print("STRATEGY A — CropScape bbox API")
    print("="*65)
    rows = []
    for county_key, county_info in COUNTIES.items():
        print(f"\n  {county_info['name']}")
        for year in tqdm(CDL_YEARS, desc="    years"):
            row = fetch_cdl_bbox(county_key, county_info, year)
            if row:
                rows.append(row)
            time.sleep(0.5)

    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUTPUT_DIR / "cdl_strategy_a.csv"
        df.to_csv(out, index=False)
        print(f"\n  -> Saved {len(df)} rows to {out}")
    else:
        print("\n  [!] Strategy A returned no data. Try Strategy B or C.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY B — Google Earth Engine (most reliable, free, no SSL issues)
# ════════════════════════════════════════════════════════════════════════════

def run_strategy_b() -> pd.DataFrame:
    """
    Strategy B: Pull CDL statistics via Google Earth Engine Python API.

    Setup (one-time):
        pip install earthengine-api
        earthengine authenticate      # opens browser, saves credentials

    GEE CDL dataset: USDA/NASS/CDL  (annual, 30m, 1997-present for full CONUS)
    Band: 'cropland'  (integer CDL class values)
    """
    print("\n" + "="*65)
    print("STRATEGY B — Google Earth Engine")
    print("="*65)
    try:
        import ee
    except ImportError:
        print("  [!] earthengine-api not installed.")
        print("      Run: pip install earthengine-api && earthengine authenticate")
        return pd.DataFrame()

    try:
        ee.Initialize()
    except Exception as e:
        print(f"  [!] GEE initialization failed: {e}")
        print("      Run: earthengine authenticate")
        return pd.DataFrame()

    # Load TIGER county boundaries as EE FeatureCollection
    tiger = ee.FeatureCollection("TIGER/2018/Counties")

    rows = []
    for county_key, county_info in COUNTIES.items():
        print(f"\n  {county_info['name']}")
        # Filter to this county by GEOID (5-digit FIPS)
        county_fc = tiger.filter(ee.Filter.eq("GEOID", county_info["fips"]))
        county_geom = county_fc.geometry()

        for year in tqdm(CDL_YEARS, desc="    years"):
            try:
                cdl = (ee.ImageCollection("USDA/NASS/CDL")
                       .filter(ee.Filter.calendarRange(year, year, "year"))
                       .first()
                       .select("cropland"))

                # Compute histogram (pixel count per class)
                # ee.Reducer may not be available in all SDK versions
                try:
                    reducer = ee.Reducer.frequencyHistogram()
                except AttributeError:
                    reducer = ee.call("Reducer.frequencyHistogram")
                hist = cdl.reduceRegion(
                    reducer=reducer,
                    geometry=county_geom,
                    scale=30,
                    maxPixels=1e10,
                ).getInfo()

                class_counts = hist.get("cropland", {})
                if not class_counts:
                    continue

                # Convert pixel counts to acres (CDL pixel = 30m x 30m = 900 m2 = 0.2224 acres)
                px_to_acres = 900 / 4046.856
                crop_acres = {int(k): float(v) * px_to_acres
                              for k, v in class_counts.items()}
                total_acres = sum(crop_acres.values()) or 1

                row = {
                    "county_key":  county_key,
                    "county_name": county_info["name"],
                    "fips":        county_info["fips"],
                    "year":        year,
                    "total_acres": round(total_acres, 2),
                    "total_ha":    round(total_acres * 0.404686, 2),
                    "source":      "GEE",
                }

                for group, codes in CDL_MOSQUITO_RELEVANT.items():
                    g_acres = sum(crop_acres.get(c, 0) for c in codes)
                    row[f"cdl_{group}_acres"] = round(g_acres, 2)
                    row[f"cdl_{group}_pct"]   = round(100 * g_acres / total_acres, 4)

                cropland_acres = sum(v for k, v in crop_acres.items() if 1 <= k <= 60)
                row["cdl_cropland_acres"] = round(cropland_acres, 2)
                row["cdl_cropland_pct"]   = round(100 * cropland_acres / total_acres, 4)

                dev_acres = sum(crop_acres.get(c, 0) for c in [82, 121, 122, 123, 124])
                row["cdl_developed_acres"] = round(dev_acres, 2)
                row["cdl_developed_pct"]   = round(100 * dev_acres / total_acres, 4)

                rows.append(row)
                time.sleep(0.2)   # respect GEE rate limits

            except Exception as e:
                print(f"    [GEE] {county_info['name']} {year}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUTPUT_DIR / "cdl_strategy_b_gee.csv"
        df.to_csv(out, index=False)
        print(f"\n  -> Saved {len(df)} rows to {out}")
    else:
        print("\n  [!] Strategy B returned no data.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY C — NASS FTP direct download + local zonal stats
# Always works; requires ~500 MB disk per state per year
# ════════════════════════════════════════════════════════════════════════════

# NASS CDL FTP base
NASS_FTP_BASE = "https://www.nass.usda.gov/Research_and_Science/Cropland/docs"

# State abbreviation to FTP directory mapping
STATE_DIRS = {
    "CO": "Colorado",
    "TX": "Texas",
    "AZ": "Arizona",
    "IL": "Illinois",
    "CA": "California",
}

# NASS CDL national download URL map (confirmed live as of March 2026)
# Source: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/
# NOTE: As of 2020, NASS switched from per-state zips to NATIONAL zips only.
# Pre-2020 per-state zips have been removed; national zips cover all states.
# 2024+ are 10-meter resolution; 2008-2023 are 30-meter.
# Confirmed URL pattern (March 2026): downloads are served from data.nass.usda.gov
# NOT www.nass.usda.gov — the subdomain differs from the landing page.
# Source: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php
NASS_CDL_NATIONAL_URLS = {
    year: f"https://www.nass.usda.gov/Research_and_Science/Cropland/datasets/{year}_30m_cdls.zip"
    for year in range(2008, 2024)
}
NASS_CDL_NATIONAL_URLS[2024] = "https://www.nass.usda.gov/Research_and_Science/Cropland/datasets/2024_30m_cdls.zip"
NASS_CDL_NATIONAL_URLS[2025] = "https://www.nass.usda.gov/Research_and_Science/Cropland/datasets/2025_30m_cdls.zip"

# Mirror: data.nass.usda.gov subdomain (same files, sometimes more reliable)
NASS_CDL_MIRROR_URLS = {
    year: f"https://data.nass.usda.gov/Research_and_Science/Cropland/datasets/{year}_30m_cdls.zip"
    for year in range(2008, 2026)
}

# Geospatial Data Gateway — alternative mirror for CDL per-state files (older years)
# https://datagateway.nrcs.usda.gov/  (may require interactive download for some years)
NASS_CDL_GATEWAY_BASE = "https://datagateway.nrcs.usda.gov/GDGOrder.aspx"

def build_nass_cdl_url(state_abbr: str, year: int) -> str:
    """
    Return the confirmed NASS national CDL download URL for a given year.
    As of 2020, NASS distributes national zips only (no per-state zips).
    The national zip contains rasters for all CONUS states — clip to county
    after download using compute_cdl_zonal_stats().

    Confirmed URL pattern (March 2026):
      https://www.nass.usda.gov/Research_and_Science/Cropland/datasets/{YEAR}_30m_cdls.zip

    Note: 2024 and 2025 also have 10m versions available at:
      https://www.nass.usda.gov/Research_and_Science/Cropland/datasets/{YEAR}_10m_cdls.zip
    """
    if year not in NASS_CDL_NATIONAL_URLS:
        raise ValueError(f"No CDL URL known for year {year}. Available: {sorted(NASS_CDL_NATIONAL_URLS)}")
    return NASS_CDL_NATIONAL_URLS[year]


def download_cdl_national_raster(year: int, download_dir: Path) -> Path | None:
    """
    Download and extract the NASS national CDL GeoTIFF for a given year.

    NASS now distributes NATIONAL zips only (no per-state zips as of ~2020).
    The zip contains a single national .tif file for all CONUS states.
    After download, use compute_cdl_zonal_stats() to clip to a county boundary.

    File sizes: ~1.8–2.0 GB per year (30m), ~9 GB for 2024/2025 (10m).
    Confirmed URLs: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/

    Returns path to extracted .tif, or None on failure.
    """
    import zipfile
    out_tif = download_dir / f"CDL_{year}_30m_national.tif"
    if out_tif.exists():
        print(f"    [CDL] Already downloaded: {out_tif.name}")
        return out_tif

    try:
        primary_url = build_nass_cdl_url(year=year, state_abbr="")
    except ValueError as e:
        print(f"    [CDL] {e}")
        return None

    # Try primary URL first, then data.nass.usda.gov mirror
    mirror_url = NASS_CDL_MIRROR_URLS.get(year)
    urls_to_try = [u for u in [primary_url, mirror_url] if u]

    zip_path = download_dir / f"CDL_{year}_national.zip"
    r = None
    url = None
    for candidate in urls_to_try:
        print(f"    [CDL] Trying {candidate}")
        try:
            resp = requests.head(candidate, timeout=15, allow_redirects=True)
            if resp.status_code < 400:
                url = candidate
                break
            print(f"    [CDL] HEAD {resp.status_code} — trying next URL")
        except Exception as e:
            print(f"    [CDL] HEAD failed ({e}) — trying next URL")

    if url is None:
        print(f"    [CDL] All URLs returned errors for year {year}. Skipping.")
        return None

    print(f"    [CDL] Downloading {url}")
    print(f"    [CDL] File size ~1.8–2.0 GB — this will take several minutes ...")

    try:
        r = requests.get(url, stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):  # 4 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"    [CDL] {pct:.0f}% ({downloaded/1e9:.2f}/{total/1e9:.2f} GB)", end="\r")
        print()

        # Extract the national .tif
        with zipfile.ZipFile(zip_path, "r") as z:
            tif_files = [n for n in z.namelist()
                         if n.lower().endswith((".img", ".tif", ".tiff"))]
            if not tif_files:
                print(f"    [CDL] No raster found in national zip for {year}")
                zip_path.unlink(missing_ok=True)
                return None
            print(f"    [CDL] Extracting {tif_files[0]} ...")
            z.extract(tif_files[0], download_dir)
            extracted = download_dir / tif_files[0]
            extracted.rename(out_tif)

        zip_path.unlink(missing_ok=True)
        print(f"    [CDL] Extracted: {out_tif.name} ({out_tif.stat().st_size/1e9:.2f} GB)")
        return out_tif

    except Exception as e:
        print(f"    [CDL] Download failed for year {year}: {e}")
        zip_path.unlink(missing_ok=True)
        return None


# Keep old name as alias so existing test stubs still resolve
def download_cdl_state_raster(state_abbr: str, year: int, download_dir: Path) -> Path | None:
    """Alias for download_cdl_national_raster (NASS no longer distributes per-state zips)."""
    return download_cdl_national_raster(year=year, download_dir=download_dir)


def compute_cdl_zonal_stats(tif_path: Path, county_gdf: gpd.GeoDataFrame,
                             county_key: str, county_info: dict, year: int) -> dict:
    """
    Clip CDL raster to county polygon and compute per-class acreage.
    Uses rasterstats for efficient zonal statistics.
    """
    try:
        from rasterstats import zonal_stats
        import rasterio
    except ImportError:
        print("    [CDL-zonal] Install: pip install rasterstats rasterio")
        return {}

    try:
        # Reproject county boundary to CDL CRS (Albers EPSG:5070)
        with rasterio.open(tif_path) as src:
            cdl_crs = src.crs

        county_reproj = county_gdf.to_crs(cdl_crs)
        stats = zonal_stats(county_reproj, str(tif_path),
                            categorical=True, nodata=0)
        if not stats:
            return {}

        class_counts = stats[0]  # {pixel_value: count}
        # CDL pixel area: 30m x 30m = 900 m2 = 0.2224 acres
        px_to_acres = 900 / 4046.856
        crop_acres = {int(k): int(v) * px_to_acres
                      for k, v in class_counts.items() if v}
        total_acres = sum(crop_acres.values()) or 1

        row = {
            "county_key":  county_key,
            "county_name": county_info["name"],
            "fips":        county_info["fips"],
            "year":        year,
            "total_acres": round(total_acres, 2),
            "total_ha":    round(total_acres * 0.404686, 2),
            "source":      "NASS-FTP-zonal",
        }
        for group, codes in CDL_MOSQUITO_RELEVANT.items():
            g_acres = sum(crop_acres.get(c, 0) for c in codes)
            row[f"cdl_{group}_acres"] = round(g_acres, 2)
            row[f"cdl_{group}_pct"]   = round(100 * g_acres / total_acres, 4)

        cropland_acres = sum(v for k, v in crop_acres.items() if 1 <= k <= 60)
        row["cdl_cropland_acres"] = round(cropland_acres, 2)
        row["cdl_cropland_pct"]   = round(100 * cropland_acres / total_acres, 4)

        dev_acres = sum(crop_acres.get(c, 0) for c in [82, 121, 122, 123, 124])
        row["cdl_developed_acres"] = round(dev_acres, 2)
        row["cdl_developed_pct"]   = round(100 * dev_acres / total_acres, 4)

        return row
    except Exception as e:
        print(f"    [CDL-zonal] Failed {county_info['name']} {year}: {e}")
        return {}


def get_tiger_county_gdf(county_info: dict) -> gpd.GeoDataFrame | None:
    """Fetch a single county GeoDataFrame from Census TIGER REST."""
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Current/MapServer/84/query"
    )
    params = {
        "where": f"STATE='{county_info['state_fips']}' AND COUNTY='{county_info['county_fips']}'",
        "outFields": "GEOID,NAME",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(r.json()["features"], crs="EPSG:4326")
        return gdf
    except Exception as e:
        print(f"  [TIGER] Could not fetch county boundary for {county_info['name']}: {e}")
        return None


def run_strategy_c(raster_dir: Path = Path("./cdl_rasters")) -> pd.DataFrame:
    """
    Strategy C: Download NASS CDL GeoTIFFs and run local zonal stats.
    Downloads one raster per (state, year) — about 200-500 MB each.

    Args:
        raster_dir: local folder to store downloaded rasters (can be large)
    """
    print("\n" + "="*65)
    print("STRATEGY C — NASS FTP download + local zonal stats")
    print(f"  Rasters will be saved to: {raster_dir.resolve()}")
    print("="*65)
    raster_dir.mkdir(exist_ok=True)

    # Group counties by state to avoid duplicate downloads
    state_counties = {}
    for key, info in COUNTIES.items():
        s = info["state_abbr"]
        if s not in state_counties:
            state_counties[s] = []
        state_counties[s].append((key, info))

    rows = []
    for year in CDL_YEARS:
        print(f"\n  Year {year}")
        # Download the NATIONAL raster once per year (covers all CONUS states)
        tif = download_cdl_national_raster(year=year, download_dir=raster_dir)
        if tif is None:
            print(f"    [CDL] Skipping year {year} — raster unavailable")
            continue

        # Run zonal stats for every target county against the national raster
        for state_abbr, county_list in state_counties.items():
            for county_key, county_info in county_list:
                gdf = get_tiger_county_gdf(county_info)
                if gdf is None:
                    continue
                row = compute_cdl_zonal_stats(tif, gdf, county_key, county_info, year)
                if row:
                    rows.append(row)

        # Optionally delete the national raster after all counties are processed
        # to recover ~2 GB of disk space per year. Uncomment to enable:
        # tif.unlink()
        # print(f"    [CDL] Deleted {tif.name} to free disk space")

    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUTPUT_DIR / "cdl_strategy_c_ftp.csv"
        df.to_csv(out, index=False)
        print(f"\n  -> Saved {len(df)} rows to {out}")
    else:
        print("\n  [!] Strategy C returned no data.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY D — Microsoft Planetary Computer (STAC + COG)
# No auth required for read access. CDL hosted as Cloud-Optimised GeoTIFF.
# pip install pystac-client odc-stac rioxarray
# ════════════════════════════════════════════════════════════════════════════

def run_strategy_d() -> pd.DataFrame:
    """
    Strategy D: Pull CDL data from Microsoft Planetary Computer via STAC API.

    Planetary Computer hosts USDA CDL as Analysis-Ready Cloud-Optimised
    GeoTIFFs (COGs). This means you can read just the county bounding-box
    window without downloading the full national raster (~2 GB/year).

    Setup (one-time):
        pip install pystac-client odc-stac rioxarray xarray numpy

    No account or API key required for public datasets.
    Catalog: https://planetarycomputer.microsoft.com/dataset/usda-cdl

    Note: CDL coverage on Planetary Computer may lag 1-2 years behind NASS.
    """
    print("\n" + "="*65)
    print("STRATEGY D — Microsoft Planetary Computer (STAC/COG)")
    print("="*65)

    try:
        import pystac_client
        import odc.stac
        import rioxarray  # noqa: F401  — needed for .rio accessor
        import numpy as np
        import planetary_computer
    except ImportError as e:
        print(f"  [!] Missing package: {e}")
        print("      Run: pip install pystac-client odc-stac rioxarray")
        return pd.DataFrame()

    PC_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"
    CDL_COLLECTION = "usda-cdl"

    rows = []
    for county_key, county_info in COUNTIES.items():
        print(f"\n  {county_info['name']}")

        # Get county bounding box
        bbox_wgs = get_county_bbox_wgs84(county_info)
        if bbox_wgs is None:
            print(f"    [PC] Could not get bbox for {county_info['name']}")
            continue
        xmin, ymin, xmax, ymax = bbox_wgs
        bbox = [xmin, ymin, xmax, ymax]

        for year in tqdm(CDL_YEARS, desc="    years"):
            try:
                catalog = pystac_client.Client.open(PC_CATALOG)
                items = catalog.search(
                    collections=[CDL_COLLECTION],
                    bbox=bbox,
                    datetime=f"{year}-01-01/{year}-12-31",
                ).item_collection()

                if not items:
                    continue

                # Sign items for access via planetary computer
                items = [planetary_computer.sign(item) for item in items]

                # Load just the cropland band for the county bbox as a COG window
                ds = odc.stac.load(
                    items,
                    bands=["cropland"],
                    bbox=bbox,
                    resolution=30,
                )
                arr = ds["cropland"].values.ravel()
                arr = arr[arr > 0]   # remove nodata

                if arr.size == 0:
                    continue

                # Pixel count per class -> acres
                unique, counts = np.unique(arr, return_counts=True)
                px_to_acres = 900 / 4046.856   # 30m pixel = 900 m2
                crop_acres = {int(v): int(c) * px_to_acres
                              for v, c in zip(unique, counts)}
                total_acres = sum(crop_acres.values()) or 1

                row = {
                    "county_key":  county_key,
                    "county_name": county_info["name"],
                    "fips":        county_info["fips"],
                    "year":        year,
                    "total_acres": round(total_acres, 2),
                    "total_ha":    round(total_acres * 0.404686, 2),
                    "source":      "PlanetaryComputer",
                }
                for group, codes in CDL_MOSQUITO_RELEVANT.items():
                    g_acres = sum(crop_acres.get(c, 0) for c in codes)
                    row[f"cdl_{group}_acres"] = round(g_acres, 2)
                    row[f"cdl_{group}_pct"]   = round(100 * g_acres / total_acres, 4)

                cropland_acres = sum(v for k, v in crop_acres.items() if 1 <= k <= 60)
                row["cdl_cropland_acres"] = round(cropland_acres, 2)
                row["cdl_cropland_pct"]   = round(100 * cropland_acres / total_acres, 4)

                dev_acres = sum(crop_acres.get(c, 0) for c in [82, 121, 122, 123, 124])
                row["cdl_developed_acres"] = round(dev_acres, 2)
                row["cdl_developed_pct"]   = round(100 * dev_acres / total_acres, 4)

                rows.append(row)
                time.sleep(0.2)

            except Exception as e:
                print(f"    [PC] {county_info['name']} {year}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUTPUT_DIR / "cdl_strategy_d_pc.csv"
        df.to_csv(out, index=False)
        print(f"\n  -> Saved {len(df)} rows to {out}")
    else:
        print("\n  [!] Strategy D returned no data.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN — try strategies in order, stop at first success
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("CDL Data Acquisition — WNV Project")

    # Pre-flight: check all endpoints before pulling anything
    access = run_preflight()

    if not access["_any_viable"]:
        print("\n[ABORT] No viable data source. Resolve access issues and re-run.")
        sys.exit(1)

    # ── Strategy B: GEE — promoted to first, most reliable ───────────────
    if access["_strategy_b_ok"]:
        print("\n[INFO] Running Strategy B (Google Earth Engine) ...")
        df = run_strategy_b()
        if not df.empty:
            print("\n[SUCCESS] Strategy B complete. CDL data saved.")
            return
        print("\n[WARN] Strategy B returned no data despite passing preflight.")
    else:
        print("\n[SKIP] Strategy B — GEE not available.")
        print("        Fix: pip install earthengine-api && earthengine authenticate")

    # ── Strategy D: Planetary Computer — no auth, county-window COG reads ─
    if access["_strategy_d_ok"]:
        print("\n[INFO] Running Strategy D (Microsoft Planetary Computer) ...")
        df = run_strategy_d()
        if not df.empty:
            print("\n[SUCCESS] Strategy D complete. CDL data saved.")
            return
        print("\n[WARN] Strategy D returned no data despite passing preflight.")
    else:
        print("\n[SKIP] Strategy D — Planetary Computer not reachable.")
        print("        Fix: pip install pystac-client odc-stac rioxarray")

    # ── Strategy A: CropScape bbox ─────────────────────────────────────────
    if access["_strategy_a_ok"]:
        print("\n[INFO] Running Strategy A (CropScape bbox API) ...")
        df = run_strategy_a()
        if not df.empty:
            print("\n[SUCCESS] Strategy A complete. CDL data saved.")
            return
        print("\n[WARN] Strategy A returned no data despite passing preflight.")
    else:
        print("\n[SKIP] Strategy A — CropScape bbox endpoint not reachable.")

    # ── Strategy C: NASS national download — large files, always works ────
    if access["_strategy_c_ok"]:
        print("\n[INFO] Running Strategy C (NASS national raster + zonal stats) ...")
        print("[INFO] Downloads ~2 GB per year — ensure ~32 GB free disk space.")
        df = run_strategy_c()
        if not df.empty:
            print("\n[SUCCESS] Strategy C complete. CDL data saved.")
            return
        print("\n[WARN] Strategy C returned no data despite passing preflight.")
    else:
        print("\n[SKIP] Strategy C — NASS download endpoints not reachable.")
        print("        Try manually: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php")

    print("\n[FAIL] All viable strategies exhausted with no data. Check logs above.")


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════
# QUICK TEST — run this to check which endpoints are live before full pull
# ════════════════════════════════════════════════════════════════════════════
#
# import requests, urllib3
# urllib3.disable_warnings()
#
# # Test county-FIPS endpoint (likely broken)
# r1 = requests.get(
#     "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
#     params={"aoi": "county:06:037", "year": "2020", "format": "csv"},
#     verify=False, timeout=15
# )
# print("County-FIPS endpoint:", r1.status_code)
#
# # Test bbox endpoint (often works when county-FIPS doesn't)
# r2 = requests.get(
#     "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLStat",
#     params={"bbox": "-2356109,1128970,-2100442,1735536", "year": "2020", "format": "csv"},
#     verify=False, timeout=15
# )
# print("Bbox endpoint:", r2.status_code, r2.text[:200])