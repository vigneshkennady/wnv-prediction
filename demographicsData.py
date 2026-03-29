"""
Demographics Data Acquisition — WNV County-Level Prediction Project
====================================================================
Counties : Larimer CO (08069) | Boulder CO (08013) | Dallas TX (48113)
           Maricopa AZ (04013) | Cook IL (17031) | Los Angeles CA (06037)

Variables:
  ACS 5-Year Estimates (Census API):
    B01003  — Total population
    B01001  — Sex by age (age 65+ derived)
    B19013  — Median household income
    S1701   — Poverty rate
    B25001  — Total housing units
    DP05    — Population (cross-check + density denominator)

  SAIPE (Small Area Income & Poverty Estimates):
    SAEMHI_PT      — Median household income (more timely than ACS)
    SAEPOVRTALL_PT — Poverty rate
    SAEPOVALL_PT   — Poverty count

  TIGER/Line (county area for population and housing density):
    AREALAND — Land area in sq metres

Sources:
  Census API    https://api.census.gov/data/
  SAIPE API     https://api.census.gov/data/timeseries/poverty/saipe
  TIGER REST    https://tigerweb.geo.census.gov/arcgis/rest/services/

Python 3.9 compatible — uses `from __future__ import annotations`
No geopandas required.

Requirements:
    pip install requests pandas tqdm

Usage:
    python demographics_data.py
    Outputs written to ./output/:
      demographics_acs.csv
      demographics_saipe.csv
      demographics_combined.csv

API Key (free):
    https://api.census.gov/data/key_signup.html
    export CENSUS_API_KEY=your_key_here
    (anonymous requests are rate-limited to ~500/day)
====================================================================
"""

from __future__ import annotations

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Target counties ──────────────────────────────────────────────────────────
COUNTIES = {
    "Larimer_CO":    {"fips": "08069", "state_fips": "08", "county_fips": "069", "name": "Larimer County, CO"},
    "Boulder_CO":    {"fips": "08013", "state_fips": "08", "county_fips": "013", "name": "Boulder County, CO"},
    "Dallas_TX":     {"fips": "48113", "state_fips": "48", "county_fips": "113", "name": "Dallas County, TX"},
    "Maricopa_AZ":   {"fips": "04013", "state_fips": "04", "county_fips": "013", "name": "Maricopa County, AZ"},
    "Cook_IL":       {"fips": "17031", "state_fips": "17", "county_fips": "031", "name": "Cook County, IL"},
    "LosAngeles_CA": {"fips": "06037", "state_fips": "06", "county_fips": "037", "name": "Los Angeles County, CA"},
}

# ── Survey years ──────────────────────────────────────────────────────────────
# ACS 5-year: first full release is 2009 (covers 2005-2009), latest is 2023
ACS_YEARS   = list(range(2009, 2024))
# SAIPE: available annually from 1989; WNV-relevant window starts ~2000
SAIPE_YEARS = list(range(2000, 2024))

# ── Census API key ────────────────────────────────────────────────────────────
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")


# ════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT ACCESS CHECKS
# ════════════════════════════════════════════════════════════════════════════

ACCESS_CHECKS = {
    "census_acs": {
        "label":       "Census ACS 5-Year API (B-tables)",
        "url":         "https://api.census.gov/data/2022/acs/acs5",
        "params":      {"get": "B01003_001E", "for": "county:069", "in": "state:08"},
        "expect_text": "B01003_001E",
    },
    "census_subject": {
        "label":       "Census ACS Subject Tables (S-tables, poverty rate)",
        "url":         "https://api.census.gov/data/2022/acs/acs5/subject",
        "params":      {"get": "S1701_C03_001E", "for": "county:069", "in": "state:08"},
        "expect_text": "S1701_C03_001E",
    },
    "census_profile": {
        "label":       "Census ACS Data Profile (DP-tables)",
        "url":         "https://api.census.gov/data/2022/acs/acs5/profile",
        "params":      {"get": "DP05_0001E", "for": "county:069", "in": "state:08"},
        "expect_text": "DP05_0001E",
    },
    "census_saipe": {
        "label":       "Census SAIPE API (income & poverty timeseries)",
        "url":         "https://api.census.gov/data/timeseries/poverty/saipe",
        "params":      {"get": "SAEMHI_PT,SAEPOVRTALL_PT", "for": "county:069",
                        "in": "state:08", "time": "2022"},
        "expect_text": "SAEMHI_PT",
    },
    "tiger_rest": {
        "label":       "Census TIGER REST API (county land area)",
        "url":         "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Current/MapServer/84/query",
        "params":      {"where": "GEOID='08069'", "outFields": "GEOID,NAME,AREALAND",
                        "returnGeometry": "false", "f": "json"},
        "expect_text": "AREALAND",
    },
}


def run_preflight() -> dict:
    """
    Test all Census API endpoints before pulling data.
    Returns dict with per-check booleans and strategy flags.
    """
    print("\n" + "=" * 65)
    print("PRE-FLIGHT ACCESS CHECKS")
    print("=" * 65)

    results: dict = {}
    for check_id, cfg in ACCESS_CHECKS.items():
        params = dict(cfg["params"])
        if CENSUS_API_KEY:
            params["key"] = CENSUS_API_KEY
        try:
            r = requests.get(cfg["url"], params=params, timeout=20)
            if r.status_code >= 400:
                ok, detail = False, f"HTTP {r.status_code}"
            elif cfg.get("expect_text") and cfg["expect_text"] not in r.text:
                ok, detail = False, f"HTTP {r.status_code} — expected key not in response"
            else:
                ok, detail = True, f"HTTP {r.status_code}"
        except requests.exceptions.Timeout:
            ok, detail = False, "Timed out after 20s"
        except requests.exceptions.ConnectionError as exc:
            ok, detail = False, f"Connection error: {exc}"
        except Exception as exc:
            ok, detail = False, str(exc)

        print(f"  {'[OK]  ' if ok else '[FAIL]'} {cfg['label']}")
        if not ok:
            print(f"         -> {detail}")
        results[check_id] = ok

    acs_ok   = all(results.get(k) for k in ["census_acs", "census_subject", "census_profile"])
    saipe_ok = results.get("census_saipe", False)
    tiger_ok = results.get("tiger_rest", False)

    print()
    viable = (
        (["ACS 5-Year"] if acs_ok else [])
        + (["SAIPE"] if saipe_ok else [])
        + (["TIGER (density)"] if tiger_ok else [])
    )
    if viable:
        print(f"  Viable sources: {', '.join(viable)}")
        if not CENSUS_API_KEY:
            print("  NOTE: No API key — anonymous requests capped at ~500/day.")
            print("        Free key: https://api.census.gov/data/key_signup.html")
            print("        Set:      export CENSUS_API_KEY=your_key_here")
    else:
        print("  [!!] NO VIABLE DATA SOURCE. Check internet connection.")

    print("=" * 65)

    results["_acs_ok"]   = acs_ok
    results["_saipe_ok"] = saipe_ok
    results["_tiger_ok"] = tiger_ok
    results["_any_ok"]   = acs_ok or saipe_ok
    return results


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def census_get(url: str, params: dict, retries: int = 3) -> Optional[list]:
    """
    GET a Census API endpoint.
    Returns parsed JSON list or None on failure.
    Retries on 5xx and 429 with exponential backoff.
    """
    if CENSUS_API_KEY:
        params = {**params, "key": CENSUS_API_KEY}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                wait = 2 ** attempt * 5
                print(f"    [Census] Rate-limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            # 4xx other than 429: variable unavailable for this year/county
            return None
        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
        except Exception as exc:
            print(f"    [Census] Request error: {exc}")
            return None
    return None


def safe_float(val: Optional[str], missing: float = -999.0) -> float:
    """
    Convert Census API string value to float.
    Returns missing sentinel (-999.0) for null / suppressed codes.
    Census null codes: -666666666, -999999999, -888888888.
    """
    if val is None or str(val).strip() in ("-666666666", "-999999999", "-888888888", "null", ""):
        return missing
    try:
        return float(val)
    except (ValueError, TypeError):
        return missing


# ════════════════════════════════════════════════════════════════════════════
# TIGER — county land area for density calculations
# ════════════════════════════════════════════════════════════════════════════

def fetch_county_areas() -> dict:
    """
    Fetch land area for all target counties from TIGER REST.
    Returns {fips: land_area_sqmi}.
    """
    print("\n[TIGER] Fetching county land areas ...")
    url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
        "tigerWMS_Current/MapServer/84/query"
    )
    fips_list = ",".join(f"'{v['fips']}'" for v in COUNTIES.values())
    params = {
        "where":          f"GEOID IN ({fips_list})",
        "outFields":      "GEOID,NAME,AREALAND",
        "returnGeometry": "false",
        "f":              "json",
    }
    areas: dict = {}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        for feat in r.json().get("features", []):
            attr   = feat["attributes"]
            fips   = attr["GEOID"]
            sqm    = attr.get("AREALAND") or 0
            sqmi   = sqm / 2_589_988.11     # sq metres -> sq miles
            areas[fips] = round(sqmi, 4)
        print(f"  -> {len(areas)} counties fetched")
        for info in COUNTIES.values():
            sqmi = areas.get(info["fips"], 0)
            print(f"     {info['name']}: {sqmi:>10,.1f} sq mi")
    except Exception as exc:
        print(f"  [TIGER] Failed: {exc}")
    return areas


# ════════════════════════════════════════════════════════════════════════════
# ACS 5-YEAR ESTIMATES
# ════════════════════════════════════════════════════════════════════════════
#
# B01003_001E   Total population
# B01001_020-025E  Male age 65-66, 67-69, 70-74, 75-79, 80-84, 85+
# B01001_044-049E  Female age 65-66, 67-69, 70-74, 75-79, 80-84, 85+
# B19013_001E   Median household income ($)
# B25001_001E   Total housing units
# S1701_C03_001E  Poverty rate (%, subject table)
# DP05_0001E    Total population cross-check (profile table)

MALE_65_VARS = [
    "B01001_020E", "B01001_021E", "B01001_022E",
    "B01001_023E", "B01001_024E", "B01001_025E",
]
FEMALE_65_VARS = [
    "B01001_044E", "B01001_045E", "B01001_046E",
    "B01001_047E", "B01001_048E", "B01001_049E",
]
ACS_B_VARS = (
    ["B01003_001E", "B19013_001E", "B25001_001E"]
    + MALE_65_VARS + FEMALE_65_VARS
)
ACS_S_VARS  = ["S1701_C03_001E"]
ACS_DP_VARS = ["DP05_0001E"]


def fetch_acs_year(
    county_key: str,
    county_info: dict,
    year: int,
    county_areas: dict,
) -> Optional[dict]:
    """
    Fetch one county x one ACS 5-year vintage.
    Calls three endpoints (B, S, DP tables) and merges into a flat row.
    """
    state  = county_info["state_fips"]
    county = county_info["county_fips"]
    fips   = county_info["fips"]
    base   = f"https://api.census.gov/data/{year}/acs/acs5"

    row: dict = {
        "county_key":  county_key,
        "county_name": county_info["name"],
        "fips":        fips,
        "year":        year,
    }

    # ── B-tables ─────────────────────────────────────────────────────────
    b = census_get(base, {"get": ",".join(ACS_B_VARS),
                          "for": f"county:{county}", "in": f"state:{state}"})
    if b and len(b) >= 2:
        d = dict(zip(b[0], b[1]))
        row["total_pop"]        = safe_float(d.get("B01003_001E"))
        row["median_hh_income"] = safe_float(d.get("B19013_001E"))
        row["housing_units"]    = safe_float(d.get("B25001_001E"))
        male_65   = sum(safe_float(d.get(v), 0.0) for v in MALE_65_VARS)
        female_65 = sum(safe_float(d.get(v), 0.0) for v in FEMALE_65_VARS)
        pop_65    = male_65 + female_65
        pop       = row["total_pop"]
        row["pop_65_plus"] = pop_65
        row["pct_65_plus"] = round(100.0 * pop_65 / pop, 4) if pop > 0 else -999.0
    else:
        for col in ["total_pop", "median_hh_income", "housing_units", "pop_65_plus", "pct_65_plus"]:
            row[col] = -999.0

    # ── S-tables: poverty ─────────────────────────────────────────────────
    s = census_get(f"{base}/subject",
                   {"get": ",".join(ACS_S_VARS),
                    "for": f"county:{county}", "in": f"state:{state}"})
    d_s = dict(zip(s[0], s[1])) if s and len(s) >= 2 else {}
    row["poverty_rate"] = safe_float(d_s.get("S1701_C03_001E"))

    # ── DP-tables: population cross-check ─────────────────────────────────
    dp = census_get(f"{base}/profile",
                    {"get": ",".join(ACS_DP_VARS),
                     "for": f"county:{county}", "in": f"state:{state}"})
    d_dp = dict(zip(dp[0], dp[1])) if dp and len(dp) >= 2 else {}
    row["total_pop_dp05"] = safe_float(d_dp.get("DP05_0001E"))

    # ── Density (requires TIGER land area) ───────────────────────────────
    sqmi = county_areas.get(fips, 0.0)
    row["land_area_sqmi"] = sqmi if sqmi else -999.0
    pop   = row.get("total_pop", -999.0)
    units = row.get("housing_units", -999.0)
    row["pop_density_per_sqmi"]     = round(pop   / sqmi, 4) if pop   > 0 and sqmi > 0 else -999.0
    row["housing_density_per_sqmi"] = round(units / sqmi, 4) if units > 0 and sqmi > 0 else -999.0

    row["data_complete"] = row["total_pop"] > 0
    return row


def fetch_acs_all(county_areas: dict) -> pd.DataFrame:
    """Fetch ACS 5-year for all counties and years."""
    print("\n" + "=" * 65)
    print("ACS 5-YEAR ESTIMATES")
    print("=" * 65)

    rows = []
    for county_key, county_info in COUNTIES.items():
        print(f"\n  {county_info['name']}")
        for year in tqdm(ACS_YEARS, desc="    years"):
            row = fetch_acs_year(county_key, county_info, year, county_areas)
            if row:
                rows.append(row)
            time.sleep(0.25)   # ~4 req/s, well within Census limits

    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUTPUT_DIR / "demographics_acs.csv"
        df.to_csv(out, index=False)
        n_complete = int(df["data_complete"].sum()) if "data_complete" in df.columns else 0
        print(f"\n  -> {len(df)} rows ({n_complete} complete) saved to {out}")
    else:
        print("\n  [!] ACS: No data retrieved.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# SAIPE — Small Area Income and Poverty Estimates
# ════════════════════════════════════════════════════════════════════════════

SAIPE_VARS = ["SAEMHI_PT", "SAEPOVRTALL_PT", "SAEPOVALL_PT", "NAME"]
SAIPE_URL  = "https://api.census.gov/data/timeseries/poverty/saipe"


def fetch_saipe_year(county_key: str, county_info: dict, year: int) -> Optional[dict]:
    """Fetch one county x one year from SAIPE."""
    data = census_get(
        SAIPE_URL,
        {
            "get":  ",".join(SAIPE_VARS),
            "for":  f"county:{county_info['county_fips']}",
            "in":   f"state:{county_info['state_fips']}",
            "time": str(year),
        },
    )
    if not data or len(data) < 2:
        return None
    d = dict(zip(data[0], data[1]))
    return {
        "county_key":          county_key,
        "county_name":         county_info["name"],
        "fips":                county_info["fips"],
        "year":                year,
        "saipe_median_income": safe_float(d.get("SAEMHI_PT")),
        "saipe_poverty_rate":  safe_float(d.get("SAEPOVRTALL_PT")),
        "saipe_poverty_count": safe_float(d.get("SAEPOVALL_PT")),
    }


def fetch_saipe_all() -> pd.DataFrame:
    """Fetch SAIPE for all counties and years."""
    print("\n" + "=" * 65)
    print("SAIPE — SMALL AREA INCOME & POVERTY ESTIMATES")
    print("=" * 65)

    rows = []
    for county_key, county_info in COUNTIES.items():
        print(f"\n  {county_info['name']}")
        for year in tqdm(SAIPE_YEARS, desc="    years"):
            row = fetch_saipe_year(county_key, county_info, year)
            if row:
                rows.append(row)
            time.sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUTPUT_DIR / "demographics_saipe.csv"
        df.to_csv(out, index=False)
        print(f"\n  -> {len(df)} rows saved to {out}")
    else:
        print("\n  [!] SAIPE: No data retrieved.")
    return df


# ════════════════════════════════════════════════════════════════════════════
# COMBINE — merge ACS + SAIPE
# ════════════════════════════════════════════════════════════════════════════

def combine(df_acs: pd.DataFrame, df_saipe: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join ACS (primary) with SAIPE on (fips, year).
    SAIPE rows for years 2000-2008 (pre-ACS) are appended as-is.
    """
    if df_acs.empty and df_saipe.empty:
        return pd.DataFrame()
    if df_acs.empty:
        return df_saipe.sort_values(["county_key", "year"]).reset_index(drop=True)
    if df_saipe.empty:
        return df_acs.sort_values(["county_key", "year"]).reset_index(drop=True)

    saipe_cols = ["fips", "year", "saipe_median_income", "saipe_poverty_rate", "saipe_poverty_count"]
    combined = df_acs.merge(df_saipe[saipe_cols], on=["fips", "year"], how="left")

    # Append SAIPE-only pre-ACS rows (years 2000-2008)
    saipe_early = df_saipe[df_saipe["year"] < 2009].copy()
    if not saipe_early.empty:
        combined = pd.concat([combined, saipe_early], ignore_index=True, sort=False)

    return combined.sort_values(["county_key", "year"]).reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Demographics Data Acquisition — WNV Project")

    access = run_preflight()
    if not access["_any_ok"]:
        print("\n[ABORT] No Census API endpoints reachable. Check connection and re-run.")
        sys.exit(1)

    # TIGER county areas (needed for density columns)
    county_areas: dict = {}
    if access["_tiger_ok"]:
        county_areas = fetch_county_areas()
    else:
        print("\n[SKIP] TIGER — density columns will be -999")

    # ACS 5-Year
    df_acs = pd.DataFrame()
    if access["_acs_ok"]:
        df_acs = fetch_acs_all(county_areas)
    else:
        print("\n[SKIP] ACS — Census B/S/DP endpoints not all reachable.")
        if not CENSUS_API_KEY:
            print("       Set CENSUS_API_KEY to raise the anonymous rate limit.")

    # SAIPE
    df_saipe = pd.DataFrame()
    if access["_saipe_ok"]:
        df_saipe = fetch_saipe_all()
    else:
        print("\n[SKIP] SAIPE — endpoint not reachable.")

    # Combine and save
    print("\n[Step] Combining ACS + SAIPE ...")
    df_combined = combine(df_acs, df_saipe)
    if not df_combined.empty:
        out = OUTPUT_DIR / "demographics_combined.csv"
        df_combined.to_csv(out, index=False)
        print(f"  -> {df_combined.shape[0]} rows x {df_combined.shape[1]} cols -> {out}")
    else:
        print("  [!] Combined table is empty.")

    # Summary
    print("\n" + "=" * 65)
    print("Output files:")
    for f in sorted(OUTPUT_DIR.glob("demographics*.csv")):
        print(f"  {f.name:42s} {f.stat().st_size:>8,} bytes")
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()


# ════════════════════════════════════════════════════════════════════════════
# COLUMN REFERENCE
# ════════════════════════════════════════════════════════════════════════════
#
# demographics_acs.csv  (one row per county x year, 2009-2023)
# ─────────────────────────────────────────────────────────────
# county_key                 e.g. "Larimer_CO"
# county_name                e.g. "Larimer County, CO"
# fips                       5-digit FIPS
# year                       ACS 5-year vintage
# total_pop                  B01003_001E  total population
# total_pop_dp05             DP05_0001E   cross-check (should match total_pop)
# median_hh_income           B19013_001E  median household income ($)
# housing_units              B25001_001E  total housing units
# pop_65_plus                Sum B01001 male+female 65+ age bins
# pct_65_plus                pop_65_plus / total_pop * 100
# poverty_rate               S1701_C03_001E  % persons below poverty line
# land_area_sqmi             County land area from TIGER (static)
# pop_density_per_sqmi       total_pop / land_area_sqmi
# housing_density_per_sqmi   housing_units / land_area_sqmi
# data_complete              True if total_pop retrieved successfully
#
# demographics_saipe.csv  (one row per county x year, 2000-2023)
# ─────────────────────────────────────────────────────────────
# county_key, county_name, fips, year
# saipe_median_income        SAEMHI_PT   median household income ($)
# saipe_poverty_rate         SAEPOVRTALL_PT  poverty rate (%)
# saipe_poverty_count        SAEPOVALL_PT    count below poverty line
#
# demographics_combined.csv  (ACS years 2009-2023 + SAIPE-only 2000-2008)
# ─────────────────────────────────────────────────────────────
# All ACS columns + SAIPE columns joined on (fips, year).
# Pre-ACS rows (2000-2008) contain only SAIPE columns; ACS columns are NaN.
# Missing / suppressed values encoded as -999.0.
# Filter: df[df["total_pop"] > 0] to remove missing rows before modelling.