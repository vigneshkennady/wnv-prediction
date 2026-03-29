"""
=============================================================================
WNV Research Project — NOAA Weather Data Retrieval Pipeline  (v2)
=============================================================================
FIXES vs v1:
  - Read timeout increased 30s → 180s  (root cause of the error)
  - Connect timeout kept at 30s (connection itself is fast)
  - PRIMARY path: NCEI daily-summaries bulk CSV endpoint — downloads one
    full-year CSV per station in a single HTTP call, no pagination needed.
    Much faster and more reliable than CDO /data for large date ranges.
  - FALLBACK path: CDO v2 API in 3-month quarterly chunks (never full-year)
    so each individual CDO request stays well within server timeout.
  - Hardcoded best-station IDs per county — eliminates the slow FIPS-based
    station-discovery CDO query that was also timing out.
  - Year-level parquet cache for fine-grained resume on interruption.
  - ISD-Lite dew point pull unchanged, same retry logic applied.

Variables in final output:
  date, county, state, fips, TAVG, TMAX, TMIN, PRCP, WIND, DEWP_mean, RH

Usage:
  pip install requests pandas numpy pyarrow
  python wnv_weather_retrieval.py

API Key : LBimaMFZNDcQkAKiOjwzokGzzmQSMkHZ
=============================================================================
"""

import gzip
import math
import time
import logging
import requests
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

API_TOKEN  = "LBimaMFZNDcQkAKiOjwzokGzzmQSMkHZ"
CDO_BASE   = "https://www.ncei.noaa.gov/cdo-web/api/v2"
NCEI_BASE  = "https://www.ncei.noaa.gov/access/services/data/v1"  # bulk CSV
ISD_PUB    = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"   # dew point

START_YEAR = 2000
END_YEAR   = 2024

OUT_DIR    = Path("wnv_weather_data")
OUT_DIR.mkdir(exist_ok=True)

# ── Timeout settings ─────────────────────────────────────────────────────────
# THE FIX: read timeout was 30s — NCEI bulk CSV can take 60-120s to prepare.
CONNECT_TIMEOUT = 30    # seconds to establish TCP connection
READ_TIMEOUT    = 180   # seconds to wait for data after connection (was 30!)
MAX_RETRIES     = 6
CDO_DELAY       = 0.35  # stay under CDO 5 req/s limit

# ─── Best Stations Per County (hardcoded to skip slow discovery queries) ──────
#
# Using known airport/ASOS stations avoids the FIPS-based station-discovery
# CDO call that was itself timing out on slow county scans.
#
# ghcnd_primary / ghcnd_backup : GHCND station IDs for NCEI bulk CSV
# isd_stations                 : USAF-WBAN for ISD-Lite dew point files

COUNTIES = {
    "Larimer_CO": {
        "name"          : "Larimer County, CO",
        "fips"          : "08069",
        "ghcnd_primary" : "GHCND:USW00094018",   # Fort Collins / NW CO Regional (KFNL)
        "ghcnd_backup"  : "GHCND:USC00053005",   # Fort Collins COOP
        "isd_stations"  : ["720533-99999", "726400-24018"],
    },
    "Boulder_CO": {
        "name"          : "Boulder County, CO",
        "fips"          : "08013",
        "ghcnd_primary" : "GHCND:USW00094075",   # Boulder Municipal (KBDU)
        "ghcnd_backup"  : "GHCND:USC00050848",   # Boulder COOP
        "isd_stations"  : ["726396-99999", "726400-24018"],
    },
    "Dallas_TX": {
        "name"          : "Dallas County, TX",
        "fips"          : "48113",
        "ghcnd_primary" : "GHCND:USW00013960",   # Dallas Love Field (KDAL)
        "ghcnd_backup"  : "GHCND:USW00003927",   # DFW Airport
        "isd_stations"  : ["722590-03927", "722583-12960"],
    },
    "Maricopa_AZ": {
        "name"          : "Maricopa County, AZ",
        "fips"          : "04013",
        "ghcnd_primary" : "GHCND:USW00023183",   # Phoenix Sky Harbor (KPHX)
        "ghcnd_backup"  : "GHCND:USW00023184",   # Phoenix Deer Valley
        "isd_stations"  : ["722780-23183", "722784-99999"],
    },
    "Cook_IL": {
        "name"          : "Cook County, IL",
        "fips"          : "17031",
        "ghcnd_primary" : "GHCND:USW00094846",   # Chicago O'Hare (KORD)
        "ghcnd_backup"  : "GHCND:USW00014819",   # Chicago Midway
        "isd_stations"  : ["725300-94846", "725340-14819"],
    },
    "LosAngeles_CA": {
        "name"          : "Los Angeles County, CA",
        "fips"          : "06037",
        "ghcnd_primary" : "GHCND:USW00023174",   # LAX Airport (KLAX)
        "ghcnd_backup"  : "GHCND:USW00003167",   # Burbank Airport
        "isd_stations"  : ["722950-23174", "722956-03167"],
    },
}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  [%(levelname)s]  %(message)s",
    handlers = [
        logging.FileHandler(OUT_DIR / "retrieval.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ─── HTTP Helper with Retry + Backoff ─────────────────────────────────────────

def http_get(url: str, params: dict = None, headers: dict = None) -> requests.Response | None:
    """
    GET request with increased timeouts and exponential backoff.
    Handles ReadTimeout, ConnectTimeout, 429, and 5xx errors gracefully.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                url,
                params  = params or {},
                headers = headers or {},
                timeout = (CONNECT_TIMEOUT, READ_TIMEOUT),
            )
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = 60 * attempt
                log.warning(f"    Rate-limited (429). Sleeping {wait}s …")
                time.sleep(wait)
            elif r.status_code in (500, 502, 503, 504):
                wait = 15 * attempt
                log.warning(f"    Server error {r.status_code}. "
                            f"Retry {attempt}/{MAX_RETRIES} in {wait}s …")
                time.sleep(wait)
            else:
                log.warning(f"    HTTP {r.status_code}: {url}")
                return None   # non-retriable client error

        except requests.exceptions.ReadTimeout:
            wait = 20 * attempt
            log.warning(f"    ReadTimeout attempt {attempt}/{MAX_RETRIES}. "
                        f"Retrying in {wait}s (read_timeout={READ_TIMEOUT}s) …")
            time.sleep(wait)

        except requests.exceptions.ConnectTimeout:
            wait = 10 * attempt
            log.warning(f"    ConnectTimeout attempt {attempt}. Retrying in {wait}s …")
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            wait = 5 * attempt
            log.warning(f"    Request error attempt {attempt}: {e}. Retrying in {wait}s …")
            time.sleep(wait)

    log.error(f"    All {MAX_RETRIES} attempts failed: {url}")
    return None


# ─── Path A: NCEI Bulk CSV (PRIMARY) ─────────────────────────────────────────
#
# Endpoint: https://www.ncei.noaa.gov/access/services/data/v1
#   ?dataset=daily-summaries&stations=GHCND:USW00023183
#   &startDate=2023-01-01&endDate=2023-12-31
#   &dataTypes=TMAX,TMIN,TAVG,PRCP,AWND&format=csv&units=metric
#
# Returns one CSV with all variables for the full year in a single request.
# This is far faster than CDO /data which requires pagination.

def fetch_ncei_year(station_id: str, year: int) -> pd.DataFrame:
    """Download full-year daily summaries via NCEI bulk CSV endpoint."""
    params = {
        "dataset"   : "daily-summaries",
        "stations"  : station_id,
        "startDate" : f"{year}-01-01",
        "endDate"   : f"{year}-12-31",
        "dataTypes" : "TMAX,TMIN,TAVG,PRCP,AWND,AWSP,DEWP",
        "format"    : "csv",
        "units"     : "metric",
        "includeStationName"    : "false",
        "includeStationLocation": "false",
    }
    r = http_get(NCEI_BASE, params=params)
    if r is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(StringIO(r.text))
        if df.empty or "DATE" not in df.columns:
            log.warning(f"      NCEI CSV empty or missing DATE for {station_id} {year}")
            return pd.DataFrame()
        df = df.rename(columns={"DATE": "date", "STATION": "station"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        log.info(f"      NCEI bulk CSV: {len(df)} rows ← {station_id} {year}")
        return df
    except Exception as e:
        log.warning(f"      NCEI CSV parse error {station_id} {year}: {e}")
        return pd.DataFrame()


# ─── Path B: CDO v2 API — Quarterly Chunks (FALLBACK) ────────────────────────
#
# When NCEI bulk CSV fails, hit CDO /data in 3-month chunks.
# Small windows prevent server-side query timeouts on CDO.

QUARTER_DATES = {
    1: ("01-01", "03-31"),
    2: ("04-01", "06-30"),
    3: ("07-01", "09-30"),
    4: ("10-01", "12-31"),
}

def fetch_cdo_quarter(station_id: str, year: int, quarter: int) -> pd.DataFrame:
    s0, s1 = QUARTER_DATES[quarter]
    params  = {
        "datasetid"  : "GHCND",
        "stationid"  : station_id,
        "datatypeid" : "TMAX,TMIN,TAVG,PRCP,AWND,AWSP",
        "startdate"  : f"{year}-{s0}",
        "enddate"    : f"{year}-{s1}",
        "units"      : "metric",
        "limit"      : 1000,
        "offset"     : 1,
    }
    headers  = {"token": API_TOKEN}
    all_rows = []
    offset   = 1

    while True:
        params["offset"] = offset
        r = http_get(f"{CDO_BASE}/data", params=params, headers=headers)
        time.sleep(CDO_DELAY)
        if r is None:
            break
        data = r.json()
        rows = data.get("results", [])
        if not rows:
            break
        all_rows.extend(rows)
        meta  = data.get("metadata", {}).get("resultset", {})
        total = meta.get("count", 0)
        lim   = meta.get("limit", 1000)
        off   = meta.get("offset", offset)
        if off + lim - 1 >= total:
            break
        offset += 1000

    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fetch_cdo_year(station_id: str, year: int) -> pd.DataFrame:
    """Fetch full year via 4 quarterly CDO calls; pivot to wide format."""
    frames = []
    for q in range(1, 5):
        log.info(f"        CDO Q{q}/{year} → {station_id} …")
        qdf = fetch_cdo_quarter(station_id, year, q)
        if not qdf.empty:
            frames.append(qdf)
    if not frames:
        return pd.DataFrame()
    raw  = pd.concat(frames, ignore_index=True)
    wide = (raw.pivot_table(index="date", columns="datatype",
                            values="value", aggfunc="mean")
               .reset_index())
    wide.columns.name = None
    wide["station"] = station_id
    return wide


# ─── GHCND Fetch Orchestrator (NCEI → CDO fallback) ──────────────────────────

def fetch_ghcnd_year(county: dict, year: int) -> pd.DataFrame:
    """Try NCEI bulk CSV first, fall back to CDO quarterly for both stations."""
    for sid in [county["ghcnd_primary"], county["ghcnd_backup"]]:
        log.info(f"    Trying NCEI bulk CSV: {sid} …")
        df = fetch_ncei_year(sid, year)
        if not df.empty and len(df) >= 50:
            return df

    log.info(f"    NCEI empty — falling back to CDO quarterly …")
    for sid in [county["ghcnd_primary"], county["ghcnd_backup"]]:
        df = fetch_cdo_year(sid, year)
        if not df.empty:
            return df

    log.warning(f"    No GHCND data for {county['name']} {year}.")
    return pd.DataFrame()


# ─── ISD-Lite: Dew Point → RH ─────────────────────────────────────────────────

def fetch_isd_year(station_id: str, year: int) -> pd.DataFrame:
    """Download ISD-Lite gzip; return daily DEWP_mean, TAIR_ISD, WIND_ISD."""
    url = f"{ISD_PUB}/{year}/{station_id}-{year}.gz"
    r   = http_get(url)
    if r is None:
        return pd.DataFrame()
    try:
        with gzip.open(BytesIO(r.content), "rt") as f:
            df = pd.read_fwf(
                f,
                widths    = [4,3,3,3,6,6,6,6,6,6,6,6],
                header    = None,
                names     = ["yr","mo","dy","hr","air_x10","dewp_x10",
                             "slp","wdir","wspd_x10","sky","p1","p6"],
                na_values = ["-9999"],
            )
        df["date"]      = pd.to_datetime(
            df[["yr","mo","dy"]].rename(columns={"yr":"year","mo":"month","dy":"day"}))
        df["dewpoint"]  = df["dewp_x10"]  / 10.0
        df["air_temp"]  = df["air_x10"]   / 10.0
        df["wind_speed"]= df["wspd_x10"]  / 10.0

        daily = (df.groupby("date")
                   .agg(DEWP_mean  = ("dewpoint",   "mean"),
                        TAIR_ISD   = ("air_temp",   "mean"),
                        WIND_ISD   = ("wind_speed", "mean"))
                   .reset_index())
        daily["date"] = daily["date"].dt.date
        return daily
    except Exception as e:
        log.debug(f"    ISD parse error {station_id} {year}: {e}")
        return pd.DataFrame()


# ─── RH Derivation (August-Roche-Magnus) ──────────────────────────────────────

def dewpoint_to_rh(temp_c, dewp_c) -> float:
    if pd.isna(temp_c) or pd.isna(dewp_c):
        return np.nan
    num   = math.exp(17.625 * dewp_c / (243.04 + dewp_c))
    denom = math.exp(17.625 * temp_c / (243.04 + temp_c))
    return round(min(100.0, 100.0 * num / denom), 1)


# ─── Merge GHCND + ISD, Compute Final Variables ───────────────────────────────

def standardise_ghcnd(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and fix tenths-of-degree encoding if present."""
    df = df.copy()
    for col in ["TMAX","TMIN","TAVG","DEWP"]:
        if col in df.columns:
            s = df[col].dropna()
            if len(s) and s.abs().max() > 100:   # likely in tenths °C
                df[col] = df[col] / 10.0
    # TAVG from average if missing
    if "TAVG" not in df.columns or df["TAVG"].isna().mean() > 0.6:
        if {"TMAX","TMIN"}.issubset(df.columns):
            df["TAVG"] = ((df["TMAX"] + df["TMIN"]) / 2).round(1)
    # Wind: AWND preferred, AWSP as backup
    if "AWND" in df.columns and "AWSP" in df.columns:
        df["AWND"] = df["AWND"].fillna(df["AWSP"])
    elif "AWSP" in df.columns:
        df["AWND"] = df.pop("AWSP")
    # DEWP direct column
    if "DEWP" in df.columns and "DEWP_mean" not in df.columns:
        df["DEWP_mean"] = df["DEWP"]
    return df


def build_daily(ghcnd: pd.DataFrame, isd: pd.DataFrame) -> pd.DataFrame:
    """Merge GHCND wide + ISD daily; compute RH and WIND."""
    if not ghcnd.empty:
        ghcnd = standardise_ghcnd(ghcnd)

    if ghcnd.empty and isd.empty:
        return pd.DataFrame()
    if ghcnd.empty:
        merged = isd.copy()
    elif isd.empty:
        merged = ghcnd.copy()
    else:
        merged = pd.merge(ghcnd, isd, on="date", how="outer")

    # Temperature source for RH
    temp_col = next((c for c in ["TAVG","TAIR_ISD","TMAX"] if c in merged.columns
                     and merged[c].notna().sum() > 0), None)
    dewp_col = "DEWP_mean" if "DEWP_mean" in merged.columns else None

    if temp_col and dewp_col:
        merged["RH"] = [dewpoint_to_rh(t, d)
                        for t, d in zip(merged[temp_col], merged[dewp_col])]
    else:
        merged["RH"] = np.nan

    # Final wind
    if "AWND" in merged.columns and "WIND_ISD" in merged.columns:
        merged["WIND"] = merged["AWND"].fillna(merged["WIND_ISD"])
    elif "AWND" in merged.columns:
        merged["WIND"] = merged["AWND"]
    elif "WIND_ISD" in merged.columns:
        merged["WIND"] = merged["WIND_ISD"]
    else:
        merged["WIND"] = np.nan

    return merged


FINAL_COLS = ["date","county","state","fips",
              "TAVG","TMAX","TMIN","PRCP","WIND","DEWP_mean","RH","station"]

def clean_final(df: pd.DataFrame, county: dict) -> pd.DataFrame:
    df = df.copy()
    df["county"] = county["name"].split(",")[0]
    df["state"]  = county["name"].split(", ")[1] if ", " in county["name"] else ""
    df["fips"]   = county["fips"]
    df["date"]   = pd.to_datetime(df["date"])
    if "station" not in df.columns:
        df["station"] = np.nan

    for col, lo, hi in [("TAVG",-60,60),("TMAX",-60,70),("TMIN",-60,60),
                         ("PRCP",0,1000),("WIND",0,80),("RH",0,100),
                         ("DEWP_mean",-60,40)]:
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    for col in ["TAVG","TMAX","TMIN","DEWP_mean"]:
        if col in df.columns: df[col] = df[col].round(1)
    for col in ["PRCP","WIND","RH"]:
        if col in df.columns: df[col] = df[col].round(2)

    for col in FINAL_COLS:
        if col not in df.columns: df[col] = np.nan

    return df[FINAL_COLS].sort_values("date").reset_index(drop=True)


# ─── Per-County Pipeline ──────────────────────────────────────────────────────

def process_county(county_key: str, county: dict) -> pd.DataFrame:
    log.info(f"\n{'='*62}")
    log.info(f"  {county['name']}  (FIPS {county['fips']})")
    log.info(f"{'='*62}")

    # County-level cache (skip entirely if already done)
    cache_path = OUT_DIR / f"{county_key}_raw.parquet"
    if cache_path.exists():
        log.info(f"  County cache found — loading {cache_path.name}")
        return pd.read_parquet(cache_path)

    # Year-level cache directory for fine-grained resume
    yr_dir = OUT_DIR / f"_cache_{county_key}"
    yr_dir.mkdir(exist_ok=True)

    all_dfs = []
    for year in range(START_YEAR, END_YEAR + 1):
        ypath = yr_dir / f"{year}.parquet"
        if ypath.exists():
            log.info(f"  {year} — from year cache")
            all_dfs.append(pd.read_parquet(ypath))
            continue

        log.info(f"  {year}:")
        ghcnd = fetch_ghcnd_year(county, year)

        # ISD dew point
        isd_frames = []
        for sid in county.get("isd_stations", []):
            idf = fetch_isd_year(sid, year)
            if not idf.empty:
                isd_frames.append(idf)
        if isd_frames:
            isd = (pd.concat(isd_frames, ignore_index=True)
                     .groupby("date")
                     .agg(DEWP_mean=("DEWP_mean","mean"),
                          TAIR_ISD =("TAIR_ISD","mean"),
                          WIND_ISD =("WIND_ISD","mean"))
                     .reset_index())
        else:
            isd = pd.DataFrame()

        year_df = build_daily(ghcnd, isd)
        if year_df.empty:
            log.warning(f"    Skipping {year} — no data from any source.")
            continue

        year_df.to_parquet(ypath, index=False)
        all_dfs.append(year_df)

    if not all_dfs:
        log.error(f"  No data for {county['name']}!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    final    = clean_final(combined, county)
    final.to_parquet(cache_path, index=False)
    log.info(f"  Saved {len(final):,} records → {cache_path.name}")
    return final


# ─── Coverage Report ──────────────────────────────────────────────────────────

def coverage_report(df: pd.DataFrame):
    print("\n" + "="*70)
    print("DATA COVERAGE REPORT")
    print("="*70)
    for county, grp in df.groupby("county"):
        n  = len(grp)
        ok = grp[["TAVG","TMAX","TMIN","PRCP","WIND","RH"]].notna().all(axis=1).sum()
        pct = int(100 * ok / n) if n else 0
        print(f"\n  {county:<25}  {n:>6} days  |  complete rows: {ok:>5} ({pct}%)")
        for col in ["TAVG","TMAX","TMIN","PRCP","WIND","DEWP_mean","RH"]:
            if col in grp.columns:
                m    = grp[col].isna().sum()
                mpct = int(100 * m / n) if n else 0
                print(f"    {col:<12} missing {m:>4} ({mpct}%)")
    print("="*70)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("WNV Weather Retrieval v2")
    log.info(f"  Timeouts   : connect={CONNECT_TIMEOUT}s  read={READ_TIMEOUT}s")
    log.info(f"  Years      : {START_YEAR}–{END_YEAR}")
    log.info(f"  Output dir : {OUT_DIR.resolve()}\n")

    all_dfs = []
    for county_key, county in COUNTIES.items():
        try:
            df = process_county(county_key, county)
            if not df.empty:
                all_dfs.append(df)
                csv_path = OUT_DIR / f"{county_key}_{START_YEAR}_{END_YEAR}.csv"
                df.to_csv(csv_path, index=False)
                log.info(f"  → {csv_path.name}")
        except Exception as e:
            log.error(f"Failed {county['name']}: {e}", exc_info=True)

    if not all_dfs:
        log.error("No data collected. Check retrieval.log for details.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = OUT_DIR / f"ALL_COUNTIES_weather_{START_YEAR}_{END_YEAR}.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"\nCombined file: {out_path.name}  ({len(combined):,} rows)")

    # Summary stats
    combined["year"] = combined["date"].dt.year
    numeric  = [c for c in ["TAVG","TMAX","TMIN","PRCP","WIND","RH"]
                if c in combined.columns]
    stats = (combined.groupby(["county","state","fips","year"])[numeric]
                     .agg(["mean","min","max"])
                     .round(2)
                     .reset_index())
    stats.columns = ["_".join(c).strip("_") for c in stats.columns]
    stats.to_csv(OUT_DIR / "summary_stats.csv", index=False)

    coverage_report(combined)
    print(f"\nAll files written to: {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()