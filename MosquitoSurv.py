"""
Mosquito Surveillance Data Acquisition — WNV County-Level Prediction Project
=============================================================================
Counties  : Larimer CO (08069) | Boulder CO (08013) | Dallas TX (48113)
            Maricopa AZ (04013) | Cook IL (17031)   | Los Angeles CA (06037)

Variables : trap_count, mosquito_species (Culex), positive_pools,
            total_pools_tested, infection_rate / MIR, week, year, county

Data sources (per county)
--------------------------
Source A  Cook County, IL
          Chicago Data Portal — Socrata REST API
          Dataset: "West Nile Virus (WNV) Mosquito Test Results"  ID: jqe8-8r6s
          URL: https://data.cityofchicago.org/resource/jqe8-8r6s.json
          Coverage: 2007–present | No API key required
          Variables: trap, test_date, number_of_mosquitoes, result, species

Source B  Los Angeles County, CA
          westnile.ca.gov annual CSV downloads
          URL: https://westnile.ca.gov/download.php?download_id={ID}
          Annual CSVs 2004–2024 (pool-level, filtered to LA County)
          Columns: County, Species, # in Pool, Collected, Result

Source C  Maricopa County, AZ
          VectorSurv REST API (requires free agency token)
          Token request: https://vectorsurv.org/signup
          Set env var: VECTORSURV_TOKEN=your_token
          Falls back to manual download if token not set
          API: https://api.vectorsurv.org/v1/arthropod/pool

Source D  Colorado (Larimer + Boulder)
          CDPHE WNV data page — weekly HTML summary (current year)
          + manual CSV loader for historical CDPHE annual files
          URL: https://cdphe.colorado.gov/animal-related-diseases/
               west-nile-virus/west-nile-virus-data

Source E  Dallas County, TX
          DSHS weekly arbovirus PDF reports (pdfplumber extraction)
          Index: https://www.dshs.texas.gov/mosquito-borne-diseases/
                 dshs-arbovirus-weekly-activity-reports
          Requires: pip install pdfplumber

Source F  All counties — CDC ArboNET (annual backstop)
          Annual county-level: positive mosquito pools, infection rate
          Coverage: 1999–2024 | No API key required

Python compatibility: 3.9+
Requirements: pip install requests pandas
Optional:     pip install pdfplumber          (Source E)
              pip install beautifulsoup4 lxml  (Sources D + E)
=============================================================================
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── API tokens ────────────────────────────────────────────────────────────────
# VectorSurv (Maricopa AZ, optionally LA CA):
#   Register free at https://vectorsurv.org/signup
#   export VECTORSURV_TOKEN=your_token_here
VECTORSURV_TOKEN = os.environ.get("VECTORSURV_TOKEN", "")

# Chicago Data Portal (optional — raises rate limit):
#   Register free at https://data.cityofchicago.org/signup
#   export CHICAGO_APP_TOKEN=your_token_here
CHICAGO_APP_TOKEN = os.environ.get("CHICAGO_APP_TOKEN", "")

# ── Year range ────────────────────────────────────────────────────────────────
START_YEAR = 2007
END_YEAR   = datetime.now().year

# ── Target FIPS ───────────────────────────────────────────────────────────────
TARGET_FIPS = {"17031", "06037", "48113", "04013", "08069", "08013"}
FIPS_META: dict = {
    "17031": ("Cook_IL",        "Cook County, IL"),
    "06037": ("LosAngeles_CA",  "Los Angeles County, CA"),
    "48113": ("Dallas_TX",      "Dallas County, TX"),
    "04013": ("Maricopa_AZ",    "Maricopa County, AZ"),
    "08069": ("Larimer_CO",     "Larimer County, CO"),
    "08013": ("Boulder_CO",     "Boulder County, CO"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Shared HTTP helper
# ══════════════════════════════════════════════════════════════════════════════

def _get(url: str,
         params: Optional[dict] = None,
         headers: Optional[dict] = None,
         retries: int = 3,
         backoff: float = 2.0,
         timeout: int = 30) -> Optional[requests.Response]:
    """GET with retry/backoff. Returns Response or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=timeout, allow_redirects=True)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                log.warning("Rate limited — waiting %.0fs (attempt %d)", wait, attempt)
                time.sleep(wait)
                continue
            if r.status_code >= 400:
                log.warning("HTTP %d: %s (attempt %d)", r.status_code, url, attempt)
                if attempt == retries:
                    return None
                time.sleep(backoff * attempt)
                continue
            return r
        except requests.exceptions.Timeout:
            log.warning("Timeout attempt %d: %s", attempt, url)
        except Exception as exc:
            log.warning("Request error attempt %d: %s", attempt, exc)
        time.sleep(backoff * attempt)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Shared: MIR calculation and county-week aggregation
# ══════════════════════════════════════════════════════════════════════════════

def _date_to_mmwr_week(d: Optional[datetime]) -> Optional[int]:
    """Return the MMWR epidemiological week number for a date (approx ISO week)."""
    if d is None or pd.isna(d):
        return None
    try:
        return int(d.strftime("%W"))
    except Exception:
        return None


def _compute_mir(positive_pools: int,
                 total_pools: int,
                 avg_pool_size: float = 50.0) -> Optional[float]:
    """
    Minimum Infection Rate (MIR) per 1,000 mosquitoes tested.
    MIR = (positive_pools / (total_pools * avg_pool_size)) * 1000
    """
    if total_pools <= 0:
        return None
    return round(1000.0 * positive_pools / (total_pools * avg_pool_size), 6)


def _aggregate_to_county_week(df: pd.DataFrame,
                               result_col: str = "is_positive",
                               pool_size_col: str = "mosquitoes_in_pool",
                               species_col: str = "species") -> pd.DataFrame:
    """
    Aggregate pool-level records to county x year x week summary.

    Output columns:
      county_key, county_name, fips, year, mmwr_week,
      total_pools_tested, positive_pools, pct_positive,
      total_mosquitoes, culex_pools, culex_pct, mir_per_1000, source
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df[result_col]    = pd.to_numeric(df.get(result_col, 0),    errors="coerce").fillna(0)
    df[pool_size_col] = pd.to_numeric(df.get(pool_size_col, 0), errors="coerce").fillna(0)

    if "is_culex" not in df.columns:
        df["is_culex"] = (
            df.get(species_col, pd.Series(dtype=str))
            .str.upper().str.contains("CULEX", na=False).astype(int)
        )

    group_cols = ["county_key", "county_name", "fips", "year"]
    if "mmwr_week" in df.columns:
        group_cols.append("mmwr_week")

    agg = (df.groupby(group_cols, dropna=False)
           .agg(
               total_pools_tested = (result_col,    "count"),
               positive_pools     = (result_col,    "sum"),
               total_mosquitoes   = (pool_size_col, "sum"),
               culex_pools        = ("is_culex",    "sum"),
           )
           .reset_index())

    agg["pct_positive"] = (
        100.0 * agg["positive_pools"] / agg["total_pools_tested"].replace(0, float("nan"))
    ).round(4)
    agg["culex_pct"] = (
        100.0 * agg["culex_pools"] / agg["total_pools_tested"].replace(0, float("nan"))
    ).round(4)
    agg["mir_per_1000"] = agg.apply(
        lambda r: _compute_mir(
            int(r["positive_pools"]),
            int(r["total_pools_tested"]),
            float(r["total_mosquitoes"]) / float(r["total_pools_tested"])
            if r["total_pools_tested"] > 0 else 50.0
        ),
        axis=1,
    )

    if "source" in df.columns:
        src = df.groupby(group_cols, dropna=False)["source"].first().reset_index()
        agg = agg.merge(src, on=group_cols, how="left")

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE A — Cook County, IL (Chicago Data Portal / Socrata)
# ══════════════════════════════════════════════════════════════════════════════

CHICAGO_API_BASE = "https://data.cityofchicago.org/resource/jqe8-8r6s.json"


def fetch_chicago_wnv(start_year: int = START_YEAR,
                      end_year:   int = END_YEAR) -> tuple:
    """
    Pull all WNV pool test records from Chicago Data Portal (Socrata).
    Returns (county_week_df, pool_level_df).
    """
    log.info("[Chicago] Fetching WNV mosquito pool data %d-%d ...", start_year, end_year)
    headers = {}
    if CHICAGO_APP_TOKEN:
        headers["X-App-Token"] = CHICAGO_APP_TOKEN

    all_rows: list = []
    offset    = 0
    page_size = 50_000

    while True:
        params = {
            "$limit":  page_size,
            "$offset": offset,
            "$where":  f"season_year >= {start_year} AND season_year <= {end_year}",
            "$order":  "season_year ASC, week ASC",
        }
        r = _get(CHICAGO_API_BASE, params=params, headers=headers, timeout=60)
        if r is None:
            break
        page = r.json()
        if not page:
            break
        all_rows.extend(page)
        log.info("  fetched %d records (total: %d)", len(page), len(all_rows))
        if len(page) < page_size:
            break
        offset += page_size
        time.sleep(0.2)

    if not all_rows:
        log.warning("[Chicago] No records returned")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={
        "season_year":          "year",
        "week":                 "mmwr_week",
        "trap":                 "trap_id",
        "trap_type":            "trap_type",
        "test_date":            "test_date",
        "species":              "species",
        "number_of_mosquitoes": "mosquitoes_in_pool",
        "result":               "wnv_result",
        "latitude":             "latitude",
        "longitude":            "longitude",
    })
    df["year"]               = pd.to_numeric(df.get("year"),               errors="coerce")
    df["mmwr_week"]          = pd.to_numeric(df.get("mmwr_week"),           errors="coerce")
    df["mosquitoes_in_pool"] = pd.to_numeric(df.get("mosquitoes_in_pool"),  errors="coerce")
    df["is_positive"]        = df["wnv_result"].str.upper().eq("POSITIVE").astype(int)
    df["is_culex"]           = (
        df.get("species", pd.Series(dtype=str))
        .str.upper().str.contains("CULEX", na=False).astype(int)
    )
    df["county_key"]  = "Cook_IL"
    df["county_name"] = "Cook County, IL"
    df["fips"]        = "17031"
    df["source"]      = "Chicago_DataPortal"

    agg = _aggregate_to_county_week(df, result_col="is_positive",
                                    pool_size_col="mosquitoes_in_pool")
    log.info("[Chicago] %d pool records -> %d county-week rows", len(df), len(agg))
    return agg, df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE B — Los Angeles County, CA (westnile.ca.gov annual CSVs)
# ══════════════════════════════════════════════════════════════════════════════

# Download IDs for annual pool-level CSVs on westnile.ca.gov (2004–2024)
CA_WNV_CSV_IDS: dict = {
    2004: 1048, 2005: 1049, 2006: 1050, 2007: 1051, 2008: 1052,
    2009: 1053, 2010: 1054, 2011: 1055, 2012: 1056, 2013: 1057,
    2014: 1058, 2015: 1059, 2016: 1060, 2017: 1061, 2018: 1062,
    2019: 1063, 2020: 1064, 2021: 1065, 2022: 1066, 2023: 1067,
    2024: 1068,
}
CA_WNV_CSV_BASE = "https://westnile.ca.gov/download.php"


def fetch_ca_wnv_year(year: int, download_id: int) -> Optional[pd.DataFrame]:
    """Download one year of CA WNV pool data and filter to Los Angeles County."""
    r = _get(CA_WNV_CSV_BASE, params={"download_id": download_id}, timeout=60)
    if r is None:
        return None
    try:
        text = r.content.decode("utf-8", errors="replace")
        df   = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        log.warning("[CA WNV] Parse error year %d: %s", year, exc)
        return None

    df.columns = [c.strip().lower().replace(" ", "_").replace("#", "n") for c in df.columns]
    col_map = {
        "county":          "county",
        "species":         "species",
        "n_in_pool":       "mosquitoes_in_pool",
        "n_pool":          "mosquitoes_in_pool",
        "collected":       "collection_date",
        "result":          "wnv_result",
        "site_code":       "trap_id",
        "pool_n":          "pool_id",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "county" in df.columns:
        df = df[df["county"].str.strip().str.upper().str.contains("LOS ANGELES", na=False)]
    if df.empty:
        return None

    df["year"]        = year
    df["county_key"]  = "LosAngeles_CA"
    df["county_name"] = "Los Angeles County, CA"
    df["fips"]        = "06037"
    df["source"]      = "westnile_ca_gov"
    return df


def fetch_ca_wnv(start_year: int = max(2004, START_YEAR),
                 end_year:   int = END_YEAR) -> tuple:
    """Download all available CA WNV pool data and filter to LA County."""
    log.info("[CA WNV] Fetching pool data for Los Angeles County %d-%d ...",
             start_year, end_year)
    pool_frames: list = []

    for year, dl_id in CA_WNV_CSV_IDS.items():
        if not (start_year <= year <= end_year):
            continue
        log.info("  Year %d (download_id=%d)", year, dl_id)
        df = fetch_ca_wnv_year(year, dl_id)
        if df is not None:
            pool_frames.append(df)
        time.sleep(0.5)

    if not pool_frames:
        log.warning("[CA WNV] No data retrieved for LA County")
        return pd.DataFrame(), pd.DataFrame()

    df_pools = pd.concat(pool_frames, ignore_index=True)

    if "collection_date" in df_pools.columns:
        df_pools["collection_date"] = pd.to_datetime(
            df_pools["collection_date"], errors="coerce"
        )
        df_pools["mmwr_week"] = df_pools["collection_date"].apply(
            lambda d: _date_to_mmwr_week(d) if pd.notna(d) else None
        )

    df_pools["mosquitoes_in_pool"] = pd.to_numeric(
        df_pools.get("mosquitoes_in_pool"), errors="coerce"
    )
    df_pools["is_positive"] = (
        df_pools.get("wnv_result", pd.Series(dtype=str))
        .str.upper().str.contains("POSITIVE|POS", na=False).astype(int)
    )
    df_pools["is_culex"] = (
        df_pools.get("species", pd.Series(dtype=str))
        .str.upper().str.contains("CULEX", na=False).astype(int)
    )

    agg = _aggregate_to_county_week(df_pools, result_col="is_positive",
                                    pool_size_col="mosquitoes_in_pool")
    log.info("[CA WNV] %d pool records -> %d county-week rows", len(df_pools), len(agg))
    return agg, df_pools


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE C — Maricopa County, AZ (VectorSurv REST API)
# ══════════════════════════════════════════════════════════════════════════════

VECTORSURV_POOL_URL = "https://api.vectorsurv.org/v1/arthropod/pool"


def fetch_maricopa_vectorsurv(start_year: int = START_YEAR,
                               end_year:   int = END_YEAR) -> tuple:
    """
    Pull Maricopa County WNV pool data from VectorSurv REST API.

    Setup (one-time):
        1. Register at https://vectorsurv.org/signup
        2. Generate API token in VectorSurv Gateway settings
        3. export VECTORSURV_TOKEN=your_token
    """
    if not VECTORSURV_TOKEN:
        log.warning(
            "[Maricopa/VectorSurv] VECTORSURV_TOKEN not set.\n"
            "  Steps to obtain Maricopa County data:\n"
            "  1. Register at https://vectorsurv.org/signup\n"
            "  2. Generate token in VectorSurv Gateway > Settings > API Tokens\n"
            "  3. export VECTORSURV_TOKEN=your_token\n"
            "  Alternatively, contact Maricopa County Environmental Services\n"
            "  Vector Control Division: https://www.maricopa.gov/1094/\n"
            "  Or use CDC ArboNET (Source F) for annual county counts."
        )
        return pd.DataFrame(), pd.DataFrame()

    headers = {
        "Authorization": f"Bearer {VECTORSURV_TOKEN}",
        "Accept":        "application/json",
    }
    all_pools: list = []
    page = 1
    per_page = 500
    log.info("[Maricopa/VectorSurv] Fetching pool data %d-%d ...", start_year, end_year)

    while True:
        params = {
            "start_date":  f"{start_year}-01-01",
            "end_date":    f"{end_year}-12-31",
            "county_fips": "04013",
            "per_page":    per_page,
            "page":        page,
        }
        r = _get(VECTORSURV_POOL_URL, params=params, headers=headers, timeout=60)
        if r is None:
            break
        try:
            data = r.json()
        except Exception:
            break

        records = data.get("data", [])
        if not records:
            break
        all_pools.extend(records)
        log.info("  page %d: %d records (total: %d)", page, len(records), len(all_pools))
        meta = data.get("meta", {})
        if page >= meta.get("last_page", 1):
            break
        page += 1
        time.sleep(0.3)

    if not all_pools:
        log.warning("[Maricopa/VectorSurv] No records returned")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_pools)
    vs_map = {
        "collection_date": "collection_date",
        "trap_id":         "trap_id",
        "trap_type":       "trap_type",
        "species_display": "species",
        "num_mosquitoes":  "mosquitoes_in_pool",
        "wnv_result":      "wnv_result",
        "pool_num":        "pool_id",
        "latitude":        "latitude",
        "longitude":       "longitude",
    }
    df = df.rename(columns={k: v for k, v in vs_map.items() if k in df.columns})

    df["year"] = pd.to_datetime(df.get("collection_date"), errors="coerce").dt.year
    df["mmwr_week"] = df.get("collection_date", pd.Series(dtype=str)).apply(
        lambda d: _date_to_mmwr_week(pd.to_datetime(d, errors="coerce"))
        if pd.notna(d) else None
    )
    df["is_positive"] = (
        df.get("wnv_result", pd.Series(dtype=str))
        .astype(str).str.upper().str.contains("POS", na=False).astype(int)
    )
    df["is_culex"] = (
        df.get("species", pd.Series(dtype=str))
        .str.upper().str.contains("CULEX", na=False).astype(int)
    )
    df["mosquitoes_in_pool"] = pd.to_numeric(df.get("mosquitoes_in_pool"), errors="coerce")
    df["county_key"]  = "Maricopa_AZ"
    df["county_name"] = "Maricopa County, AZ"
    df["fips"]        = "04013"
    df["source"]      = "VectorSurv"

    agg = _aggregate_to_county_week(df, result_col="is_positive",
                                    pool_size_col="mosquitoes_in_pool")
    log.info("[Maricopa/VectorSurv] %d pool records -> %d county-week rows",
             len(df), len(agg))
    return agg, df


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE D — Colorado: Larimer + Boulder (CDPHE HTML + manual files)
# ══════════════════════════════════════════════════════════════════════════════

CDPHE_DATA_URL = (
    "https://cdphe.colorado.gov/animal-related-diseases/"
    "west-nile-virus/west-nile-virus-data"
)
CO_TARGET_COUNTIES = {"LARIMER", "BOULDER"}


def _parse_cdphe_html_table(html: str, year: int) -> pd.DataFrame:
    """Parse CDPHE WNV county mosquito table from HTML (current year)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("[CO CDPHE] beautifulsoup4 not installed: pip install beautifulsoup4 lxml")
        return pd.DataFrame()

    soup  = BeautifulSoup(html, "lxml")
    rows: list = []

    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not any("pool" in h or "county" in h for h in headers):
            continue
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) < 3:
                continue
            county = cells[0].upper().strip()
            if not any(c in county for c in CO_TARGET_COUNTIES):
                continue
            row: dict = {"county_raw": cells[0], "year": year, "source": "CDPHE_HTML"}
            for i, h in enumerate(headers[1:], start=1):
                if i < len(cells):
                    row[h] = cells[i]
            rows.append(row)

    return pd.DataFrame(rows)


def fetch_colorado_cdphe() -> pd.DataFrame:
    """
    Fetch Colorado WNV mosquito data from CDPHE.
    Current year: parses HTML summary table.
    Historical years: load manually downloaded CDPHE annual files (see below).

    For historical Larimer/Boulder data, download annual Excel/CSV from:
      https://cdphe.colorado.gov/animal-related-diseases/west-nile-virus/west-nile-virus-data
    Place files in ./manual_data/co_cdphe/ and re-run.
    """
    log.info("[CO CDPHE] Fetching current-year WNV summary from CDPHE ...")
    frames: list = []

    r = _get(CDPHE_DATA_URL, timeout=30)
    if r is not None:
        year   = datetime.now().year
        df_html = _parse_cdphe_html_table(r.text, year)
        if not df_html.empty:
            frames.append(df_html)
            log.info("[CO CDPHE] Parsed %d rows from current-year HTML", len(df_html))
    else:
        log.warning("[CO CDPHE] Could not reach CDPHE data page")

    # Load manual historical files
    manual_dir = Path("./manual_data/co_cdphe")
    if manual_dir.exists():
        for f in sorted(manual_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                df["source"] = f"CDPHE_manual_{f.stem}"
                frames.append(df)
                log.info("[CO manual] Loaded %s (%d rows)", f.name, len(df))
            except Exception as exc:
                log.warning("[CO manual] Failed %s: %s", f.name, exc)
        for f in sorted(manual_dir.glob("*.xlsx")):
            try:
                df = pd.read_excel(f)
                df["source"] = f"CDPHE_manual_{f.stem}"
                frames.append(df)
                log.info("[CO manual] Loaded %s (%d rows)", f.name, len(df))
            except Exception as exc:
                log.warning("[CO manual] Failed %s: %s", f.name, exc)

    if not frames:
        log.warning(
            "[CO CDPHE] No data available. For historical CO data:\n"
            "  1. Download annual files from the CDPHE WNV data page\n"
            "  2. Place in ./manual_data/co_cdphe/\n"
            "  3. Re-run — or use CDC ArboNET (Source F) for annual counts."
        )
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    def assign_key(county_raw: str) -> str:
        u = str(county_raw).upper()
        if "LARIMER" in u: return "Larimer_CO"
        if "BOULDER" in u: return "Boulder_CO"
        return "Other_CO"

    df["county_key"]  = df.get("county_raw", df.get("county", "")).apply(assign_key)
    df["county_name"] = df["county_key"].map({
        "Larimer_CO": "Larimer County, CO",
        "Boulder_CO": "Boulder County, CO",
    }).fillna("Colorado (other)")
    df["fips"] = df["county_key"].map({
        "Larimer_CO": "08069",
        "Boulder_CO": "08013",
    }).fillna("")

    return df[df["county_key"].isin(["Larimer_CO", "Boulder_CO"])].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE E — Dallas County, TX (DSHS weekly arbovirus PDF reports)
# ══════════════════════════════════════════════════════════════════════════════

DSHS_REPORTS_URL = (
    "https://www.dshs.texas.gov/mosquito-borne-diseases/"
    "dshs-arbovirus-weekly-activity-reports"
)


def _list_dshs_pdf_urls(year: int) -> list:
    """Scrape DSHS report index for PDF links matching the given year."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("[Dallas/DSHS] beautifulsoup4 not installed: pip install beautifulsoup4 lxml")
        return []

    r = _get(DSHS_REPORTS_URL, timeout=30)
    if r is None:
        return []

    soup     = BeautifulSoup(r.text, "lxml")
    pdf_urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if str(year) in href and href.lower().endswith(".pdf"):
            if not href.startswith("http"):
                href = "https://www.dshs.texas.gov" + href
            pdf_urls.append(href)
    return list(dict.fromkeys(pdf_urls))


# DSHS Table 3 column layout for WNV (confirmed from live PDFs March 2026):
#   County | CAL_M | CAL_H | EEEV_M | EEEV_V | EEEV_SC | EEEV_H
#          | SLEV_M | SLEV_SC | SLEV_H
#          | WNV_M | WNV_A | WNV_V | WNV_SC | WNV_WNF | WNV_WNND | WNV_PVD | TOTAL
#
# IMPORTANT facts about DSHS PDFs:
#   1. WNV_M = positive mosquito pools (NOT total pools tested).
#      Total pools tested is never published in these PDFs.
#   2. All values are CUMULATIVE season totals, not weekly increments.
#      To get weekly increments: subtract previous week's PDF values.
#   3. The year comes from the PDF header, never from the table.
#   4. MIR cannot be computed from these PDFs (no total pools column).
#
# We extract per-week: WNV_M (positive pools), WNV_WNF (human fever cases),
# WNV_WNND (human neuroinvasive cases) for Dallas County.
# Weekly increment = this week's cumulative - last week's cumulative.


def _parse_dshs_pdf(pdf_bytes: bytes, year: int, week_num: int) -> Optional[dict]:
    """
    Extract Dallas County WNV data from a DSHS weekly arbovirus PDF.

    Parses Table 3 ("Other Arbovirus Activity by County") which contains
    cumulative season totals. Returns one dict per call with cumulative values;
    the caller computes weekly increments by differencing consecutive weeks.

    Columns captured:
      wnv_positive_pools  WNV_M  — cumulative positive mosquito pools
      wnv_avian           WNV_A  — cumulative positive avian detections
      wnv_vet             WNV_V  — cumulative positive veterinary cases
      wnv_sentinel_chk    WNV_SC — cumulative positive sentinel chickens
      wnv_fever_cases     WNV_WNF  — cumulative human WNV fever cases
      wnv_neuro_cases     WNV_WNND — cumulative human WNV neuroinvasive cases

    Requires: pip install pdfplumber
    NOTE: MIR cannot be computed — total pools tested is not in these PDFs.
    """
    try:
        import pdfplumber
    except ImportError:
        log.warning("[Dallas/DSHS] pdfplumber not installed: pip install pdfplumber")
        return None

    import logging
    # Suppress pdfminer gray color warnings (invalid float in old PDFs)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        log.warning("[Dallas/DSHS] pdfplumber failed to open PDF: %s", exc)
        return None

    # Table 3 starts after the "Table 3." header.
    # Find the section between "Table 3." and "Total Number of Reports"
    t3_match = re.search(
        r"Table 3\..*?County.*?WNV(.*?)Total Number of\s*Reports",
        full_text, re.DOTALL | re.IGNORECASE
    )
    if not t3_match:
        # Fallback: search in full text
        table_text = full_text
    else:
        table_text = t3_match.group(1)

    # Dallas row pattern in Table 3.
    # The row starts with "Dallas" followed by up to 17 numeric tokens.
    # Many counties have only WNV columns populated; CAL/EEEV/SLEV will be 0.
    # We anchor on "Dallas" at the start of a line (possibly with leading space).
    #
    # Pattern captures all numeric tokens on the Dallas line.
    # Table 3 column order (WNV section only, after SLEV):
    #   WNV_M  WNV_A  WNV_V  WNV_SC  WNV_WNF  WNV_WNND  [WNV_PVD]  TOTAL
    #
    # For counties active only in WNV, the CAL/EEEV/SLEV columns will be absent
    # or show "0", so we look for the Dallas line with >= 2 numbers and take
    # the first number as WNV_M (positive pools — always the largest number for Dallas).
    dallas_line_pat = re.compile(
        r"^[ \t]*Dallas[ \t]+((?:\d+[ \t]*)+)",
        re.MULTILINE | re.IGNORECASE
    )
    m = dallas_line_pat.search(table_text)
    if not m:
        log.debug("[Dallas/DSHS] No Dallas row found in year=%d week=%d", year, week_num)
        return None

    tokens = [int(x) for x in m.group(1).split()]
    if not tokens:
        return None

    # Table 3 WNV column order (confirmed from live DSHS PDFs March 2026):
    #   WNV_M  WNV_A  WNV_V  WNV_SC  WNV_WNF  WNV_WNND  [WNV_PVD]  TOTAL
    #   tok[0] tok[1] tok[2] tok[3]  tok[4]   tok[5]
    #
    # For Dallas, all 6 WNV sub-columns are typically present:
    #   283 2 2 6 2 8  (week 50 2025 example)
    #
    # For counties with fewer detections, some columns may be absent —
    # we use safe indexing with fallback to 0.
    wnv_m    = tokens[0]
    wnv_a    = tokens[1] if len(tokens) > 1 else 0
    wnv_v    = tokens[2] if len(tokens) > 2 else 0
    wnv_sc   = tokens[3] if len(tokens) > 3 else 0
    wnv_wnf  = tokens[4] if len(tokens) > 4 else 0
    wnv_wnnd = tokens[5] if len(tokens) > 5 else 0

    return {
        "year":                year,
        "week_num":            week_num,
        "report_week_label":   f"Week{week_num:02d}",
        "county_key":          "Dallas_TX",
        "county_name":         "Dallas County, TX",
        "fips":                "48113",
        # Cumulative season totals as of this week
        # Column order: WNV_M WNV_A WNV_V WNV_SC WNV_WNF WNV_WNND
        "cumul_positive_pools":  wnv_m,   # WNV_M: positive mosquito pools
        "cumul_avian":           wnv_a,   # WNV_A: positive avian detections
        "cumul_veterinary":      wnv_v,   # WNV_V: positive veterinary cases
        "cumul_sentinel_chk":    wnv_sc,  # WNV_SC: positive sentinel chickens
        "cumul_wnv_fever":       wnv_wnf, # human WNV fever cases
        "cumul_wnv_neuro":       wnv_wnnd,# human WNV neuroinvasive cases
        # pools_tested not available in DSHS PDFs (only positive pools reported)
        "pools_tested":          None,
        "positive_pools":        None,    # filled in after differencing
        "infection_rate":        None,    # cannot compute without total pools tested
        "source":                "DSHS_TX_PDF",
        "source_note":           "Cumulative season totals — use _diff_cumulative() for weekly counts.",
    }


def _week_num_from_url(url: str) -> int:
    """Extract MMWR week number from a DSHS PDF URL. Returns 0 if not found."""
    # URL patterns: week28.pdf  |  week_28.pdf  |  28aug2023week28.pdf
    m = re.search(r"week[_\s]*(\d{1,2})\.pdf", url, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: last 1-2 digit number before .pdf
    m = re.search(r"(\d{1,2})\.pdf$", url, re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _diff_cumulative(rows: list) -> list:
    """
    Convert cumulative DSHS season-total rows into weekly incremental rows.
    Input rows must be sorted by week_num ascending for one season/year.
    Returns new list with positive_pools = this_week - prev_week.
    """
    result = []
    prev_pools = 0
    for row in sorted(rows, key=lambda r: r.get("week_num", 0)):
        cumul = row.get("cumul_positive_pools") or 0
        row["positive_pools"] = max(0, cumul - prev_pools)   # never negative
        row["cumul_positive_pools_prev"] = prev_pools
        prev_pools = cumul
        result.append(row)
    return result


def fetch_dallas_dshs(start_year: int = START_YEAR,
                      end_year:   int = END_YEAR) -> pd.DataFrame:
    """
    Download DSHS weekly PDF reports and extract Dallas County WNV data.

    DSHS PDF facts (confirmed from live PDFs):
      - Table 3 contains cumulative season totals, not weekly counts.
      - Only WNV_M (positive pools) is reported — total pools tested is absent.
      - MIR cannot be computed from these PDFs alone.
      - Year comes from PDF header; week number comes from the URL filename.

    Strategy:
      - Parse EVERY weekly PDF for each year (not just the last).
      - Difference consecutive cumulative values to get weekly increments.
      - Weeks 1-19: expect 0 or very few positive pools (pre-season).
      - Weeks 20-45: active WNV season for Dallas.
      - Weeks 50-53: season-end summaries; treat week_num > 49 with caution.

    For full historical data:
      - Email wnv@dshs.texas.gov for county-level CSV
      - Place manual CSVs in ./manual_data/dallas_tx/ and re-run
    """
    log.info("[Dallas/DSHS] Scraping DSHS arbovirus report index ...")
    all_rows: list = []

    for year in range(start_year, end_year + 1):
        pdf_urls = _list_dshs_pdf_urls(year)
        if not pdf_urls:
            log.info("  Year %d: no PDFs found", year)
            continue

        # Sort by week number ascending so differencing works correctly
        url_week_pairs = []
        for url in pdf_urls:
            wn = _week_num_from_url(url)
            url_week_pairs.append((wn, url))
        url_week_pairs.sort(key=lambda x: x[0])

        log.info("  Year %d: %d PDFs — parsing all weekly reports ...",
                 year, len(url_week_pairs))

        year_rows = []
        for week_num, url in url_week_pairs:
            r = _get(url, timeout=60)
            if r is None:
                log.warning("    Week %d: download failed", week_num)
                continue
            row = _parse_dshs_pdf(r.content, year, week_num)
            if row:
                year_rows.append(row)
                log.debug("    Week %02d: cumul_positive_pools=%d",
                          week_num, row.get("cumul_positive_pools", 0))
            time.sleep(0.4)   # be polite to DSHS servers

        # Compute weekly increments from cumulative values
        if year_rows:
            year_rows = _diff_cumulative(year_rows)
            log.info("  Year %d: %d weekly records extracted", year, len(year_rows))
            all_rows.extend(year_rows)

    # Load manual files
    manual_dir = Path("./manual_data/dallas_tx")
    if manual_dir.exists():
        for f in sorted(manual_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                df["source"]      = f"Dallas_manual_{f.stem}"
                df["county_key"]  = "Dallas_TX"
                df["county_name"] = "Dallas County, TX"
                df["fips"]        = "48113"
                all_rows.extend(df.to_dict("records"))
                log.info("[Dallas manual] Loaded %s (%d rows)", f.name, len(df))
            except Exception as exc:
                log.warning("[Dallas manual] Failed %s: %s", f.name, exc)

    if not all_rows:
        log.warning(
            "[Dallas/DSHS] No data parsed.\n"
            "  Options for Dallas County WNV surveillance data:\n"
            "  1. Visit https://www.dallascounty.org/departments/dchhs/"
            "data-reports/arbovirus-surveillance.php\n"
            "  2. Email wnv@dshs.texas.gov for county-level historical data\n"
            "  3. Place CSVs in ./manual_data/dallas_tx/ and re-run\n"
            "  4. Use CDC ArboNET (Source F) for annual county-level counts."
        )
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE F — CDC ArboNET (all-county annual backstop)
# ══════════════════════════════════════════════════════════════════════════════

# Public ArboNET county-level WNV data via data.cdc.gov Socrata endpoint
CDC_ARBONET_COUNTY_URL = (
    "https://data.cdc.gov/api/views/5tbi-ypts/rows.csv?accessType=DOWNLOAD"
)
CDC_ARBONET_HUMAN_URL  = (
    "https://data.cdc.gov/api/views/d4h9-28fr/rows.csv?accessType=DOWNLOAD"
)


def fetch_cdc_arbonet() -> pd.DataFrame:
    """
    Download CDC ArboNET county-level WNV data (1999–2024).
    This is the annual backstop for TX, AZ, and CO where direct APIs
    require manual steps or are unavailable programmatically.
    Filters to the 6 target counties.
    """
    log.info("[CDC ArboNET] Downloading county-level WNV data ...")
    r = _get(CDC_ARBONET_COUNTY_URL, timeout=120)
    if r is None:
        log.info("[CDC ArboNET] County URL failed — trying human disease data ...")
        r = _get(CDC_ARBONET_HUMAN_URL, timeout=120)
    if r is None:
        log.error("[CDC ArboNET] Could not download data")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="replace")))
    except Exception as exc:
        log.error("[CDC ArboNET] CSV parse error: %s", exc)
        return pd.DataFrame()

    log.info("[CDC ArboNET] Downloaded %d rows, %d cols", len(df), len(df.columns))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    fips_col = next((c for c in df.columns if "fips" in c or "county_code" in c), None)
    if fips_col:
        df[fips_col] = df[fips_col].astype(str).str.zfill(5)
        df = df[df[fips_col].isin(TARGET_FIPS)].copy()
        df["county_key"]  = df[fips_col].map(lambda f: FIPS_META.get(f, (None, None))[0])
        df["county_name"] = df[fips_col].map(lambda f: FIPS_META.get(f, (None, ""))[1])
        df["fips"]        = df[fips_col]

    if df.empty:
        log.warning("[CDC ArboNET] No matching rows for target counties")
        return pd.DataFrame()

    df["source"] = "CDC_ArboNET"
    log.info("[CDC ArboNET] %d target-county rows", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Pre-flight access checks
# ══════════════════════════════════════════════════════════════════════════════

ACCESS_CHECKS: dict = {
    "chicago_api": {
        "label":  "Chicago Data Portal — Cook County IL (Socrata)",
        "url":    "https://data.cityofchicago.org/resource/jqe8-8r6s.json",
        "params": {"$limit": "1"},
    },
    "ca_wnv_csv": {
        "label":  "westnile.ca.gov — Los Angeles County CA (CSV download)",
        "url":    "https://westnile.ca.gov/download.php",
        "params": {"download_id": "1068"},
    },
    "vectorsurv_api": {
        "label":           "VectorSurv API — Maricopa County AZ (requires token)",
        "url":             "https://api.vectorsurv.org/v1/arthropod/pool",
        "params":          {},
        "requires_token":  True,
    },
    "cdphe_co": {
        "label":  "CDPHE — Colorado (Larimer + Boulder) HTML summary",
        "url":    CDPHE_DATA_URL,
        "params": {},
    },

    "cdc_arbonet": {
        "label":  "CDC ArboNET — all-county annual backstop",
        "url":    CDC_ARBONET_COUNTY_URL,
        "params": {},
    },
}


def run_preflight() -> dict:
    """Check all surveillance endpoints before pulling data."""
    print("\n" + "=" * 65)
    print("PRE-FLIGHT ACCESS CHECKS — Mosquito Surveillance")
    print("=" * 65)
    results: dict = {}

    for check_id, cfg in ACCESS_CHECKS.items():
        label          = cfg["label"]
        requires_token = cfg.get("requires_token", False)

        if requires_token and not VECTORSURV_TOKEN:
            print(f"  [SKIP] {label}")
            print("         -> VECTORSURV_TOKEN not set")
            results[check_id] = False
            continue

        req_headers: dict = {}
        if requires_token and VECTORSURV_TOKEN:
            req_headers["Authorization"] = f"Bearer {VECTORSURV_TOKEN}"

        try:
            r = requests.get(cfg["url"], params=cfg.get("params", {}),
                             headers=req_headers, timeout=20, allow_redirects=True)
            ok     = r.status_code < 400
            detail = f"HTTP {r.status_code}"
            if ok and cfg["url"].endswith(".json"):
                try:
                    body = r.json()
                    if not body:
                        ok, detail = False, "HTTP 200 but empty response"
                except Exception:
                    ok, detail = False, "HTTP 200 but non-JSON body"
        except requests.exceptions.Timeout:
            ok, detail = False, "Timed out after 20s"
        except Exception as exc:
            ok, detail = False, str(exc)

        symbol = "[OK]  " if ok else "[FAIL]"
        print(f"  {symbol} {label}")
        if not ok:
            print(f"         -> {detail}")
        results[check_id] = ok

    print()
    print("  Token status:")
    print(f"    VECTORSURV_TOKEN  : {'SET' if VECTORSURV_TOKEN  else 'NOT SET — Maricopa will be empty'}")
    print(f"    CHICAGO_APP_TOKEN : {'SET' if CHICAGO_APP_TOKEN else 'not set  (optional, higher rate limit)'}")
    print()

    viable = [k for k, v in results.items() if v]
    if viable:
        print(f"  Viable sources: {', '.join(viable)}")
    else:
        print("  [!!] NO VIABLE SOURCES FOUND. Check network access.")
    print("=" * 65)
    results["_any_viable"] = bool(viable)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Mosquito Surveillance Data Acquisition — WNV Project")
    print(f"Year range : {START_YEAR}–{END_YEAR}")

    access = run_preflight()
    if not access["_any_viable"]:
        log.error("No accessible sources. Check network access.")
        sys.exit(1)

    all_agg:   list = []   # county-week aggregates (model features)
    all_pools: list = []   # pool-level records (where available)

    # ── Source A: Cook County / Chicago ──────────────────────────────────
    if access.get("chicago_api"):
        log.info("\n[Step 1] Cook County IL — Chicago Data Portal")
        agg, pools = fetch_chicago_wnv()
        if not agg.empty:   all_agg.append(agg)
        if not pools.empty: all_pools.append(pools)

    # ── Source B: Los Angeles / CA WNV ───────────────────────────────────
    if access.get("ca_wnv_csv"):
        log.info("\n[Step 2] Los Angeles County CA — westnile.ca.gov")
        agg, pools = fetch_ca_wnv()
        if not agg.empty:   all_agg.append(agg)
        if not pools.empty: all_pools.append(pools)

    # ── Source C: Maricopa / VectorSurv ──────────────────────────────────
    if access.get("vectorsurv_api"):
        log.info("\n[Step 3] Maricopa County AZ — VectorSurv API")
        agg, pools = fetch_maricopa_vectorsurv()
        if not agg.empty:   all_agg.append(agg)
        if not pools.empty: all_pools.append(pools)
    else:
        log.info("\n[Step 3] Maricopa County AZ — SKIPPED (no token)")

    # ── Source D: Colorado / CDPHE ────────────────────────────────────────
    log.info("\n[Step 4] Larimer + Boulder CO — CDPHE")
    df_co = fetch_colorado_cdphe()
    if not df_co.empty: all_agg.append(df_co)

    # ── Source E: Dallas — manual files only (PDF pull done separately) ─────
    log.info("\n[Step 5] Dallas County TX — loading from manual_data/dallas_tx/")
    manual_dir = Path("./manual_data/dallas_tx")
    if manual_dir.exists():
        for f in sorted(manual_dir.glob("*.csv")):
            try:
                df = pd.read_csv(f)
                df["county_key"]  = "Dallas_TX"
                df["county_name"] = "Dallas County, TX"
                df["fips"]        = "48113"
                df["source"]      = f"Dallas_manual_{f.stem}"
                all_agg.append(df)
                log.info("  Loaded %s (%d rows)", f.name, len(df))
            except Exception as exc:
                log.warning("  Failed to load %s: %s", f.name, exc)
    else:
        log.info("  No manual_data/dallas_tx/ directory found — skipping Dallas")
        log.info("  Place dallas CSV file(s) there and re-run to include Dallas data")

    # ── Source F: CDC ArboNET backstop ────────────────────────────────────
    if access.get("cdc_arbonet"):
        log.info("\n[Step 6] CDC ArboNET — all-county annual backstop")
        df_arbonet = fetch_cdc_arbonet()
        if not df_arbonet.empty: all_agg.append(df_arbonet)

    # ── Save outputs — timestamped filenames prevent overwriting ─────────
    log.info("\n[Step 7] Saving outputs ...")
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_csv(df: pd.DataFrame, base_name: str) -> Optional[Path]:
        """Save df to output/<base_name>_<timestamp>.csv — never overwrites."""
        if df.empty:
            return None
        out = OUTPUT_DIR / f"{base_name}_{run_ts}.csv"
        df.to_csv(out, index=False)
        log.info("  -> %-40s %d rows x %d cols", out.name, len(df), len(df.columns))
        return out

    saved: list = []

    if all_agg:
        df_out = pd.concat(all_agg, ignore_index=True)
        p = _save_csv(df_out, "mosquito_surveillance_county_week")
        if p: saved.append(p)
    else:
        log.warning("  No aggregate data collected across any source")

    if all_pools:
        df_pools_out = pd.concat(all_pools, ignore_index=True)
        p = _save_csv(df_pools_out, "mosquito_surveillance_pool_level")
        if p: saved.append(p)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"Output files (run: {run_ts}):")
    for f in saved:
        print(f"  {f.name:60s}  {f.stat().st_size:>10,} bytes")
    if not saved:
        print("  (no files written — no data collected)")
    print()
    print("Column reference — county-week aggregate (WNV model features):")
    col_ref = [
        ("county_key",         "County identifier (e.g. Cook_IL)"),
        ("fips",               "5-digit county FIPS code"),
        ("year",               "Surveillance year"),
        ("mmwr_week",          "MMWR epidemiological week number"),
        ("total_pools_tested", "Total mosquito pools tested that week"),
        ("positive_pools",     "Pools testing WNV-positive"),
        ("pct_positive",       "% of pools that tested positive"),
        ("total_mosquitoes",   "Total mosquitoes across all tested pools"),
        ("culex_pools",        "Number of Culex species pools"),
        ("culex_pct",          "% of pools that are Culex spp."),
        ("mir_per_1000",       "Minimum Infection Rate per 1,000 mosquitoes"),
        ("source",             "Data source identifier"),
    ]
    for col, desc in col_ref:
        print(f"  {col:26s}  {desc}")
    print()
    print("Manual data fallback directories:")
    print("  ./manual_data/co_cdphe/   <- CDPHE CO annual Excel/CSV files")
    print("  ./manual_data/dallas_tx/  <- Dallas County surveillance CSVs")
    print("=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()