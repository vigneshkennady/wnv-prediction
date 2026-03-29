# WNV Weather Data Retrieval — Setup & Run Guide

## Overview

Two-script pipeline to retrieve and engineer NOAA weather features for WNV prediction research.

| Script | Purpose |
|---|---|
| `wnv_weather_retrieval.py` | Hits NOAA CDO v2 API + ISD-Lite to download raw daily weather data (2000–2024) |
| `wnv_feature_engineering.py` | Aggregates to epiweeks, adds lag/rolling features |

---

## Setup

```bash
pip install requests pandas numpy tqdm pyarrow
```

---

## Run Order

```bash
# Step 1 — Download (takes ~30–90 min; API rate limited to 5 req/sec)
python wnv_weather_retrieval.py

# Step 2 — Feature engineering
python wnv_feature_engineering.py
```

The retrieval script is **resume-safe**: each county is cached as a `.parquet` file.
If the script is interrupted, re-running will skip already-completed counties.

---

## Variables Retrieved

| Column | Source | Description |
|---|---|---|
| `TAVG` | GHCND | Average daily temperature (°C) |
| `TMAX` | GHCND | Maximum daily temperature (°C) |
| `TMIN` | GHCND | Minimum daily temperature (°C) |
| `PRCP` | GHCND | Daily precipitation (mm) |
| `WIND` | GHCND (AWND) or ISD | Average daily wind speed (m/s) |
| `DEWP_mean` | ISD-Lite | Mean daily dew point temperature (°C) |
| `RH` | Derived | Relative humidity (%) via August-Roche-Magnus formula |

### RH Derivation Formula
```
RH = 100 × exp(17.625 × Td / (243.04 + Td))
           / exp(17.625 × T  / (243.04 + T ))
```
Where `T` = TAVG (°C) and `Td` = DEWP_mean (°C). This is the standard meteorological formula (WMO-approved).

---

## Output Files

```
wnv_weather_data/
├── Larimer_CO_2000_2024.csv
├── Boulder_CO_2000_2024.csv
├── Dallas_TX_2000_2024.csv
├── Maricopa_AZ_2000_2024.csv
├── Cook_IL_2000_2024.csv
├── LosAngeles_CA_2000_2024.csv
├── ALL_COUNTIES_weather_2000_2024.csv         ← combined raw daily
├── ALL_COUNTIES_weather_daily_enriched.csv    ← daily + heat index + GDD
├── ALL_COUNTIES_weather_features.csv          ← weekly epiweek features + lags
├── summary_stats.csv                           ← QA coverage stats
└── retrieval.log
```

---

## ISD Stations Used

| County | ISD Station IDs | Airport |
|---|---|---|
| Larimer CO | 720533-99999, 726400-24018 | Fort Collins / DEN area |
| Boulder CO | 726396-99999, 726400-24018 | Boulder / DEN |
| Dallas TX | 722590-03927, 722583-12960 | DFW / Love Field |
| Maricopa AZ | 722780-23183, 722784-99999 | PHX Sky Harbor |
| Cook IL | 725300-94846, 725340-14819 | O'Hare / Midway |
| Los Angeles CA | 722950-23174, 722956-03167 | LAX / Burbank |

---

## Notes on Data Coverage

- **TAVG**: Not all GHCND stations report TAVG directly. If missing, compute as `(TMAX + TMIN) / 2` in post-processing.
- **RH**: GHCND does not include relative humidity. This pipeline derives it from ISD-Lite dew point data.
- **WIND**: GHCND AWND variable covers ~70% of stations. Falls back to ISD wind speed where missing.
- **Historical gap**: Pre-2000 data exists but station coverage is sparser.

---

## County FIPS Reference

| County | FIPS |
|---|---|
| Larimer CO | 08069 |
| Boulder CO | 08013 |
| Dallas TX | 48113 |
| Maricopa AZ | 04013 |
| Cook IL | 17031 |
| Los Angeles CA | 06037 |
