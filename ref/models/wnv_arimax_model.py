#!/usr/bin/env python3
"""
West Nile Virus (WNV) County-Level Human Case Prediction — ARIMAX Model
========================================================================
Research Question:
    Can we predict county-level WNV human case counts several weeks in
    advance using historical surveillance, meteorological, land-use,
    and demographic data?

Model   : ARIMAX — AutoRegressive Integrated Moving Average with
          eXogenous regressors (SARIMAX from statsmodels).

          Pure ARIMA is a univariate model that uses only the target
          series' own past values.  ARIMAX extends ARIMA with an
          exogenous (X) regressor matrix, allowing weather, mosquito
          surveillance, land-use, and demographic covariates to drive
          predictions alongside the autoregressive/moving-average
          components — making it well-suited for this multi-source
          epidemiological forecasting task.

          Library : statsmodels  (SARIMAX class)
                    pandas, numpy, scikit-learn (preprocessing & metrics)

Target  : Weekly new WNV human cases, FORECAST_LEAD weeks ahead
Counties: Boulder-CO, Cook-IL, Dallas-TX, Larimer-CO,
          LosAngeles-CA, Maricopa-AZ

Strategy: One ARIMAX model is fitted PER COUNTY on that county's
          time series.  Predictions are collected across counties
          and aggregated for reporting.

Data files (read from  ref/  folder inside the repo):
    ref/wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv
    ref/demographics_combined.csv
    ref/cdl_strategy_d_pc.csv
    ref/mosquito_surveillance_county_week.csv
    ref/wnv_human_cases_county_year.csv

Install:
    pip install statsmodels scikit-learn pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ── statsmodels ARIMAX (SARIMAX API) ─────────────────────────────────────────
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools         import adfuller, acf, pacf

# ── scikit-learn — preprocessing & evaluation metrics ────────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

DIVIDER  = "=" * 80
DIVIDER2 = "-" * 80

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# >>>  Set BASE to the path of your local clone of the repo  <<<
BASE = "ref/"

FORECAST_LEAD = 4    # weeks ahead to forecast (the prediction horizon)
RANDOM_STATE  = 42

# ARIMA order (p, d, q) applied to every county series.
# p=2  : autoregressive order — model looks back 2 steps
# d=0  : differencing order  — series is made stationary via log1p transform
#         instead of differencing (preserves interpretability)
# q=1  : moving-average order — smooths one-step residual shocks
ARIMA_ORDER = (2, 0, 1)

# Seasonal ARIMA order (P, D, Q, S) — weekly seasonality period = 52
# P=1, D=0, Q=1, S=52 captures annual WNV peak cycles
SEASONAL_ORDER = (1, 0, 1, 52)

# Seasonal distribution params per county  (peak MMWR week, sigma)
SEASON_PARAMS = {
    "Cook_IL"      : (33, 4.5),
    "LosAngeles_CA": (33, 6.0),
    "Maricopa_AZ"  : (31, 5.5),
    "Larimer_CO"   : (32, 4.0),
    "Boulder_CO"   : (32, 4.0),
    "Dallas_TX"    : (33, 5.0),
}

# Active season window per county  (first active MMWR week, last active week)
SEASON_WINDOW = {
    "Cook_IL"      : (22, 42),
    "LosAngeles_CA": (20, 44),
    "Maricopa_AZ"  : (20, 44),
    "Larimer_CO"   : (24, 40),
    "Boulder_CO"   : (24, 40),
    "Dallas_TX"    : (22, 44),
}

COUNTIES = [
    "Boulder_CO", "Cook_IL", "Dallas_TX",
    "Larimer_CO", "LosAngeles_CA", "Maricopa_AZ",
]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("STEP 1: DATA LOADING")
print(DIVIDER)

weather = pd.read_csv(
    BASE + "wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv",
    parse_dates=["date"])

demographics = pd.read_csv(BASE + "demographics_combined.csv")
demographics.replace(-999, np.nan, inplace=True)

land_use     = pd.read_csv(BASE + "cdl_strategy_d_pc.csv")
surveillance = pd.read_csv(BASE + "mosquito_surveillance_county_week.csv")
human_cases  = pd.read_csv(BASE + "wnv_human_cases_county_year.csv")

print(f"  Weather         : {len(weather):>6,} rows")
print(f"  Demographics    : {len(demographics):>6,} rows")
print(f"  Land Use        : {len(land_use):>6,} rows")
print(f"  Mosquito Surv.  : {len(surveillance):>6,} rows  "
      f"({surveillance['county_key'].nunique()} counties)")
print(f"  WNV Human Cases : {len(human_cases):>6,} rows  "
      f"({human_cases['county_key'].nunique()} counties, "
      f"{human_cases['year'].min()}–{human_cases['year'].max()})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATA EXPLORATION / EDA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 2: DATA EXPLORATION / EDA")
print(DIVIDER)

datasets = {
    "Weather"        : weather,
    "Demographics"   : demographics,
    "Land Use"       : land_use,
    "Mosquito Surv." : surveillance,
    "WNV Human Cases": human_cases,
}

# ── 2a. First 5 rows ─────────────────────────────────────────────────────────
print("\n── 2a. First 5 Rows of Each Dataset ──\n")
for name, df in datasets.items():
    print(f"  >>> {name}")
    print(df.head().to_string())
    print()

# ── 2b. Data types & null counts ─────────────────────────────────────────────
print("\n── 2b. Data Types & Null Value Counts ──\n")
for name, df in datasets.items():
    print(f"  >>> {name}")
    info = pd.DataFrame({
        "dtype"     : df.dtypes,
        "null_count": df.isnull().sum(),
        "null_pct"  : (df.isnull().sum() / len(df) * 100).round(2),
    })
    print(info.to_string())
    print()

# ── 2c. Summary statistics ────────────────────────────────────────────────────
print("\n── 2c. Summary Statistics ──\n")
for name, df in datasets.items():
    print(f"  >>> {name}")
    print(df.describe(include="all").to_string())
    print()

# ── 2d. Neuroinvasive spotlight by county ─────────────────────────────────────
print("\n── 2d. WNV Human Cases — Neuroinvasive Summary by County ──\n")
hc_active = human_cases[human_cases["total_cases"] > 0].copy()
hc_active["pct_neuroinvasive"] = (
    hc_active["neuroinvasive"] / hc_active["total_cases"] * 100).round(1)

spotlight = hc_active.groupby("county_key").agg(
    active_years      = ("year",             "count"),
    total_cases_sum   = ("total_cases",       "sum"),
    neuro_sum         = ("neuroinvasive",      "sum"),
    deaths_sum        = ("deaths",             "sum"),
    peak_year         = ("total_cases",
                         lambda x: hc_active.loc[x.idxmax(), "year"]),
    peak_cases        = ("total_cases",        "max"),
    avg_pct_neuro     = ("pct_neuroinvasive",  "mean"),
    case_fatality_pct = ("deaths",
                         lambda x: round(
                             x.sum() /
                             hc_active.loc[x.index, "total_cases"].sum() * 100,
                             1)),
).reset_index()
print(spotlight.to_string(index=False))

# ── 2e. Time-series stationarity check (ADF test) per county ─────────────────
print("\n── 2e. Augmented Dickey-Fuller Stationarity Test (per county) ──\n")
print("  A p-value < 0.05 indicates the series is stationary (no unit root).")
print(f"  {'County':<18} {'ADF Stat':>12} {'p-value':>10} {'Stationary?':>12}")
print("  " + DIVIDER2)

# Build a quick annual-series placeholder for the ADF test
# (full weekly series built in Step 3; here we use annual totals)
for ck in COUNTIES:
    sub = human_cases[human_cases["county_key"] == ck].sort_values("year")
    if len(sub) < 8:
        print(f"  {ck:<18} {'(insufficient data)':>24}")
        continue
    series = sub["total_cases"].values.astype(float)
    try:
        adf_stat, p_val, *_ = adfuller(series, autolag="AIC")
        stat_label = "Yes" if p_val < 0.05 else "No  (d≥1 needed)"
        print(f"  {ck:<18} {adf_stat:>12.4f} {p_val:>10.4f} {stat_label:>12}")
    except Exception as e:
        print(f"  {ck:<18}  ADF failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 3: FEATURE ENGINEERING & PREPROCESSING")
print(DIVIDER)

# ── 3a. Weekly weather aggregation ───────────────────────────────────────────
def make_county_key(row):
    return (f"{row['county'].replace(' County','').replace(' ','')}"
            f"_{row['state']}")

weather["county_key"] = weather.apply(make_county_key, axis=1)
weather["year"]       = weather["date"].dt.year
weather["week_num"]   = weather["date"].dt.isocalendar().week.astype(int)

weather_weekly = (
    weather
    .groupby(["county_key", "year", "week_num"])
    .agg(
        TAVG_mean  = ("TAVG",      "mean"),
        TAVG_std   = ("TAVG",      "std"),
        TMAX_max   = ("TMAX",      "max"),
        TMIN_min   = ("TMIN",      "min"),
        PRCP_sum   = ("PRCP",      "sum"),
        PRCP_max   = ("PRCP",      "max"),
        WIND_mean  = ("WIND",      "mean"),
        DEWP_mean  = ("DEWP_mean", "mean"),
        RH_mean    = ("RH",        "mean"),
        temp_range = ("TAVG",      lambda x: x.max() - x.min()),
    )
    .reset_index()
)
weather_weekly.sort_values(["county_key", "year", "week_num"], inplace=True)

# Lagged weather (1, 2, 4 weeks) — key ARIMAX exogenous covariates
for lag in [1, 2, 4]:
    for col in ["TAVG_mean", "PRCP_sum", "RH_mean", "DEWP_mean", "WIND_mean"]:
        weather_weekly[f"{col}_lag{lag}"] = (
            weather_weekly.groupby("county_key")[col].shift(lag))

# Rolling aggregates
weather_weekly["TAVG_roll4"] = (
    weather_weekly.groupby("county_key")["TAVG_mean"]
    .transform(lambda x: x.rolling(4, min_periods=1).mean()))
weather_weekly["PRCP_cumul4"] = (
    weather_weekly.groupby("county_key")["PRCP_sum"]
    .transform(lambda x: x.rolling(4, min_periods=1).sum()))

# Growing-degree days — biological proxy for Culex mosquito activity
weather_weekly["gdd_proxy"]  = (weather_weekly["TAVG_mean"] - 14).clip(lower=0)
weather_weekly["gdd_cumul8"] = (
    weather_weekly.groupby("county_key")["gdd_proxy"]
    .transform(lambda x: x.rolling(8, min_periods=1).sum()))

print(f"\n  [3a] Weekly weather: {weather_weekly.shape[1]} columns, "
      f"{len(weather_weekly):,} rows")

# ── 3b. Distribute annual human case totals → weekly estimates ────────────────
def gaussian_weights(peak_wk, sigma, start_wk, end_wk):
    wks = np.arange(start_wk, end_wk + 1)
    w   = np.exp(-0.5 * ((wks - peak_wk) / sigma) ** 2)
    w  /= w.sum()
    return wks, w

print("\n  [3b] Distributing annual CDC/ArboNET totals to weekly estimates…")
weekly_human_rows = []
for _, yr_row in human_cases.iterrows():
    ck    = yr_row["county_key"]
    yr    = yr_row["year"]
    total = int(yr_row["total_cases"])
    neuro = int(yr_row["neuroinvasive"])

    pk, sg      = SEASON_PARAMS.get(ck,    (33, 5.0))
    s_wk, e_wk  = SEASON_WINDOW.get(ck,   (22, 44))
    act_wks, wts = gaussian_weights(pk, sg, s_wk, e_wk)

    seed = int(abs(hash((ck, yr)))) % (2 ** 31)
    rng  = np.random.default_rng(seed)

    total_wkly = rng.multinomial(total, wts) if total > 0 else np.zeros(len(wts), int)
    neuro_wkly = rng.multinomial(neuro, wts) if neuro > 0 else np.zeros(len(wts), int)
    non_n_wkly = np.clip(total_wkly - neuro_wkly, 0, None)

    act_map = dict(zip(act_wks.tolist(),
                       zip(total_wkly, neuro_wkly, non_n_wkly)))
    for wk in range(1, 53):
        tv, nv, nn = act_map.get(wk, (0, 0, 0))
        weekly_human_rows.append({
            "county_key"          : ck,
            "year"                : yr,
            "week_num"            : wk,
            "wk_human_total"      : int(tv),
            "wk_neuroinvasive"    : int(nv),
            "wk_non_neuroinvasive": int(nn),
        })

df_wk_human = pd.DataFrame(weekly_human_rows)
df_wk_human.sort_values(["county_key", "year", "week_num"], inplace=True)
print(f"     Weekly human-case rows generated: {len(df_wk_human):,}")

# ── 3c. Annual human-case lagged features ─────────────────────────────────────
hc_feats = human_cases[
    ["county_key", "year", "total_cases", "neuroinvasive",
     "non_neuroinvasive", "deaths"]
].copy()
hc_feats.sort_values(["county_key", "year"], inplace=True)

hc_feats["prev_yr_neuroinvasive"] = hc_feats.groupby("county_key")["neuroinvasive"].shift(1)
hc_feats["prev_yr_total_cases"]   = hc_feats.groupby("county_key")["total_cases"].shift(1)
hc_feats["prev_yr_deaths"]        = hc_feats.groupby("county_key")["deaths"].shift(1)
hc_feats["neuro_3yr_avg"]         = (
    hc_feats.groupby("county_key")["neuroinvasive"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()))
hc_feats["neuro_yoy_change"]      = (
    hc_feats["prev_yr_neuroinvasive"]
    - hc_feats.groupby("county_key")["neuroinvasive"].shift(2))
hc_feats["prev_yr_neuro_pct"]     = (
    hc_feats["prev_yr_neuroinvasive"]
    / hc_feats["prev_yr_total_cases"].replace(0, np.nan)
).fillna(0)

hc_lags = hc_feats[[
    "county_key", "year",
    "prev_yr_neuroinvasive", "prev_yr_total_cases", "prev_yr_deaths",
    "neuro_3yr_avg", "neuro_yoy_change", "prev_yr_neuro_pct",
]].copy()
print(f"\n  [3c] Annual human-case lag features: {len(hc_lags.columns)-2} variables")

# ── 3d. Mosquito surveillance lagged features ──────────────────────────────────
surveillance["positive_pools"] = pd.to_numeric(
    surveillance["positive_pools"], errors="coerce").fillna(0)
surveillance["infection_rate"] = pd.to_numeric(
    surveillance["infection_rate"], errors="coerce").fillna(0)
surveillance.sort_values(["county_key", "year", "week_num"], inplace=True)

for lag in [1, 2, 4]:
    surveillance[f"pos_pools_lag{lag}"]   = (
        surveillance.groupby(["county_key", "year"])["positive_pools"].shift(lag))
    surveillance[f"infect_rate_lag{lag}"] = (
        surveillance.groupby(["county_key", "year"])["infection_rate"].shift(lag))

surv_cols = (
    ["county_key", "year", "week_num", "positive_pools", "infection_rate"]
    + [f"pos_pools_lag{l}"   for l in [1, 2, 4]]
    + [f"infect_rate_lag{l}" for l in [1, 2, 4]]
)
surv_feats = surveillance[surv_cols].copy()
print(f"\n  [3d] Mosquito surveillance lag features: {len(surv_cols)-3} variables")

# ── 3e. Assemble master weekly frame ──────────────────────────────────────────
backbone = weather_weekly[["county_key", "year", "week_num"]].copy()

merged = backbone.merge(df_wk_human,  on=["county_key", "year", "week_num"], how="left")
merged = merged.merge(weather_weekly, on=["county_key", "year", "week_num"], how="left")
merged = merged.merge(hc_lags,        on=["county_key", "year"],             how="left")
merged = merged.merge(surv_feats,     on=["county_key", "year", "week_num"], how="left")

demo_keep = ["county_key", "year", "total_pop", "median_hh_income",
             "housing_units", "pop_65_plus", "pct_65_plus", "poverty_rate",
             "land_area_sqmi", "pop_density_per_sqmi",
             "housing_density_per_sqmi", "saipe_median_income",
             "saipe_poverty_rate"]
merged = merged.merge(
    demographics[[c for c in demo_keep if c in demographics.columns]],
    on=["county_key", "year"], how="left")

lu_keep = ["county_key", "year", "total_acres",
           "cdl_cropland_pct", "cdl_developed_pct", "cdl_wetlands_pct",
           "cdl_pasture_hay_pct", "cdl_fallow_idle_pct",
           "cdl_corn_pct", "cdl_rice_pct", "cdl_aquaculture_pct"]
merged = merged.merge(
    land_use[[c for c in lu_keep if c in land_use.columns]],
    on=["county_key", "year"], how="left")

merged.sort_values(["county_key", "year", "week_num"], inplace=True)
print(f"\n  [3e] Master merged dataset: "
      f"{merged.shape[0]:,} rows × {merged.shape[1]} columns")

# ── 3f. Cyclical temporal features ────────────────────────────────────────────
merged["sin_week"]      = np.sin(2 * np.pi * merged["week_num"] / 52)
merged["cos_week"]      = np.cos(2 * np.pi * merged["week_num"] / 52)
merged["in_wnv_season"] = merged["week_num"].between(26, 40).astype(int)

# ── 3g. FORECAST_LEAD-week-ahead target (shift target forward) ────────────────
TARGET_COL = f"target_{FORECAST_LEAD}wk_ahead"
merged[TARGET_COL] = (
    merged.groupby("county_key")["wk_human_total"].shift(-FORECAST_LEAD))

print(f"\n  [3f–3g] Cyclical temporal encodings + "
      f"{FORECAST_LEAD}-week-ahead target created.")

# ── 3h. Exogenous feature selection for ARIMAX ────────────────────────────────
# ARIMAX works best with a compact, well-curated exogenous (X) matrix.
# Too many correlated regressors inflate the covariance matrix and cause
# numerical instability.  We select the most epidemiologically meaningful
# and empirically least collinear subset.
EXOG_COLS = [
    # Weather (most WNV-predictive variables)
    "TAVG_mean",          # current weekly temperature
    "PRCP_sum",           # current weekly precipitation
    "gdd_cumul8",         # 8-week growing-degree days (Culex activity proxy)
    "RH_mean",            # relative humidity
    "TAVG_roll4",         # 4-week rolling temperature
    "PRCP_cumul4",        # 4-week cumulative precipitation
    # Lagged weather
    "TAVG_mean_lag1",     # temperature 1 week ago
    "TAVG_mean_lag2",     # temperature 2 weeks ago
    "PRCP_sum_lag1",      # precipitation 1 week ago
    "PRCP_sum_lag4",      # precipitation 4 weeks ago
    # Mosquito surveillance
    "positive_pools",     # current positive pools
    "pos_pools_lag1",     # positive pools 1 week ago
    "infect_rate_lag1",   # infection rate 1 week ago
    # Annual human-case history
    "prev_yr_neuroinvasive",  # prior-year neuroinvasive count (★ strongest signal)
    "neuro_3yr_avg",          # 3-year rolling mean neuroinvasive
    "neuro_yoy_change",       # year-over-year neuroinvasive delta
    # Demographics (slow-changing contextual)
    "pop_density_per_sqmi",
    "poverty_rate",
    # Land-use
    "cdl_developed_pct",
    "cdl_wetlands_pct",
    # Cyclical time
    "sin_week",
    "cos_week",
    "in_wnv_season",
]
# Retain only columns that actually exist after the merge
EXOG_COLS = [c for c in EXOG_COLS if c in merged.columns]
print(f"\n  [3h] ARIMAX exogenous feature set: {len(EXOG_COLS)} variables")
for i, c in enumerate(EXOG_COLS, 1):
    print(f"       {i:>2}. {c}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINAL DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 4: FINAL DATASET PREPARATION")
print(DIVIDER)

model_df = merged.dropna(subset=[TARGET_COL]).copy()

# Keep active-season window per county (±4 week buffer)
def in_active_window(row):
    s, e = SEASON_WINDOW.get(row["county_key"], (20, 45))
    return (s - 4) <= row["week_num"] <= (e + 4)

mask     = model_df.apply(in_active_window, axis=1)
model_df = model_df[mask].copy()
model_df.sort_values(["county_key", "year", "week_num"], inplace=True)

print(f"  Rows after active-season filter : {len(model_df):,}")
print(f"  Counties                        : "
      f"{sorted(model_df['county_key'].unique())}")
print(f"  Year range                      : "
      f"{int(model_df['year'].min())}–{int(model_df['year'].max())}")
print(f"  Non-zero target weeks           : "
      f"{(model_df[TARGET_COL] > 0).sum():,}")

# ── Impute NaNs in exogenous columns with column medians ─────────────────────
for col in EXOG_COLS:
    if col in model_df.columns:
        model_df[col] = model_df[col].fillna(model_df[col].median())

# ── log1p-transform the target ────────────────────────────────────────────────
# ARIMAX assumes a (weakly) stationary, near-Gaussian series.
# WNV counts are zero-inflated and right-skewed; log1p compresses the range
# and reduces heteroscedasticity without requiring integer differencing.
model_df["target_log"] = np.log1p(model_df[TARGET_COL])

# ── StandardScaler on exogenous block ────────────────────────────────────────
scaler = StandardScaler()

# Fit scaler on training portion (80 %) to avoid data leakage
all_sorted = model_df.sort_values(["year", "week_num"])
global_split = int(len(all_sorted) * 0.80)
scaler.fit(all_sorted[EXOG_COLS].iloc[:global_split])

model_df[EXOG_COLS] = scaler.transform(model_df[EXOG_COLS])

print(f"\n  Target log-transformed;  exogenous columns StandardScaled.")
print(f"  Exogenous feature count : {len(EXOG_COLS)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL BUILDING  (per-county ARIMAX / SARIMAX)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print(f"STEP 5: MODEL BUILDING — ARIMAX  "
      f"(order={ARIMA_ORDER}, seasonal={SEASONAL_ORDER}, "
      f"lead={FORECAST_LEAD} weeks)")
print(DIVIDER)

print(f"""
  ARIMAX architecture notes
  ─────────────────────────
  Model class : statsmodels SARIMAX
  ARIMA order : (p={ARIMA_ORDER[0]}, d={ARIMA_ORDER[1]}, q={ARIMA_ORDER[2]})
    p = {ARIMA_ORDER[0]}  → AR terms: model uses its own 2 lagged values
    d = {ARIMA_ORDER[1]}  → no integer differencing (series stationarised via log1p)
    q = {ARIMA_ORDER[2]}  → MA term: smooths 1-step forecast error shocks
  Seasonal    : (P={SEASONAL_ORDER[0]}, D={SEASONAL_ORDER[1]}, Q={SEASONAL_ORDER[2]}, S={SEASONAL_ORDER[3]})
    S = 52 captures annual WNV transmission cycles
  Exogenous X : {len(EXOG_COLS)} weather / surveillance / demographic / land-use regressors
  Strategy    : one model fitted per county on its own time series
  Fit method  : maximum likelihood (Kalman-filter state-space representation)
  Train/Test  : temporal 80/20 split per county
""")

county_results = {}   # store per-county fitted models + test data

for ck in COUNTIES:
    print(f"  {'─'*60}")
    print(f"  County: {ck}")

    sub = model_df[model_df["county_key"] == ck].sort_values(["year","week_num"]).copy()
    if len(sub) < 30:
        print(f"    ⚠  Skipped — insufficient rows ({len(sub)})")
        continue

    y   = sub["target_log"].values
    X   = sub[EXOG_COLS].values
    n   = len(sub)
    sp  = int(n * 0.80)

    y_train, y_test_raw = y[:sp],  y[sp:]
    X_train, X_test_raw = X[:sp],  X[sp:]
    meta_test           = sub.iloc[sp:][["year", "week_num", TARGET_COL]].copy()

    print(f"    Train: {sp} weeks  |  Test: {n - sp} weeks")

    try:
        mdl = SARIMAX(
            endog          = y_train,
            exog           = X_train,
            order          = ARIMA_ORDER,
            seasonal_order = SEASONAL_ORDER,
            enforce_stationarity    = False,
            enforce_invertibility   = False,
            concentrate_scale       = True,
        )
        res = mdl.fit(
            disp            = False,
            maxiter         = 200,
            method          = "lbfgs",
        )
        print(f"    AIC  : {res.aic:.2f}   BIC : {res.bic:.2f}   "
              f"Log-L : {res.llf:.2f}")

        # One-step-ahead forecasts on the test window
        # apply_model extends the fitted Kalman filter to new observations
        test_res = res.apply(
            endog = y_test_raw,
            exog  = X_test_raw,
            refit = False,
        )
        pred_log = test_res.fittedvalues

        # Inverse log1p → original case-count scale
        pred_raw = np.expm1(pred_log)
        actual_raw = np.expm1(y_test_raw)

        pred_raw   = np.clip(pred_raw, 0, None)

        county_results[ck] = {
            "model"      : res,
            "y_train"    : y_train,
            "X_train"    : X_train,
            "y_test_log" : y_test_raw,
            "X_test"     : X_test_raw,
            "pred_raw"   : pred_raw,
            "actual_raw" : actual_raw,
            "meta_test"  : meta_test,
        }

        rmse_c = np.sqrt(mean_squared_error(actual_raw, pred_raw))
        mae_c  = mean_absolute_error(actual_raw, pred_raw)
        r2_c   = r2_score(actual_raw, pred_raw) if len(actual_raw) > 1 else float("nan")
        print(f"    RMSE : {rmse_c:.4f}   MAE : {mae_c:.4f}   R² : {r2_c:.4f}")

    except Exception as exc:
        print(f"    ✗  Fitting failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 6: PREDICTION & EVALUATION")
print(DIVIDER)

if not county_results:
    print("  No county models were successfully fitted.  "
          "Check data paths and column names.")
else:
    # ── Aggregate all counties into one prediction dataframe ─────────────────
    all_rows = []
    for ck, res_dict in county_results.items():
        meta = res_dict["meta_test"].reset_index(drop=True)
        for i in range(len(res_dict["pred_raw"])):
            row = {
                "county_key" : ck,
                "year"       : int(meta.iloc[i]["year"])       if i < len(meta) else np.nan,
                "week_num"   : int(meta.iloc[i]["week_num"])   if i < len(meta) else np.nan,
                "actual"     : res_dict["actual_raw"][i],
                "predicted"  : res_dict["pred_raw"][i],
            }
            all_rows.append(row)

    pred_df = pd.DataFrame(all_rows)

    # ── Overall metrics ───────────────────────────────────────────────────────
    rmse_all = np.sqrt(mean_squared_error(pred_df["actual"], pred_df["predicted"]))
    mae_all  = mean_absolute_error(pred_df["actual"],        pred_df["predicted"])
    r2_all   = r2_score(pred_df["actual"],                   pred_df["predicted"])

    print(f"\n  ── Overall Evaluation Metrics (all counties, test set) ──")
    print(f"  RMSE : {rmse_all:.4f}")
    print(f"  MAE  : {mae_all:.4f}")
    print(f"  R²   : {r2_all:.4f}")

    # ── Per-county summary ────────────────────────────────────────────────────
    print(f"\n  ── Per-County Test Metrics ──")
    print(f"  {'County':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}")
    print("  " + DIVIDER2)
    for ck, grp in pred_df.groupby("county_key"):
        if len(grp) < 3:
            continue
        cr  = np.sqrt(mean_squared_error(grp["actual"], grp["predicted"]))
        cm  = mean_absolute_error(grp["actual"],         grp["predicted"])
        cr2 = r2_score(grp["actual"], grp["predicted"]) if len(grp) > 1 else float("nan")
        print(f"  {ck:<18} {cr:>8.3f} {cm:>8.3f} {cr2:>8.3f} {len(grp):>6}")

    # ── ARIMA model diagnostics per county ───────────────────────────────────
    print(f"\n  ── ARIMAX Model Diagnostics ──")
    print(f"  {'County':<18} {'AIC':>10} {'BIC':>10} {'LogL':>12}")
    print("  " + DIVIDER2)
    for ck, res_dict in county_results.items():
        r = res_dict["model"]
        print(f"  {ck:<18} {r.aic:>10.2f} {r.bic:>10.2f} {r.llf:>12.2f}")

    # ── First 10 actual vs predicted (across all counties, sorted) ───────────
    print(f"\n  ── First 10 Actual vs Predicted  "
          f"(WNV human cases, {FORECAST_LEAD} weeks ahead) ──")
    print(f"  {'#':<5} {'County':<16} {'Year':>5} {'Wk':>4} "
          f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print("  " + DIVIDER2)

    display_df = pred_df.dropna(subset=["year", "week_num"]).head(10)
    for rank, (_, row) in enumerate(display_df.iterrows(), 1):
        err = row["actual"] - row["predicted"]
        print(f"  {rank:<5} {row['county_key']:<16} "
              f"{int(row['year']):>5} {int(row['week_num']):>4} "
              f"{row['actual']:>10.2f} {row['predicted']:>10.4f} "
              f"{err:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 7: SUMMARY")
print(DIVIDER)

n_fitted = len(county_results)
yr_span  = int(model_df["year"].max() - model_df["year"].min() + 1)

print(f"""
  Research Question:
    Can we predict county-level WNV human case counts several weeks in
    advance using historical surveillance, meteorological, land-use,
    and demographic data?

  Target Variable:
    Weekly new WNV human cases, {FORECAST_LEAD} weeks ahead
    (all 6 counties — Boulder-CO, Cook-IL, Dallas-TX, Larimer-CO,
     LosAngeles-CA, Maricopa-AZ — distributed from CDC ArboNET
     annual totals via Gaussian seasonal bell curve)

  Model:
    ARIMAX — AutoRegressive Integrated Moving Average with eXogenous
             regressors (statsmodels SARIMAX class)

    One model is fitted independently per county:
      ARIMA order   : {ARIMA_ORDER}
        p={ARIMA_ORDER[0]} — AR: uses 2 lagged values of the target series
        d={ARIMA_ORDER[1]} — no integer differencing (log1p-transformed target)
        q={ARIMA_ORDER[2]} — MA: smooths 1-step residual shocks
      Seasonal order: {SEASONAL_ORDER}
        S=52 captures annual WNV transmission season cycles
      Estimation    : Maximum likelihood via Kalman filter (state-space)
      Prediction    : One-step-ahead using apply() on the test window

  Preprocessing (scikit-learn):
    Target         : log1p transform → near-Gaussian, reduced skew
    Exogenous X    : StandardScaler (fit on train, applied to full series)
    Missing values : Median imputation per column

  Exogenous Feature Groups ({len(EXOG_COLS)} total):
    Weather        : TAVG, PRCP, GDD, RH, rolling averages, lags 1–4
    Surveillance   : positive pools, infection rate lags
    Human-case hist: prev_yr_neuroinvasive, neuro_3yr_avg, neuro_yoy_change
    Demographics   : pop_density, poverty_rate
    Land-use       : developed %, wetlands %
    Temporal       : sin/cos week, WNV season indicator

  Results (all counties, test set):
    Models fitted : {n_fitted} / {len(COUNTIES)} counties
    Coverage      : {yr_span} years of weekly data
""")

if county_results:
    print(f"    RMSE = {rmse_all:.4f}")
    print(f"    MAE  = {mae_all:.4f}")
    print(f"    R²   = {r2_all:.4f}")

print(f"\n{DIVIDER}")
print("Done.")
