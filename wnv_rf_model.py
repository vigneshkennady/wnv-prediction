#!/usr/bin/env python3
"""
West Nile Virus (WNV) County-Level Human Case Prediction
=========================================================
Research Question:
    Can we predict county-level WNV human case counts several weeks in
    advance using historical surveillance, meteorological, land-use,
    and demographic data?

Model   : Random Forest Regression (scikit-learn)
Target  : Weekly new WNV human cases, 4 weeks ahead
Counties: Boulder-CO, Cook-IL, Dallas-TX, Larimer-CO, LosAngeles-CA,
          Maricopa-AZ

Key upgrade in this version
────────────────────────────
  • Integrates wnv_human_cases_county_year.csv (CDC ArboNET + state
    health dept data, 1999-2024, all 6 counties).
  • Most important new variable: NEUROINVASIVE case count.
    Neuroinvasive disease (encephalitis / meningitis) is the CDC's
    primary WNV severity metric, is the most consistently reported
    variable across all 5 state surveillance systems, and is the
    strongest inter-year predictor of community risk because it reflects
    both viral prevalence and host-immune landscape.
  • Prior-year neuroinvasive count, 3-year rolling average, and
    year-over-year trend are engineered as lagged contextual features.
  • Annual human case totals are distributed across weeks via a
    Gaussian seasonal bell curve to generate weekly targets for all
    6 counties (previously only Dallas was covered).
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE          = "ref/"
FORECAST_LEAD = 4          # weeks ahead to predict
RANDOM_STATE  = 42
N_TREES       = 300
MAX_DEPTH     = 15

# Seasonal distribution parameters per county (peak MMWR week, sigma)
# Used to convert annual human case totals → weekly estimates
SEASON_PARAMS = {
    "Cook_IL":       (33, 4.5),
    "LosAngeles_CA": (33, 6.0),
    "Maricopa_AZ":   (31, 5.5),
    "Larimer_CO":    (32, 4.0),
    "Boulder_CO":    (32, 4.0),
    "Dallas_TX":     (33, 5.0),
}
SEASON_WINDOW = {          # (first active week, last active week)
    "Cook_IL":       (22, 42),
    "LosAngeles_CA": (20, 44),
    "Maricopa_AZ":   (20, 44),
    "Larimer_CO":    (24, 40),
    "Boulder_CO":    (24, 40),
    "Dallas_TX":     (22, 44),
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("STEP 1: DATA LOADING")
print("=" * 80)

weather      = pd.read_csv(BASE + "wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv",
                           parse_dates=["date"])
demographics = pd.read_csv(BASE + "demographics_combined.csv")
demographics.replace(-999, np.nan, inplace=True)
land_use     = pd.read_csv(BASE + "cdl_strategy_d_pc.csv")
surveillance = pd.read_csv(BASE + "mosquito_surveillance_county_week.csv")
human_cases  = pd.read_csv(BASE + "wnv_human_cases_county_year.csv")

print(f"  Weather            : {len(weather):>6,} rows")
print(f"  Demographics       : {len(demographics):>6,} rows")
print(f"  Land Use           : {len(land_use):>6,} rows")
print(f"  Mosquito Surv.     : {len(surveillance):>6,} rows  "
      f"({surveillance['county_key'].nunique()} counties)")
print(f"  WNV Human Cases    : {len(human_cases):>6,} rows  "
      f"({human_cases['county_key'].nunique()} counties, "
      f"{human_cases['year'].min()}–{human_cases['year'].max()})")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATA EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 2: DATA EXPLORATION")
print("=" * 80)

datasets = {
    "Weather":          weather,
    "Demographics":     demographics,
    "Land Use":         land_use,
    "Mosquito Surv.":   surveillance,
    "WNV Human Cases":  human_cases,
}

# 2a. First 5 rows
for name, df in datasets.items():
    print(f"\n─── {name}: First 5 Rows ───")
    print(df.head().to_string())

# 2b. Data types & null counts
print("\n\n─── Data Types & Null Value Counts ───")
for name, df in datasets.items():
    print(f"\n>> {name}")
    info = pd.DataFrame({
        "dtype":      df.dtypes,
        "null_count": df.isnull().sum(),
        "null_pct":   (df.isnull().sum() / len(df) * 100).round(2),
    })
    print(info.to_string())

# 2c. Summary statistics
print("\n\n─── Summary Statistics ───")
for name, df in datasets.items():
    print(f"\n>> {name}")
    print(df.describe().to_string())

# 2d. Human cases spotlight — neuroinvasive analysis
print("\n\n─── WNV Human Cases: Neuroinvasive Analysis by County ───")
hc_active = human_cases[human_cases["total_cases"] > 0].copy()
hc_active["pct_neuroinvasive"] = (
    hc_active["neuroinvasive"] / hc_active["total_cases"] * 100
).round(1)
spotlight = hc_active.groupby("county_key").agg(
    active_years    = ("year",             "count"),
    total_cases_sum = ("total_cases",      "sum"),
    neuro_sum       = ("neuroinvasive",    "sum"),
    non_neuro_sum   = ("non_neuroinvasive","sum"),
    deaths_sum      = ("deaths",           "sum"),
    peak_year       = ("total_cases",      lambda x: hc_active.loc[x.idxmax(), "year"]),
    peak_cases      = ("total_cases",      "max"),
    avg_pct_neuro   = ("pct_neuroinvasive","mean"),
    case_fatality_pct=("deaths",           lambda x:
                       round(x.sum() / hc_active.loc[x.index,"total_cases"].sum() * 100, 1)),
).reset_index()
print(spotlight.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 3: FEATURE ENGINEERING & PREPROCESSING")
print("=" * 80)

# ── 3a. Weekly weather aggregation ──────────────────────────────────────────
def make_county_key(row):
    return f"{row['county'].replace(' County','').replace(' ','')}_{row['state']}"

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

# Lagged weather (1, 2, 4 weeks)
for lag in [1, 2, 4]:
    for col in ["TAVG_mean", "PRCP_sum", "RH_mean", "DEWP_mean", "WIND_mean"]:
        weather_weekly[f"{col}_lag{lag}"] = (
            weather_weekly.groupby("county_key")[col].shift(lag)
        )

# Rolling aggregates
weather_weekly["TAVG_roll4"]  = (
    weather_weekly.groupby("county_key")["TAVG_mean"]
    .transform(lambda x: x.rolling(4, min_periods=1).mean())
)
weather_weekly["PRCP_cumul4"] = (
    weather_weekly.groupby("county_key")["PRCP_sum"]
    .transform(lambda x: x.rolling(4, min_periods=1).sum())
)
# Growing-degree days proxy (days above 14°C = 57.2°F, threshold for Culex)
weather_weekly["gdd_proxy"] = (weather_weekly["TAVG_mean"] - 14).clip(lower=0)
weather_weekly["gdd_cumul8"] = (
    weather_weekly.groupby("county_key")["gdd_proxy"]
    .transform(lambda x: x.rolling(8, min_periods=1).sum())
)

print(f"  Weekly weather: {weather_weekly.shape[1]} features, "
      f"{len(weather_weekly):,} rows")

# ── 3b. WNV Human Cases → weekly distribution ───────────────────────────────
# The KEY new step: distribute annual neuroinvasive & total case counts into
# weekly estimates for ALL 6 counties using a Gaussian seasonal bell curve.
# This replaces the previous approach (Dallas-only from mosquito surveillance).
#
# WHY NEUROINVASIVE is the primary target driver:
#   • Most clinically severe and consistently reported category
#   • CDC's principal WNV surveillance metric (ArboNET primary series)
#   • Least subject to under-reporting (hospitalised / lab-confirmed cases)
#   • Strongest cross-year predictor of community transmission intensity

print("\n  Distributing annual human case totals to weekly estimates...")
rng_global = np.random.default_rng(RANDOM_STATE)

def gaussian_weights(peak_wk, sigma, start_wk, end_wk):
    wks = np.arange(start_wk, end_wk + 1)
    w   = np.exp(-0.5 * ((wks - peak_wk) / sigma) ** 2)
    w  /= w.sum()
    return wks, w

weekly_human_rows = []

for _, yr_row in human_cases.iterrows():
    ck    = yr_row["county_key"]
    yr    = yr_row["year"]
    total = int(yr_row["total_cases"])
    neuro = int(yr_row["neuroinvasive"])
    non_n = int(yr_row["non_neuroinvasive"])
    dths  = int(yr_row["deaths"])

    pk, sg         = SEASON_PARAMS.get(ck, (33, 5.0))
    s_wk, e_wk     = SEASON_WINDOW.get(ck, (22, 44))
    act_wks, wts   = gaussian_weights(pk, sg, s_wk, e_wk)

    seed = int(abs(hash((ck, yr)))) % (2 ** 31)
    rng  = np.random.default_rng(seed)

    # Distribute counts across active season weeks
    total_wkly = rng.multinomial(total, wts) if total > 0 else np.zeros(len(wts), int)
    neuro_wkly = rng.multinomial(neuro, wts) if neuro > 0 else np.zeros(len(wts), int)
    non_n_wkly = total_wkly - neuro_wkly          # ensures sum consistency
    non_n_wkly = np.clip(non_n_wkly, 0, None)

    # Extend over all 52 weeks (off-season = 0)
    act_set = set(act_wks.tolist())
    act_map = dict(zip(act_wks.tolist(),
                       zip(total_wkly, neuro_wkly, non_n_wkly)))

    for wk in range(1, 53):
        tv, nv, nn = act_map.get(wk, (0, 0, 0))
        weekly_human_rows.append({
            "county_key":          ck,
            "year":                yr,
            "week_num":            wk,
            "wk_human_total":      int(tv),
            "wk_neuroinvasive":    int(nv),
            "wk_non_neuroinvasive":int(nn),
        })

df_wk_human = pd.DataFrame(weekly_human_rows)
df_wk_human.sort_values(["county_key", "year", "week_num"], inplace=True)
print(f"  Weekly human cases generated: {len(df_wk_human):,} rows")

# ── 3c. Human-case ANNUAL lagged features (most predictive new variables) ────
# Merge prior-year annual totals as contextual features.
# These capture WNV multi-year amplification cycles.
hc_feats = human_cases[
    ["county_key", "year", "total_cases", "neuroinvasive",
     "non_neuroinvasive", "deaths"]
].copy()
hc_feats.sort_values(["county_key", "year"], inplace=True)

# Lag 1 year (most important: last year's neuroinvasive count)
hc_feats["prev_yr_neuroinvasive"]  = (
    hc_feats.groupby("county_key")["neuroinvasive"].shift(1)
)
hc_feats["prev_yr_total_cases"]    = (
    hc_feats.groupby("county_key")["total_cases"].shift(1)
)
hc_feats["prev_yr_deaths"]         = (
    hc_feats.groupby("county_key")["deaths"].shift(1)
)
# 3-year rolling average of neuroinvasive (trend signal)
hc_feats["neuro_3yr_avg"] = (
    hc_feats.groupby("county_key")["neuroinvasive"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
# Year-over-year change (amplification / attenuation signal)
hc_feats["neuro_yoy_change"] = (
    hc_feats["prev_yr_neuroinvasive"]
    - hc_feats.groupby("county_key")["neuroinvasive"].shift(2)
)
# Neuroinvasive fraction (severity index)
hc_feats["prev_yr_neuro_pct"] = (
    hc_feats["prev_yr_neuroinvasive"]
    / hc_feats["prev_yr_total_cases"].replace(0, np.nan)
).fillna(0)

# Keep only the engineered columns (not raw counts — those go in as target)
hc_lag_cols = [
    "county_key", "year",
    "prev_yr_neuroinvasive",    # ★ PRIMARY new feature
    "prev_yr_total_cases",
    "prev_yr_deaths",
    "neuro_3yr_avg",
    "neuro_yoy_change",
    "prev_yr_neuro_pct",
]
hc_lags = hc_feats[hc_lag_cols].copy()

print(f"\n  Human-case annual lag features: {len(hc_lag_cols)-2} variables")
print(f"    ★ prev_yr_neuroinvasive  — prior-year neuroinvasive count")
print(f"      prev_yr_total_cases    — prior-year total human cases")
print(f"      prev_yr_deaths         — prior-year WNV deaths")
print(f"      neuro_3yr_avg          — 3-year rolling mean neuroinvasive")
print(f"      neuro_yoy_change       — year-over-year neuroinvasive delta")
print(f"      prev_yr_neuro_pct      — prior-year neuroinvasive fraction")

# ── 3d. Mosquito surveillance features (weekly, lagged) ─────────────────────
surveillance["positive_pools"] = pd.to_numeric(
    surveillance["positive_pools"], errors="coerce").fillna(0)
surveillance["infection_rate"] = pd.to_numeric(
    surveillance["infection_rate"], errors="coerce").fillna(0)
surveillance.sort_values(["county_key", "year", "week_num"], inplace=True)

for lag in [1, 2, 4]:
    surveillance[f"pos_pools_lag{lag}"] = (
        surveillance.groupby(["county_key", "year"])["positive_pools"].shift(lag)
    )
    surveillance[f"infect_rate_lag{lag}"] = (
        surveillance.groupby(["county_key", "year"])["infection_rate"].shift(lag)
    )

surv_cols = (["county_key", "year", "week_num", "positive_pools", "infection_rate"]
             + [f"pos_pools_lag{l}"    for l in [1, 2, 4]]
             + [f"infect_rate_lag{l}"  for l in [1, 2, 4]])
surv_feats = surveillance[surv_cols].copy()

# ── 3e. Build the master weekly frame ───────────────────────────────────────
# Backbone: all county-year-week combos from weather (2000-2024)
backbone = weather_weekly[["county_key", "year", "week_num"]].copy()

# Attach weekly human case target
merged = backbone.merge(df_wk_human, on=["county_key", "year", "week_num"],
                        how="left")
# Attach weather features
merged = merged.merge(weather_weekly, on=["county_key", "year", "week_num"],
                      how="left")
# Attach annual human-case lags (broadcast to all weeks of that year)
merged = merged.merge(hc_lags, on=["county_key", "year"], how="left")
# Attach mosquito surveillance lags (sparse — mostly Dallas)
merged = merged.merge(surv_feats, on=["county_key", "year", "week_num"],
                      how="left")
# Attach demographics (annual)
demo_keep = ["county_key", "year", "total_pop", "median_hh_income",
             "housing_units", "pop_65_plus", "pct_65_plus", "poverty_rate",
             "land_area_sqmi", "pop_density_per_sqmi",
             "housing_density_per_sqmi", "saipe_median_income",
             "saipe_poverty_rate"]
merged = merged.merge(
    demographics[[c for c in demo_keep if c in demographics.columns]],
    on=["county_key", "year"], how="left")
# Attach land-use (annual)
lu_keep = ["county_key", "year", "total_acres",
           "cdl_cropland_pct", "cdl_developed_pct", "cdl_wetlands_pct",
           "cdl_pasture_hay_pct", "cdl_fallow_idle_pct",
           "cdl_corn_pct", "cdl_rice_pct", "cdl_aquaculture_pct"]
merged = merged.merge(
    land_use[[c for c in lu_keep if c in land_use.columns]],
    on=["county_key", "year"], how="left")

# ── 3f. Create the N-week-ahead target ──────────────────────────────────────
# Shift wk_human_total FORWARD by FORECAST_LEAD weeks (i.e. each row now
# holds the case count that will occur FORECAST_LEAD weeks later).
merged.sort_values(["county_key", "year", "week_num"], inplace=True)

merged[f"target_{FORECAST_LEAD}wk_ahead"] = (
    merged.groupby("county_key")["wk_human_total"]
    .shift(-FORECAST_LEAD)
)

# ── 3g. Encode county + temporal features ───────────────────────────────────
le = LabelEncoder()
merged["county_encoded"] = le.fit_transform(merged["county_key"])
merged["sin_week"]        = np.sin(2 * np.pi * merged["week_num"] / 52)
merged["cos_week"]        = np.cos(2 * np.pi * merged["week_num"] / 52)
# Season indicator (peak WNV months = weeks 26-40)
merged["in_wnv_season"]   = merged["week_num"].between(26, 40).astype(int)

print(f"\n  Master merged dataset: {merged.shape[0]:,} rows × "
      f"{merged.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINAL DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 4: FINAL DATASET PREPARATION")
print("=" * 80)

target_col = f"target_{FORECAST_LEAD}wk_ahead"

# Restrict to rows inside active season window where we have a valid target
# (off-season weeks with 0 cases are kept — they matter for seasonality)
model_df = merged.dropna(subset=[target_col]).copy()

# Keep only active-season range per county (trim dead-of-winter noise)
def in_active_window(row):
    s, e = SEASON_WINDOW.get(row["county_key"], (20, 45))
    return (row["week_num"] >= s - 4) and (row["week_num"] <= e + 4)

mask = model_df.apply(in_active_window, axis=1)
model_df = model_df[mask].copy()

print(f"  Rows after active-season filter : {len(model_df):,}")
print(f"  Counties                        : "
      f"{sorted(model_df['county_key'].unique())}")
print(f"  Year range                      : "
      f"{int(model_df['year'].min())}–{int(model_df['year'].max())}")
print(f"  Target ('{target_col}') — non-zero weeks: "
      f"{(model_df[target_col] > 0).sum():,}")

# Feature matrix
exclude = {
    "county_key", "year", "week_num",
    target_col,
    "wk_human_total", "wk_neuroinvasive", "wk_non_neuroinvasive",
    "gdd_proxy",       # intermediate; gdd_cumul8 is the rolled version
}
feature_cols = [
    c for c in model_df.columns
    if c not in exclude
    and model_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
]

print(f"\n  Total features: {len(feature_cols)}")
print("  Feature list:")
for i, fc in enumerate(feature_cols, 1):
    marker = " ★" if fc in {
        "prev_yr_neuroinvasive", "neuro_3yr_avg",
        "neuro_yoy_change", "prev_yr_neuro_pct",
        "prev_yr_total_cases", "prev_yr_deaths",
    } else ""
    print(f"    {i:>2}. {fc}{marker}")

X = model_df[feature_cols].copy()
y = model_df[target_col].copy()

# Impute remaining NaNs with column medians
X.fillna(X.median(), inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"STEP 5: MODEL BUILDING — Random Forest Regression "
      f"({FORECAST_LEAD}-week forecast lead)")
print("=" * 80)

# Temporal train/test split (80/20 — earlier years train, later years test)
sorted_idx = model_df.sort_values(["year", "week_num"]).index
X = X.loc[sorted_idx]
y = y.loc[sorted_idx]

split = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"\n  Train : {len(X_train):,} samples  "
      f"(years {int(model_df.loc[sorted_idx].iloc[:split]['year'].min())}–"
      f"{int(model_df.loc[sorted_idx].iloc[:split]['year'].max())})")
print(f"  Test  : {len(X_test):,} samples  "
      f"(years {int(model_df.loc[sorted_idx].iloc[split:]['year'].min())}–"
      f"{int(model_df.loc[sorted_idx].iloc[split:]['year'].max())})")

rf = RandomForestRegressor(
    n_estimators     = N_TREES,
    max_depth        = MAX_DEPTH,
    min_samples_split= 5,
    min_samples_leaf = 2,
    max_features     = "sqrt",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)
rf.fit(X_train, y_train)
print("\n  ✓ Model trained successfully")

# Feature importance
importances = pd.Series(rf.feature_importances_, index=feature_cols)
imp_sorted  = importances.sort_values(ascending=False)

print(f"\n  Top 20 Feature Importances  (★ = new human-cases variable):")
print("  " + "─" * 58)
print(f"  {'Rank':<5} {'Feature':<38} {'Importance'}")
print("  " + "─" * 58)
human_feats = {
    "prev_yr_neuroinvasive", "neuro_3yr_avg", "neuro_yoy_change",
    "prev_yr_neuro_pct", "prev_yr_total_cases", "prev_yr_deaths",
}
for rank, (feat, imp) in enumerate(imp_sorted.head(20).items(), 1):
    marker = " ★" if feat in human_feats else ""
    print(f"  {rank:<5} {feat:<38}{imp:.4f}{marker}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 6: PREDICTION & EVALUATION")
print("=" * 80)

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"\n  ── Evaluation Metrics on Test Set ──")
print(f"    RMSE  : {rmse:.4f}")
print(f"    MAE   : {mae:.4f}")
print(f"    R²    : {r2:.4f}")

# Per-county metrics
test_meta = model_df.loc[sorted_idx].iloc[split:][
    ["county_key", "year", "week_num"]
].reset_index(drop=True)
test_meta["actual"]    = y_test.values
test_meta["predicted"] = y_pred

print(f"\n  ── Per-County Test Metrics ──")
print(f"  {'County':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8}  {'N':>6}")
print("  " + "─" * 54)
for ck, grp in test_meta.groupby("county_key"):
    if len(grp) < 5:
        continue
    cr   = np.sqrt(mean_squared_error(grp["actual"], grp["predicted"]))
    cm   = mean_absolute_error(grp["actual"], grp["predicted"])
    cr2  = r2_score(grp["actual"], grp["predicted"]) if len(grp) > 1 else float("nan")
    print(f"  {ck:<18} {cr:>8.3f} {cm:>8.3f} {cr2:>8.3f}  {len(grp):>6}")

# First 10 actual vs predicted
print(f"\n  ── First 10 Actual vs Predicted "
      f"(WNV human cases, {FORECAST_LEAD} weeks ahead) ──")
print(f"  {'#':<5} {'County':<16} {'Year':>5} {'Wk':>4} "
      f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("  " + "─" * 65)
for i, row in test_meta.head(10).iterrows():
    err = row["actual"] - row["predicted"]
    print(f"  {i+1:<5} {row['county_key']:<16} {int(row['year']):>5} "
          f"{int(row['week_num']):>4} {row['actual']:>10.2f} "
          f"{row['predicted']:>10.4f} {err:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("STEP 7: SUMMARY")
print("=" * 80)

top5 = imp_sorted.head(5)
top5_str = "\n".join(
    f"      {i+1}. {f} ({v:.4f})" for i, (f, v) in enumerate(top5.items())
)

print(f"""
  Research Question:
    Can we predict county-level WNV human case counts several weeks
    in advance using historical surveillance, meteorological, land-use,
    and demographic data?

  Target Variable:
    Weekly new WNV human cases, {FORECAST_LEAD} weeks ahead
    (all 6 counties, distributed from CDC ArboNET annual totals)

  Key New Variable — NEUROINVASIVE CASES:
    The most important addition is prev_yr_neuroinvasive — the prior
    year's county-level neuroinvasive WNV case count (encephalitis /
    meningitis). This variable:
      • Is the CDC's primary WNV severity metric (ArboNET series)
      • Captures multi-year host-immunity / amplification dynamics
      • Is the least under-reported WNV category (hospitalised & lab-confirmed)
      • Drives the 3-year rolling average and YoY trend features

  Feature Groups ({len(feature_cols)} total):
    ★ Human-case history  : prev_yr_neuroinvasive, neuro_3yr_avg,
                            neuro_yoy_change, prev_yr_neuro_pct,
                            prev_yr_total_cases, prev_yr_deaths
      Weather             : TAVG lags, PRCP lags, RH, dew point,
                            growing-degree days, rolling averages
      Mosquito surveillance: positive pool lags, infection rate lags
      Demographics        : population density, income, % age 65+
      Land-use            : developed %, wetlands %, cropland %
      Temporal            : sin/cos week encoding, season indicator

  Model:
    Random Forest Regressor — {N_TREES} trees, max_depth={MAX_DEPTH}
    Temporal 80/20 split (train on earlier years, test on later years)

  Top 5 Features:
{top5_str}

  Results (test set):
    RMSE = {rmse:.4f}   MAE = {mae:.4f}   R² = {r2:.4f}

  Coverage: All 6 counties × {int(model_df['year'].max() - model_df['year'].min() + 1)} years
    (Previously: Dallas TX only)
""")
print("=" * 80)
print("Done.")
