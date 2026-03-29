#!/usr/bin/env python3
"""
West Nile Virus (WNV) County-Level Human Case Prediction — SVR Model
=====================================================================
Research Question:
    Can we predict county-level WNV human case counts several weeks in
    advance using historical surveillance, meteorological, land-use,
    and demographic data?

Model   : Support Vector Regression (SVR) — scikit-learn
          SVR finds a regression hyperplane in a kernel-induced feature
          space that fits the data within an epsilon-insensitive tube.
          Predictions outside the tube are penalised by regularisation
          parameter C.  The RBF (Radial Basis Function) kernel maps
          inputs non-linearly, capturing complex interactions between
          weather, mosquito surveillance, and demographic variables
          without explicit feature crossing.

          All scikit-learn components used:
            SVR                  — core regressor
            StandardScaler       — feature & target normalisation
            LabelEncoder         — county string → integer
            Pipeline             — scaler + SVR in one object
            GridSearchCV         — hyper-parameter tuning
            mean_squared_error,
            mean_absolute_error,
            r2_score             — evaluation metrics

Target  : Weekly new WNV human cases, FORECAST_LEAD weeks ahead
Counties: Boulder-CO, Cook-IL, Dallas-TX, Larimer-CO,
          LosAngeles-CA, Maricopa-AZ

Data files (read from the repo's  ref/  folder):
    ref/wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv
    ref/demographics_combined.csv
    ref/cdl_strategy_d_pc.csv
    ref/mosquito_surveillance_county_week.csv
    ref/wnv_human_cases_county_year.csv

Install:
    pip install scikit-learn pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ── scikit-learn — all components ────────────────────────────────────────────
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics         import (mean_squared_error,
                                     mean_absolute_error,
                                     r2_score)
from sklearn.inspection      import permutation_importance

np.random.seed(42)

DIVIDER  = "=" * 80
DIVIDER2 = "-" * 80

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# >>>  Set BASE to the path of your local clone of the repo  <<<
BASE = "ref/"

FORECAST_LEAD = 4      # weeks ahead to forecast
RANDOM_STATE  = 42

# SVR hyper-parameter search grid
# C      : regularisation — larger C = tighter fit, less margin
# epsilon: width of the insensitive tube around the regression line
# gamma  : RBF kernel bandwidth ('scale' = 1 / (n_features * X.var()))
PARAM_GRID = {
    "svr__C"      : [0.1, 1.0, 10.0, 100.0],
    "svr__epsilon": [0.01, 0.1, 0.5],
    "svr__gamma"  : ["scale", "auto"],
    "svr__kernel" : ["rbf"],
}

# Number of TimeSeriesSplit folds for cross-validation
CV_FOLDS = 3

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

# ── 2d. WNV human-case spotlight by county ───────────────────────────────────
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

# ── 2e. Target distribution analysis ─────────────────────────────────────────
print("\n── 2e. WNV Human Cases — Annual Distribution by County ──\n")
print(f"  {'County':<18} {'Mean':>8} {'Median':>8} {'Max':>8} "
      f"{'Std':>8} {'Zero-yrs':>10}")
print("  " + DIVIDER2)
for ck in sorted(human_cases["county_key"].unique()):
    sub   = human_cases[human_cases["county_key"] == ck]["total_cases"]
    zeros = (sub == 0).sum()
    print(f"  {ck:<18} {sub.mean():>8.1f} {sub.median():>8.1f} "
          f"{sub.max():>8.0f} {sub.std():>8.1f} {zeros:>10}")


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

# Lagged weather (1, 2, 4 weeks)
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

# Growing-degree days — Culex mosquito activity proxy (threshold 14 °C)
weather_weekly["gdd_proxy"]  = (weather_weekly["TAVG_mean"] - 14).clip(lower=0)
weather_weekly["gdd_cumul8"] = (
    weather_weekly.groupby("county_key")["gdd_proxy"]
    .transform(lambda x: x.rolling(8, min_periods=1).sum()))

# Heat-wave indicator: weeks where TMAX > 35 °C
weather_weekly["heat_wave"]  = (weather_weekly["TMAX_max"] > 35).astype(int)

# Wet-week indicator: precipitation above seasonal 75th percentile
prcp_75 = weather_weekly["PRCP_sum"].quantile(0.75)
weather_weekly["wet_week"]   = (weather_weekly["PRCP_sum"] > prcp_75).astype(int)

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

    pk, sg       = SEASON_PARAMS.get(ck,   (33, 5.0))
    s_wk, e_wk   = SEASON_WINDOW.get(ck,  (22, 44))
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

hc_feats["prev_yr_neuroinvasive"] = (
    hc_feats.groupby("county_key")["neuroinvasive"].shift(1))
hc_feats["prev_yr_total_cases"]   = (
    hc_feats.groupby("county_key")["total_cases"].shift(1))
hc_feats["prev_yr_deaths"]        = (
    hc_feats.groupby("county_key")["deaths"].shift(1))
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

print(f"\n  [3e] Master merged dataset: "
      f"{merged.shape[0]:,} rows × {merged.shape[1]} columns")

# ── 3f. County encoding + cyclical temporal features ──────────────────────────
le = LabelEncoder()
merged["county_encoded"] = le.fit_transform(merged["county_key"])
merged["sin_week"]        = np.sin(2 * np.pi * merged["week_num"] / 52)
merged["cos_week"]        = np.cos(2 * np.pi * merged["week_num"] / 52)
merged["in_wnv_season"]   = merged["week_num"].between(26, 40).astype(int)

# Year normalised (trends in urbanisation, climate, surveillance)
yr_min = merged["year"].min()
yr_max = merged["year"].max()
merged["year_norm"] = (merged["year"] - yr_min) / max(yr_max - yr_min, 1)

# ── 3g. FORECAST_LEAD-week-ahead target ───────────────────────────────────────
TARGET_COL = f"target_{FORECAST_LEAD}wk_ahead"
merged.sort_values(["county_key", "year", "week_num"], inplace=True)
merged[TARGET_COL] = (
    merged.groupby("county_key")["wk_human_total"].shift(-FORECAST_LEAD))

# log1p-transform target: SVR assumes roughly symmetric residuals;
# log1p compresses the right tail of the zero-inflated WNV count distribution
merged["target_log"] = np.log1p(merged[TARGET_COL])

print(f"\n  [3f–3g] Temporal encodings + {FORECAST_LEAD}-week-ahead "
      f"target (log1p-transformed) created.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINAL DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 4: FINAL DATASET PREPARATION")
print(DIVIDER)

model_df = merged.dropna(subset=["target_log"]).copy()

# Filter to active-season window per county (±4 week buffer)
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

# ── Feature matrix definition ─────────────────────────────────────────────────
EXCLUDE = {
    "county_key", "year", "week_num",
    TARGET_COL, "target_log",
    "wk_human_total", "wk_neuroinvasive", "wk_non_neuroinvasive",
    "gdd_proxy",                     # intermediate; gdd_cumul8 is the rolled form
}
feature_cols = [
    c for c in model_df.columns
    if c not in EXCLUDE
    and model_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
]

print(f"\n  Total features : {len(feature_cols)}")
print("  Feature list   :")
hc_feat_set = {
    "prev_yr_neuroinvasive", "neuro_3yr_avg", "neuro_yoy_change",
    "prev_yr_neuro_pct", "prev_yr_total_cases", "prev_yr_deaths",
}
for i, fc in enumerate(feature_cols, 1):
    marker = "  ★" if fc in hc_feat_set else ""
    print(f"    {i:>2}. {fc}{marker}")

X_all = model_df[feature_cols].copy()
y_all = model_df["target_log"].copy()        # log1p-scaled target

# Median imputation for any residual NaNs
X_all.fillna(X_all.median(), inplace=True)

# ── Temporal 80/20 train-test split ───────────────────────────────────────────
sorted_idx  = model_df.sort_values(["year", "week_num"]).index
X_all       = X_all.loc[sorted_idx]
y_all       = y_all.loc[sorted_idx]
meta_sorted = model_df.loc[sorted_idx].reset_index(drop=True)

split   = int(len(X_all) * 0.80)
X_train = X_all.iloc[:split].values
X_test  = X_all.iloc[split:].values
y_train = y_all.iloc[:split].values
y_test  = y_all.iloc[split:].values

# Original (un-logged) test targets — used for final metric reporting
y_test_orig  = meta_sorted.iloc[split:][TARGET_COL].values

print(f"\n  Train : {len(X_train):,} samples  "
      f"(years {int(meta_sorted.iloc[:split]['year'].min())}–"
      f"{int(meta_sorted.iloc[:split]['year'].max())})")
print(f"  Test  : {len(X_test):,} samples  "
      f"(years {int(meta_sorted.iloc[split:]['year'].min())}–"
      f"{int(meta_sorted.iloc[split:]['year'].max())})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print(f"STEP 5: MODEL BUILDING — Support Vector Regression (SVR)  "
      f"({FORECAST_LEAD}-week forecast lead)")
print(DIVIDER)

print(f"""
  SVR design notes
  ────────────────
  Kernel   : RBF (Radial Basis Function)
    Maps inputs into an infinite-dimensional feature space via the
    Gaussian kernel k(x,z) = exp(-gamma * ||x-z||²), allowing SVR
    to learn non-linear relationships between weather/surveillance
    inputs and WNV case counts without explicit feature engineering.

  Key hyper-parameters:
    C       (regularisation): trades off margin width vs. training error.
                              Large C = smaller tube = closer fit.
    epsilon (tube width)    : predictions within ε of the true value
                              incur zero penalty.
    gamma   (kernel width)  : controls how far individual training
                              examples reach; 'scale' = 1/(n_feat·σ²).

  Pipeline:
    StandardScaler → SVR(kernel='rbf')
    Scaler is fitted only on the training fold to prevent leakage.

  Tuning:
    GridSearchCV with TimeSeriesSplit ({CV_FOLDS} folds)
    TimeSeriesSplit respects temporal ordering — later folds always
    test on data that follows the training data chronologically.
    Scoring: neg_mean_squared_error (minimise MSE on log1p target).

  Target:
    log1p(WNV case count) — SVR assumes symmetric residuals; log1p
    compresses the heavy right tail of the zero-inflated count
    distribution and prevents support vectors from being dominated
    by rare outbreak-year peaks.
""")

# ── Build sklearn Pipeline: scaler + SVR ──────────────────────────────────────
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svr",    SVR(kernel="rbf")),
])

# ── TimeSeriesSplit cross-validation for hyper-parameter search ───────────────
tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

print(f"  Running GridSearchCV  "
      f"({len(PARAM_GRID['svr__C']) * len(PARAM_GRID['svr__epsilon']) * len(PARAM_GRID['svr__gamma'])} "
      f"combinations × {CV_FOLDS} folds)…")
print("  This may take several minutes on large datasets.\n")

grid_search = GridSearchCV(
    estimator  = pipe,
    param_grid = PARAM_GRID,
    cv         = tscv,
    scoring    = "neg_mean_squared_error",
    n_jobs     = -1,
    verbose    = 1,
    refit      = True,       # refit best estimator on full training set
)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_cv_mse = -grid_search.best_score_

print(f"\n  ✓ GridSearchCV complete")
print(f"\n  Best hyper-parameters:")
for k, v in best_params.items():
    print(f"    {k:<24}: {v}")
print(f"\n  Best CV MSE (log1p scale): {best_cv_mse:.6f}")
print(f"  Best CV RMSE             : {np.sqrt(best_cv_mse):.6f}")

best_model = grid_search.best_estimator_

# ── Number of support vectors ──────────────────────────────────────────────────
n_sv = best_model.named_steps["svr"].support_vectors_.shape[0]
ratio_sv = n_sv / len(X_train) * 100
print(f"\n  Support vectors used : {n_sv:,}  ({ratio_sv:.1f}% of training samples)")
print(f"  (Lower ratio = wider margin, better generalisation)")

# ── Cross-validation results summary ──────────────────────────────────────────
cv_results = pd.DataFrame(grid_search.cv_results_)
top5_cv = (
    cv_results[["param_svr__C", "param_svr__epsilon",
                "param_svr__gamma", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .head(5)
)
top5_cv["rmse"] = np.sqrt(-top5_cv["mean_test_score"])
print(f"\n  Top 5 CV configurations:")
print(f"  {'C':<8} {'epsilon':<10} {'gamma':<8} {'CV RMSE':>10}")
print("  " + DIVIDER2)
for _, r in top5_cv.iterrows():
    print(f"  {str(r['param_svr__C']):<8} "
          f"{str(r['param_svr__epsilon']):<10} "
          f"{str(r['param_svr__gamma']):<8} "
          f"{r['rmse']:>10.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 6: PREDICTION & EVALUATION")
print(DIVIDER)

# Predict on test set (log1p scale), then inverse-transform
y_pred_log  = best_model.predict(X_test)
y_pred_orig = np.expm1(y_pred_log)          # back to case-count scale
y_pred_orig = np.clip(y_pred_orig, 0, None) # case counts ≥ 0

# ── Metrics on log1p scale (what the model optimised) ────────────────────────
rmse_log = np.sqrt(mean_squared_error(y_test,      y_pred_log))
mae_log  = mean_absolute_error(y_test,             y_pred_log)
r2_log   = r2_score(y_test,                        y_pred_log)

# ── Metrics on original case-count scale (most interpretable) ────────────────
y_test_orig_clean = np.nan_to_num(y_test_orig, nan=0.0)
rmse_orig = np.sqrt(mean_squared_error(y_test_orig_clean, y_pred_orig))
mae_orig  = mean_absolute_error(y_test_orig_clean,        y_pred_orig)
r2_orig   = r2_score(y_test_orig_clean,                   y_pred_orig)

print(f"\n  ── Overall Evaluation Metrics on Test Set ──")
print(f"\n  On log1p-transformed scale (model objective):")
print(f"    RMSE : {rmse_log:.4f}")
print(f"    MAE  : {mae_log:.4f}")
print(f"    R²   : {r2_log:.4f}")
print(f"\n  On original case-count scale (interpretable):")
print(f"    RMSE : {rmse_orig:.4f}")
print(f"    MAE  : {mae_orig:.4f}")
print(f"    R²   : {r2_orig:.4f}")

# ── Per-county metrics ─────────────────────────────────────────────────────────
test_meta = meta_sorted.iloc[split:][
    ["county_key", "year", "week_num", TARGET_COL]
].reset_index(drop=True)
test_meta["actual"]    = y_test_orig_clean
test_meta["predicted"] = y_pred_orig

print(f"\n  ── Per-County Test Metrics (original scale) ──")
print(f"  {'County':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}")
print("  " + DIVIDER2)
for ck, grp in test_meta.groupby("county_key"):
    if len(grp) < 5:
        continue
    cr  = np.sqrt(mean_squared_error(grp["actual"], grp["predicted"]))
    cm  = mean_absolute_error(grp["actual"],         grp["predicted"])
    cr2 = r2_score(grp["actual"], grp["predicted"]) if len(grp) > 1 else float("nan")
    print(f"  {ck:<18} {cr:>8.3f} {cm:>8.3f} {cr2:>8.3f} {len(grp):>6}")

# ── Permutation feature importance ────────────────────────────────────────────
# SVR has no native feature_importances_ attribute (unlike tree models).
# Permutation importance shuffles one feature at a time and measures the
# drop in R² — model-agnostic and valid for any sklearn estimator.
print(f"\n  ── Permutation Feature Importance (top 15, on test set) ──")
print("  Computing permutation importance on the test set…")

perm_result = permutation_importance(
    best_model, X_test, y_test,
    n_repeats = 10,
    random_state = RANDOM_STATE,
    scoring   = "r2",
    n_jobs    = -1,
)
perm_imp = pd.Series(
    perm_result.importances_mean, index=feature_cols
).sort_values(ascending=False)

print(f"\n  {'Rank':<5} {'Feature':<38} {'Importance (mean Δ R²)':>22}")
print("  " + DIVIDER2)
for rank, (feat, imp) in enumerate(perm_imp.head(15).items(), 1):
    marker = "  ★" if feat in hc_feat_set else ""
    print(f"  {rank:<5} {feat:<38} {imp:>22.4f}{marker}")

# ── First 10 actual vs predicted ──────────────────────────────────────────────
print(f"\n  ── First 10 Actual vs Predicted  "
      f"(WNV human cases, {FORECAST_LEAD} weeks ahead) ──")
print(f"  {'#':<5} {'County':<16} {'Year':>5} {'Wk':>4} "
      f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("  " + DIVIDER2)
for i, row in test_meta.head(10).iterrows():
    err = row["actual"] - row["predicted"]
    print(f"  {i+1:<5} {row['county_key']:<16} "
          f"{int(row['year']):>5} {int(row['week_num']):>4} "
          f"{row['actual']:>10.2f} {row['predicted']:>10.4f} "
          f"{err:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 7: SUMMARY")
print(DIVIDER)

top5_feat_str = "\n".join(
    f"    {i+1}. {f}  ({v:.4f})"
    for i, (f, v) in enumerate(perm_imp.head(5).items()))

yr_span = int(model_df["year"].max() - model_df["year"].min() + 1)

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
    Support Vector Regression — RBF kernel (scikit-learn SVR)
    Pipeline: StandardScaler → SVR
    Hyper-parameter tuning: GridSearchCV + TimeSeriesSplit ({CV_FOLDS} folds)
    Temporal 80/20 split — train on earlier years, test on later years

  Best hyper-parameters found:
    C       = {best_params.get('svr__C', 'N/A')}
    epsilon = {best_params.get('svr__epsilon', 'N/A')}
    gamma   = {best_params.get('svr__gamma', 'N/A')}
    kernel  = {best_params.get('svr__kernel', 'rbf')}
    Support vectors: {n_sv:,} ({ratio_sv:.1f}% of training samples)

  Feature Groups ({len(feature_cols)} total):
    ★ Human-case history  : prev_yr_neuroinvasive, neuro_3yr_avg,
                            neuro_yoy_change, prev_yr_neuro_pct,
                            prev_yr_total_cases, prev_yr_deaths
      Weather             : TAVG lags, PRCP lags, RH, dew point,
                            growing-degree days, heat wave, wet week
      Mosquito surveillance: positive pool lags, infection rate lags
      Demographics        : population density, income, age 65+
      Land-use            : developed %, wetlands %, cropland %
      Temporal            : sin/cos week, season indicator, year trend,
                            county encoding

  Top 5 Features (permutation importance — mean Δ R²):
{top5_feat_str}

  Final Test-Set Metrics (original case-count scale):
    RMSE = {rmse_orig:.4f}
    MAE  = {mae_orig:.4f}
    R²   = {r2_orig:.4f}

  Coverage: All 6 counties × {yr_span} years
""")

print(DIVIDER)
print("Done.")
