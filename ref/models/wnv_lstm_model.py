#!/usr/bin/env python3
"""
West Nile Virus (WNV) County-Level Human Case Prediction — LSTM Model
======================================================================
Research Question:
    Can we predict county-level WNV human case counts several weeks in
    advance using historical surveillance, meteorological, land-use,
    and demographic data?

Model   : Long Short-Term Memory (LSTM) recurrent neural network
          Built with Keras/TensorFlow; scikit-learn used for all
          preprocessing (StandardScaler, LabelEncoder) and evaluation
          metrics (RMSE, MAE, R²).

          NOTE: scikit-learn does not implement LSTM natively —
          LSTMs require a sequential/recurrent layer architecture
          unavailable in the sklearn estimator API. This script uses
          Keras for the network while keeping every other step
          (preprocessing, scaling, metrics) in the sklearn ecosystem.

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
    pip install tensorflow scikit-learn pandas numpy
"""

import warnings
warnings.filterwarnings("ignore")
os_environ_key = "TF_CPP_MIN_LOG_LEVEL"
import os
os.environ[os_environ_key] = "3"          # silence TF C++ logs

import pandas as pd
import numpy as np

# ── scikit-learn utilities ────────────────────────────────────────────────────
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.metrics        import mean_squared_error, mean_absolute_error, r2_score

# ── Keras / TensorFlow ───────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models    import Sequential
from tensorflow.keras.layers    import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

DIVIDER = "=" * 80

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# >>>  Set BASE to the root of your local clone of the repo  <<<
BASE = "ref/"

FORECAST_LEAD  = 4     # weeks ahead to forecast
SEQ_LEN        = 8     # look-back window (weeks) fed into the LSTM
RANDOM_STATE   = 42
EPOCHS         = 100
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
PATIENCE       = 15    # early-stopping patience

# Seasonal distribution params per county  (peak MMWR week, sigma)
SEASON_PARAMS = {
    "Cook_IL"      : (33, 4.5),
    "LosAngeles_CA": (33, 6.0),
    "Maricopa_AZ"  : (31, 5.5),
    "Larimer_CO"   : (32, 4.0),
    "Boulder_CO"   : (32, 4.0),
    "Dallas_TX"    : (33, 5.0),
}

# Active season window per county  (first active week, last active week)
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

# ── 2b. Data types & null counts ────────────────────────────────────────────
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

# ── 2c. Summary statistics ───────────────────────────────────────────────────
print("\n── 2c. Summary Statistics ──\n")
for name, df in datasets.items():
    print(f"  >>> {name}")
    print(df.describe(include="all").to_string())
    print()

# ── 2d. Neuroinvasive spotlight by county ────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 3: FEATURE ENGINEERING & PREPROCESSING")
print(DIVIDER)

# ── 3a. Weekly weather aggregation ──────────────────────────────────────────
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

# Lagged weather  (1, 2, 4 weeks) — within-county shift
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

# ── 3b. Distribute annual human case totals → weekly estimates ───────────────
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

# ── 3c. Annual human-case lagged features ────────────────────────────────────
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

# ── 3d. Mosquito surveillance lagged features ────────────────────────────────
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

# ── 3e. Assemble master weekly frame ────────────────────────────────────────
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

# ── 3f. Create FORECAST_LEAD-week-ahead target ───────────────────────────────
merged.sort_values(["county_key", "year", "week_num"], inplace=True)
TARGET_COL = f"target_{FORECAST_LEAD}wk_ahead"
merged[TARGET_COL] = (
    merged.groupby("county_key")["wk_human_total"].shift(-FORECAST_LEAD))

# ── 3g. County label encoding + cyclical temporal features ───────────────────
le = LabelEncoder()
merged["county_encoded"] = le.fit_transform(merged["county_key"])
merged["sin_week"]        = np.sin(2 * np.pi * merged["week_num"] / 52)
merged["cos_week"]        = np.cos(2 * np.pi * merged["week_num"] / 52)
merged["in_wnv_season"]   = merged["week_num"].between(26, 40).astype(int)

print(f"\n  [3g] Temporal encodings added: county_encoded, sin_week, "
      f"cos_week, in_wnv_season")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — FINAL DATASET PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 4: FINAL DATASET PREPARATION")
print(DIVIDER)

model_df = merged.dropna(subset=[TARGET_COL]).copy()

# Filter to active season window (±4 week buffer) per county
def in_active_window(row):
    s, e = SEASON_WINDOW.get(row["county_key"], (20, 45))
    return (s - 4) <= row["week_num"] <= (e + 4)

mask     = model_df.apply(in_active_window, axis=1)
model_df = model_df[mask].copy()

print(f"  Rows after active-season filter : {len(model_df):,}")
print(f"  Counties                        : "
      f"{sorted(model_df['county_key'].unique())}")
print(f"  Year range                      : "
      f"{int(model_df['year'].min())}–{int(model_df['year'].max())}")
print(f"  Non-zero target weeks           : "
      f"{(model_df[TARGET_COL] > 0).sum():,}")

# ── Feature matrix ──────────────────────────────────────────────────────────
EXCLUDE = {
    "county_key", "year", "week_num",
    TARGET_COL,
    "wk_human_total", "wk_neuroinvasive", "wk_non_neuroinvasive",
    "gdd_proxy",
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

X_raw = model_df[feature_cols].copy()
y_raw = model_df[TARGET_COL].copy()
X_raw.fillna(X_raw.median(), inplace=True)

# ── Temporal 80/20 train-test split ─────────────────────────────────────────
sorted_idx  = model_df.sort_values(["year", "week_num"]).index
X_raw       = X_raw.loc[sorted_idx]
y_raw       = y_raw.loc[sorted_idx]
meta_sorted = model_df.loc[sorted_idx].reset_index(drop=True)

split   = int(len(X_raw) * 0.80)
X_tr_raw, X_te_raw = X_raw.iloc[:split].values, X_raw.iloc[split:].values
y_tr_raw, y_te_raw = y_raw.iloc[:split].values, y_raw.iloc[split:].values

# ── scikit-learn StandardScaler ──────────────────────────────────────────────
feat_scaler   = StandardScaler()
target_scaler = StandardScaler()

X_tr_sc = feat_scaler.fit_transform(X_tr_raw)
X_te_sc = feat_scaler.transform(X_te_raw)

y_tr_sc = target_scaler.fit_transform(y_tr_raw.reshape(-1, 1)).ravel()
y_te_sc = target_scaler.transform(y_te_raw.reshape(-1, 1)).ravel()

print(f"\n  Train samples (pre-sequence) : {len(X_tr_sc):,}  "
      f"(years {int(meta_sorted.iloc[:split]['year'].min())}–"
      f"{int(meta_sorted.iloc[:split]['year'].max())})")
print(f"  Test  samples (pre-sequence) : {len(X_te_sc):,}  "
      f"(years {int(meta_sorted.iloc[split:]['year'].min())}–"
      f"{int(meta_sorted.iloc[split:]['year'].max())})")

# ── Build rolling LSTM sequences  [samples, SEQ_LEN, n_features] ─────────────
def build_sequences(X_scaled, y_scaled, seq_len):
    """
    Slide a fixed window of length `seq_len` over the time series.
    Each sample = the last `seq_len` feature rows;
    each label  = the target at the current (last) time step.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X_scaled)):
        Xs.append(X_scaled[i - seq_len : i])
        ys.append(y_scaled[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_tr_seq, y_tr_seq = build_sequences(X_tr_sc, y_tr_sc, SEQ_LEN)
X_te_seq, y_te_seq = build_sequences(X_te_sc, y_te_sc, SEQ_LEN)

n_features = X_tr_seq.shape[2]
print(f"\n  LSTM input shape — train : {X_tr_seq.shape}  "
      f"(samples × SEQ_LEN × features)")
print(f"  LSTM input shape — test  : {X_te_seq.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print(f"STEP 5: MODEL BUILDING — Stacked LSTM  "
      f"({FORECAST_LEAD}-week forecast lead, look-back = {SEQ_LEN} weeks)")
print(DIVIDER)

def build_lstm(n_features, seq_len):
    """
    Two-layer stacked LSTM with BatchNorm, Dropout, and a Dense output.
    Designed for weekly epidemiological time-series regression.

    Architecture rationale
    ──────────────────────
    Layer 1  LSTM(128, return_sequences=True)
        Captures short-range temporal patterns (week-to-week weather,
        mosquito-pool trends).  return_sequences=True passes the full
        hidden-state sequence to layer 2.

    BatchNormalization
        Stabilises activations between stacked LSTM layers, reduces
        internal covariate shift, and speeds up training on sparse
        epidemiological data.

    Dropout(0.3)
        Regularisation — prevents over-fitting on zero-inflated
        WNV case counts.

    Layer 2  LSTM(64, return_sequences=False)
        Distils the temporal context into a single fixed-size vector
        that captures season-level dynamics (multi-week accumulation,
        annual neuroinvasive history).

    BatchNormalization + Dropout(0.2)

    Dense(32, relu)  →  Dense(1)
        Maps the learned representation to a scalar case-count prediction.
        Linear output (no activation on the final layer) suits regression.
    """
    model = Sequential([
        LSTM(128, input_shape=(seq_len, n_features),
             return_sequences=True, name="lstm_1"),
        BatchNormalization(name="bn_1"),
        Dropout(0.30, name="drop_1"),

        LSTM(64, return_sequences=False, name="lstm_2"),
        BatchNormalization(name="bn_2"),
        Dropout(0.20, name="drop_2"),

        Dense(32, activation="relu", name="dense_hidden"),
        Dense(1,  activation="linear", name="output"),
    ])
    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE),
        loss      = "mse",
        metrics   = ["mae"],
    )
    return model

model = build_lstm(n_features, SEQ_LEN)

print("\n  ── Model Architecture ──")
model.summary()

# ── Callbacks ────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor   = "val_loss",
        patience  = PATIENCE,
        restore_best_weights = True,
        verbose   = 1,
    ),
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 7,
        min_lr   = 1e-6,
        verbose  = 1,
    ),
]

print(f"\n  Training configuration:")
print(f"    Max epochs     : {EPOCHS}")
print(f"    Batch size     : {BATCH_SIZE}")
print(f"    Learning rate  : {LEARNING_RATE}")
print(f"    Early stopping : patience = {PATIENCE}")
print(f"    Sequence length: {SEQ_LEN} weeks")
print(f"    Validation     : 15 % of training data (temporal tail)")

# Temporal validation split inside training set (last 15 % of train)
val_cut = int(len(X_tr_seq) * 0.85)
X_tr_fit, X_val = X_tr_seq[:val_cut], X_tr_seq[val_cut:]
y_tr_fit, y_val = y_tr_seq[:val_cut], y_tr_seq[val_cut:]

print(f"\n    Train fit samples : {len(X_tr_fit):,}")
print(f"    Val samples       : {len(X_val):,}")
print(f"    Test samples      : {len(X_te_seq):,}\n")

history = model.fit(
    X_tr_fit, y_tr_fit,
    validation_data = (X_val, y_val),
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    callbacks       = callbacks,
    verbose         = 1,
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"\n  ✓ LSTM training complete  "
      f"(best epoch: {best_epoch} / {len(history.history['loss'])})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — PREDICTION & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 6: PREDICTION & EVALUATION")
print(DIVIDER)

# Raw (scaled) predictions → inverse-transform to original case-count scale
y_pred_sc  = model.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0).ravel()
y_pred     = target_scaler.inverse_transform(
                 y_pred_sc.reshape(-1, 1)).ravel()
y_actual   = target_scaler.inverse_transform(
                 y_te_seq.reshape(-1, 1)).ravel()

# Clip predictions to non-negative (case counts cannot be < 0)
y_pred = np.clip(y_pred, 0, None)

# ── Overall metrics (scikit-learn) ───────────────────────────────────────────
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)
r2   = r2_score(y_actual, y_pred)

print(f"\n  ── Overall Evaluation Metrics on Test Set ──")
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")

# ── Per-county metrics ───────────────────────────────────────────────────────
# The sequence build consumed the first SEQ_LEN rows, so align metadata
test_meta = meta_sorted.iloc[split + SEQ_LEN :][
    ["county_key", "year", "week_num"]
].reset_index(drop=True)

# Guard: ensure lengths match (edge case if test set < SEQ_LEN)
n_out = min(len(test_meta), len(y_actual))
test_meta = test_meta.iloc[:n_out].copy()
test_meta["actual"]    = y_actual[:n_out]
test_meta["predicted"] = y_pred[:n_out]

print(f"\n  ── Per-County Test Metrics ──")
print(f"  {'County':<18} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'N':>6}")
print("  " + "─" * 54)
for ck, grp in test_meta.groupby("county_key"):
    if len(grp) < 5:
        continue
    cr  = np.sqrt(mean_squared_error(grp["actual"], grp["predicted"]))
    cm  = mean_absolute_error(grp["actual"],         grp["predicted"])
    cr2 = r2_score(grp["actual"], grp["predicted"]) if len(grp) > 1 else float("nan")
    print(f"  {ck:<18} {cr:>8.3f} {cm:>8.3f} {cr2:>8.3f} {len(grp):>6}")

# ── First 10 actual vs predicted ────────────────────────────────────────────
print(f"\n  ── First 10 Actual vs Predicted  "
      f"(WNV human cases, {FORECAST_LEAD} weeks ahead) ──")
print(f"  {'#':<5} {'County':<16} {'Year':>5} {'Wk':>4} "
      f"{'Actual':>10} {'Predicted':>10} {'Error':>10}")
print("  " + "─" * 65)
for i, row in test_meta.head(10).iterrows():
    err = row["actual"] - row["predicted"]
    print(f"  {i+1:<5} {row['county_key']:<16} {int(row['year']):>5} "
          f"{int(row['week_num']):>4} {row['actual']:>10.2f} "
          f"{row['predicted']:>10.4f} {err:>10.4f}")

# ── Training-curve summary ───────────────────────────────────────────────────
final_train_loss = history.history["loss"][-1]
final_val_loss   = history.history["val_loss"][-1]
best_val_loss    = min(history.history["val_loss"])
print(f"\n  ── Training Curve Summary ──")
print(f"  Total epochs trained  : {len(history.history['loss'])}")
print(f"  Best epoch            : {best_epoch}")
print(f"  Best val loss (MSE)   : {best_val_loss:.6f}")
print(f"  Final train loss      : {final_train_loss:.6f}")
print(f"  Final val   loss      : {final_val_loss:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + DIVIDER)
print("STEP 7: SUMMARY")
print(DIVIDER)

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
    Stacked LSTM (2 layers: 128 → 64 units)
    Keras/TensorFlow for LSTM layers
    scikit-learn StandardScaler for feature & target normalisation
    scikit-learn metrics (RMSE, MAE, R²) for evaluation
    Temporal 80/20 split — train on earlier years, test on later years
    Look-back window : {SEQ_LEN} weeks
    Forecast horizon : {FORECAST_LEAD} weeks ahead
    Best epoch       : {best_epoch}

  Feature Groups ({len(feature_cols)} total):
    ★ Human-case history  : prev_yr_neuroinvasive, neuro_3yr_avg,
                            neuro_yoy_change, prev_yr_neuro_pct,
                            prev_yr_total_cases, prev_yr_deaths
      Weather             : TAVG lags, PRCP lags, RH, dew point,
                            growing-degree days, rolling averages
      Mosquito surveillance: positive pool lags, infection rate lags
      Demographics        : population density, income, age 65+
      Land-use            : developed %, wetlands %, cropland %
      Temporal            : sin/cos week encoding, season indicator,
                            county encoding

  Final Test-Set Metrics:
    RMSE = {rmse:.4f}
    MAE  = {mae:.4f}
    R²   = {r2:.4f}

  Coverage: All 6 counties × {int(model_df['year'].max() - model_df['year'].min() + 1)} years
""")

print(DIVIDER)
print("Done.")
