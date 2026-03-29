"""
=============================================================================
WNV Research Project — Weather Feature Engineering
=============================================================================
Reads the raw daily weather CSVs produced by wnv_weather_retrieval.py and
engineers WNV-predictive features:
  - Weekly aggregates (mean, max, sum)
  - Lag features: 1, 2, 3, 4, 6, 8 week lags
  - Rolling windows: 2-week, 4-week, 8-week rolling means
  - Degree-day accumulation (base 14°C Culex development threshold)
  - Heat index (apparent temperature)
  - Extreme heat days (TMAX >= 35°C)

Output:
  wnv_weather_data/ALL_COUNTIES_weather_features.csv

Usage:
  python wnv_feature_engineering.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

IN_PATH  = Path("wnv_weather_data/ALL_COUNTIES_weather_{START}_{END}.csv")
OUT_DIR  = Path("wnv_weather_data")
START_YEAR = 2000
END_YEAR   = 2024

BASE_TEMP_CULEX = 14.3   # °C — minimum development threshold for Culex pipiens


def load_data() -> pd.DataFrame:
    path = Path(f"wnv_weather_data/ALL_COUNTIES_weather_{START_YEAR}_{END_YEAR}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Run wnv_weather_retrieval.py first to generate: {path}"
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["county","date"]).reset_index(drop=True)
    return df


def add_heat_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Heat Index (°C) from TMAX and RH using Rothfusz regression.
    Valid when TMAX >= 27°C and RH >= 40%.
    """
    T  = df["TMAX"]
    RH = df["RH"]
    HI = (-8.78469475556
          + 1.61139411   * T
          + 2.33854883889 * RH
          - 0.14611605   * T * RH
          - 0.012308094  * T**2
          - 0.016424828  * RH**2
          + 0.002211732  * T**2 * RH
          + 0.00072546   * T * RH**2
          - 0.000003582  * T**2 * RH**2)
    df["heat_index"] = np.where((T >= 27) & (RH >= 40), HI.round(1), T)
    return df


def add_degree_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Growing degree days for Culex development.
    GDD = max(0, (TMAX + TMIN)/2 - base_temp)
    Cumulative sum within each county-year (reset Jan 1).
    """
    df["GDD_daily"] = ((df["TMAX"] + df["TMIN"]) / 2 - BASE_TEMP_CULEX).clip(lower=0)
    df["year"]      = df["date"].dt.year
    df["GDD_cumul"] = df.groupby(["county","year"])["GDD_daily"].cumsum()
    return df


def weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data to ISO week-level per county.
    Week = ISO week ending Sunday. Adds epidemiological week column (epiweek).
    """
    df = df.copy()
    df["year"]    = df["date"].dt.isocalendar().year.astype(int)
    df["epiweek"] = df["date"].dt.isocalendar().week.astype(int)

    agg_funcs = {
        "TAVG"       : ["mean", "max", "min"],
        "TMAX"       : ["mean", "max"],
        "TMIN"       : ["mean", "min"],
        "PRCP"       : ["sum", "mean", "max"],
        "WIND"       : ["mean", "max"],
        "RH"         : ["mean", "max", "min"],
        "heat_index" : ["mean", "max"],
        "GDD_daily"  : "sum",
        "GDD_cumul"  : "last",
    }
    # Only keep columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}

    weekly = (df.groupby(["county","state","fips","year","epiweek"])
                .agg(agg_funcs)
                .reset_index())

    weekly.columns = [
        "_".join(c).strip("_") if c[1] else c[0]
        for c in weekly.columns
    ]

    # Consecutive weeks with PRCP == 0 (drought proxy)
    weekly["week_start"] = pd.to_datetime(
        weekly["year"].astype(str) + weekly["epiweek"].astype(str) + "1",
        format="%G%V%u"
    )

    # Count extreme heat days (TMAX >= 35°C) per week
    weekly_heat = (df.groupby(["county","year","epiweek"])
                     .apply(lambda g: (g["TMAX"] >= 35).sum())
                     .reset_index(name="extreme_heat_days"))
    weekly = pd.merge(weekly, weekly_heat, on=["county","year","epiweek"], how="left")

    # Count wet days (PRCP > 1 mm) per week
    if "PRCP" in df.columns:
        weekly_wet = (df.groupby(["county","year","epiweek"])
                        .apply(lambda g: (g["PRCP"] > 1.0).sum())
                        .reset_index(name="wet_days"))
        weekly = pd.merge(weekly, weekly_wet, on=["county","year","epiweek"], how="left")

    weekly = weekly.sort_values(["county","year","epiweek"]).reset_index(drop=True)
    return weekly


def add_lag_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features (1–8 weeks) and rolling averages for key variables.
    Lags are within each county group.
    WNV transmission typically lags weather by 2–6 weeks.
    """
    VARS_TO_LAG = [
        "TAVG_mean", "TMAX_mean", "TMIN_mean",
        "PRCP_sum", "RH_mean", "WIND_mean",
        "GDD_daily_sum", "extreme_heat_days", "wet_days"
    ]
    LAG_WEEKS   = [1, 2, 3, 4, 6, 8]
    ROLLING     = [2, 4, 8]   # weeks

    VARS_TO_LAG = [v for v in VARS_TO_LAG if v in df.columns]

    df = df.sort_values(["county","year","epiweek"]).copy()

    for var in VARS_TO_LAG:
        grp = df.groupby("county")[var]
        for lag in LAG_WEEKS:
            df[f"{var}_lag{lag}w"] = grp.shift(lag)
        for win in ROLLING:
            df[f"{var}_roll{win}w"] = grp.transform(
                lambda x: x.shift(1).rolling(win, min_periods=max(1, win//2)).mean()
            )

    return df


def main():
    print("Loading raw daily weather data …")
    daily = load_data()

    print(f"Loaded {len(daily):,} daily records across "
          f"{daily['county'].nunique()} counties.")

    # Daily feature additions
    print("Computing heat index …")
    daily = add_heat_index(daily)
    print("Computing degree days …")
    daily = add_degree_days(daily)

    # Save enriched daily
    daily_out = OUT_DIR / f"ALL_COUNTIES_weather_daily_enriched.csv"
    daily.to_csv(daily_out, index=False)
    print(f"Enriched daily data saved → {daily_out}")

    # Weekly aggregation
    print("Aggregating to weekly epiweeks …")
    weekly = weekly_features(daily)
    print(f"Weekly records: {len(weekly):,}")

    # Lag / rolling features
    print("Adding lag and rolling features …")
    weekly = add_lag_and_rolling(weekly)

    weekly_out = OUT_DIR / "ALL_COUNTIES_weather_features.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"\nFinal feature set saved → {weekly_out}")
    print(f"Shape: {weekly.shape[0]:,} rows × {weekly.shape[1]} columns")

    # Column summary
    print("\nFeature columns:")
    for col in sorted(weekly.columns):
        missing = weekly[col].isna().sum()
        pct     = round(100 * missing / len(weekly), 1) if len(weekly) else 0
        print(f"  {col:<45} missing: {missing:>6} ({pct}%)")


if __name__ == "__main__":
    main()
