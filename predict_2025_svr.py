#!/usr/bin/env python3
"""
West Nile Virus (WNV) 2025 Prediction Script - SVR Version
=============================================================
This script extends the trained SVR model to predict 2025 WNV cases
for all 6 counties using available weather, demographic, and historical data.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (same as original SVR model)
# ─────────────────────────────────────────────────────────────────────────────
BASE          = "ref/"
FORECAST_LEAD = 4      # weeks ahead to forecast
RANDOM_STATE  = 42
CV_FOLDS      = 5

# SVR hyper-parameter search grid (simplified)
PARAM_GRID = {
    "svr__C"      : [1.0, 10.0, 100.0],
    "svr__epsilon": [0.01, 0.1, 0.5],
    "svr__gamma"  : ["scale", "auto"],
    "svr__kernel" : ["rbf"],
}

# Counties
COUNTIES = [
    "Boulder_CO", "Cook_IL", "Dallas_TX",
    "Larimer_CO", "LosAngeles_CA", "Maricopa_AZ",
]

def load_and_preprocess_data():
    """Load and preprocess all data with proper NaN handling"""
    
    print("Loading and preprocessing data...")
    
    # Load data files
    weather = pd.read_csv(f"{BASE}wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv", parse_dates=["date"])
    demographics = pd.read_csv(f"{BASE}demographics_combined.csv")
    land_use = pd.read_csv(f"{BASE}cdl_strategy_d_pc.csv")
    surveillance = pd.read_csv(f"{BASE}mosquito_surveillance_county_week.csv")
    human_cases = pd.read_csv(f"{BASE}wnv_human_cases_county_year.csv")
    
    # Clean demographics
    demographics.replace(-999, np.nan, inplace=True)
    
    print(f"  Weather         : {len(weather):>6,} rows")
    print(f"  Demographics    : {len(demographics):>6,} rows")
    print(f"  Land Use        : {len(land_use):>6,} rows")
    print(f"  Mosquito Surv.  : {len(surveillance):>6,} rows")
    print(f"  WNV Human Cases : {len(human_cases):>6,} rows")
    
    return weather, demographics, land_use, surveillance, human_cases

def create_weekly_features(weather):
    """Create weekly weather features"""
    
    print("Creating weekly weather features...")
    
    # Extract week and year
    weather['week_num'] = weather['date'].dt.isocalendar().week
    weather['year'] = weather['date'].dt.year
    
    # Aggregate to weekly level
    weather_weekly = weather.groupby(['county', 'year', 'week_num']).agg({
        'TAVG': ['mean', 'std'],
        'TMAX': 'max',
        'TMIN': 'min',
        'PRCP': ['sum', 'max'],
        'RH': 'mean',
        'DEWP_mean': 'mean',
        'WIND': 'mean'
    }).reset_index()
    
    # Flatten column names
    weather_weekly.columns = ['county', 'year', 'week_num', 'TAVG_mean', 'TAVG_std',
                              'TMAX_max', 'TMIN_min', 'PRCP_sum', 'PRCP_max',
                              'RH_mean', 'DEWP_mean', 'WIND_mean']
    
    # Create rolling averages
    weather_weekly = weather_weekly.sort_values(['county', 'year', 'week_num'])
    weather_weekly['TAVG_roll4'] = weather_weekly.groupby('county')['TAVG_mean'].rolling(4, min_periods=1).mean().reset_index(0, drop=True)
    weather_weekly['PRCP_cumul4'] = weather_weekly.groupby('county')['PRCP_sum'].rolling(4, min_periods=1).sum().reset_index(0, drop=True)
    
    # Calculate growing degree days
    weather_weekly['gdd_cumul8'] = weather_weekly.groupby('county').apply(
        lambda x: np.maximum(0, x['TAVG_mean'] - 10).cumsum()
    ).reset_index(0, drop=True)
    
    # Create county_key
    weather_weekly['county_key'] = weather_weekly['county'].str.replace(' County', '').str.replace(' ', '_') + '_' + weather_weekly['year'].astype(str).str[:2]
    
    return weather_weekly

def create_lagged_features(df):
    """Create lagged features"""
    
    print("Creating lagged features...")
    
    df = df.sort_values(['county_key', 'year', 'week_num'])
    
    # Weather lags
    for lag in [1, 2, 4]:
        df[f'TAVG_mean_lag{lag}'] = df.groupby('county_key')['TAVG_mean'].shift(lag)
        df[f'PRCP_sum_lag{lag}'] = df.groupby('county_key')['PRCP_sum'].shift(lag)
    
    # Human case lags
    df['prev_yr_neuroinvasive'] = df.groupby('county_key')['neuroinvasive'].shift(52)  # Previous year
    df['neuro_3yr_avg'] = df.groupby('county_key')['neuroinvasive'].rolling(156, min_periods=1).mean().reset_index(0, drop=True)
    df['neuro_yoy_change'] = df.groupby('county_key')['neuroinvasive'].diff(52)
    
    # Mosquito surveillance lags
    df['pos_pools_lag1'] = df.groupby('county_key')['positive_pools'].shift(1)
    df['infect_rate_lag1'] = df.groupby('county_key')['infection_rate'].shift(1)
    
    return df

def create_temporal_features(df):
    """Create temporal features"""
    
    print("Creating temporal features...")
    
    # Cyclical encoding
    df['sin_week'] = np.sin(2 * np.pi * df['week_num'] / 52)
    df['cos_week'] = np.cos(2 * np.pi * df['week_num'] / 52)
    
    # WNV season indicator
    df['in_wnv_season'] = ((df['week_num'] >= 20) & (df['week_num'] <= 45)).astype(int)
    
    return df

def merge_datasets(weather_weekly, demographics, land_use, surveillance, human_cases):
    """Merge all datasets"""
    
    print("Merging datasets...")
    
    # Start with weather data
    merged = weather_weekly.copy()
    
    # Add human cases
    merged = merged.merge(human_cases, on=['county_key', 'year'], how='left')
    
    # Add demographics (use most recent available)
    demo_latest = demographics.loc[demographics.groupby('county_key')['year'].idxmax()]
    demo_cols = ['county_key'] + [col for col in demographics.columns if col not in ['county_key', 'year', 'county_name', 'fips', 'state']]
    merged = merged.merge(demo_latest[demo_cols], on='county_key', how='left')
    
    # Add land use (use most recent available)
    land_latest = land_use.loc[land_use.groupby('county_key')['year'].idxmax()]
    land_cols = ['county_key'] + [col for col in land_use.columns if col not in ['county_key', 'year', 'county_name', 'fips', 'state']]
    merged = merged.merge(land_latest[land_cols], on='county_key', how='left')
    
    # Add mosquito surveillance
    surv_cols = ['county_key', 'year', 'week_num'] + [col for col in surveillance.columns if col not in ['county_key', 'year', 'week_num', 'county_name', 'fips', 'state']]
    merged = merged.merge(surveillance[surv_cols], on=['county_key', 'year', 'week_num'], how='left')
    
    # Fill missing values
    for col in ['total_cases', 'neuroinvasive', 'non_neuroinvasive', 'deaths']:
        merged[col] = merged[col].fillna(0)
    
    for col in ['positive_pools', 'infection_rate']:
        merged[col] = merged[col].fillna(0)
    
    return merged

def create_target_variable(df):
    """Create 4-week ahead target variable"""
    
    print("Creating target variable...")
    
    # Create 4-week ahead target
    df = df.sort_values(['county_key', 'year', 'week_num'])
    df['target'] = df.groupby('county_key')['total_cases'].shift(-FORECAST_LEAD)
    
    return df

def clean_data(df):
    """Clean data - handle NaN and inf values"""
    
    print("Cleaning data...")
    
    # Fill NaN values with column medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def train_svr_model(df):
    """Train SVR model with proper data cleaning"""
    
    print("Training SVR model...")
    
    # Select feature columns (exclude target and metadata)
    feature_cols = [col for col in df.columns if col not in [
        'county', 'county_key', 'year', 'week_num', 'target', 'data_confidence', 
        'notes', 'source_primary', 'source_secondary', 'county_name', 'state', 'fips'
    ]]
    
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(X.median())
    y = y.fillna(0)
    
    # Log transform target
    y_log = np.log1p(y)
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_log.iloc[:split_idx], y_log.iloc[split_idx:]
    
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(feature_cols)}")
    
    # Ensure no NaN values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    # Replace inf values properly
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    # Create pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf")),
    ])
    
    # Simplified grid search
    print("  Running simplified grid search...")
    
    try:
        # Use simpler parameters to avoid convergence issues
        simple_grid = {
            "svr__C": [10.0, 100.0],
            "svr__epsilon": [0.1, 0.5],
            "svr__gamma": ["scale"],
        }
        
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced folds
        grid_search = GridSearchCV(
            pipe, 
            simple_grid, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"  ✓ Best parameters: C={grid_search.best_params_['svr__C']}, "
              f"epsilon={grid_search.best_params_['svr__epsilon']}")
        
    except Exception as e:
        print(f"  ✗ Grid search failed: {str(e)}")
        print("  Using default parameters...")
        
        # Fallback to default parameters
        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.1)),
        ])
        best_model.fit(X_train, y_train)
    
    return best_model, feature_cols

def create_2025_features(feature_cols):
    """Create features for 2025 predictions"""
    
    print("Creating 2025 features...")
    
    # Load historical data for training
    weather, demographics, land_use, surveillance, human_cases = load_and_preprocess_data()
    
    # Create historical dataset
    weather_weekly = create_weekly_features(weather)
    
    # Create 2025 template
    weeks_2025 = list(range(1, 53))
    data_2025 = []
    
    for county in COUNTIES:
        # Get 2024 weather data as baseline
        county_base = county.split('_')[0]
        weather_2024 = weather_weekly[(weather_weekly['county'].str.contains(county_base)) & 
                                       (weather_weekly['year'] == 2024)]
        
        if weather_2024.empty:
            # Use default weather patterns
            weather_defaults = {
                'TAVG_mean': 20, 'TAVG_std': 5, 'TMAX_max': 30, 'TMIN_min': 10,
                'PRCP_sum': 50, 'PRCP_max': 100, 'RH_mean': 60, 'DEWP_mean': 10,
                'WIND_mean': 10, 'TAVG_roll4': 20, 'PRCP_cumul4': 200, 'gdd_cumul8': 100
            }
        else:
            weather_defaults = weather_2024[['TAVG_mean', 'TAVG_std', 'TMAX_max', 'TMIN_min',
                                           'PRCP_sum', 'PRCP_max', 'RH_mean', 'DEWP_mean',
                                           'WIND_mean', 'TAVG_roll4', 'PRCP_cumul4', 'gdd_cumul8']].mean().to_dict()
        
        # Get 2024 human cases
        human_2024 = human_cases[human_cases['county_key'] == county]
        if human_2024.empty:
            human_defaults = {'total_cases': 0, 'neuroinvasive': 0, 'non_neuroinvasive': 0, 'deaths': 0}
        else:
            human_defaults = human_2024.iloc[0].to_dict()
        
        for week in weeks_2025:
            row = {
                'county_key': county,
                'year': 2025,
                'week_num': week,
                'total_cases': human_defaults['total_cases'],
                'neuroinvasive': human_defaults['neuroinvasive'],
                'non_neuroinvasive': human_defaults['non_neuroinvasive'],
                'deaths': human_defaults['deaths'],
                'positive_pools': 0,
                'infection_rate': 0.01,
            }
            
            # Add weather features
            row.update(weather_defaults)
            
            # Add temporal features
            row['sin_week'] = np.sin(2 * np.pi * week / 52)
            row['cos_week'] = np.cos(2 * np.pi * week / 52)
            row['in_wnv_season'] = 1 if 20 <= week <= 45 else 0
            
            # Add lagged features (use current values as approximation)
            row['TAVG_mean_lag1'] = weather_defaults['TAVG_mean']
            row['TAVG_mean_lag2'] = weather_defaults['TAVG_mean']
            row['PRCP_sum_lag1'] = weather_defaults['PRCP_sum']
            row['PRCP_sum_lag4'] = weather_defaults['PRCP_sum']
            row['pos_pools_lag1'] = 0
            row['infect_rate_lag1'] = 0.01
            row['prev_yr_neuroinvasive'] = human_defaults['neuroinvasive']
            row['neuro_3yr_avg'] = human_defaults['neuroinvasive']
            row['neuro_yoy_change'] = 0
            
            # Add demographic and land use features (use reasonable defaults)
            row.update({
                'pop_density_per_sqmi': 1000,
                'poverty_rate': 0.12,
                'cdl_developed_pct': 0.25,
                'cdl_wetlands_pct': 0.05,
            })
            
            data_2025.append(row)
    
    df_2025 = pd.DataFrame(data_2025)
    
    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in df_2025.columns:
            df_2025[col] = 0
    
    print(f"2025 dataset shape: {df_2025.shape}")
    
    return df_2025

def predict_2025_svr():
    """Main function to predict 2025 WNV cases using SVR"""
    
    print("=" * 80)
    print("WNV 2025 PREDICTION - SVR VERSION")
    print("=" * 80)
    
    # Load and prepare historical data
    weather, demographics, land_use, surveillance, human_cases = load_and_preprocess_data()
    
    # Create historical features
    weather_weekly = create_weekly_features(weather)
    merged = merge_datasets(weather_weekly, demographics, land_use, surveillance, human_cases)
    merged = create_lagged_features(merged)
    merged = create_temporal_features(merged)
    merged = create_target_variable(merged)
    merged = clean_data(merged)
    
    # Train SVR model
    model, feature_cols = train_svr_model(merged)
    
    # Create 2025 features
    df_2025 = create_2025_features(feature_cols)
    
    # Make predictions for 2025
    print("\nMaking 2025 predictions...")
    
    # Prepare 2025 data
    X_2025 = df_2025[feature_cols].copy()
    X_2025 = X_2025.fillna(X_2025.median())
    X_2025 = X_2025.replace([np.inf, -np.inf], X_2025.median())
    
    try:
        # Make predictions
        y_pred_log = model.predict(X_2025)
        y_pred = np.expm1(y_pred_log)  # Reverse log transform
        y_pred = np.maximum(0, y_pred)  # No negative cases
        
        # Store predictions
        predictions = []
        for i, (idx, row) in enumerate(df_2025.iterrows()):
            predictions.append({
                'county_key': row['county_key'],
                'year': 2025,
                'week_num': row['week_num'],
                'predicted_cases': y_pred[i]
            })
        
        results_df = pd.DataFrame(predictions)
        
        # Summary by county
        print("\n" + "=" * 80)
        print("2025 SVR PREDICTION SUMMARY")
        print("=" * 80)
        
        print("\nPredicted 2025 WNV Cases by County (SVR):")
        print("-" * 50)
        for county in results_df['county_key'].unique():
            county_data = results_df[results_df['county_key'] == county]
            total_cases = county_data['predicted_cases'].sum()
            max_weekly = county_data['predicted_cases'].max()
            peak_week = county_data.loc[county_data['predicted_cases'].idxmax(), 'week_num']
            
            print(f"{county:<18}: Total={total_cases:>6.1f}, Peak Week={peak_week:>2} ({max_weekly:.2f} cases)")
        
        # Save predictions
        output_file = "files_rf/wnv_2025_svr_predictions.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSVR predictions saved to: {output_file}")
        
        # Save weekly predictions
        weekly_summary = results_df.pivot(index='week_num', columns='county_key', values='predicted_cases').round(2)
        weekly_file = "files_rf/wnv_2025_svr_weekly_predictions.csv"
        weekly_summary.to_csv(weekly_file)
        print(f"Weekly SVR predictions saved to: {weekly_file}")
        
        return results_df
        
    except Exception as e:
        print(f"Prediction failed: {str(e)}")
        return None

if __name__ == "__main__":
    predictions_2025 = predict_2025_svr()
    print("\n" + "=" * 80)
    print("2025 SVR PREDICTION COMPLETE")
    print("=" * 80)
