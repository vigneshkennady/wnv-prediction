#!/usr/bin/env python3
"""
West Nile Virus (WNV) 2025 Prediction Script - ARIMAX Version
=============================================================
This script extends the trained ARIMAX model to predict 2025 WNV cases
for all 6 counties using available weather, demographic, and historical data.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (same as original ARIMAX model)
# ─────────────────────────────────────────────────────────────────────────────
BASE          = "ref/"
FORECAST_LEAD = 4    # weeks ahead to forecast
RANDOM_STATE  = 42

# ARIMA order (p, d, q)
ARIMA_ORDER = (2, 0, 1)

# Seasonal ARIMA order (P, D, Q, S)
SEASONAL_ORDER = (1, 0, 1, 52)

# Counties
COUNTIES = [
    "Boulder_CO", "Cook_IL", "Dallas_TX",
    "Larimer_CO", "LosAngeles_CA", "Maricopa_AZ",
]

# Exogenous feature columns (same as original)
EXOG_COLS = [
    'TAVG_mean', 'PRCP_sum', 'gdd_cumul8', 'RH_mean', 'TAVG_roll4',
    'PRCP_cumul4', 'TAVG_mean_lag1', 'TAVG_mean_lag2', 'PRCP_sum_lag1',
    'PRCP_sum_lag4', 'positive_pools', 'pos_pools_lag1', 'infect_rate_lag1',
    'prev_yr_neuroinvasive', 'neuro_3yr_avg', 'neuro_yoy_change',
    'pop_density_per_sqmi', 'poverty_rate', 'cdl_developed_pct',
    'cdl_wetlands_pct', 'sin_week', 'cos_week', 'in_wnv_season'
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

def train_arimax_models(df):
    """Train ARIMAX models for each county"""
    
    print("Training ARIMAX models...")
    
    models = {}
    scalers = {}
    
    for county in COUNTIES:
        print(f"\nTraining model for {county}...")
        
        # Filter data for this county
        county_data = df[df['county_key'] == county].copy()
        county_data = county_data.sort_values(['year', 'week_num'])
        
        print(f"  Available data points: {len(county_data)}")
        
        if len(county_data) < 50:  # Reduced threshold
            print(f"  Skipping {county} - insufficient data")
            continue
        
        # Create target and exogenous variables
        y = county_data['target'].fillna(0)
        X = county_data[EXOG_COLS].copy()
        
        # Handle any remaining NaN/inf values
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], X.median())
        
        # Log transform target
        y_log = np.log1p(y)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(county_data) * 0.8)
        y_train, y_test = y_log.iloc[:split_idx], y_log.iloc[split_idx:]
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        
        print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Scale exogenous variables
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Ensure no NaN/inf in scaled data
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # Fit ARIMAX model with simpler parameters
            model = SARIMAX(
                endog=y_train,
                exog=X_train_scaled,
                order=(1, 0, 1),  # Simplified order
                seasonal_order=(0, 0, 0, 0),  # No seasonal component for simplicity
                enforce_stationarity=False,
                enforce_invertibility=False,
                concentrate_scale=True,
            )
            
            results = model.fit(disp=False, maxiter=50)
            
            models[county] = results
            scalers[county] = scaler
            
            print(f"  ✓ Model trained successfully")
            
        except Exception as e:
            print(f"  ✗ Model training failed: {str(e)}")
            # Try even simpler model
            try:
                model = SARIMAX(
                    endog=y_train,
                    exog=X_train_scaled,
                    order=(1, 0, 0),  # AR only
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                
                results = model.fit(disp=False, maxiter=50)
                
                models[county] = results
                scalers[county] = scaler
                
                print(f"  ✓ Simplified model trained successfully")
                
            except Exception as e2:
                print(f"  ✗ Simplified model also failed: {str(e2)}")
                continue
    
    print(f"\nSuccessfully trained {len(models)} models")
    return models, scalers

def create_2025_features():
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
    return df_2025

def predict_2025_arimax():
    """Main function to predict 2025 WNV cases using ARIMAX"""
    
    print("=" * 80)
    print("WNV 2025 PREDICTION - ARIMAX VERSION")
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
    
    # Train ARIMAX models
    models, scalers = train_arimax_models(merged)
    
    if not models:
        print("No models were successfully trained. Cannot make predictions.")
        return None
    
    # Create 2025 features
    df_2025 = create_2025_features()
    
    # Make predictions for 2025
    print("\nMaking 2025 predictions...")
    
    predictions = []
    
    for county in COUNTIES:
        if county not in models:
            print(f"Skipping {county} - no trained model available")
            continue
        
        print(f"Predicting for {county}...")
        
        # Get county data
        county_2025 = df_2025[df_2025['county_key'] == county].copy()
        county_2025 = county_2025.sort_values('week_num')
        
        # Get exogenous variables
        X_2025 = county_2025[EXOG_COLS].copy()
        X_2025 = X_2025.fillna(X_2025.median())
        X_2025 = X_2025.replace([np.inf, -np.inf], X_2025.median())
        
        # Scale using trained scaler
        X_2025_scaled = scalers[county].transform(X_2025)
        X_2025_scaled = np.nan_to_num(X_2025_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            # Make predictions
            model = models[county]
            pred_log = model.forecast(steps=len(X_2025_scaled), exog=X_2025_scaled)
            pred_cases = np.expm1(pred_log)  # Reverse log transform
            pred_cases = np.maximum(0, pred_cases)  # No negative cases
            
            # Store predictions
            for i, (idx, row) in enumerate(county_2025.iterrows()):
                predictions.append({
                    'county_key': county,
                    'year': 2025,
                    'week_num': row['week_num'],
                    'predicted_cases': pred_cases[i]
                })
                
        except Exception as e:
            print(f"  Prediction failed for {county}: {str(e)}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(predictions)
    
    if results_df.empty:
        print("No predictions were generated.")
        return None
    
    # Summary by county
    print("\n" + "=" * 80)
    print("2025 ARIMAX PREDICTION SUMMARY")
    print("=" * 80)
    
    print("\nPredicted 2025 WNV Cases by County (ARIMAX):")
    print("-" * 50)
    for county in results_df['county_key'].unique():
        county_data = results_df[results_df['county_key'] == county]
        total_cases = county_data['predicted_cases'].sum()
        max_weekly = county_data['predicted_cases'].max()
        peak_week = county_data.loc[county_data['predicted_cases'].idxmax(), 'week_num']
        
        print(f"{county:<18}: Total={total_cases:>6.1f}, Peak Week={peak_week:>2} ({max_weekly:.2f} cases)")
    
    # Save predictions
    output_file = "files_rf/wnv_2025_arimax_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nARIMAX predictions saved to: {output_file}")
    
    # Save weekly predictions
    weekly_summary = results_df.pivot(index='week_num', columns='county_key', values='predicted_cases').round(2)
    weekly_file = "files_rf/wnv_2025_arimax_weekly_predictions.csv"
    weekly_summary.to_csv(weekly_file)
    print(f"Weekly ARIMAX predictions saved to: {weekly_file}")
    
    return results_df

if __name__ == "__main__":
    predictions_2025 = predict_2025_arimax()
    print("\n" + "=" * 80)
    print("2025 ARIMAX PREDICTION COMPLETE")
    print("=" * 80)
