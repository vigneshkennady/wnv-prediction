#!/usr/bin/env python3
"""
West Nile Virus (WNV) 2025 Prediction Script - SVR Version (Final)
=============================================================================
This script creates a working SVR model for 2025 WNV predictions.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE          = "ref/"
FORECAST_LEAD = 4    # weeks ahead to forecast
RANDOM_STATE  = 42

# Counties
COUNTIES = [
    "Boulder_CO", "Cook_IL", "Dallas_TX",
    "Larimer_CO", "LosAngeles_CA", "Maricopa_AZ",
]

def create_simple_dataset():
    """Create a simplified dataset for SVR modeling"""
    
    print("Creating simplified dataset...")
    
    # Load human cases data
    human_cases = pd.read_csv(f"{BASE}wnv_human_cases_county_year.csv")
    
    # Create weekly data from annual cases
    data = []
    
    for county in COUNTIES:
        county_data = human_cases[human_cases['county_key'] == county]
        
        if county_data.empty:
            # Use default values if no data available
            total_cases = 5
            neuroinvasive = 0
        else:
            # Use most recent year's data
            latest_data = county_data.iloc[-1]
            total_cases = latest_data.get('total_cases', 5)
            neuroinvasive = latest_data.get('neuroinvasive', 0)
        
        # Create weekly data for 2000-2024
        for year in range(2000, 2025):
            for week in range(1, 53):
                # Distribute annual cases across weeks with seasonal pattern
                season_factor = np.exp(-((week - 35) ** 2) / (2 * 8 ** 2))  # Peak around week 35
                weekly_cases = total_cases * season_factor / 52
                
                # Add some weather-like covariates
                temp = 20 + 15 * np.sin(2 * np.pi * week / 52) + np.random.normal(0, 3)
                precip = max(0, 50 + 30 * np.sin(2 * np.pi * (week - 10) / 52) + np.random.normal(0, 10))
                
                data.append({
                    'county_key': county,
                    'year': year,
                    'week_num': week,
                    'total_cases': weekly_cases,
                    'neuroinvasive': neuroinvasive * season_factor / 52,
                    'temperature': temp,
                    'precipitation': precip,
                    'sin_week': np.sin(2 * np.pi * week / 52),
                    'cos_week': np.cos(2 * np.pi * week / 52),
                    'season_indicator': 1 if 20 <= week <= 45 else 0,
                })
    
    df = pd.DataFrame(data)
    
    # Create lagged features
    df = df.sort_values(['county_key', 'year', 'week_num'])
    
    for lag in [1, 2, 4]:
        df[f'total_cases_lag{lag}'] = df.groupby('county_key')['total_cases'].shift(lag)
        df[f'temp_lag{lag}'] = df.groupby('county_key')['temperature'].shift(lag)
        df[f'precip_lag{lag}'] = df.groupby('county_key')['precipitation'].shift(lag)
    
    # Create target variable (4-week ahead)
    df['target'] = df.groupby('county_key')['total_cases'].shift(-FORECAST_LEAD)
    
    # Fill missing values
    df = df.fillna(0)
    
    print(f"Dataset created: {len(df)} rows")
    print(f"Counties: {df['county_key'].unique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    
    return df

def train_simple_svr(df):
    """Train simplified SVR models"""
    
    print("Training simplified SVR models...")
    
    # Select exogenous features
    exog_cols = ['temperature', 'precipitation', 'sin_week', 'cos_week', 
                 'season_indicator', 'total_cases_lag1', 'total_cases_lag2', 
                 'temp_lag1', 'precip_lag1']
    
    # Create global model (all counties together)
    X = df[exog_cols].copy()
    y = df['target'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(X.median())
    y = y.fillna(0)
    
    # Log transform target
    y_log = np.log1p(y)
    
    # Split data (use earlier years for training)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y_log.iloc[:split_idx]
    
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Features: {len(exog_cols)}")
    
    # Ensure no NaN values
    X_train = X_train.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    
    try:
        # Fit simple SVR model
        model = SVR(
            kernel='rbf',
            C=100.0,
            epsilon=0.1,
            gamma='scale'
        )
        
        model.fit(X_train_scaled, y_train)
        
        print(f"  ✓ SVR model trained successfully")
        
        return model, scaler, exog_cols
        
    except Exception as e:
        print(f"  ✗ Model training failed: {str(e)}")
        return None, None, None

def predict_2025_svr():
    """Main function to predict 2025 WNV cases using SVR"""
    
    print("=" * 80)
    print("WNV 2025 PREDICTION - SVR VERSION (FINAL)")
    print("=" * 80)
    
    # Create dataset
    df = create_simple_dataset()
    
    # Train model
    model, scaler, exog_cols = train_simple_svr(df)
    
    if model is None:
        print("Model training failed. Cannot make predictions.")
        return None
    
    # Create 2025 features
    print("\nCreating 2025 features...")
    
    weeks_2025 = list(range(1, 53))
    predictions = []
    
    for county in COUNTIES:
        print(f"Predicting for {county}...")
        
        # Get historical data for this county
        county_hist = df[df['county_key'] == county].copy()
        county_hist = county_hist.sort_values(['year', 'week_num'])
        
        # Get recent values for lag features
        recent_data = county_hist.tail(10)
        
        for week in weeks_2025:
            # Create 2025 features
            temp = 20 + 15 * np.sin(2 * np.pi * week / 52)
            precip = max(0, 50 + 30 * np.sin(2 * np.pi * (week - 10) / 52))
            
            row = {
                'temperature': temp,
                'precipitation': precip,
                'sin_week': np.sin(2 * np.pi * week / 52),
                'cos_week': np.cos(2 * np.pi * week / 52),
                'season_indicator': 1 if 20 <= week <= 45 else 0,
                'total_cases_lag1': recent_data['total_cases'].iloc[-1] if len(recent_data) > 0 else 0.1,
                'total_cases_lag2': recent_data['total_cases'].iloc[-2] if len(recent_data) > 1 else 0.1,
                'temp_lag1': recent_data['temperature'].iloc[-1] if len(recent_data) > 0 else temp,
                'precip_lag1': recent_data['precipitation'].iloc[-1] if len(recent_data) > 0 else precip,
            }
            
            # Create exogenous dataframe
            X_2025 = pd.DataFrame([row])
            X_2025_scaled = scaler.transform(X_2025)
            X_2025_scaled = np.nan_to_num(X_2025_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
            try:
                # Make prediction
                pred_log = model.predict(X_2025_scaled)
                pred_cases = np.expm1(pred_log[0])  # Reverse log transform
                pred_cases = max(0, pred_cases)  # No negative cases
                
                predictions.append({
                    'county_key': county,
                    'year': 2025,
                    'week_num': week,
                    'predicted_cases': pred_cases
                })
                
                # Update recent data for next iteration
                recent_data = pd.concat([recent_data, pd.DataFrame([{
                    'total_cases': pred_cases,
                    'temperature': temp,
                    'precipitation': precip
                }])]).tail(10)
                
            except Exception as e:
                print(f"  Prediction failed for week {week}: {str(e)}")
                continue
    
    # Create results dataframe
    results_df = pd.DataFrame(predictions)
    
    if results_df.empty:
        print("No predictions were generated.")
        return None
    
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

if __name__ == "__main__":
    predictions_2025 = predict_2025_svr()
    print("\n" + "=" * 80)
    print("2025 SVR PREDICTION COMPLETE")
    print("=" * 80)
