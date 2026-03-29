#!/usr/bin/env python3
"""
West Nile Virus (WNV) 2025 Prediction Script - LSTM Version
=============================================================
This script extends the trained LSTM model to predict 2025 WNV cases
for all 6 counties using available weather, demographic, and historical data.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION (same as original LSTM model)
# ─────────────────────────────────────────────────────────────────────────────
BASE          = "ref/"
FORECAST_LEAD = 4
SEQ_LEN       = 8          # Look-back window (weeks)
LEARNING_RATE = 0.001
RANDOM_STATE  = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def load_and_preprocess_historical_data():
    """Load and preprocess historical data for training"""
    
    print("Loading and preprocessing historical data...")
    
    # Load data files
    weather_data = pd.read_csv(f"{BASE}wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv")
    human_cases = pd.read_csv(f"{BASE}wnv_human_cases_county_year.csv")
    
    # Extract year from date and convert to datetime
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    weather_data['year'] = weather_data['date'].dt.year
    
    # Create weekly features from weather data
    weather_weekly = weather_data.groupby(['county', 'year', weather_data['date'].dt.isocalendar().week]).agg({
        'TAVG': ['mean', 'std', 'max', 'min'],
        'TMAX': ['max'],
        'TMIN': ['min'],
        'PRCP': ['sum', 'max'],
        'WIND': ['mean'],
        'RH': ['mean'],
        'DEWP_mean': ['mean']
    }).reset_index()
    
    # Flatten column names
    weather_weekly.columns = ['county', 'year', 'week_num', 'TAVG_mean', 'TAVG_std', 
                              'TAVG_max', 'TAVG_min', 'TMAX_max', 'TMIN_min', 
                              'PRCP_sum', 'PRCP_max', 'WIND_mean', 'RH_mean', 'DEWP_mean']
    
    # Create county_key to match other datasets
    weather_weekly['county_key'] = weather_weekly['county'].str.replace(' County', '').str.replace(' ', '_') + '_' + weather_weekly['year'].astype(str).str[:2]
    
    # Merge with human cases data
    merged = weather_weekly.merge(human_cases, on=['county_key', 'year'], how='left')
    
    # Fill missing human case data with 0
    human_cols = ['total_cases', 'neuroinvasive', 'non_neuroinvasive', 'deaths']
    for col in human_cols:
        merged[col] = merged[col].fillna(0)
    
    # Create target variable (4-week ahead prediction)
    merged['target'] = merged.groupby('county_key')['total_cases'].shift(-FORECAST_LEAD)
    
    # Add temporal features
    merged['sin_week'] = np.sin(2 * np.pi * merged['week_num'] / 52)
    merged['cos_week'] = np.cos(2 * np.pi * merged['week_num'] / 52)
    merged['in_wnv_season'] = ((merged['week_num'] >= 20) & (merged['week_num'] <= 45)).astype(int)
    
    # Add lagged features
    for lag in [1, 2, 4]:
        merged[f'TAVG_mean_lag{lag}'] = merged.groupby('county_key')['TAVG_mean'].shift(lag)
        merged[f'PRCP_sum_lag{lag}'] = merged.groupby('county_key')['PRCP_sum'].shift(lag)
        merged[f'total_cases_lag{lag}'] = merged.groupby('county_key')['total_cases'].shift(lag)
    
    # Handle NaN values in lagged features
    numeric_cols = merged.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        merged[col] = merged[col].fillna(merged[col].median())
        # Replace any remaining inf values
        merged[col] = merged[col].replace([np.inf, -np.inf], merged[col].median())
    
    # Drop rows with NaN target
    merged = merged.dropna(subset=['target'])
    
    print(f"Historical dataset shape: {merged.shape}")
    print(f"Counties: {merged['county_key'].unique()}")
    
    return merged

def create_2025_features():
    """Create feature dataset for 2025 predictions"""
    
    print("Creating 2025 feature dataset...")
    
    # Load historical data for training
    historical_df = load_and_preprocess_historical_data()
    
    # Load 2024 weather data for creating 2025 features
    weather_data = pd.read_csv(f"{BASE}wnv_weather_data/ALL_COUNTIES_weather_2000_2024.csv")
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    weather_data['year'] = weather_data['date'].dt.year
    
    # Get 2024 data
    weather_2024 = weather_data[weather_data['year'] == 2024].copy()
    
    # Create weekly features from 2024 weather data
    weather_weekly_2024 = weather_2024.groupby(['county', weather_2024['date'].dt.isocalendar().week]).agg({
        'TAVG': ['mean', 'std', 'max', 'min'],
        'TMAX': ['max'],
        'TMIN': ['min'],
        'PRCP': ['sum', 'max'],
        'WIND': ['mean'],
        'RH': ['mean'],
        'DEWP_mean': ['mean']
    }).reset_index()
    
    # Flatten column names
    weather_weekly_2024.columns = ['county', 'week_num', 'TAVG_mean', 'TAVG_std', 
                                   'TAVG_max', 'TAVG_min', 'TMAX_max', 'TMIN_min', 
                                   'PRCP_sum', 'PRCP_max', 'WIND_mean', 'RH_mean', 'DEWP_mean']
    
    # Create county_key
    weather_weekly_2024['county_key'] = weather_weekly_2024['county'].str.replace(' County', '').str.replace(' ', '_') + '_CO'
    
    # Load 2024 human cases
    human_cases = pd.read_csv(f"{BASE}wnv_human_cases_county_year.csv")
    human_2024 = human_cases[human_cases['year'] == 2024].copy()
    
    # Create 2025 template
    counties = ["Boulder_CO", "Cook_IL", "Dallas_TX", "Larimer_CO", "LosAngeles_CA", "Maricopa_AZ"]
    weeks_2025 = list(range(1, 53))  # All 52 weeks of 2025
    
    data_2025 = []
    for county in counties:
        # Get 2024 weather data for this county (use similar county patterns)
        county_base = county.split('_')[0]
        county_weather_2024 = weather_weekly_2024[weather_weekly_2024['county'].str.contains(county_base)]
        
        if county_weather_2024.empty:
            # Use default weather patterns if no 2024 data available
            weather_means = {
                'TAVG_mean': 20, 'TAVG_std': 5, 'TAVG_max': 30, 'TAVG_min': 10,
                'TMAX_max': 35, 'TMIN_min': 5, 'PRCP_sum': 50, 'PRCP_max': 100,
                'WIND_mean': 10, 'RH_mean': 60, 'DEWP_mean': 10
            }
        else:
            weather_means = county_weather_2024[['TAVG_mean', 'TAVG_std', 'TAVG_max', 'TAVG_min', 
                                               'TMAX_max', 'TMIN_min', 'PRCP_sum', 'PRCP_max',
                                               'WIND_mean', 'RH_mean', 'DEWP_mean']].mean().to_dict()
        
        # Get 2024 human cases for this county
        county_human_2024 = human_2024[human_2024['county_key'] == county]
        if county_human_2024.empty:
            human_2024_data = {'total_cases': 0, 'neuroinvasive': 0, 'non_neuroinvasive': 0, 'deaths': 0}
        else:
            human_2024_data = county_human_2024.iloc[0].to_dict()
        
        for week in weeks_2025:
            # Create feature row for 2025
            row = {
                'county_key': county,
                'year': 2025,
                'week_num': week,
                'TAVG_mean': weather_means['TAVG_mean'],
                'TAVG_std': weather_means['TAVG_std'],
                'TAVG_max': weather_means['TAVG_max'],
                'TAVG_min': weather_means['TAVG_min'],
                'TMAX_max': weather_means['TMAX_max'],
                'TMIN_min': weather_means['TMIN_min'],
                'PRCP_sum': weather_means['PRCP_sum'],
                'PRCP_max': weather_means['PRCP_max'],
                'WIND_mean': weather_means['WIND_mean'],
                'RH_mean': weather_means['RH_mean'],
                'DEWP_mean': weather_means['DEWP_mean'],
                'total_cases': human_2024_data['total_cases'],
                'neuroinvasive': human_2024_data['neuroinvasive'],
                'non_neuroinvasive': human_2024_data['non_neuroinvasive'],
                'deaths': human_2024_data['deaths'],
                'sin_week': np.sin(2 * np.pi * week / 52),
                'cos_week': np.cos(2 * np.pi * week / 52),
                'in_wnv_season': 1 if 20 <= week <= 45 else 0,
            }
            
            # Add lagged features (use same values as approximation)
            for lag in [1, 2, 4]:
                row[f'TAVG_mean_lag{lag}'] = weather_means['TAVG_mean']
                row[f'PRCP_sum_lag{lag}'] = weather_means['PRCP_sum']
                row[f'total_cases_lag{lag}'] = human_2024_data['total_cases']
            
            data_2025.append(row)
    
    df_2025 = pd.DataFrame(data_2025)
    print(f"2025 dataset shape: {df_2025.shape}")
    
    return df_2025, historical_df

def build_lstm_model(n_features, seq_len):
    """Build LSTM model with same architecture as training"""
    
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features), return_sequences=True, name="lstm_1"),
        BatchNormalization(name="bn_1"),
        Dropout(0.2, name="drop_1"),
        
        LSTM(32, return_sequences=False, name="lstm_2"),
        BatchNormalization(name="bn_2"),
        Dropout(0.2, name="drop_2"),
        
        Dense(16, activation="relu", name="dense_hidden"),
        Dense(1, activation="linear", name="output"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    
    return model

def create_sequences_for_prediction(df, seq_len):
    """Create sequences for prediction from 2025 data"""
    
    sequences = []
    sequence_info = []
    
    # Group by county
    for county in df['county_key'].unique():
        county_data = df[df['county_key'] == county].copy()
        county_data = county_data.sort_values('week_num')
        
        # Create sequences for this county
        if len(county_data) >= seq_len:
            for i in range(seq_len, len(county_data)):
                seq_data = county_data.iloc[i - seq_len : i]
                target_week = county_data.iloc[i]
                
                sequences.append(seq_data[feature_cols].values)
                sequence_info.append({
                    'county_key': county,
                    'week_num': target_week['week_num'],
                    'year': target_week['year']
                })
    
    return np.array(sequences, dtype=np.float32), sequence_info

def train_lstm_and_predict_2025():
    """Main function to train LSTM and predict 2025 cases"""
    
    print("=" * 80)
    print("WNV 2025 PREDICTION - LSTM VERSION")
    print("=" * 80)
    
    # Create 2025 features and get historical data
    df_2025, historical_df = create_2025_features()
    
    # Prepare historical training data
    print("Preparing historical training data...")
    
    # Select feature columns (same as training)
    global feature_cols
    feature_cols = [col for col in historical_df.columns if col not in [
        'county', 'county_key', 'year', 'week_num', 'target', 'data_confidence', 
        'notes', 'source_primary', 'source_secondary', 'county_name', 'state', 'fips'
    ]]
    
    X_hist = historical_df[feature_cols]
    y_hist = historical_df['target']
    
    # Encode county if present
    le = LabelEncoder()
    if 'county_key' in X_hist.columns:
        X_hist['county_encoded'] = le.fit_transform(historical_df['county_key'])
        if 'county_key' in X_hist.columns:
            X_hist = X_hist.drop('county_key', axis=1)
    
    # Scale features and target
    feat_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_hist_scaled = feat_scaler.fit_transform(X_hist)
    y_hist_scaled = target_scaler.fit_transform(y_hist.values.reshape(-1, 1)).ravel()
    
    # Build sequences for training
    def build_sequences(X_scaled, y_scaled, seq_len):
        Xs, ys = [], []
        for i in range(seq_len, len(X_scaled)):
            seq_X = X_scaled[i - seq_len : i]
            seq_y = y_scaled[i]
            if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
                Xs.append(seq_X)
                ys.append(seq_y)
        return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
    
    X_train_seq, y_train_seq = build_sequences(X_hist_scaled, y_hist_scaled, SEQ_LEN)
    
    print(f"Training sequences shape: {X_train_seq.shape}")
    
    # Train LSTM model
    print("Training LSTM model...")
    model = build_lstm_model(X_train_seq.shape[2], SEQ_LEN)
    
    # Train for a few epochs (simplified for prediction)
    model.fit(X_train_seq, y_train_seq, epochs=10, batch_size=32, verbose=0)
    
    # Prepare 2025 data for prediction
    print("Preparing 2025 data for prediction...")
    
    # Encode county in 2025 data (need to handle unseen counties)
    if 'county_key' in df_2025.columns:
        # Check if label encoder was fitted
        if hasattr(le, 'classes_'):
            unique_counties_train = set(le.classes_)
            df_2025['county_encoded'] = df_2025['county_key'].apply(
                lambda x: le.transform([x])[0] if x in unique_counties_train else -1
            )
        else:
            # If not fitted, create a simple mapping
            unique_counties = df_2025['county_key'].unique()
            county_mapping = {county: i for i, county in enumerate(unique_counties)}
            df_2025['county_encoded'] = df_2025['county_key'].map(county_mapping)
    
    # Scale 2025 features
    X_2025 = df_2025[feature_cols].copy()
    if 'county_key' in X_2025.columns:
        X_2025 = X_2025.drop('county_key', axis=1)
    
    X_2025_scaled = feat_scaler.transform(X_2025)
    
    # Create sequences for 2025 prediction
    X_2025_seq, seq_info = create_sequences_for_prediction(df_2025, SEQ_LEN)
    
    print(f"2025 prediction sequences shape: {X_2025_seq.shape}")
    
    # Make predictions
    print("Making 2025 predictions...")
    y_pred_scaled = model.predict(X_2025_seq, verbose=0).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    # Ensure no negative predictions
    y_pred = np.maximum(0, y_pred)
    
    # Create results dataframe
    results = []
    for i, info in enumerate(seq_info):
        results.append({
            'county_key': info['county_key'],
            'year': info['year'],
            'week_num': info['week_num'],
            'predicted_cases': y_pred[i]
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary by county
    print("\n" + "=" * 80)
    print("2025 LSTM PREDICTION SUMMARY")
    print("=" * 80)
    
    county_summary = results_df.groupby('county_key').agg({
        'predicted_cases': ['sum', 'mean', 'max']
    }).round(2)
    
    print("\nPredicted 2025 WNV Cases by County (LSTM):")
    print("-" * 50)
    for county in results_df['county_key'].unique():
        county_data = results_df[results_df['county_key'] == county]
        if len(county_data) > 0 and not county_data['predicted_cases'].isna().all():
            total_cases = county_data['predicted_cases'].sum()
            max_weekly = county_data['predicted_cases'].max()
            peak_week_idx = county_data['predicted_cases'].idxmax()
            peak_week = county_data.loc[peak_week_idx, 'week_num']
            
            print(f"{county:<18}: Total={total_cases:>6.1f}, Peak Week={peak_week:>2} ({max_weekly:.2f} cases)")
        else:
            print(f"{county:<18}: Total={0:>6.1f}, Peak Week={0:>2} ({0:.2f} cases)")
    
    # Save predictions
    output_file = "files_rf/wnv_2025_lstm_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nLSTM predictions saved to: {output_file}")
    
    # Save weekly predictions for each county
    weekly_summary = results_df.pivot(index='week_num', columns='county_key', values='predicted_cases').round(2)
    weekly_file = "files_rf/wnv_2025_lstm_weekly_predictions.csv"
    weekly_summary.to_csv(weekly_file)
    print(f"Weekly LSTM predictions saved to: {weekly_file}")
    
    return results_df

if __name__ == "__main__":
    predictions_2025 = train_lstm_and_predict_2025()
    print("\n" + "=" * 80)
    print("2025 LSTM PREDICTION COMPLETE")
    print("=" * 80)
