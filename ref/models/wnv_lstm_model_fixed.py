#!/usr/bin/env python3
"""
West Nile Virus (WNV) County-Level Human Case Prediction — LSTM Model (Fixed)
======================================================================
Fixed version that handles NaN values and improves training stability.
"""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BASE          = "ref/"
FORECAST_LEAD = 4
SEQ_LEN       = 8          # Look-back window (weeks)
EPOCHS        = 50         # Reduced for faster debugging
BATCH_SIZE    = 32
LEARNING_RATE = 0.001      # Lower learning rate for stability
PATIENCE      = 10
RANDOM_STATE  = 42

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def load_and_preprocess_data():
    """Load and preprocess all data with NaN handling"""
    
    print("Loading and preprocessing data...")
    
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
    
    print(f"Dataset shape after preprocessing: {merged.shape}")
    print(f"Counties: {merged['county_key'].unique()}")
    print(f"Year range: {merged['year'].min()} - {merged['year'].max()}")
    
    return merged

def create_features_and_target(df):
    """Create feature matrix and target vector"""
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in [
        'county', 'county_key', 'year', 'week_num', 'target', 'data_confidence', 
        'notes', 'source_primary', 'source_secondary', 'county_name', 'state', 'fips'
    ]]
    
    X = df[feature_cols]
    y = df['target']
    
    # Encode county if present
    if 'county_key' in feature_cols:
        le = LabelEncoder()
        X['county_encoded'] = le.fit_transform(df['county_key'])
        if 'county_key' in X.columns:
            X = X.drop('county_key', axis=1)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def build_sequences(X_scaled, y_scaled, seq_len):
    """Build LSTM sequences with NaN checking"""
    
    Xs, ys = [], []
    for i in range(seq_len, len(X_scaled)):
        # Check for NaN in the sequence
        seq_X = X_scaled[i - seq_len : i]
        seq_y = y_scaled[i]
        
        if not (np.isnan(seq_X).any() or np.isnan(seq_y)):
            Xs.append(seq_X)
            ys.append(seq_y)
    
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_lstm_model(n_features, seq_len):
    """Build LSTM model with better architecture"""
    
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
        optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),  # Gradient clipping
        loss="mse",
        metrics=["mae"],
    )
    
    return model

def main():
    """Main training and evaluation function"""
    
    print("=" * 80)
    print("WNV LSTM MODEL - FIXED VERSION")
    print("=" * 80)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Create features and target
    X, y = create_features_and_target(df)
    
    # Temporal split (80% train, 20% test)
    split = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features and target
    feat_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    X_train_scaled = feat_scaler.fit_transform(X_train)
    X_test_scaled = feat_scaler.transform(X_test)
    
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    
    # Build sequences
    X_train_seq, y_train_seq = build_sequences(X_train_scaled, y_train_scaled, SEQ_LEN)
    X_test_seq, y_test_seq = build_sequences(X_test_scaled, y_test_scaled, SEQ_LEN)
    
    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Test sequences: {X_test_seq.shape}")
    
    # Build and train model
    model = build_lstm_model(X_train_seq.shape[2], SEQ_LEN)
    
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]
    
    # Train model
    print(f"\nTraining LSTM model...")
    history = model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_scaled = model.predict(X_test_seq, verbose=0).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_actual = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).ravel()
    
    # Ensure no negative predictions
    y_pred = np.maximum(0, y_pred)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Show some predictions
    print(f"\nFirst 10 Predictions vs Actual:")
    for i in range(min(10, len(y_pred))):
        print(f"Predicted: {y_pred[i]:.2f}, Actual: {y_actual[i]:.2f}, Error: {y_actual[i] - y_pred[i]:.2f}")
    
    print("\n" + "=" * 80)
    print("LSTM TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
