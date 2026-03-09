import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# 1. SETTINGS & REPRODUCIBILITY
# ---------------------------------------------------------
def set_seeds(seed=42):
    """Fix random seeds for reproducible results."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

COINS = ['BTC-USD', 'ETH-USD', 'XRP-USD']
FUTURE_DAYS = 30  # Forecast horizon
TIME_STEP = 60    # Look-back window (Memory)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def get_data(ticker):
    """Downloads historical data from Yahoo Finance."""
    start_date = '2020-01-01'
    print(f"\nFetching data: {ticker} ({start_date} -> TODAY)...")
    
    # Download data
    df = yf.download(ticker, start=start_date, progress=False)
    
    # Handle MultiIndex columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    else:
        df = df[['Close']]
    
    # Rename column for consistency
    if ticker in df.columns:
        df.rename(columns={ticker: 'Close'}, inplace=True)
        
    df.dropna(inplace=True)
    return df

def build_lstm_model(time_step):
    """Builds the LSTM Deep Learning model architecture."""
    model = Sequential()
    # LSTM Layers
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    # Dense Layers
    model.add(Dense(25))
    model.add(Dense(1)) # Output layer (Price prediction)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------------------------------------------------
# HYBRID ANALYSIS (ARIMA + LSTM)
# ---------------------------------------------------------
def analyze_hybrid(ticker):
    # 1. DATA PREPARATION
    df = get_data(ticker)
    
    # Check for sufficient data
    if len(df) < TIME_STEP + 2:
        print(f"WARNING: Insufficient data for {ticker}!")
        return None

    data = df.values 
    dates = df.index # Capture dates for plotting
    
    # Get Current Price (Last available data point)
    current_price = data[-1][0] if isinstance(data[-1], (list, np.ndarray)) else data[-1]

    # --- ARIMA MODEL (Linear Trend) ---
    print(f" > {ticker}: Calculating ARIMA (Trend)...")
    try:
        arima_model = ARIMA(data, order=(5,1,0)) 
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=FUTURE_DAYS)
        arima_end_price = arima_forecast[-1]
    except:
        print("   -> ARIMA error, using fallback (flat line).")
        arima_forecast = np.full(FUTURE_DAYS, current_price)
        arima_end_price = current_price

    # --- LSTM MODEL (Non-Linear / AI) ---
    print(f" > {ticker}: Training LSTM (Deep Learning)...")
    
    # Scaling (Normalization)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Create Training Set
    X_train, y_train = [], []
    for i in range(TIME_STEP, len(scaled_data)):
        X_train.append(scaled_data[i-TIME_STEP:i, 0])
        y_train.append(scaled_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Train Model
    lstm_model = build_lstm_model(TIME_STEP)
    lstm_model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=0)

    # Recursive Forecasting
    last_chunk = scaled_data[-TIME_STEP:]
    curr_input = last_chunk.reshape(1, -1)
    temp_input = list(curr_input[0])
    lstm_outputs = []
    
    for i in range(FUTURE_DAYS):
        input_as_array = np.array(temp_input[-TIME_STEP:]).reshape(1, TIME_STEP, 1)
        pred = lstm_model.predict(input_as_array, verbose=0)
        temp_input.append(pred[0][0])
        lstm_outputs.append(pred[0][0])
    
    lstm_forecast = scaler.inverse_transform(np.array(lstm_outputs).reshape(-1, 1))
    lstm_end_price = lstm_forecast[-1][0]

    # --- CALCULATE PERCENTAGE CHANGE ---
    lstm_change_pct = ((lstm_end_price - current_price) / current_price) * 100
    arima_change_pct = ((arima_end_price - current_price) / current_price) * 100

    # ---------------------------------------------------------
    # PLOTTING (With Correct Dates)
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Historical Data (Last 365 Days)
    plot_start = len(data) - 365 if len(data) > 365 else 0
    recent_dates = dates[plot_start:] 
    recent_data = data[plot_start:]   
    
    plt.plot(recent_dates, recent_data, label='Historical Data', color='black')
    
    # 2. Generate Future Dates
    last_date = dates[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS)
    
    # 3. Plot Forecasts
    plt.plot(future_dates, arima_forecast, label='ARIMA Forecast (Trend)', color='red', linestyle='--')
    plt.plot(future_dates, lstm_forecast.flatten(), label='LSTM Forecast (AI)', color='green', linewidth=2)
    
    plt.title(f'{ticker} - Price Forecast (Next 30 Days)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format dates nicely
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return {
        "Coin": ticker,
        "Current Price": current_price,
        "ARIMA Forecast": arima_end_price,
        "ARIMA %": arima_change_pct,
        "LSTM Forecast": lstm_end_price,
        "LSTM %": lstm_change_pct
    }

# ---------------------------------------------------------
# EXECUTION & REPORTING
# ---------------------------------------------------------
results = []
for coin in COINS:
    res = analyze_hybrid(coin)
    if res:
        results.append(res)

print("\n" + "="*95)
print(f"{'COIN':<10} {'CURRENT ($)':<12} {'ARIMA ($)':<12} {'ARIMA %':<10} {'LSTM ($)':<12} {'LSTM %':<10}")
print("="*95)

for r in results:
    print(f"{r['Coin']:<10} "
          f"{r['Current Price']:<12.2f} "
          f"{r['ARIMA Forecast']:<12.2f} "
          f"{r['ARIMA %']:<10.2f} "
          f"{r['LSTM Forecast']:<12.2f} "
          f"{r['LSTM %']:<10.2f}")
print("="*95)