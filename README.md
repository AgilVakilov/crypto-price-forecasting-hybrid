# Crypto Price Forecasting: Hybrid ARIMA-LSTM Model

A machine learning project that combines traditional statistical forecasting (**ARIMA**) with Deep Learning (**LSTM**) to predict the next 30 days of cryptocurrency prices (BTC, ETH, XRP).

## 📊 Overview
This project addresses the high volatility of crypto markets by using a "Hybrid" approach:
- **ARIMA (AutoRegressive Integrated Moving Average):** Captures linear trends and momentum in historical data.
- **LSTM (Long Short-Term Memory):** A Recurrent Neural Network (RNN) that identifies complex, non-linear patterns over a 60-day "look-back" window.

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Data Source:** Yahoo Finance API (`yfinance`)
- **Analysis:** Pandas, NumPy, Statsmodels
- **Visualization:** Matplotlib

## 🚀 How It Works
1. **Data Acquisition:** Downloads real-time historical data.
2. **Preprocessing:** Normalizes data using `MinMaxScaler` for the Neural Network.
3. **Hybrid Modeling:** Executes the ARIMA forecast and trains the LSTM model simultaneously.
4. **Visualization:** Generates a 30-day forecast plot comparing both models.

## 📈 Example Results
*(Tip: Once you upload your forecast image, replace this line with: ![Forecast Plot](forecast_plot.png))*
