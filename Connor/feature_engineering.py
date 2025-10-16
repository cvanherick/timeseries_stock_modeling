import yfinance as yf
import pandas as pd

# 1. Download stock data
ticker = yf.Ticker("AAPL")
data = ticker.history(period="6mo")

# Display raw data
print("Raw Data Sample:")
print(data.head())

# 2. Define a feature engineering pipeline
def feature_engineering_pipeline(df):
    df = df.copy()
    
    # Ensure Date is a datetime index
    df = df.sort_index()
    df = df.asfreq('B')      # business-day frequency
    df = df.ffill()          # forward fill missing values
    
    # Lag features
    for lag in [1, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    # Rolling window statistics
    df['rolling_mean_5'] = df['Close'].rolling(5).mean()
    df['rolling_std_5'] = df['Close'].rolling(5).std()
    df['rolling_mean_20'] = df['Close'].rolling(20).mean()
    
    # Returns and volatility
    df['returns'] = df['Close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()
    
    # Temporal features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Drop rows with NaNs caused by shifts/rolling
    df = df.dropna()
    
    return df

# 3. Apply feature engineering
clean_data = feature_engineering_pipeline(data)

# 4. Show results
print("\nFeature-Engineered Data Sample:")
print(clean_data.head())
