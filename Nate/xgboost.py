# XGBoost Regression Demo for Stock Forecasting
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# 1. Download Stock Data
ticker = yf.Ticker("NVDA")  # NVIDIA
data = ticker.history(period="6mo") 

# 2. Feature Engineering Functions
def feature_engineering_train(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df.index)
    df.set_index('Date', inplace=True)
    df = df.asfreq('B')  # business day frequency

    # Lag features (smaller lags because dataset is shorter)
    for lag in [1, 2, 7, 14, 21]:
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag)

    # Rolling statistical features (smaller windows)
    df['roll_mean_5'] = df['Close'].rolling(5).mean()
    df['roll_std_5'] = df['Close'].rolling(5).std()

    # Calendar features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    df.dropna(inplace=True)
    return df

def feature_engineering_future(last_df, future_days=5):
    future_dates = pd.bdate_range(start=last_df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    repeated_vals = np.tile(last_df.iloc[-1].values, (future_days, 1))
    future_exog = pd.DataFrame(repeated_vals, index=future_dates, columns=last_df.columns)
    
    future_exog['day_of_week'] = future_exog.index.dayofweek
    future_exog['month'] = future_exog.index.month
    return future_exog

# 3. Prepare Train/Test Data
clean_data = feature_engineering_train(data)

test_days = 5
train_data = clean_data.iloc[:-test_days]
test_data = clean_data.iloc[-test_days:]

y_train = train_data['Close']
X_train = train_data.drop(columns=['Close'])
y_test = test_data['Close']
X_test = test_data.drop(columns=['Close'])

# Standardize features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# 4. XGBoost Model Training (balanced params for small dataset)
model = XGBRegressor(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 5. Evaluate on Test Set
forecast_test = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, forecast_test))
mae = mean_absolute_error(y_test, forecast_test)
mape = np.mean(np.abs((y_test - forecast_test) / y_test)) * 100

print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test MAPE: {mape:.2f}%")

# 6. Forecast Future Days
full_X = clean_data.drop(columns=['Close'])
full_X_scaled = pd.DataFrame(scaler.fit_transform(full_X), index=full_X.index, columns=full_X.columns)

future_days = 5
X_future_scaled = feature_engineering_future(full_X_scaled, future_days=future_days)
forecast_future = model.predict(X_future_scaled)

print(f"\nNext {future_days} Business Days Forecast:")
print(pd.Series(forecast_future, index=X_future_scaled.index))

# 7. Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label='Training Close', color='blue')
plt.plot(y_test.index, y_test, label='Actual Test Close', color='green', marker='o')
plt.plot(y_test.index, forecast_test, label='Test Forecast', color='orange', marker='x')
plt.plot(X_future_scaled.index, forecast_future, label='Future Forecast', color='red', marker='^')
plt.title('XGBoost Forecast vs Actuals for NVDA (6mo)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

