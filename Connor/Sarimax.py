import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------
# 1. Download stock data
# -------------------------------
ticker = yf.Ticker("AAPL")
raw_data = ticker.history(period="6mo")
raw_data.reset_index(inplace=True)

# -------------------------------
# 2. Feature Engineering Pipeline
# -------------------------------
def feature_engineering_pipeline(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df.asfreq('B').ffill()
    
    for lag in [1, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
        
    df['rolling_mean_5'] = df['Close'].rolling(5).mean()
    df['rolling_std_5'] = df['Close'].rolling(5).std()
    df['rolling_mean_20'] = df['Close'].rolling(20).mean()
    
    df['returns'] = df['Close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df = df.dropna()
    return df

clean_data = feature_engineering_pipeline(raw_data)

# -------------------------------
# 3. Train-Test Split (last week as test)
# -------------------------------
test_days = 5  # Last week
train_data = clean_data.iloc[:-test_days]
test_data = clean_data.iloc[-test_days:]

y_train = train_data['Close']
X_train = train_data.drop(columns=['Close'])

y_test = test_data['Close']
X_test = test_data.drop(columns=['Close'])

# Scale exogenous features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# -------------------------------
# 4. Fit SARIMAX Model on Training Data
# -------------------------------
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 5)

model = SARIMAX(endog=y_train, exog=X_train_scaled, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)
print(results.summary())

# -------------------------------
# 5. Forecast over Test Set
# -------------------------------
forecast_test = results.get_forecast(steps=test_days, exog=X_test_scaled)
forecast_test_mean = forecast_test.predicted_mean
forecast_test_ci = forecast_test.conf_int()

# Evaluate Test Forecast
rmse = np.sqrt(mean_squared_error(y_test, forecast_test_mean))
mae = mean_absolute_error(y_test, forecast_test_mean)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")

# -------------------------------
# 6. Forecast Next Day Closing Price
# -------------------------------
# Use all data (including test set) for next-day forecast
full_y = clean_data['Close']
full_X = clean_data.drop(columns=['Close'])
full_X_scaled = pd.DataFrame(scaler.fit_transform(full_X), index=full_X.index, columns=full_X.columns)

model_full = SARIMAX(endog=full_y, exog=full_X_scaled, order=order, seasonal_order=seasonal_order)
results_full = model_full.fit(disp=False)

next_day_index = pd.bdate_range(start=clean_data.index[-1]+pd.Timedelta(days=1), periods=1)
X_next_day = pd.DataFrame(np.tile(full_X_scaled.values[-1], (1,1)), index=next_day_index, columns=full_X_scaled.columns)

forecast_next_day = results_full.get_forecast(steps=1, exog=X_next_day)
forecast_next_day_mean = forecast_next_day.predicted_mean
forecast_next_day_ci = forecast_next_day.conf_int()

print(f"Next Day Forecast: {forecast_next_day_mean.values[0]:.2f}")
print(f"Confidence Interval: [{forecast_next_day_ci.iloc[0,0]:.2f}, {forecast_next_day_ci.iloc[0,1]:.2f}]")

# -------------------------------
# 7. Plot Training, Test, and Next-Day Forecast
# -------------------------------
plt.figure(figsize=(12,6))
plt.plot(y_train.index, y_train, label='Training Data')
plt.plot(y_test.index, y_test, label='Actual Test Data', color='green')
plt.plot(forecast_test_mean.index, forecast_test_mean, label='Test Forecast', color='orange')
plt.fill_between(forecast_test_ci.index, forecast_test_ci.iloc[:,0], forecast_test_ci.iloc[:,1], color='orange', alpha=0.2)

plt.scatter(forecast_next_day_mean.index, forecast_next_day_mean, color='red', label='Next-Day Forecast', zorder=5)
plt.fill_between(forecast_next_day_ci.index, forecast_next_day_ci.iloc[:,0], forecast_next_day_ci.iloc[:,1], color='red', alpha=0.2)

plt.title('SARIMAX Forecast: Last Week Test Set + Next Day')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
