import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("ADBE")

# Download recent data (e.g. 6 months)
raw_data = ticker.history(period="6mo")
raw_data.reset_index(inplace=True)

# Show the first few rows
plt.figure(figsize=(10, 5))
plt.plot(raw_data.index, raw_data["Close"])

# ---------------------------------------
# 2. Feature Engineering Pipeline
# ---------------------------------------

def feature_engineering_training(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = df.asfreq('B').ffill()

    for lag in [1, 3, 5, 10]:
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag)

    df['rolling_mean_5'] = df['Open'].rolling(5).mean()
    df['rolling_std_5'] = df['Open'].rolling(5).std()
    df['rolling_mean_20'] = df['Open'].rolling(20).mean()

    df['returns'] = df['Open'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()
    
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df = df.dropna()
    
    return df

def feature_engineering_future(last_df, future_days=5):
    """
    Generate exogenous features for future days beyond the available dataset.
    Reuses scaled features and updates calendar-based fields. 
    This function bassicailly pulls the data from the last day to use as the matrix when forecasting future values.
    """
    future_dates = pd.bdate_range(start=last_df.index[-1] + pd.Timedelta(days=1), periods=future_days) #timedelta allows arithmetic with datetime objects

    
    repeated_vals = np.tile(last_df.iloc[-1].values, (future_days, 1))
    future_exog = pd.DataFrame(repeated_vals, index=future_dates, columns=last_df.columns)

    if 'day_of_week' in future_exog.columns:
        future_exog['day_of_week'] = future_exog.index.dayofweek
    if 'month' in future_exog.columns:
        future_exog['month'] = future_exog.index.month

    return future_exog

# ---------------------------------------
# 3. Prepare Train/Test Data
# ---------------------------------------

clean_data = feature_engineering_training(raw_data)
test_days = 5
train_data = clean_data.iloc[:-test_days]
test_data = clean_data.iloc[-test_days:]

y_train = train_data['Close'] #target variable
X_train = train_data.drop(columns=['Close']) #dropping target variable
y_test = test_data['Close']#target variable
X_test = test_data.drop(columns=['Close'])#dropping target variable

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# ---------------------------------------
# 4. Fit SARIMAX Model (Training)
# ---------------------------------------
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 5)

model = SARIMAX(endog=y_train, exog=X_train_scaled, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)
print(results.summary())

# ---------------------------------------
# 5. Evaluate on Test Set
# ---------------------------------------

forecast_test = results.get_forecast(steps=test_days, exog=X_test_scaled)
forecast_test_mean = forecast_test.predicted_mean
forecast_test_ci = forecast_test.conf_int()

rmse = np.sqrt(mean_squared_error(y_test, forecast_test_mean))
mae = mean_absolute_error(y_test, forecast_test_mean)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")

# ---------------------------------------
# 6. Forecast Future Unseen Days
# ---------------------------------------

# Refit model on full dataset
full_y = clean_data['Close']
full_X = clean_data.drop(columns=['Close'])
full_X_scaled = pd.DataFrame(scaler.fit_transform(full_X), index=full_X.index, columns=full_X.columns)

model_full = SARIMAX(endog=full_y, exog=full_X_scaled, order=order, seasonal_order=seasonal_order)
results_full = model_full.fit(disp=False)

# Generate future features
future_days = 5
X_future_scaled = feature_engineering_future(full_X_scaled, scaler, future_days=future_days)

forecast_future = results_full.get_forecast(steps=future_days, exog=X_future_scaled)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_future.conf_int()

print(f"\nNext {future_days} Business Days Forecast:")
print(forecast_future_mean)


# 7. Plot Test Forecast vs Actuals and Future Forecast
plt.figure(figsize=(12,6))

# Historical training data
plt.plot(y_train.index, y_train, label='Training Close', color='blue')

# Actual test data
plt.plot(y_test.index, y_test, label='Actual Test Close', color='green', marker='o')

# Forecast on test set
plt.plot(forecast_test_mean.index, forecast_test_mean, label='Test Forecast', color='orange', marker='x')
plt.fill_between(forecast_test_ci.index,
                 forecast_test_ci.iloc[:,0],
                 forecast_test_ci.iloc[:,1],
                 color='orange', alpha=0.2)

# Forecast for future days
plt.plot(forecast_future_mean.index, forecast_future_mean, label='Future Forecast', color='red', marker='^')
plt.fill_between(forecast_future_ci.index,
                 forecast_future_ci.iloc[:,0],
                 forecast_future_ci.iloc[:,1],
                 color='red', alpha=0.2)

plt.title(f'SARIMAX Forecast vs Actuals for {ticker.info["symbol"]}')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()