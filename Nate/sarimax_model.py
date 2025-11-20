import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# 1. Download Stock Data
ticker = yf.Ticker("PLTR")
raw_data = ticker.history(period="6mo", interval="1d")

# 2. Feature Engineering Pipelines
def feature_engineering_train(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("B")  # business-day frequency
    df = df.ffill()
 
    # Lag features
    for lag in [1, 2, 7, 14, 21]:
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag)

    # Numeric encodings for day_of_week and month (instead of strings)
    df["day_of_week"] = df.index.weekday  # 0 = Monday
    df["month"] = df.index.month          # 1 = January

    df = df.dropna()
    return df

def feature_engineering_future(last_df, future_days=5):
    """Generate exogenous features for future business days."""
    future_dates = pd.bdate_range(start=last_df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    
    # Repeat last row values
    repeated_vals = np.tile(last_df.iloc[-1].values, (future_days, 1))
    future_exog = pd.DataFrame(repeated_vals, index=future_dates, columns=last_df.columns)
    
    # Update calendar numeric features
    future_exog["day_of_week"] = future_exog.index.weekday
    future_exog["month"] = future_exog.index.month
    return future_exog

# 3. Prepare Train/Test Data
clean_data = feature_engineering_train(raw_data)
test_days = 5
train_data = clean_data.iloc[:-test_days]
test_data = clean_data.iloc[-test_days:]

y_train = train_data["Close"]
X_train = train_data.drop(columns=["Close"])
y_test = test_data["Close"]
X_test = test_data.drop(columns=["Close"])

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# 4. Fit SARIMAX Model (Training)
order = (1, 1, 1)
seasonal_order = (0, 0, 0, 5)

model = SARIMAX(endog=y_train, exog=X_train_scaled, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)
print(results.summary())

# 5. Evaluate on Test Set
forecast_test = results.get_forecast(steps=test_days, exog=X_test_scaled)
forecast_test_mean = forecast_test.predicted_mean
forecast_test_ci = forecast_test.conf_int()


rmse = np.sqrt(mean_squared_error(y_test, forecast_test_mean))
mae = mean_absolute_error(y_test, forecast_test_mean)
mape = np.mean(np.abs((y_test - forecast_test_mean) / y_test)) * 100
r2 = r2_score(y_test, forecast_test_mean)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test MAPE: {mape:.2f}%")
print(f"Test R²: {r2:.4f}")


# 6. Forecast Future Unseen Days
full_y = clean_data["Close"]
full_X = clean_data.drop(columns=["Close"])
full_X_scaled = pd.DataFrame(scaler.fit_transform(full_X), index=full_X.index, columns=full_X.columns)

model_full = SARIMAX(endog=full_y, exog=full_X_scaled, order=order, seasonal_order=seasonal_order)
results_full = model_full.fit(disp=False)

future_days = 5
X_future = feature_engineering_future(full_X, future_days=future_days)
X_future_scaled = pd.DataFrame(scaler.transform(X_future), index=X_future.index, columns=X_future.columns)

forecast_future = results_full.get_forecast(steps=future_days, exog=X_future_scaled)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_future.conf_int()


# 7. Plot Historical, Test, and Forecast Results
plt.figure(figsize=(12, 6))

# --- Actual Prices ---
plt.plot(full_y.index, full_y, label="Actual Close", color="blue", linewidth=2)

# --- Test Forecast (last 5 days of known data) ---
plt.plot(forecast_test_mean.index, forecast_test_mean, label="Test Forecast", color="green", linestyle="--", linewidth=2)
plt.fill_between(forecast_test_ci.index,
                 forecast_test_ci["lower Close"],
                 forecast_test_ci["upper Close"],
                 color="green", alpha=0.1)

# --- Future Forecast (unseen data) ---
plt.plot(forecast_future_mean.index, forecast_future_mean, label="Future Forecast", color="orange", linewidth=2)
plt.fill_between(forecast_future_ci.index,
                 forecast_future_ci["lower Close"],
                 forecast_future_ci["upper Close"],
                 color="orange", alpha=0.2)

# --- Visual Formatting ---
plt.title("PLTR Closing Price: Actual vs Test Forecast vs Future Forecast (SARIMAX)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nNext 5 Business Days Forecast:")
for date, price in forecast_future_mean.items():
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")