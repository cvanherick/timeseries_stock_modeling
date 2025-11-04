import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Download Stock Data

ticker = yf.Ticker("GOOG") #TO DO
#TO DO EXPERIMENT WITH DIFFERNT TIME PERIOD AFTER TRAINING YOUR MODEL
raw_data = ticker.history(period="6mo") #TO DO

# 2. Feature Engineering Pipelines
def feature_engineering_train(df):
    """Feature engineering for training data."""
    df = df.copy()
    df["Date"] = df.index
    df["Date"] = pd.to_datetime(df["Date"]) #TO DO SET THE DATE COLUMN TO A DATETIME OBJECT
    df = df.set_index("Date").sort_index()  #TO DO SET THE DATE AS THE INDEX
    df = df.asfreq('B') #TO DO SET TO BUSINESS DAY FREQUENCY
    df = df.ffill()
    
    #CHOOSE THIS BASED ON TRENDS YOU SEE IN YOUR INDIVIDUAL STOCK
    #this is the lag periods
    for lag in [1, 5, 10, 20]: #TO DO
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag) #TO DO

    #TO DO below this it what i did for mine you can try differnt things for your data depending on what trends you see in your data
    #df['rolling_mean_5'] = df['Open'].rolling(5).mean() #5 days = 1 week
    #df['rolling_std_5'] = df['Open'].rolling(5).std() #5 days = 1 week
    #df['rolling_mean_20'] = df['Open'].rolling(20).mean() #5 days = ~1 month
    df['rolling_mean_5'] = df['Open'].rolling(5).mean()
    df['rolling_std_5'] = df['Open'].rolling(5).std()
    
    #df['returns'] = df['Open'].pct_change()
    #df['volatility_10'] = df['returns'].rolling(10).std()
    df['returns'] = df['Open'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()
    
    df['day_of_week'] = df.index.dayofweek #TO DO
    df['month'] = df.index.month #TO DO
    
    df = df.dropna() #TO DO drop nan values
    
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




# 3. Prepare Train/Test Data

clean_data = feature_engineering_train(raw_data)
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


# 4. Fit SARIMAX Model (Training)
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 5)

model = SARIMAX(
    endog=y_train,
    exog=X_train_scaled,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit()

model = SARIMAX(endog=y_train, exog=X_train_scaled, order=order, seasonal_order=seasonal_order)
results = model.fit(disp=False)
print(results.summary())


# 5. Evaluate on Test Set

forecast_test = results.get_forecast(steps=test_days, exog=X_test_scaled)
forecast_test_mean = forecast_test.predicted_mean
forecast_test_ci = forecast_test.conf_int()

rmse = np.sqrt(mean_squared_error(y_test, forecast_test_mean))
mae = mean_absolute_error(y_test, forecast_test_mean)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAE: {mae:.2f}")


# 6. Forecast Future Unseen Days

# Refit model on full dataset
full_y = clean_data['Close']
full_X = clean_data.drop(columns=['Close'])
full_X_scaled = pd.DataFrame(scaler.fit_transform(full_X), index=full_X.index, columns=full_X.columns)

model_full = SARIMAX(endog=full_y, exog=full_X_scaled, order=order, seasonal_order=seasonal_order)
results_full = model_full.fit(disp=False)

# Generate future features
future_days = 5
X_future_scaled = feature_engineering_future(full_X_scaled, future_days=future_days) # Deleted scaler argument


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