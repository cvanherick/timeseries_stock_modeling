import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("META")

# Download recent data (e.g. 6 months)
meta = ticker.history(period="6mo")

# Show the first few rows
print(meta.head())

plt.figure(figsize = (10,5))
plt.plot(meta.index, meta['Close'], label = 'META')
plt.title('Meta Stock Price (Last 6 Months)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.xticks(meta.index[::11], rotation = 45)
plt.show()

# fills in missing values with the average of the day before and after
#eta = meta.ffill()

# Only looks at data during business days
#meta = meta.asfreq('B')

# Adding lag features for tomorrow even if you don't know what the features mean
#meta['Shifted by 5'] = meta['High'].shift(5)
#print(meta)

# rolling average / rolling sd = average over the past x amount of days
#meta['Rolling Avg by 20'] = meta['High'].rolling(20).mean()

def feature_engineering(df):
    
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.asfreq('B')
    df = df.ffill()
    
    for lag in [1, 3, 5, 10]:
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag)

    df['rolling_mean_5'] = df['Open'].rolling(5).mean()
    df['rolling_std_5'] = df['Open'].rolling(5).std()
    df['rolling_mean_20'] = df['Open'].rolling(20).mean()

    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['returns'] = df['Close'].pct_change()
    df['volatility_10'] = df['returns'].rolling(10).std()

    df = df.dropna()
    return df

meta = feature_engineering(meta)
train_data = meta.iloc[:-5]
test_data = meta.iloc[-5:]

y_train = train_data['Close']
X_train = train_data.drop(columns = ['Close'])
y_test = test_data['Close']
X_test = test_data.drop(columns = ['Close'])

# Seasonal AutoRegressive Integrated Moving-Average with Exogenous factors
# SARIMAX
# Choose seasonality with the arguments
# autoregressive part is the lag features for the target
# (p, d, q) 
# p is the number of lag features
# d is the difference
# q is the number of means

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    index = X_train.index,
    columns = X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    index = X_test.index,
    columns = X_test.columns
)


# Create the SARIMAX Model
order = (1, 1, 1) # Simple AR, differencing, and MA
seasonal_order = (1, 1, 1, 5) # Weekly seasonality (5 business days)

model = SARIMAX(
    endog = y_train,              
    exog = X_train_scaled, 
    order = order,
    seasonal_order = seasonal_order,
    enforce_stationarity = False, 
    enforce_invertibility = False
)

results = model.fit(disp = False)
print(results.summary())

# Evalulating Test Set
forecast_test = results.get_forecast(steps = 5, exog=X_test_scaled)
predictions = forecast_test.predicted_mean
conf_int = forecast_test.conf_int()

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

# Forecasting
y_full = meta['Close']
X_full = meta.drop(columns = ['Close'])
X_full_scaled = pd.DataFrame(
    scaler.fit_transform(X_full),
    index=X_full.index,
    columns=X_full.columns
)

model_full = SARIMAX(
    endog = y_full,
    exog = X_full_scaled,
    order = order,
    seasonal_order = seasonal_order,
    enforce_stationarity = False,
    enforce_invertibility = False
)

results_full = model_full.fit(disp = False)
last_date = meta.index[-1]
future_dates = pd.bdate_range(start = last_date + pd.Timedelta(days = 1), periods = 5)
last_features = X_full_scaled.iloc[-1].values
future_features = pd.DataFrame(
    np.tile(last_features, (5, 1)),
    index=future_dates,
    columns=X_full_scaled.columns
)

future_features['day_of_week'] = future_features.index.dayofweek
future_features['month'] = future_features.index.month
forecast_future = results_full.get_forecast(steps = 5, exog = future_features)
future_predictions = forecast_future.predicted_mean
future_conf_int = forecast_future.conf_int()

plt.figure(figsize=(15, 8))

plt.plot(y_train.index, y_train, label='Training Data', color='blue', linewidth=1.5)
plt.plot(y_test.index, y_test, label='Actual (Test)', color='green', 
         marker='o', linewidth=2, markersize=8)

plt.plot(predictions.index, predictions, label='Test Predictions', 
         color='orange', marker='x', linewidth=2, markersize=8)
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='orange', alpha=0.2)

plt.plot(future_predictions.index, future_predictions, label='Future Forecast', 
         color='red', marker='^', linewidth=2.5, markersize=10)
plt.fill_between(future_conf_int.index,
                 future_conf_int.iloc[:, 0],
                 future_conf_int.iloc[:, 1],
                 color='red', alpha=0.2)

plt.title('META Stock Price: SARIMAX Model Results', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price ($)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()