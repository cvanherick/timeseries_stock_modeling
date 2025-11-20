import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("AMD")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows

plt.plot(data.index, data['Close'])
plt.ylabel("Price in US Dollars")
plt.xlabel("Date")
plt.title('AMD Stock Price Over the Last 6 Months')
#plt.show()

#print('Starts here:')
#print(data.index[-2])
#print(data.iloc[-2])


#data['lf_prev1_open'] = data['Open'].shift(1)
#data['lf_prev3_open'] = data['Open'].shift(3)
#data['lf_prev7_open'] = data['Open'].shift(7)
#data['lf_prev14_open'] = data['Open'].shift(14)
#data['lf_prev30_open'] = data['Open'].shift(30)

#takes in dataframe and list of numbers for shifting the data
def engineered(data, shift_list):
    
    data = data.ffill()
    data = data.asfreq("B")
    
    for i in range(len(shift_list)):
        col_name = "lag_open" + str(shift_list[i])
        data[col_name] = data['Open'].shift(shift_list[i])
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month

    data["Rolling Mean 5d"] = data["Open"].rolling(window=5).mean()
    data["Rolling StD 5d"] = data["Open"].rolling(window=5).std()
    data["Rolling Mean 20d"] = data["Open"].rolling(window=20).mean()

    data = data.dropna()

    return data

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

#engineered_df = engineered(data, [1, 3, 7, 14, 30])
#print('engineered below!')
#print(engineered_df)



#.rolling(arg - period).(mean or std) and takes the average over the period
#shift 
#make above lag stuff into a function

#print(data['lf_prev1_open'])

# -------------------------------------------------- COPIED/MODELING ----------------------------------------------------

# 3. Prepare Train/Test Data

clean_data = engineered(data, [1, 3, 7, 14, 30])
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
#TO DO
#order = ()
#seasonal_order = ()

model = SARIMAX(endog=y_train, exog=X_train_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
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

model_full = SARIMAX(endog=full_y, exog=full_X_scaled, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
results_full = model_full.fit(disp=False)

# Generate future features
future_days = 5
X_future_scaled = feature_engineering_future(full_X_scaled, future_days=future_days)

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



