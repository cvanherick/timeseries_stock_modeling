import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Download Stock Data

ticker = yf.Ticker("PLTR") 
raw_data = ticker.history(period="6mo", interval="1d") 


# 2. Feature Engineering Pipelines
def feature_engineering_train(df):
    """Feature engineering for training data."""
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.asfreq("B")
    df = df.ffill()
    
    for lag in [1, 2, 7, 14, 21]: 
        df[f'Open_lag_{lag}'] = df['Open'].shift(lag) 

    df['day_of_week'] = df.index.day_name()
    df['month'] = df.index.month_name()
    
    df = df.dropna()
    
    return df


def feature_engineering_future(last_df, future_days=5):
    """
    Generate exogenous features for future days beyond the available dataset.
    Reuses scaled features and updates calendar-based fields. 
    This function bassicailly pulls the data from the last day to use as the matrix when forecasting future values.
    """
    future_dates = pd.bdate_range(start=last_df.index[-1] + pd.Timedelta(days=1), periods=future_days) 

    
    repeated_vals = np.tile(last_df.iloc[-1].values, (future_days, 1))
    future_exog = pd.DataFrame(repeated_vals, index=future_dates, columns=last_df.columns)

    if 'day_of_week' in future_exog.columns:
        future_exog['day_of_week'] = future_exog.index.day_name()
    if 'month' in future_exog.columns:
        future_exog['month'] = future_exog.index.month_name()

    return future_exog




# 3. Prepare Train/Test Data

clean_data = feature_engineering_train(raw_data)
test_days = 5
train_data = clean_data.iloc[:-test_days]
test_data = clean_data.iloc[-test_days:]

y_train = train_data['Close'] 
X_train = train_data.drop(columns=['Close']) 
y_test = test_data['Close']
X_test = test_data.drop(columns=['Close'])

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)


# 4. Fit SARIMAX Model (Training)
order = (1, 1, 1)
seasonal_order = (0, 0, 0, 0)

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
X_future_scaled = feature_engineering_future(full_X_scaled, future_days=future_days)

forecast_future = results_full.get_forecast(steps=future_days, exog=X_future_scaled)
forecast_future_mean = forecast_future.predicted_mean
forecast_future_ci = forecast_f_
