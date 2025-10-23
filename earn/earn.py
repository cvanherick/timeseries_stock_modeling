import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("META")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows
print(data.head())

plt.plot(data.index, data["Close"])
plt.title("META Close Price (Last 6 Months)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.show()


def feature_engineering(data):
    data.index = pd.to_datetime(data.index)
    data = data.ffill() #to fil out any missing values in the data frame for time series

    #set the frequecy to the business day frequency beceause they are only open on business days, ignoring weekends and holidays
    data = data.asfreq('B')


        # data["Open shifted 2d"] = data["Open"].shift(2)
    # data["Open shifted 3d"] = data["Open"].shift(3)
    # data["Open shifted 7d"] = data["Open"].shift(7)
    # data["Open shifted 14d"] = data["Open"].shift(14)
    # data["Open shifted 25d"] = data["Open"].shift(25)

    # data["Close shifted 2d"] = data["Close"].shift(2)
    # data["Close shifted 3d"] = data["Close"].shift(3)
    # data["Close shifted 7d"] = data["Close"].shift(7)
    # data["Close shifted 14d"] = data["Close"].shift(14)
    # data["Close shifted 25d"] = data["Close"].shift(25)

    "Rate of change from previous point in time, looking at the stock price today and comparing it to the stock price yesterday &"
    "also comparing the price today to the price from 2 weeks ago" 

    shift_list = [2,3,7,14,25]
    for num_days in range(len(shift_list)):
        data["lag_open" + str(shift_list[num_days])] = data["Open"].shift(shift_list[num_days])

    'rolling is the past 20 days average'
    'can do mean or std deviation'
    data["Rolling Mean 20d"] = data["Close"].rolling(window=20).mean()
    data["Rolling Std 20d"] = data["Close"].rolling(window=20).std()
    data["Month"] = data.index.month
    data["Day"] = data.index.day
    data = data.dropna()

    return data

print(data.head(10))

cleaned_data = feature_engineering(data)
train_data = cleaned_data.iloc[:-10]
test_data = cleaned_data.iloc[-10:]

y_train = train_data["Close"]
X_train = train_data.drop(columns=["Close"])

y_test = test_data["Close"]
X_test = test_data.drop(columns=["Close"])



