import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("MSFT")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows
# print(data.head())

print(data.shape)

plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Close"], label="MSFT Close Price")
plt.title("Microsoft Stock Price (Last 6 Months)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.xticks(data.index[::11], rotation=45)
# plt.show()

# fills in missing values with the average of the day before and after
# .ffill()

# tells the data to only look at business days - sets frequency of the data to business days
# .asfreq('B)

# lag feature looks at data in the context of other times - some predictor variable for tomorrow without knowing what the actual valeus are going to be
# .shift



#rolling average / rolling sd = average over the past x amount of days
data["rolling average by 20 (high)"] = data["High"].rolling(20).mean()

print(data.head(20))

def feature_engineering(df):
    df["Date"] = pd.to_datetime[df["Date"]]
    df = df.set_index["Date"].sort_index()
    # tells the data to only look at business days - sets frequency of the data to business days
    df = df.asfreq('B')
    # fills in missing values with the average of the day before and after
    df = df.ffill()
    # lag feature looks at data in the context of other times - some predictor variable for tomorrow without knowing what the actual values are going to be
    for lag in [1, 3, 5, 10]:
        df["high_lag_{lag}]"] = df["High"].shift(lag)
    df["rolling_mean_5"] = df["High"].rolling(5).mean()
    df["rolling_std_5"] = df["High"].rolling(5).std()

    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df = df.dropna()

cleaned_data = feature_engineering(data)
train_data = cleaned_data.iloc[:-5]
test_data = cleaned_data.iloc[-5:]

y_train = train_data['Close']
x_train = train_data.drop(columns=["Close"])

y_test = test_data["Close"]
x_test = test_data.drop(columns=["Close"])

    