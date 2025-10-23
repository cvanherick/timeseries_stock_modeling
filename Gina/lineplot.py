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
data = data.ffill()

# tells the data to only look at business days - sets frequency of the data to business days
data = data.asfreq('B')

# lag feature looks at data in the context of other times - some predictor variable for tomorrow without knowing what the actual valeus are going to be
data["High sft by 1"] = data["High"].shift(1)
data["High sft by 3"] = data["High"].shift(3)
data["High sft by 5"] = data["High"].shift(5)
data["High sft by 10"] = data["High"].shift(10)
data["High sft by 15"] = data["High"].shift(15)
data["High sft by 20"] = data["High"].shift(20)


#rolling average / rolling sd = average over the past x amount of days
data["rolling average by 20 (high)"] = data["High"].rolling(20).mean()

print(data.head())