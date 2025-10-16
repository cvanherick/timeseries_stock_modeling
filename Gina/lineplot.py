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
data["Close shift by 1"] = data["Close"].shift(1)
data["Close shift by 3"] = data["Close"].shift(3)
data["Close shift by 5"] = data["Close"].shift(5)
data["Close shift by 10"] = data["Close"].shift(10)
data["Close shift by 15"] = data["Close"].shift(15)
data["Close shift by 20"] = data["Close"].shift(20)

print(data.head())