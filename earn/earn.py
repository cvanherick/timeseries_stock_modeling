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

data.index = pd.to_datetime(data.index)
data = data.ffill() #to fil out any missing values in the data frame for time series

#set the frequecy to the business day frequency beceause they are only open on business days, ignoring weekends and holidays
data = data.asfreq('B')

print(data.head())

"Rate of change from previous point in time, looking at the stock price today and comparing it to the stock price yesterday &"
"also comparing the price today to the price from 2 weeks ago" 

data["Open shifted 2d"] = data["Open"].shift(2)
data["Open shifted 3d"] = data["Open"].shift(3)
data["Open shifted 7d"] = data["Open"].shift(7)
data["Open shifted 14d"] = data["Open"].shift(14)
data["Open shifted 25d"] = data["Open"].shift(25)

data["Close shifted 2d"] = data["Close"].shift(2)
data["Close shifted 3d"] = data["Close"].shift(3)
data["Close shifted 7d"] = data["Close"].shift(7)
data["Close shifted 14d"] = data["Close"].shift(14)
data["Close shifted 25d"] = data["Close"].shift(25)






