import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("MSFT")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows
print(data.head())

print(data.shape)

plt.figure(figsize=(10, 5))
plt.plot(data.index, data["Close"], label="MSFT Close Price")
plt.title("Microsoft Stock Price (Last 6 Months)")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.xticks(data.index[::11], rotation=45)
plt.show()
