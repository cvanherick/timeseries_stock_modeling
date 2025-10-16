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



