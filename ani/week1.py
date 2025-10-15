import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("AMD")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows
print(data.head())

plt.plot(data.index, data['Close'])
plt.ylabel("Price in US Dollars")
plt.xlabel("Date")
plt.title('AMD Stock Price Over the Last 6 Months')
plt.show()
