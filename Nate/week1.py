import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt

SYMBOL = "PLTR" 

ticker = yf.Ticker(SYMBOL)
data = ticker.history(period="6mo", interval="1d")

print(data.head())

x = data.index
y = data["Close"]

plt.figure()
plt.plot(x,y)
plt.title(f"{SYMBOL} - Closing Prices: Last 6 Months")
plt.ylabel("Closing Price ($)")
plt.xlabel ("Date (Month)")
plt.show()



