import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("AMD")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows

plt.plot(data.index, data['Close'])
plt.ylabel("Price in US Dollars")
plt.xlabel("Date")
plt.title('AMD Stock Price Over the Last 6 Months')
#plt.show()

print('Starts here:')
print(data.index[-2])
print(data.iloc[-2])

data = data.ffill()
data = data.asfreq("B")
data['lf_prev1_open'] = data['Open'].shift(1)
data['lf_prev3_open'] = data['Open'].shift(3)
data['lf_prev7_open'] = data['Open'].shift(7)
data['lf_prev14_open'] = data['Open'].shift(14)
data['lf_prev30_open'] = data['Open'].shift(30)

#.rolling(arg - period).(mean or std) and takes the average over the period
#shift 
#make above lag stuff into a function


print(data['lf_prev1_open'])
