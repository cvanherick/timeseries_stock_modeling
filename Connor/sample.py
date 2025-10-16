import yfinance as yf

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("AAPL")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows
print(data.head())
for lag in [1, 3, 5, 10, 21]:
    data[f'Close_lag_{lag}'] = data['Close'].shift(lag)
data.head()
