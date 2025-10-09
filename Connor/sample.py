import yfinance as yf

# Choose a stock ticker — for example, Apple
ticker = yf.Ticker("AAPL")

# Download recent data (e.g. 6 months)
data = ticker.history(period="6mo")

# Show the first few rows
print(data.head())