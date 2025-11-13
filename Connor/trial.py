import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Load data
ticker = yf.Ticker("MSFT")
data = ticker.history(period="6mo")

# Closing price series
close_prices = data['Close']

# Define maximum lags for ACF/PACF (increase to see possible seasonal cycles)
max_lag = 40   # For weekday seasonality, lags 5, 10, 15... will be visible

# Plot ACF and PACF on original series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(close_prices, lags=max_lag, ax=axes[0])
plot_pacf(close_prices, lags=max_lag, ax=axes[1])
axes[0].set_title('ACF (Original Series)')
axes[1].set_title('PACF (Original Series)')
# Add vertical lines for hypothesized seasonality (e.g., 5 days/week)
for lag in range(5, max_lag+1, 5):
    axes[0].axvline(lag, color='red', linestyle='--', alpha=0.4)
    axes[1].axvline(lag, color='red', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# First difference to induce stationarity
diff_1 = close_prices.diff().dropna()

# Plot ACF and PACF on differenced series
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(diff_1, lags=max_lag, ax=axes[0])
plot_pacf(diff_1, lags=max_lag, ax=axes[1])
axes[0].set_title('ACF (First-Differenced Series)')
axes[1].set_title('PACF (First-Differenced Series)')
for lag in range(5, max_lag+1, 5):
    axes[0].axvline(lag, color='red', linestyle='--', alpha=0.4)
    axes[1].axvline(lag, color='red', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
