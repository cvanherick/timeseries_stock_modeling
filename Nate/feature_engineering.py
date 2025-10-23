import yfinance as yf
import pandas as pd 
import webbrowser, tempfile, pathlib

SYMBOL = "PLTR" 

ticker = yf.Ticker(SYMBOL)
data = ticker.history(period="6mo", interval="1d")


x = data.index
y = data["Close"]

#data.index = data["date"] (index is already set as date)

data = data.ffill() #for time series, take previous day's result
data = data.asfreq('B') # set frequency of data, (business day frequence),,,completely ignores weekends & holidays instead of "blank"

# lag feature: rate of change from previous point in time,,,predictive aspect 
    #periods of time that represent some market trend (5 days, 1 month, etc)

data["lag_1_open"] = data["Open"].shift(1)
data["lag_2_open"] = data["Open"].shift(2)
data["lag_7_open"] = data["Open"].shift(7)
data["lag_14_open"] = data["Open"].shift(14)
data["lag_21_open"] = data["Open"].shift(21)


print(type(data.index))         # should be DatetimeIndex for time ops
print(data[["Open","lag_1_open", "Open", "lag_2_open", "Open", "lag_7_open", "Open", "lag_14_open", "Open", "lag_21_open"]])


# Build an HTML table from your existing `data` (no column changes)
view = data.copy()

# Make the datetime index readable in the table (optional, not altering `data`)
idx = view.index
if getattr(idx, "tz", None) is not None:
    idx = idx.tz_convert("America/Los_Angeles")  # or remove this line
view.index = idx.strftime("%Y-%m-%d")
view.index.name = "date"

# Render HTML
html_table = view.to_html(border=0, classes="table", float_format=lambda x: f"{x:.2f}")

# Wrap in a simple page and open it
page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{SYMBOL} — full DataFrame</title>
  <style>
    body{{font-family:system-ui, -apple-system, Segoe UI, Arial; margin:24px}}
    table{{border-collapse:collapse; width:100%}}
    th, td{{border:1px solid #ddd; padding:6px 10px; text-align:right}}
    th{{background:#f6f6f6; position:sticky; top:0}}
    td:first-child, th:first-child{{text-align:left}}
    tbody tr:nth-child(odd){{background:#fafafa}}
  </style>
</head>
<body>
  <h2>{SYMBOL} — full DataFrame</h2>
  {html_table}
</body>
</html>
"""

out = pathlib.Path(tempfile.gettempdir()) / f"{SYMBOL}_table.html"
out.write_text(page, encoding="utf-8")
webbrowser.open_new_tab(out.as_uri())
print(f"Opened {out}")