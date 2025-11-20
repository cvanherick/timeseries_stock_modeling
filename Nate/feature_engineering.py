import yfinance as yf
import pandas as pd
import webbrowser, tempfile, pathlib


SYMBOL = "AVGO"
ticker = yf.Ticker(SYMBOL)

data = ticker.history(period="6mo", interval="1d")
data = data.copy()
data = data.ffill()
data = data.asfreq("B")


lag_periods = [1, 2, 7, 14, 21]

for lag in lag_periods:
    data[f"lag_{lag}_open"] = data["Open"].shift(lag)

data["day_of_week"] = data.index.day_name()
data["month"] = data.index.month_name()

data = data.dropna()

view = data.copy()


idx = view.index
if getattr(idx, "tz", None) is not None:
    idx = idx.tz_convert("America/Los_Angeles")
view.index = idx.strftime("%Y-%m-%d")
view.index.name = "date"

html_table = view.to_html(border=0, classes="table", float_format=lambda x: f"{x:.2f}")

page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{SYMBOL} — Cleaned DataFrame</title>
  <style>
    :root {{
      --border:#ddd; --bg:#f6f6f6; --z-head:3; --z-first:2;
    }}
    body {{
      font-family: system-ui, -apple-system, Segoe UI, Arial;
      margin: 24px;
    }}
    h2 {{
      margin: 0 0 12px 0;
    }}
    .table-wrap {{
      overflow-x: auto;               /* horizontal scroll */
      -webkit-overflow-scrolling: touch;
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(0,0,0,.05);
      padding-bottom: 4px;
    }}
    table {{
      border-collapse: collapse;
      width: max(100%, 1400px);       /* ensures wide tables scroll */
      table-layout: auto;
      white-space: nowrap;            /* keep data on one line */
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 6px 10px;
      text-align: right;
    }}
    th {{
      position: sticky;
      top: 0;
      background: var(--bg);
      z-index: var(--z-head);
    }}
    td:first-child, th:first-child {{
      position: sticky;
      left: 0;
      background: #fff;
      text-align: left;
      z-index: var(--z-first);
      box-shadow: 2px 0 0 rgba(0,0,0,0.03);
    }}
    tbody tr:nth-child(odd) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h2>{SYMBOL} — Cleaned and Enhanced DataFrame</h2>
  <div class="table-wrap">
    {html_table}
  </div>
</body>
</html>
"""

out = pathlib.Path(tempfile.gettempdir()) / f"{SYMBOL}_cleaned_table.html"
out.write_text(page, encoding="utf-8")
webbrowser.open_new_tab(out.as_uri())
print(f"Full Dtaframe: Cleaned: {out}")

cleaned_data = data
train_data = cleaned_data.iloc[:-10]
test_data = cleaned_data.iloc[-10:]

y_train = train_data["Close"]
X_train = train_data.drop(columns=["Close"])

y_test = test_data["Close"]
X_test = test_data.drop(columns=["Close"])