import sys; sys.path.insert(0,'code/src')
import numpy as np, pandas as pd
from train import _build_label_and_clean

# Build a small raw frame and test CSZScoreNorm
rows = []
dates = pd.date_range("2024-01-01", periods=6, freq="D")
np.random.seed(42)
for stock_idx in range(50):
    code = f"{stock_idx:06d}"
    base = 10.0 + stock_idx
    for day_idx, dt in enumerate(dates):
        open_price = base + day_idx + np.random.randn() * 5
        rows.append({"股票代码": code, "日期": dt.strftime("%Y-%m-%d"),
                      "开盘": open_price, "收盘": open_price + 0.5,
                      "最高": open_price + 1, "最低": open_price - 1,
                      "成交量": 1000, "成交额": 10000})

raw = pd.DataFrame(rows)
processed = _build_label_and_clean(raw.copy(), label_alpha=0.3)
labels = processed['label'].dropna()
print(f"New label: mean={labels.mean():.3f}, std={labels.std():.3f}, min={labels.min():.3f}, max={labels.max():.3f}")
print(f"Label distribution: [{labels.quantile(0.01):.3f}, {labels.quantile(0.99):.3f}] 99% range")
print(f"Samples dropped: {(raw.shape[0] - processed.shape[0])} of {raw.shape[0]}")
print("CSZScoreNorm OK")
