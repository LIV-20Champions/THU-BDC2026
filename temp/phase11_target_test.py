from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from train import _build_label_and_clean, collate_fn
from utils import LazyRankingDataset


def build_raw_frame():
    rows = []
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    for stock_idx in range(10):
        code = f"{stock_idx:06d}"
        base = 10.0 + stock_idx
        for day_idx, dt in enumerate(dates):
            open_price = base + day_idx
            rows.append(
                {
                    "股票代码": code,
                    "日期": dt.strftime("%Y-%m-%d"),
                    "开盘": open_price,
                    "收盘": open_price + 0.2,
                    "最高": open_price + 0.5,
                    "最低": open_price - 0.5,
                    "成交量": 1000 + stock_idx,
                    "成交额": 10000 + stock_idx,
                }
            )
    return pd.DataFrame(rows)


def test_build_label_and_score_target():
    raw = build_raw_frame()
    processed = _build_label_and_clean(raw.copy(), label_alpha=0.3)
    assert "label" in processed.columns
    assert "score_target" in processed.columns

    one_stock = raw[raw["股票代码"] == "000000"].reset_index(drop=True)
    expected_open_t1 = one_stock.loc[1, "开盘"]
    expected_open_t3 = one_stock.loc[3, "开盘"]
    expected_open_t5 = one_stock.loc[5, "开盘"]
    expected_label = 0.3 * ((expected_open_t3 - expected_open_t1) / expected_open_t1) + 0.7 * (
        (expected_open_t5 - expected_open_t1) / expected_open_t1
    )
    expected_score_target = (expected_open_t5 - expected_open_t1) / expected_open_t1

    first_row = processed[processed["股票代码"] == "000000"].iloc[0]
    assert np.isclose(first_row["label"], expected_label)
    assert np.isclose(first_row["score_target"], expected_score_target)


def build_dataset_frame():
    rows = []
    dates = pd.date_range("2024-02-01", periods=6, freq="D")
    for stock_idx in range(10):
        for day_idx, dt in enumerate(dates):
            rows.append(
                {
                    "instrument": stock_idx,
                    "日期": dt.strftime("%Y-%m-%d"),
                    "f1": float(stock_idx + day_idx),
                    "f2": float(stock_idx - day_idx),
                    "label": float(stock_idx + day_idx) / 100.0,
                    "score_target": float(stock_idx + 2 * day_idx) / 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_dataset_and_collate_carry_score_targets():
    data = build_dataset_frame()
    features = ["instrument", "f1", "f2"]
    dataset = LazyRankingDataset(
        data,
        features,
        sequence_length=3,
        min_window_end_date=None,
        use_per_stock_norm=False,
        use_cs_features=False,
    )

    sample = dataset[0]
    assert "score_targets" in sample
    assert sample["score_targets"].shape == sample["targets"].shape

    smaller = {
        "sequences": sample["sequences"][:8].clone(),
        "targets": sample["targets"][:8].clone(),
        "score_targets": sample["score_targets"][:8].clone(),
        "relevance": sample["relevance"][:8].clone(),
        "stock_indices": sample["stock_indices"][:8].clone(),
    }
    batch = collate_fn([sample, smaller])
    assert batch["score_targets"].shape == batch["targets"].shape
    assert batch["masks"].shape[:2] == batch["targets"].shape
    assert torch.all(batch["score_targets"][1, 8:] == 0)


if __name__ == "__main__":
    test_build_label_and_score_target()
    test_dataset_and_collate_carry_score_targets()
    print("phase11 target tests OK")
