import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from config import config, get_eff_input_dim, get_scale_features
from model import StockTransformer
from sam import SAM
from train import collate_fn, preprocess_data, set_seed, train_ranking_model
from utils import LazyRankingDataset


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


def _build_one_real_batch():
    old_workers = config.get("feature_engineering_workers")
    config["feature_engineering_workers"] = 1
    try:
        full_df = pd.read_csv(ROOT / "data/train.csv", dtype={"股票代码": str})
        full_df["股票代码"] = full_df["股票代码"].astype(str).str.zfill(6)
        stockid2idx = {
            sid: idx for idx, sid in enumerate(sorted(full_df["股票代码"].unique()))
        }
        train_data, features = preprocess_data(
            full_df.copy(), is_train=True, stockid2idx=stockid2idx
        )
        scale_features = get_scale_features(features)
        train_data[scale_features] = (
            train_data[scale_features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        dataset = LazyRankingDataset(
            train_data,
            features,
            config["sequence_length"],
            use_per_stock_norm=True,
            use_cs_features=False,
        )
        dataset._max_stocks = config.get("max_stocks_per_sample", 0)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        return next(iter(loader)), features, len(stockid2idx)
    finally:
        config["feature_engineering_workers"] = old_workers


def test_amp_request_with_sam_keeps_training_step_finite():
    if not torch.cuda.is_available():
        print("cuda unavailable, skipping AMP+SAM regression")
        return

    set_seed(42)
    old_soft_topk = config.get("use_soft_topk_return_loss")
    old_rankic = config.get("use_soft_rankic_loss")
    config["use_soft_topk_return_loss"] = True
    config["use_soft_rankic_loss"] = True
    try:
        batch, features, num_stocks = _build_one_real_batch()
        device = torch.device("cuda")
        model = StockTransformer(
            input_dim=get_eff_input_dim(len(features)),
            config=config,
            num_stocks=num_stocks,
        ).to(device)
        base_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=float(config.get("weight_decay", 1e-5)),
        )
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=float(config.get("sam_rho", 0.02)),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        loss, _ = train_ranking_model(
            model,
            [batch],
            criterion=None,
            optimizer=optimizer,
            device=device,
            epoch=0,
            writer=NullWriter(),
            accumulation_steps=1,
            ema_wrapper=None,
            use_mixup=False,
            use_label_smoothing=False,
            scaler=scaler,
            use_sam=True,
        )

        assert math.isfinite(loss), f"expected finite training loss, got {loss}"
        assert all(
            torch.isfinite(p).all().item()
            for p in model.parameters()
            if p.requires_grad
        ), "SAM step with AMP requested produced non-finite parameters"
    finally:
        config["use_soft_topk_return_loss"] = old_soft_topk
        config["use_soft_rankic_loss"] = old_rankic


if __name__ == "__main__":
    test_amp_request_with_sam_keeps_training_step_finite()
    print("phase11 AMP+SAM regression test OK")
