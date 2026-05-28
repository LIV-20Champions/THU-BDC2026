from pathlib import Path
import sys

import math
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

import train
from config import config


class NaNCriterion(nn.Module):
    def forward(self, y_pred, y_true):
        return y_pred.sum() * float("nan")


class TinyRankingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = nn.Parameter(torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, -0.1]], dtype=torch.float32))

    def forward(self, sequences, stock_indices=None, stock_mask=None):
        return self.scores.expand(sequences.size(0), -1)


def build_batch():
    return {
        "sequences": torch.zeros(1, 6, 3, 2, dtype=torch.float32),
        "targets": torch.tensor([[0.03, 0.02, 0.01, 0.00, -0.01, -0.02]], dtype=torch.float32),
        "score_targets": torch.tensor([[0.30, 0.25, 0.10, 0.05, 0.01, -0.02]], dtype=torch.float32),
        "masks": torch.ones(1, 6, dtype=torch.float32),
        "stock_indices": torch.arange(6, dtype=torch.long).unsqueeze(0),
    }


def test_soft_topk_training_path_ignores_legacy_criterion():
    original = dict(config)
    try:
        config["use_soft_topk_return_loss"] = True
        config["use_soft_rankic_loss"] = True
        config["soft_rankic_weight"] = 0.2
        config["use_mixup"] = False
        config["use_label_smoothing"] = False
        config["enable_grad_clip"] = False

        model = TinyRankingModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, metrics = train.train_ranking_model(
            model=model,
            dataloader=[build_batch()],
            criterion=NaNCriterion(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=0,
            writer=None,
            accumulation_steps=1,
            ema_wrapper=None,
            use_mixup=False,
            use_label_smoothing=False,
            scaler=None,
            use_sam=False,
        )

        assert math.isfinite(loss), f"expected finite soft-topk loss, got {loss}"
        assert "official_score_eq" in metrics
    finally:
        config.clear()
        config.update(original)


if __name__ == "__main__":
    test_soft_topk_training_path_ignores_legacy_criterion()
    print("phase11 train path test OK")
