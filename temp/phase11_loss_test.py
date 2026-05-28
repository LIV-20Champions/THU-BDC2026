from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))

from train import SoftTopKReturnLoss, MaskedSoftRankICLoss


def test_soft_topk_return_loss_prefers_better_basket():
    loss_fn = SoftTopKReturnLoss(
        top_k=5,
        rank_temperature=0.5,
        gate_temperature=0.5,
        weight_temperature=0.5,
        gate_margin=0.5,
    )
    rankic_fn = MaskedSoftRankICLoss(temperature=0.5)

    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
    score_target = torch.tensor([[0.30, 0.25, 0.10, 0.05, 0.01, 9.99]], dtype=torch.float32)
    label = torch.tensor([[0.20, 0.18, 0.08, 0.04, 0.00, -9.99]], dtype=torch.float32)
    pred_good = torch.tensor([[5.0, 4.0, 2.0, 1.0, 0.5, 100.0]], dtype=torch.float32)
    pred_bad = torch.tensor([[0.5, 1.0, 2.0, 4.0, 5.0, -100.0]], dtype=torch.float32)

    good_loss = loss_fn(pred_good, score_target, mask)
    bad_loss = loss_fn(pred_bad, score_target, mask)
    assert good_loss.item() < bad_loss.item()

    padded_changed = score_target.clone()
    padded_changed[0, 5] = -1234.0
    same_loss = loss_fn(pred_good, padded_changed, mask)
    assert torch.isclose(good_loss, same_loss, atol=1e-6)

    rankic = rankic_fn(pred_good, label, mask)
    assert torch.isfinite(rankic)


if __name__ == "__main__":
    test_soft_topk_return_loss_prefers_better_basket()
    print("phase11 loss tests OK")
