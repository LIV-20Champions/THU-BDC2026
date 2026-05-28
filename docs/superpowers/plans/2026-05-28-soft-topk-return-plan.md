# Soft Top-K Return Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align training with the final top-5 weighted return score by adding a separate `score_target`, a differentiable soft top-k return loss, and a checkpoint policy that no longer depends on validation selection.

**Architecture:** Keep the existing Transformer backbone, SAM, EMA/SWA, and ensemble flow. Split the supervision into two channels: `label` for ranking stability and `score_target` for the true 5-day return objective. Replace the old proxy-loss default with a masked soft top-k return objective plus masked RankIC, and keep validation metrics as diagnostics only.

**Tech Stack:** Python 3.12, PyTorch 2.6, pandas, numpy, joblib, tqdm, existing repo scripts in `code/src` and `temp/`

---

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `code/src/train.py` | MODIFY | Build `score_target`, pad it in `collate_fn`, add `SoftTopKReturnLoss`, masked RankIC, shared batch-loss helper, and checkpoint policy. |
| `code/src/utils.py` | MODIFY | Carry `score_target` through `LazyRankingDataset`. |
| `code/src/config.py` | MODIFY | Switch default training to the new objective and fixed-epoch, no-validation selection mode. |
| `temp/phase11_target_test.py` | CREATE | Synthetic target/dataset/collate regression test. |
| `temp/phase11_loss_test.py` | CREATE | Synthetic loss monotonicity and padding-mask regression test. |
| `temp/phase11_smoke_test.py` | CREATE | End-to-end train/predict/score smoke test with a short run. |

---

### Task 1: Add `score_target` Plumbing

**Files:**
- Modify: `code/src/train.py`
- Modify: `code/src/utils.py`
- Create: `temp/phase11_target_test.py`

- [ ] **Step 1: Write the failing test**

Create `temp/phase11_target_test.py` with a synthetic dataset that exercises both the raw label builder and the lazy dataset/collate path:

```python
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
```

- [ ] **Step 2: Run the test and confirm it fails before implementation**

Run:

```bash
source .venv/bin/activate && python temp/phase11_target_test.py
```

Expected: fail because `score_target` / `score_targets` is not wired through yet.

- [ ] **Step 3: Implement the minimal plumbing**

In `code/src/train.py`, make the raw preprocessing and collation carry both targets.

Add `score_target` in `_build_label_and_clean()`:

```python
ret_t1t3 = (processed['open_t3'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
ret_t1t5 = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
processed['label'] = label_alpha * ret_t1t3 + (1.0 - label_alpha) * ret_t1t5
processed['score_target'] = ret_t1t5
processed = processed.dropna(subset=['label', 'score_target'])
```

Update `LazyRankingDataset` sample storage:

```python
labels = group['label'].values.astype(np.float32)
score_targets = group['score_target'].values.astype(np.float32)
dates = pd.to_datetime(group['datetime']).values

for end_idx in range(self.sequence_length - 1, len(group)):
    target = labels[end_idx]
    score_target = score_targets[end_idx]
    if not np.isfinite(target) or not np.isfinite(score_target):
        continue

    end_date = dates[end_idx]
    if min_window_end_date is not None and end_date < min_window_end_date:
        continue

    bucket = sample_buckets[end_date]
    bucket['entries'].append((int(stock_code), int(end_idx)))
    bucket['targets'].append(float(target))
    bucket['score_targets'].append(float(score_target))
```

Return the new tensor from `__getitem__()`:

```python
return {
    'sequences': torch.from_numpy(sequences),
    'targets': torch.from_numpy(targets_sub.astype(np.float32)),
    'score_targets': torch.from_numpy(score_targets_sub.astype(np.float32)),
    'relevance': torch.from_numpy(relevance_sub.astype(np.float32)),
    'stock_indices': torch.from_numpy(stock_indices_sub.astype(np.int64)),
}
```

Pad `score_targets` in `collate_fn()`:

```python
score_targets = [item['score_targets'] for item in batch]
padded_score_targets = []

for seq, tgt, score_tgt, rel, stock_idx in zip(
        sequences, targets, score_targets, relevance, stock_indices):
    num_stocks = seq.size(0)
    if num_stocks < max_stocks:
        pad_size = max_stocks - num_stocks
        score_pad = torch.zeros(pad_size)
        score_tgt = torch.cat([score_tgt, score_pad], dim=0)
    padded_score_targets.append(score_tgt)

result = {
    'sequences': torch.stack(padded_sequences),
    'targets': torch.stack(padded_targets),
    'score_targets': torch.stack(padded_score_targets),
    'relevance': torch.stack(padded_relevance),
    'stock_indices': torch.stack(padded_stock_indices).long(),
    'masks': torch.stack(masks),
}
```

In `code/src/utils.py`, make `LazyRankingDataset` store and return the new target. Change the initial filtering:

```python
df = df.dropna(subset=['label', 'score_target'])
```

Add `score_targets` to `sample_buckets` and per-stock arrays:

```python
sample_buckets = defaultdict(lambda: {'entries': [], 'targets': [], 'score_targets': []})
labels = group['label'].values.astype(np.float32)
score_targets = group['score_target'].values.astype(np.float32)
```

Store the daily array in each sample:

```python
day_targets = np.asarray(bucket['targets'], dtype=np.float32)
score_targets_day = np.asarray(bucket['score_targets'], dtype=np.float32)
sorted_indices = np.argsort(day_targets)[::-1]
relevance = np.zeros_like(day_targets, dtype=np.float32)
for rank, idx in enumerate(sorted_indices):
    relevance[idx] = len(day_targets) - rank
stock_indices = np.asarray([entry[0] for entry in bucket['entries']], dtype=np.int64)

self.samples.append({
    'date': date,
    'entries': bucket['entries'],
    'targets': day_targets,
    'score_targets': score_targets_day,
    'relevance': relevance,
    'stock_indices': stock_indices,
})
```

Slice it in `__getitem__()`:

```python
if max_stocks > 0 and len(entries) > max_stocks:
    perm = np.random.permutation(len(entries))[:max_stocks]
    entries = [entries[i] for i in perm]
    targets_sub = sample['targets'][perm]
    score_targets_sub = sample['score_targets'][perm]
    relevance_sub = sample['relevance'][perm]
    stock_indices_sub = sample['stock_indices'][perm]
else:
    targets_sub = sample['targets']
    score_targets_sub = sample['score_targets']
    relevance_sub = sample['relevance']
    stock_indices_sub = sample['stock_indices']
```

- [ ] **Step 4: Re-run the test and confirm it passes**

Run:

```bash
source .venv/bin/activate && python temp/phase11_target_test.py
```

Expected: silent success or a final `OK` print if you add one.

- [ ] **Step 5: Commit**

```bash
git add code/src/train.py code/src/utils.py temp/phase11_target_test.py
git commit -m "feat: add score target plumbing"
```

---

### Task 2: Replace the Default Loss With Soft Top-K Return + Masked RankIC

**Files:**
- Modify: `code/src/train.py`
- Create: `temp/phase11_loss_test.py`

- [ ] **Step 1: Write the failing test**

Create `temp/phase11_loss_test.py` to verify the new loss is monotonic and ignores padded items:

```python
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
```

- [ ] **Step 2: Run the test and confirm it fails before implementation**

Run:

```bash
source .venv/bin/activate && python temp/phase11_loss_test.py
```

Expected: import error or assertion failure until the new loss classes and mask wiring exist.

- [ ] **Step 3: Implement the loss classes and shared batch-loss helper**

In `code/src/train.py`, add a return-aligned loss and a masked RankIC loss:

```python
class SoftTopKReturnLoss(nn.Module):
    def __init__(self, top_k=5, rank_temperature=0.5, gate_temperature=0.5,
                 weight_temperature=0.5, gate_margin=0.5, eps=1e-8):
        super().__init__()
        self.top_k = top_k
        self.rank_temperature = rank_temperature
        self.gate_temperature = gate_temperature
        self.weight_temperature = weight_temperature
        self.gate_margin = gate_margin
        self.eps = eps

    def forward(self, y_pred, score_target, mask=None):
        if mask is None:
            mask = torch.ones_like(y_pred, dtype=torch.bool)

        losses = []
        for i in range(y_pred.size(0)):
            valid = mask[i].bool()
            if valid.sum().item() == 0:
                continue
            pred = y_pred[i, valid].unsqueeze(0)
            target = score_target[i, valid].unsqueeze(0)
            diff = pred.unsqueeze(2) - pred.unsqueeze(1)
            soft_rank = 1.0 + torch.sigmoid(diff / max(self.rank_temperature, 0.01)).sum(dim=2) - 0.5
            gate = torch.sigmoid((self.top_k + self.gate_margin - soft_rank) / max(self.gate_temperature, 0.01))
            raw_weight = torch.softmax(pred / max(self.weight_temperature, 0.01), dim=1) * gate
            basket_weight = raw_weight / (raw_weight.sum(dim=1, keepdim=True) + self.eps)
            expected_return = (basket_weight * target).sum(dim=1)
            losses.append(-expected_return.mean())
        if not losses:
            return y_pred.sum() * 0.0
        return torch.stack(losses).mean()
```

```python
class MaskedSoftRankICLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, y_pred, y_true, mask=None):
        if mask is None:
            mask = torch.ones_like(y_pred, dtype=torch.bool)

        losses = []
        for i in range(y_pred.size(0)):
            valid = mask[i].bool()
            if valid.sum().item() <= 1:
                continue
            pred = y_pred[i, valid].unsqueeze(0)
            true = y_true[i, valid].unsqueeze(0)
            diff_pred = pred.unsqueeze(2) - pred.unsqueeze(1)
            soft_rank_pred = torch.sigmoid(diff_pred / max(self.temperature, 0.01)).sum(dim=2) + 0.5
            diff_true = true.unsqueeze(2) - true.unsqueeze(1)
            soft_rank_true = torch.sigmoid(diff_true / max(self.temperature, 0.01)).sum(dim=2) + 0.5
            pred_c = soft_rank_pred - soft_rank_pred.mean(dim=1, keepdim=True)
            true_c = soft_rank_true - soft_rank_true.mean(dim=1, keepdim=True)
            rho = (pred_c * true_c).sum(dim=1) / (pred_c.norm(p=2, dim=1) * true_c.norm(p=2, dim=1) + 1e-8)
            losses.append((1.0 - rho).mean())
        if not losses:
            return y_pred.sum() * 0.0
        return torch.stack(losses).mean()
```

Add a shared helper so the standard and SAM passes use the exact same logic:

```python
use_soft_topk_return_loss = bool(config.get('use_soft_topk_return_loss', True))
topk_loss_fn = SoftTopKReturnLoss(
    top_k=int(config.get('soft_topk_k', 5)),
    rank_temperature=float(config.get('soft_topk_rank_temperature', 0.5)),
    gate_temperature=float(config.get('soft_topk_gate_temperature', 0.5)),
    weight_temperature=float(config.get('soft_topk_weight_temperature', 0.5)),
    gate_margin=float(config.get('soft_topk_gate_margin', 0.5)),
)
if not use_soft_topk_return_loss:
    topk_loss_fn = SmoothNDCGLoss(
        top_k=int(config.get('ndcg_top_k', 5)),
        temperature=float(config.get('soft_sort_temperature', 0.5)),
        delta_clip=float(config.get('lambda_delta_clip', 10.0)),
        smooth_ndcg_weight=float(config.get('smooth_ndcg_weight', 0.7)),
        lambda_pairwise_weight=float(config.get('lambda_pairwise_weight', 0.3)),
        use_mse_aux=bool(config.get('use_mse_aux_loss', False)),
        mse_aux_weight=float(config.get('mse_aux_weight', 0.05)),
        temperature_start=float(config.get('temperature_anneal_start', 2.0)),
        temperature_target=float(config.get('temperature_anneal_target', 0.5)),
        temperature_anneal_epochs=int(config.get('temperature_anneal_epochs', 30)),
    )
rankic_loss_fn = MaskedSoftRankICLoss(
    temperature=float(config.get('soft_rankic_temperature', 0.5))
) if bool(config.get('use_soft_rankic_loss', True)) else None
rankic_weight = float(config.get('soft_rankic_weight', 0.2))

def _compute_grouped_batch_loss(outputs, targets, score_targets, masks,
                                topk_loss_fn, rankic_loss_fn, rankic_weight):
    group_losses = []
    for i in range(outputs.size(0)):
        valid = masks[i].bool()
        if valid.sum().item() <= 1:
            continue
        pred = outputs[i, valid].unsqueeze(0)
        label = targets[i, valid].unsqueeze(0)
        score = score_targets[i, valid].unsqueeze(0)
        group_mask = torch.ones_like(score, dtype=torch.bool)
        if isinstance(topk_loss_fn, SoftTopKReturnLoss):
            group_loss = topk_loss_fn(pred, score, group_mask)
        else:
            group_loss = topk_loss_fn(pred, label)
        if rankic_loss_fn is not None and rankic_weight > 0:
            group_loss = group_loss + rankic_weight * rankic_loss_fn(pred, label, group_mask)
        group_losses.append(group_loss)
    if not group_losses:
        return None
    return torch.stack(group_losses).mean()
```

Wire this helper into both `train_ranking_model()` and `evaluate_ranking_model()`. Change the evaluation signature to accept `topk_loss_fn`, `rankic_loss_fn`, and `rankic_weight`, and pass `score_targets` into `calculate_ranking_metrics()` so logging reflects the true 5-day objective:

```python
score_targets = batch['score_targets'].to(device)
batch_loss = _compute_grouped_batch_loss(
    masked_outputs, targets, score_targets, masks,
    topk_loss_fn, rankic_loss_fn, rankic_weight,
)
metrics = calculate_ranking_metrics(masked_outputs, score_targets, masks, k=5)
```

- [ ] **Step 4: Re-run the test and confirm it passes**

Run:

```bash
source .venv/bin/activate && python temp/phase11_loss_test.py
```

Expected: the good basket loss is lower, the padding change is ignored, and the RankIC value is finite.

- [ ] **Step 5: Commit**

```bash
git add code/src/train.py temp/phase11_loss_test.py
git commit -m "feat: add soft top-k return loss"
```

---

### Task 3: Switch the Default Training Policy to Fixed-Epoch, No-Validation Selection

**Files:**
- Modify: `code/src/config.py`
- Modify: `code/src/train.py`
- Create: `temp/phase11_smoke_test.py`

- [ ] **Step 1: Write the failing smoke test**

Create `temp/phase11_smoke_test.py` to check the default run path, final checkpoint behavior, and prediction output:

```python
from pathlib import Path
import subprocess
import sys
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from config import config


def test_train_predict_score_smoke():
    output_dir = ROOT / config["output_dir"]
    if output_dir.exists():
        shutil.rmtree(output_dir)

    subprocess.run(
        [sys.executable, "code/src/train.py", "--num_epochs_override", "2"],
        cwd=ROOT,
        check=True,
    )

    summary = (output_dir / "final_score.txt").read_text(encoding="utf-8")
    assert "Best epoch: 2" in summary

    subprocess.run([sys.executable, "code/src/predict.py"], cwd=ROOT, check=True)
    assert (ROOT / "output" / "result.csv").exists()

    subprocess.run([sys.executable, "test/score_self.py"], cwd=ROOT, check=True)
    result = pd.read_csv(ROOT / "temp" / "tmp.csv")
    assert "Final Score" in result.columns
    assert pd.notna(result.loc[0, "Final Score"])


if __name__ == "__main__":
    test_train_predict_score_smoke()
    print("phase11 smoke test OK")
```

- [ ] **Step 2: Run the smoke test and confirm it fails before the policy change**

Run:

```bash
source .venv/bin/activate && python temp/phase11_smoke_test.py
```

Expected: fail before the config and checkpoint policy switch because the current default still depends on validation selection and proxy-loss settings.

- [ ] **Step 3: Implement the default-policy switch**

In `code/src/config.py`, switch the existing default config entries to the Phase 11 regime. Do not remove architecture keys that are not listed here; keep those unchanged.

```python
'num_epochs': 30,
'output_dir': './model/phase11_soft_topk',
'val_months': 0,
'early_stopping_patience': 30,
'use_smooth_ndcg_loss': False,
'smooth_ndcg_weight': 0.0,
'lambda_pairwise_weight': 0.0,
'use_soft_rankic_loss': True,
'soft_rankic_weight': 0.2,
'soft_rankic_temperature': 0.5,
'use_soft_topk_return_loss': True,
'soft_topk_return_weight': 1.0,
'soft_topk_k': 5,
'soft_topk_rank_temperature': 0.5,
'soft_topk_gate_temperature': 0.5,
'soft_topk_weight_temperature': 0.5,
'soft_topk_gate_margin': 0.5,
'use_validation_checkpoint_selection': False,
'use_mse_aux_loss': False,
```

Then change the training loop in `code/src/train.py` so validation no longer decides the checkpoint when `use_validation_checkpoint_selection` is `False`:

```python
use_validation_selection = bool(config.get("use_validation_checkpoint_selection", False))

if no_val or not use_validation_selection:
    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
    if ema_wrapper is not None:
        torch.save(ema_wrapper.shadow, os.path.join(output_dir, "best_model_ema.pth"))
    best_epoch = epoch + 1
    best_score = float(train_metrics.get("official_score_eq", 0.0))
else:
    eval_loss, eval_metrics = evaluate_ranking_model(
        model, val_loader, topk_loss_fn, rankic_loss_fn, rankic_weight,
        device, writer, epoch, use_amp=use_amp,
    )
    if ema_wrapper is not None and epoch >= 5:
        ema_wrapper.restore()
    print(f"Eval Loss: {eval_loss:.4f}")
    for k, v in eval_metrics.items():
        print(f"Eval {k}: {v:.4f}")
    current_selection_score = float(eval_metrics.get(selection_metric_name, 0.0))
    if current_selection_score > best_score:
        best_score = current_selection_score
        best_epoch = epoch + 1
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        if ema_wrapper is not None:
            torch.save(ema_wrapper.shadow, os.path.join(output_dir, "best_model_ema.pth"))
    else:
        epochs_without_improvement += 1
        print(f"Not improved ({epochs_without_improvement}/{early_stopping_patience}), current best: {best_score:.6f}")
```

Keep the final summary writing exactly compatible with `predict.py`:

```python
with open(os.path.join(output_dir, "final_score.txt"), "w") as f:
    f.write(f"Best epoch: {best_epoch}\n")
    f.write(f"Selection metric: {selection_metric_name}\n")
    f.write(f"Best score: {best_score:.6f}\n")
```

- [ ] **Step 4: Re-run the smoke test and confirm it passes**

Run:

```bash
source .venv/bin/activate && python temp/phase11_smoke_test.py
```

Expected: `Best epoch: 2` appears in `final_score.txt`, `output/result.csv` is produced, and `temp/tmp.csv` contains a finite score.

- [ ] **Step 5: Commit**

```bash
git add code/src/config.py code/src/train.py temp/phase11_smoke_test.py
git commit -m "feat: switch phase 11 to fixed-epoch soft top-k training"
```

---

### Task 4: Full Score Regression Check

**Files:**
- No code changes expected

- [ ] **Step 1: Run one full production cycle**

Run the default end-to-end path after the code changes land:

```bash
source .venv/bin/activate && python code/src/train.py
source .venv/bin/activate && python code/src/predict.py
source .venv/bin/activate && python test/score_self.py
```

Expected: `output/result.csv` is regenerated and `temp/tmp.csv` contains a positive score.

- [ ] **Step 2: Repeat the full cycle twice more and save the scores**

Run:

```bash
source .venv/bin/activate
for run in 1 2 3; do
  rm -rf model/phase11_soft_topk output/result.csv temp/tmp.csv
  python code/src/train.py
  python code/src/predict.py
  python test/score_self.py
  cp temp/tmp.csv "temp/phase11_score_run_${run}.csv"
done
```

Expected: `temp/phase11_score_run_1.csv`, `temp/phase11_score_run_2.csv`, and `temp/phase11_score_run_3.csv` exist.

- [ ] **Step 3: Record the result**

Capture the three scores in the work log and note whether the new objective improves the stable basket or only changes variance.

Run:

```bash
source .venv/bin/activate && python -c "import glob, pandas as pd; scores=[float(pd.read_csv(p)['Final Score'].iloc[0]) for p in sorted(glob.glob('temp/phase11_score_run_*.csv'))]; print({'scores': scores, 'median': sorted(scores)[len(scores)//2], 'min': min(scores)})"
```

Expected: the median beats the current ~0.0673 reference and the minimum stays positive.

---

## Self-Review Checklist

1. `score_target` is introduced in preprocessing, dataset storage, and batching.
2. The new loss uses the true 5-day return objective and ignores padded stocks.
3. RankIC remains as a small stabilizer, but validation selection is no longer the default gatekeeper.
4. The config default now matches the Phase 11 regime: 30 epochs, no validation-based selection, and the new return-first objective.
5. Each task has a failing test first, then implementation, then rerun, then commit.
6. No placeholder text remains.
