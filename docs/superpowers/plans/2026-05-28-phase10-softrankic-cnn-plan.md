# Phase 10: SoftRankIC Loss + CNN Feature Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add differentiable Spearman RankIC loss for objective alignment + CNN feature extractor for signal enhancement, targeting self-score > 0.0673

**Architecture:** SoftRankICLoss replaces/complements SmoothNDCG in train.py. CNNFeatureExtractor (3-layer Conv1d + BN) is added as a parallel path to TA-Lib features in StockTransformer, concat-then-project into the existing Transformer encoder.

**Tech Stack:** Python 3.12, PyTorch 2.6, torch.nn.Conv1d, torch.nn.BatchNorm1d

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `code/src/train.py` | MODIFY | Add SoftRankICLoss class, integrate with loss computation |
| `code/src/model.py` | MODIFY | Add CNNFeatureExtractor class, integrate with StockTransformer |
| `code/src/config.py` | MODIFY | Add feature flags for both modules |

---

### Task 1: SoftRankIC Loss

**Files:**
- Modify: `code/src/train.py`

- [ ] **Step 1: Add SoftRankICLoss class**

Insert after the `SmoothNDCGLoss` class (after line 196):

```python
class SoftRankICLoss(nn.Module):
    """Differentiable Spearman rank correlation loss.

    Uses soft ranking (sigmoid-based) to make the discrete ranking operation
    differentiable.  Optimizing 1-SpearmanR directly aligns training with the
    ranking objective used in quantitative finance (Rank IC).
    """
    def __init__(self, temperature=0.5):
        super(SoftRankICLoss, self).__init__()
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        B, N = y_true.size()
        eps = 1e-8

        # Soft rank via pairwise sigmoid
        diff_pred = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)   # [B, N, N]
        soft_rank_pred = (torch.sigmoid(diff_pred / max(self.temperature, 0.01))
                          .sum(dim=2) + 0.5)

        diff_true = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        soft_rank_true = (torch.sigmoid(diff_true / max(self.temperature, 0.01))
                          .sum(dim=2) + 0.5)

        # Pearson correlation on ranks = Spearman
        pred_c = soft_rank_pred - soft_rank_pred.mean(dim=1, keepdim=True)
        true_c = soft_rank_true - soft_rank_true.mean(dim=1, keepdim=True)

        cov = (pred_c * true_c).sum(dim=1)
        pred_std = pred_c.norm(p=2, dim=1)
        true_std = true_c.norm(p=2, dim=1)

        rho = cov / (pred_std * true_std + eps)
        return (1.0 - rho).mean()
```

- [ ] **Step 2: Integrate into loss computation**

In `train_ranking_model`, after computing `batch_loss` from the criterion, add RankIC loss. Find the loss computation section and add after `criterion(...)`:

In the first forward pass (around line 457):
```python
                loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
```

Add after this block:
```python
            # SoftRankIC as additional ranking objective
            use_soft_rankic = bool(config.get('use_soft_rankic_loss', False))
            if use_soft_rankic and batch_loss is not None:
                from train import SoftRankICLoss
                rankic_weight = float(config.get('soft_rankic_weight', 1.0))
                rankic_temp = float(config.get('soft_rankic_temperature', 0.5))
                rankic_loss_fn = SoftRankICLoss(temperature=rankic_temp)
                # Compute rankic on the full batch (masked)
                # Use original outputs (before masking) for ranking
                full_pred = outputs.reshape(masked_outputs.size(0), -1)
                full_true = masked_targets
                rankic_loss = rankic_loss_fn(full_pred, full_true)
                batch_loss = batch_loss + rankic_weight * rankic_loss
```

In the SAM second forward pass, also add RankIC loss (same pattern around line 490 where `loss2` is computed):

```python
                if use_soft_rankic and batch_loss2 is not None:
                    full_pred2 = outputs2.reshape(masked_outputs2.size(0), -1)
                    full_true2 = masked_targets
                    rankic_loss2 = rankic_loss_fn(full_pred2, full_true2)
                    batch_loss2 = batch_loss2 + rankic_weight * rankic_loss2
```

- [ ] **Step 3: Verify import**

```bash
source .venv/bin/activate && python -c "import sys; sys.path.insert(0,'code/src'); import train; print('SoftRankIC OK')"
```
Expected: `SoftRankIC OK`

- [ ] **Step 4: Commit**

```bash
git add code/src/train.py
git commit -m "feat: add SoftRankIC loss (differentiable Spearman correlation)"
```

---

### Task 2: CNN Feature Extractor

**Files:**
- Modify: `code/src/model.py`

- [ ] **Step 1: Add CNNFeatureExtractor class**

Insert before the `StockTransformer` class (before line 330):

```python
class CNNFeatureExtractor(nn.Module):
    """1D CNN feature extractor inspired by AlphaNet.

    Applies temporal convolutions to raw stock features to learn
    new time-series patterns.  Output concat'd with original features
    before the Transformer encoder.
    """
    def __init__(self, in_channels, out_channels=32, kernel_sizes=None):
        super(CNNFeatureExtractor, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 3]
        mid = 64

        self.conv1 = nn.Conv1d(in_channels, mid, kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(mid)
        self.conv2 = nn.Conv1d(mid, mid, kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(mid)
        self.conv3 = nn.Conv1d(mid, out_channels, kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B*N, L, F] -> transpose to [B*N, F, L] for Conv1d
        x_t = x.transpose(1, 2)
        x_t = self.activation(self.bn1(self.conv1(x_t)))
        x_t = self.activation(self.bn2(self.conv2(x_t)))
        x_t = self.activation(self.bn3(self.conv3(x_t)))
        # Back to [B*N, L, out_channels]
        return x_t.transpose(1, 2)
```

- [ ] **Step 2: Integrate into StockTransformer**

In `StockTransformer.__init__`, after `self.input_proj` block (after line 365), add:

```python
        # CNN feature extractor (AlphaNet-style)
        self.use_cnn_features = bool(config.get("use_cnn_features", False))
        if self.use_cnn_features:
            cnn_out_dim = int(config.get("cnn_feature_dim", 32))
            self.cnn_extractor = CNNFeatureExtractor(
                in_channels=numeric_input_dim,
                out_channels=cnn_out_dim,
            )
            # Project CNN output to d_model for concatenation
            self.cnn_proj = nn.Sequential(
                nn.Linear(cnn_out_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )
            # After concat: original d_model + cnn d_model
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.cnn_extractor = None
```

In `StockTransformer.forward()`, after `src_proj = self.input_proj(src_reshaped)` (after line 527), add:

```python
        # CNN feature extraction (parallel path)
        if self.use_cnn_features:
            src_cnn = self.cnn_extractor(src_reshaped)
            src_cnn_proj = self.cnn_proj(src_cnn)
            src_proj = self.fusion_proj(torch.cat([src_proj, src_cnn_proj], dim=-1))
```

- [ ] **Step 3: Verify import**

```bash
source .venv/bin/activate && python -c "import sys; sys.path.insert(0,'code/src'); from model import CNNFeatureExtractor, StockTransformer; print('CNN OK')"
```
Expected: `CNN OK`

- [ ] **Step 4: Commit**

```bash
git add code/src/model.py
git commit -m "feat: add CNN feature extractor (AlphaNet-style Conv1d + BN)"
```

---

### Task 3: Config Changes

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Add config entries**

Insert after the loss-related config block (after `lambda_delta_clip`):

```python
    # --- SoftRankIC loss ---
    'use_soft_rankic_loss': True,
    'soft_rankic_weight': 0.3,
    'soft_rankic_temperature': 0.5,

    # --- CNN feature extractor ---
    'use_cnn_features': True,
    'cnn_feature_dim': 32,
```

Update `output_dir`:
```python
    'output_dir': './model/phase10_rankic_cnn',
```

- [ ] **Step 2: Verify config**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0,'code/src')
from config import config
for k in ['use_soft_rankic_loss','soft_rankic_weight','use_cnn_features','cnn_feature_dim']:
    print(f'{k}={config[k]}')
print('Config OK')
"
```
Expected: all True/configured, then `Config OK`

- [ ] **Step 3: Commit**

```bash
git add code/src/config.py
git commit -m "feat: Phase 10 config — SoftRankIC loss + CNN features"
```

---

### Task 4: Verification — Single Model Quick Test

**Files:**
- No code changes

- [ ] **Step 1: Train single model (2 epochs)**

```bash
rm -rf model/phase10_rankic_cnn
source .venv/bin/activate && python code/src/train.py --seed 42 --output_dir ./model/phase10_rankic_cnn --num_epochs_override 2 2>&1 | tail -5
```
Expected: training completes without error

- [ ] **Step 2: Verify predict works**

```bash
source .venv/bin/activate && python code/src/predict.py 2>&1 | tail -6
python test/score_self.py 2>&1 | tail -1
```
Expected: prediction and scoring complete

- [ ] **Step 3: If passes, commit test model**

```bash
git add model/phase10_rankic_cnn/
git commit -m "test: Phase 10 single-model quick verification passes"
```

---

### Task 5: Full Ensemble Training + Scoring

**Files:**
- No code changes (revert to ensemble_size=5 via config)

- [ ] **Step 1: Restore ensemble_mode defaults in config**

Ensure config.py has `ensemble_size: 5` and `_num_epochs_override` default is 30 (already set in train.py ensemble loop).

- [ ] **Step 2: Full 5-model ensemble training**

```bash
rm -rf model/phase10_rankic_cnn
source .venv/bin/activate && python code/src/train.py 2>&1 | tail -3
```

- [ ] **Step 3: Predict + score**

```bash
source .venv/bin/activate && python code/src/predict.py 2>&1 | tail -6
python test/score_self.py 2>&1 | tail -1
```

- [ ] **Step 4: Compare with Phase 9 baseline**

```bash
echo "Phase 9 baseline: 0.0673"
echo "Phase 10: [INSERT_SCORE]"
```

- [ ] **Step 5: Commit results**

```bash
git add model/phase10_rankic_cnn/
git commit -m "results: Phase 10 SoftRankIC+CNN ensemble — score [INSERT_SCORE]"
```

---

### Task 6: Ablation — Test Each Change Independently

**Files:**
- Modify: `code/src/config.py` (temporarily)

- [ ] **Step 1: SoftRankIC only (disable CNN)**

```python
'use_cnn_features': False,
```

Train, score. Record.

- [ ] **Step 2: CNN only (disable SoftRankIC)**

```python
'use_cnn_features': True,
'use_soft_rankic_loss': False,
```

Train, score. Record.

- [ ] **Step 3: Both (re-enable)**

```python
'use_cnn_features': True,
'use_soft_rankic_loss': True,
```

Train, score. Record.

- [ ] **Step 4: Commit the winner config**

```bash
git add code/src/config.py
git commit -m "feat: Phase 10 final config after ablation"
```
