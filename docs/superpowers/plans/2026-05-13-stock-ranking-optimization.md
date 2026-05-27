# Stock Ranking Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize the THU-BDC2026 stock ranking project for higher returns with reasonable training time.

**Architecture:** Bug fixes (NDCG loss, batch size, EMA) + hyperparameter tuning + gradient accumulation + Pre-LN Transformer + improved label construction.

**Tech Stack:** PyTorch, TA-Lib, scikit-learn, pandas

---

## Completed Bug Fixes (Already Applied)

- [x] Fix `_soft_rank` self-comparison offset (train.py)
- [x] Fix NDCG position calculation to use standard convention (train.py)
- [x] Increase batch_size from 1 to 4, learning rate from 2e-5 to 5e-5, reduce dropout from 0.3 to 0.2 (config.py)
- [x] Fix EMA decay from 0.99 to 0.999 (config.py)
- [x] Fix `_init_weights` to skip Transformer internal layers (model.py)
- [x] Fix `fillna(0)` to use forward-fill first (utils.py)
- [x] Vectorize `per_stock_sliding_zscore` using pandas rolling (utils.py)
- [x] Fix CrossStockAttention padding before LayerNorm (model.py)

---

### Task 1: Add gradient accumulation for stable training

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Set gradient_accumulation_steps to 4**

Change `'gradient_accumulation_steps': 1` to `'gradient_accumulation_steps': 4` in config.py.

This gives an effective batch size of 4 * 4 = 16, which is much more stable for ranking loss computation.

---

### Task 2: Switch to Pre-LN Transformer for better training stability

**Files:**
- Modify: `code/src/model.py`

- [ ] **Step 1: Add norm_first=True to TransformerEncoderLayer**

In model.py, change all `nn.TransformerEncoderLayer` instantiations to include `norm_first=True`:
- In `StockTransformer.__init__` (the single Transformer path)
- In `MultiScaleTemporalEncoder._make_encoder`

Pre-LN (norm_first=True) is the modern standard for Transformers - it puts LayerNorm before attention/FFN instead of after, which leads to much more stable training and better convergence.

---

### Task 3: Improve label construction with multi-horizon returns

**Files:**
- Modify: `code/src/train.py`

- [ ] **Step 1: Modify `_build_label_and_clean` to use weighted multi-horizon returns**

Change the label from single-horizon `(open_t5 - open_t1) / open_t1` to a weighted combination of t+1 through t+5 returns. This provides a smoother optimization target that considers short-term momentum.

---

### Task 4: Increase model capacity moderately

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Increase d_model and dim_feedforward**

Change:
- `d_model`: 128 → 192
- `dim_feedforward`: 256 → 384
- `nhead`: 4 → 6 (to keep head_dim=32)
- `num_layers`: 2 → 3

This gives the model more capacity to capture complex stock patterns without being too large.

---

### Task 5: Add label smoothing for ranking

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Enable label smoothing**

Change:
- `use_label_smoothing`: False → True
- `label_smoothing_alpha`: 0.05 → 0.03

Label smoothing prevents the model from becoming overconfident and improves generalization.

---

### Task 6: Improve ensemble with rank-based averaging

**Files:**
- Modify: `code/src/predict.py`

- [ ] **Step 1: Add rank normalization before averaging ensemble scores**

Before averaging scores from multiple models, convert each model's scores to ranks (percentiles), then average the ranks. This handles scale differences between models.

---

### Task 7: Optimize training schedule

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Adjust training hyperparameters**

Change:
- `warmup_epochs`: 3 → 5
- `early_stopping_patience`: 20 → 15
- `num_epochs`: 60 → 40 (with early stopping, more epochs just waste time)
- `swa_start_epoch`: 10 → 15

---

### Task 8: Verify all changes work together

**Files:**
- All modified files

- [ ] **Step 1: Run a quick syntax check on all modified files**
- [ ] **Step 2: Verify config consistency between train.py and predict.py**
- [ ] **Step 3: Ensure model architecture changes are reflected in predict.py**
