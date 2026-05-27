---
name: "quant-code-review"
description: "Strict code review for quantitative finance projects. Invoke when reviewing stock prediction, ranking models, or financial ML code for bugs, leakage, and numerical issues."
---

# Quantitative Finance Code Review

Strict code review checklist for quantitative finance / algorithmic trading codebases. Focuses on financial data integrity, look-ahead bias prevention, numerical stability, and ranking correctness.

## 1. Data Leakage & Look-Ahead Bias

### Label Construction
- [ ] Verify labels use ONLY future information, never current or past
- [ ] Check `groupby('股票代码').shift(-N)` operations — confirm negative shift direction
- [ ] Confirm no feature column derives from future data (e.g., future returns as features)

```python
# CORRECT: Label uses future open price shifted backward
processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)

# WRONG: Using current close to predict current close (data leakage)
processed['label'] = processed['收盘'] / processed['开盘']
```

### Train/Validation Split
- [ ] Validation split respects chronological order (no future data in training)
- [ ] Sequence context correctly handled: validation window includes `sequence_length` days before split point
- [ ] `min_window_end_date` parameter correctly filters samples in both training and validation

### Feature Normalization
- [ ] Normalization is computed ONLY on training data
- [ ] Validation/test data uses training statistics (StandardScaler) or per-stock sliding window
- [ ] Sliding window z-score uses correct lookback window (`t - sequence_length + 1` to `t`)

## 2. Label & Loss Function Correctness

### Relevance/Label Construction
- [ ] Labels have consistent scale (verify no extreme outliers dominating training)
- [ ] Relevance scores in LTR are monotonically related to returns
- [ ] Check for `inf` or `NaN` in labels after construction

### SmoothNDCGLoss Verification
- [ ] Temperature annealing starts higher than target (soft → sharp)
- [ ] `soft_sort_temperature` is in valid range [0.05, 5.0]
- [ ] DCG gains use `2^relevance - 1` formula (standard)
- [ ] IDCG computed correctly with sorted ground-truth gains
- [ ] Discount factor uses `log2(rank + 1)`, not `log2(rank)` or `log2(rank + 2)`

### LambdaRank Pairwise Loss Verification
- [ ] Delta NDCG computation is symmetric and correct
- [ ] Pairwise mask excludes equal-relevance pairs
- [ ] Gradient clipping (`delta_clip`) is applied

## 3. Numerical Stability

### Division Safety
- [ ] All divisions have epsilon guard (`+ 1e-12`, not just `+ 1e-8`)
- [ ] Logarithm inputs are positive (add epsilon before log)
- [ ] Softmax inputs are shifted by max value for stability

```python
# CORRECT
x = x / (denom + 1e-12)
log_x = torch.log(x + 1e-8)
shifted = scores - scores.max()
softmax_out = torch.softmax(shifted / temperature, dim=-1)

# DANGEROUS
x = x / denom  # denom could be zero
log_x = torch.log(x)  # x could be zero or negative
```

### Inf/NaN Handling
- [ ] Feature engineering output checked: `replace([np.inf, -np.inf], np.nan).fillna(0)`
- [ ] Model output checked for NaN before loss computation
- [ ] Gradient clipping enabled (`max_grad_norm: 5.0`)

### Float Precision
- [ ] Training uses float32; feature storage may use float16 for memory
- [ ] Loss computation uses float32 (mixed precision may cause underflow)
- [ ] Soft ranks use `float64` internally if `temperature < 0.1`

## 4. Model Architecture Checks

### Input Shape Validation
- [ ] `input_dim` calculation consistent with config (`get_eff_input_dim`)
- [ ] `stock_embedding` dimension matches `instrument` column usage
- [ ] Cross-sectional features correctly expand input dimensions

### Mask Handling
- [ ] Padding mask correctly applied in attention layers
- [ ] Mask broadcast dimensions are correct (batch, num_stocks)
- [ ] Loss computation excludes padded stocks via mask
- [ ] `stock_indices` with value `-1` (padding) handled correctly with `clamp(min=0)`

### Gradient Flow
- [ ] Check that all model branches receive gradients (no dead branches)
- [ ] `DropPath` has `training` mode check
- [ ] `LayerScale` initialized with small values (1e-5) for stable training start

## 5. Training Loop Robustness

### Gradient Accumulation
- [ ] Loss divided by `(batch_size * accumulation_steps)` before backward
- [ ] Optimizer step only called after `accumulation_steps` batches
- [ ] Metrics averaged correctly across accumulation steps

### Batch Loss Handling
- [ ] Handle case where all stocks in batch are invalid (mask all zeros)
- [ ] `batch_loss is None` path has fallback (zero tensor)
- [ ] Metric computation skips batches with < k valid stocks

### Mixed Precision (AMP)
- [ ] `GradScaler` properly used: `scaler.scale(loss).backward()` then `scaler.step(optimizer)`
- [ ] Gradient clipping uses `scaler.unscale_(optimizer)` before `clip_grad_norm_`
- [ ] AMP autocast context correctly wraps forward pass only

## 6. Prediction Pipeline Checks

### Feature Consistency
- [ ] Same feature columns used in training and inference
- [ ] Same normalization method (per-stock or StandardScaler)
- [ ] Same cross-sectional feature computation

### Sort Order
- [ ] Scores sorted descending for top-k selection
- [ ] Stock IDs correctly mapped through `stockid2idx` and back

### Weight Allocation
- [ ] Weights sum to 1.0 (within tolerance)
- [ ] No negative weights
- [ ] `_normalize_weights` handles edge cases (single stock, all zero)

## 7. Memory & Performance

### Feature Storage
- [ ] Use float16 for feature store when `use_per_stock_norm=True`
- [ ] Clear intermediate DataFrames after dataset construction (`del`, `gc.collect()`)
- [ ] `LazyRankingDataset` avoids storing full sequences in memory

### DataLoader
- [ ] `num_workers` optimized for available CPU cores (2-4)
- [ ] `pin_memory=True` on GPU, `False` on CPU
- [ ] `persistent_workers=True` when `num_workers > 0`

## 8. Docker & Reproducibility

### Determinism
- [ ] Random seeds set for Python, NumPy, PyTorch, CUDA
- [ ] `cudnn.deterministic = True`, `cudnn.benchmark = False`
- [ ] Config snapshot saved alongside model checkpoint

### Dependency Locking
- [ ] `pyproject.toml` or `requirements.txt` with exact versions
- [ ] TA-Lib system library version documented
- [ ] CUDA version compatible with Docker base image

## Quick Audit Commands

```bash
# Check for data leakage patterns in labels
grep -rn "shift.*-1\|shift.*[0-9]" code/src/

# Check for division without epsilon
grep -rn "/ (.*)" code/src/ | grep -v "1e-12\|1e-8\|1e-6\|eps"

# Check for NaN/Inf handling
grep -rn "fillna\|replace.*inf\|nan_to_num" code/src/

# Verify seed setting
grep -rn "seed\|deterministic" code/src/
```