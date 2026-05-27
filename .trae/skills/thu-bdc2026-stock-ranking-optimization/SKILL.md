---
name: thu-bdc2026-stock-ranking-optimization
description: Use when optimizing the THU-BDC2026 stock ranking project or similar quantitative stock selection models based on Transformer architecture
---

# THU-BDC2026 Stock Ranking Optimization

## Overview

Key optimization patterns for the StockTransformer-based stock ranking system. Focuses on bug fixes with highest impact, followed by architecture and hyperparameter improvements.

## Critical Bugs Found and Fixed

### 1. _soft_rank Self-Comparison Offset

`_soft_rank` included j=i self-comparison (sigmoid(0)=0.5), causing all ranks to shift by +0.5. Fix: subtract 0.5 from sum.

```python
# Before: return 1.0 + sig.sum(dim=2)
# After:
return 1.0 + sig.sum(dim=2) - 0.5
```

### 2. NDCG Position Calculation

After fixing _soft_rank to use standard convention (rank 1 = best), use `dcg_positions = soft_ranks` directly instead of `N - soft_ranks + 1`.

### 3. Batch Size Too Small

batch_size=1 causes extreme gradient variance for ranking losses. Increase to 4+ with proportional learning rate increase.

### 4. EMA Decay Too Aggressive

0.99 means only ~100 steps of smoothing. Use 0.999 (~1000 steps) or 0.9995.

### 5. CrossStockAttention Padding

Padding stocks affect LayerNorm statistics. Apply mask BEFORE LayerNorm, not after.

## Architecture Improvements

### Pre-LN Transformer

Use `norm_first=True` in TransformerEncoderLayer for more stable training and better convergence.

### Model Capacity

d_model=192, nhead=6, num_layers=3, dim_feedforward=384 provides good balance of capacity and speed.

## Training Improvements

### Gradient Accumulation

Use gradient_accumulation_steps=4 with batch_size=4 for effective batch size of 16.

### Multi-Horizon Labels

Combine t+3 and t+5 returns: `label = 0.3 * ret_t1t3 + 0.7 * ret_t1t5` for smoother optimization target.

### Label Smoothing

Enable with alpha=0.03 to prevent overconfidence and improve generalization.

## Feature Engineering

### Forward Fill Before Zero Fill

Use `fillna(method='ffill')` before `fillna(0)` to avoid introducing false zero signals from early NaN values in technical indicators.

### Vectorized Z-Score

Replace Python loop in `per_stock_sliding_zscore` with `pandas.rolling()` for 10-100x speedup.

## Ensemble

Use rank-based averaging instead of score averaging to handle scale differences between models:

```python
for s in all_scores:
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_scores.append(ranks / len(s))
scores = np.mean(rank_scores, axis=0)
```

## Config Summary

| Parameter | Before | After |
|-----------|--------|-------|
| d_model | 128 | 192 |
| nhead | 4 | 6 |
| num_layers | 2 | 3 |
| dim_feedforward | 256 | 384 |
| batch_size | 1 | 4 |
| learning_rate | 2e-5 | 5e-5 |
| dropout | 0.3 | 0.2 |
| gradient_accumulation | 1 | 4 |
| ema_decay | 0.99 | 0.999 |
| warmup_epochs | 3 | 5 |
| label_smoothing | False | True (0.03) |
