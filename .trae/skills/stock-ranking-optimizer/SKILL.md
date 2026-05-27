---
name: "stock-ranking-optimizer"
description: "Optimize StockTransformer ranking model for CSI 300 stock selection competition. Invoke when tuning hyperparameters, loss functions, feature engineering, or model architecture for quant ranking tasks."
---

# Stock Ranking Optimizer

Optimization guide for the THU-BDC2026 stock ranking learning project. Covers loss function tuning, feature engineering, architecture decisions, and regularization strategies.

## Loss Function Optimization

The project uses **SmoothNDCGLoss** (arXiv:2510.14156 confirms listwise losses outperform pointwise for stock ranking) with optional LambdaRank pairwise augmentation.

### SmoothNDCG Temperature Annealing

```python
# Conservative (stable convergence):
'temperature_anneal_start': 2.0,
'temperature_anneal_target': 0.5,
'temperature_anneal_epochs': 30,

# Aggressive (faster convergence, risk of instability):
'temperature_anneal_start': 0.5,
'temperature_anneal_target': 0.1,
'temperature_anneal_epochs': 15,
```

Rule: Start conservative. Higher temperature → softer gradients → more stable but slower. Lower target → sharper NDCG approximation → better ranking but noisier gradients.

### Loss Weight Strategy

```python
# Pure SmoothNDCG (when ranking is primary goal):
'smooth_ndcg_weight': 1.0,
'lambda_pairwise_weight': 0.0,

# Hybrid (recommended for most cases):
'smooth_ndcg_weight': 0.7,
'lambda_pairwise_weight': 0.3,

# Pairwise-dominant (when sample sizes are small):
'smooth_ndcg_weight': 0.3,
'lambda_pairwise_weight': 0.7,
```

### LambdaDeltaClip

Set `lambda_delta_clip` to 10.0 to prevent gradient explosion from extreme pairwise deltas. For volatile markets, reduce to 5.0.

## Feature Engineering

### When to Enable Cross-Sectional Features

```python
'use_cross_sectional_features': True,   # STRONGLY RECOMMENDED
'cs_feature_types': ['rank_pct', 'zscore'],
```

Cross-sectional features (rank percentile and z-score within each day's stock universe) are critical for ranking tasks. They provide relative positioning information that raw features cannot capture. Disabling them typically reduces final_score by 15-25%.

### VSN (Variable Selection Network)

```python
'use_vsn': True,       # Enable when feature_num >= 100
'vsn_num_groups': 10,
'vsn_hidden_size': 64,
```

VSN adds ~10% parameters. **Enable when**: 158+39 features (197 total). **Disable when**: 39 features only (too few for group partitioning to be meaningful).

### Multi-Scale Temporal Encoder

```python
'use_multi_scale': True,    # Enable for GPU training
'ms_short_patch': 5,        # 5-day pattern detection
'ms_medium_segment': 15,    # 15-day pattern detection
```

Multi-scale adds 3 parallel encoder branches. Enables detection of patterns at different time horizons. **Trade-off**: ~2x training time, ~40% more parameters. **Enable on GPU** only; use gradient checkpointing (`use_gradient_checkpointing: True`) to reduce memory.

## Architecture Decisions

### Cross-Stock Attention Layers

```python
'cross_stock_layers': 2,     # Recommended for max_stocks >= 100
'cross_stock_layers': 1,     # Sufficient for max_stocks < 50
```

More layers → better cross-stock relationship modeling but higher computational cost. Two layers with DropPath (0.1 rate) provide good regularization.

### Feature Interaction

```python
'use_feature_interaction': True,
'interaction_hidden_dim': 64,
```

Adds a lightweight MLP after cross-stock attention to model feature-feature interactions. Minimal overhead (~1% params), consistently improves ranking quality.

## Regularization Strategy

### EMA (Exponential Moving Average)

```python
'use_ema': True,
'ema_decay': 0.99,          # 0.99 = slow update, stable. 0.999 = very slow
```

Always enable EMA. Apply shadow weights during eval (epoch >= 5). Use EMA weights for final prediction via `best_model_ema.pth`.

### SWA (Stochastic Weight Averaging)

```python
'use_swa': True,
'swa_start_epoch': 10,      # Start collecting checkpoints early
'swa_lookahead': 5,         # Also trigger when approaching early stop
```

SWA averages model checkpoints for smoother loss landscape. Start early (epoch 10) to collect more checkpoints.

### Mixup

```python
'use_mixup': True,
'mixup_alpha': 0.2,         # Beta distribution parameter
'mixup_prob': 0.3,          # Apply to 30% of batches
```

Mixup interpolates features and labels within the same day's stock universe. It helps prevent overfitting on noisy financial data but can blur ranking signals if alpha is too large.

## Training Configuration

### Learning Rate & Scheduler

```python
'learning_rate': 2e-5,       # AdamW with Transformer
'warmup_epochs': 3,
'cosine_min_lr_ratio': 0.01,
'weight_decay': 1e-5,
```

2e-5 is a safe starting point for Transformer models. For VSN+MultiScale enabled, try 3e-5. Larger models need slightly higher LR.

### Early Stopping & Patience

```python
'early_stopping_patience': 20,  # Conservative for financial data
'val_months': 3,                # Last 3 months for validation
```

Financial data is noisy; 20-epoch patience prevents premature stopping. Use 3-month validation split for more robust evaluation.

### Batch Size & Gradient Accumulation

```python
'batch_size': 1,                         # Per-day sample
'gradient_accumulation_steps': 1,
'max_stocks_per_sample': 150,            # Cap daily stocks
```

Batch size 1 = one trading day per batch. Accumulation steps > 1 simulates larger batches. `max_stocks_per_sample` randomly subsamples stocks per day to control memory.

## Prediction Weight Allocation

```python
# Equal weight (default, most robust):
'predict_weight_mode': 'equal',

# Rank decay (top stock gets higher weight):
'predict_weight_mode': 'rank_decay',
'predict_rank_alpha': 0.8,

# Softmax (score-based weights):
'predict_weight_mode': 'softmax',
'predict_weight_temperature': 0.5,
```

Equal weight is most robust. Softmax can improve returns but increases variance. Rank decay is a good middle ground.

## Ensemble Strategy

```python
'ensemble_model_dirs': [
    './model/opt_v10_s42',
    './model/opt_v10_s123',
    './model/opt_v10_s7',
],
```

Ensemble 3 models trained with different random seeds (42, 123, 7). Score averaging reduces variance. Weighted ensemble (by validation final_score) can further improve results.

## Quick Reference: Configuration Profiles

| Profile | VSN | MultiScale | CrossStock | Feature Interact | Use Case |
|---------|-----|-----------|------------|-----------------|----------|
| Lite (CPU) | Off | Off | 1 layer | On | Quick experiments |
| Balanced | On | Off | 2 layers | On | CPU training, good baseline |
| Full (GPU) | On | On | 2 layers | On | GPU training, best performance |
| Aggressive | On | On | 3 layers | On | Max capacity, risk of overfit |

## Key Papers & References

- Kwiatkowski & Chudziak (2025): "On Evaluating Loss Functions for Stock Ranking" — confirms listwise > pairwise > pointwise for stock ranking with Transformers
- LambdaRankIC (Lin et al., 2026): Direct Rank IC optimization, relevant for alternative loss design
- Sawhney et al. (AAAI 2021): STHAN-SR hypergraph for stock selection via LTR
- Alpha158 (Qlib): Industry-standard 158 technical factors for Chinese stock market