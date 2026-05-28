# Phase 10: SoftRankIC Loss + CNN Feature Extraction

**Date**: 2026-05-28
**Goal**: 突破 0.0673 信息天花板，引入可微 RankIC 损失对齐评分目标 + CNN 特征提取层增加信号维度
**Baseline**: Phase 9 — SAM, ensemble=5, self-score 0.0673

## Problem Diagnosis

After 10 phases of optimization, the score plateaus at 0.0673 regardless of feature set (39/158+39/cross-sectional), embedding (on/off), or architecture tweaks. Root cause is twofold:

1. **Loss-evaluation misalignment**: SmoothNDCG optimizes top-k NDCG, but self-score computes weighted return of top-5 stocks. These are correlated but not identical objectives.
2. **Static feature engineering**: 39 fixed TA-Lib indicators are the ceiling. Adding more fixed indicators (158+39, cross-sectional) adds noise, not signal.

## Solution

### 1. SoftRankIC Loss (train.py — NEW)

Directly optimize Spearman rank correlation between predicted scores and true labels. Spearman IC is the standard metric in quantitative finance for evaluating factor quality — optimizing it directly aligns training with the actual ranking objective.

**Implementation**: `SoftRankICLoss(nn.Module)` in train.py. Uses soft rank (sigmoid-based) to make the discrete ranking operation differentiable. Computes `1 - Spearman_R` per batch.

**Configuration**:
- `use_soft_rankic_loss: True`
- `soft_rankic_weight: 1.0`
- `soft_rankic_temperature: 0.5`
- SmoothNDCG and pairwise loss remain configurable for combination experiments

### 2. CNN Feature Extraction Layer (model.py — NEW)

Inspired by AlphaNet (Huatai Securities, 2020-2024): treat stock OHLCV+technical data as a 2D "image" and apply 1D convolutions in the time dimension to learn new temporal patterns automatically.

**Architecture**:
```
Input [B*N, 60, 39]
  ├── TA-Lib features (existing path, unchanged)
  └── CNNFeatureExtractor (NEW):
        Conv1d(39→64, k=3) + BN + ReLU
        Conv1d(64→64, k=5) + BN + ReLU
        Conv1d(64→32, k=3) + BN + ReLU
        → Output [B*N, 60, 32]
Concat → [B*N, 60, 71] → Input Proj → Transformer
```

**Key design**:
- Only ~30K additional parameters (vs VSN's 300K that degraded)
- Batch Normalization for training stability (AlphaNet's key insight)
- TA-Lib path preserved as fallback — CNN is additive, not replacement
- Convolution only in time dimension (Conv1d), preserving feature semantics

**Configuration**:
- `use_cnn_features: True`
- `cnn_feature_dim: 32`

### Files Changed

| File | Action | Content |
|------|--------|---------|
| code/src/train.py | MODIFY | Add SoftRankICLoss class, integrate with loss computation |
| code/src/model.py | MODIFY | Add CNNFeatureExtractor class, integrate with StockTransformer |
| code/src/config.py | MODIFY | Add feature flags and hyperparameters |

### Unchanged

- SAM optimizer (rho=0.02), cosine restarts, SWA, ensemble_size=5
- All existing ON features (mixup, label_smoothing, FI, MSE aux)
- Architecture parameters (d_model=128, layers=2)
- Feature engineering pipeline (39 features)
