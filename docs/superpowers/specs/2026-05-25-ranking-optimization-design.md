# Ranking Model Optimization Design

**Date**: 2026-05-25
**Goal**: 稳定提升收益率得分 (official_score_eq)，不大幅增加训练时间
**Baseline**: opt_v11_optimized, best_score=0.014562 (epoch 2/80)

## Problem Diagnosis

Root cause: severe overfitting. 3.1M parameters on 422 training samples, model peaks at epoch 2.

Three bottlenecks:
1. Model capacity >> data size → rapid overfitting
2. 158-dim features fed without selection → noise interference
3. All advanced modules disabled → lack of structured inductive bias, counterintuitively worsens overfitting

## Solution: Two-Phase Progressive Optimization

### Phase 1 — Data & Regularization (foundation)

Goal: reduce overfitting without increasing training time.

1. **Feature selection** (158 → top 60-80 by mutual information with label)
   - New file: `code/src/feature_selection.py`
   - Run once offline, generate feature ranking
   - `LazyRankingDataset` supports column subset filtering
   - Does NOT modify `feature_engineer_func_map` logic — filtering happens at dataset construction

2. **Label optimization** — change base price from `open_t1` (unobservable) to `close` (observable):
   - `ret_close→t3 = (open_t3 - close) / close`
   - `ret_close→t5 = (open_t5 - close) / close`
   - Weight sweep `alpha * ret_close→t3 + (1-alpha) * ret_close→t5` for `alpha ∈ {0.3, 0.5, 0.7}`

3. **Regularization strengthening**:
   - d_model: 128 → 96, dim_feedforward: 256 → 192, num_layers: 2 → 1
   - dropout: 0.35 → 0.5, weight_decay: 5e-4 → 1e-3
   - Enable mixup (use_mixup=True, alpha=0.2, prob=0.3)
   - label_smoothing_alpha: 0.03 → 0.05

4. **Validation**: val_months 3 → 2 (more training data, ensure ≥30 val samples)

**Phase 1 config changes to config.py**:
```
feature_num: '39'  # unchanged, maps to engineer_features_158
use_mixup: False → True
d_model: 128 → 96
dim_feedforward: 256 → 192
num_layers: 2 → 1
dropout: 0.35 → 0.5
weight_decay: 5e-4 → 1e-3
label_smoothing_alpha: 0.03 → 0.05
val_months: 3 → 2
```

### Phase 2 — Model Upgrade & Fine-tuning

Goal: add structured modules on top of Phase 1's stable foundation.

1. **Enable VSN (Variable Selection Network)**
   - `use_vsn: False → True`
   - vsn_num_groups=10, vsn_hidden_size=64, vsn_temperature=1.0
   - Adds ~180K params; with Phase 1 reduction, total ~1.7M (still < original 3.1M)

2. **Enable Feature Interaction**
   - `use_feature_interaction: False → True`
   - interaction_hidden_dim=64
   - Adds ~12K params, negligible

3. **Loss weight tuning** — grid search over (smooth_ndcg, pairwise):
   - {(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)}
   - Select best on Phase 2 validation

4. **Prediction weight tuning** — test rank_decay variants:
   - Steeper: [0.40, 0.25, 0.18, 0.11, 0.06]
   - Flatter: [0.25, 0.22, 0.20, 0.18, 0.15]
   - Current: [0.30, 0.25, 0.20, 0.15, 0.10]

5. **LR adjustment** (smaller model tolerates higher LR):
   - lr: 5e-5 → 8e-5
   - warmup_epochs: 5 → 3

### Parameter Summary

| Parameter | Original | Phase 1 | Phase 2 |
|-----------|----------|---------|---------|
| d_model | 128 | 96 | 96 |
| num_layers | 2 | 1 | 1 |
| dim_feedforward | 256 | 192 | 192 |
| dropout | 0.35 | 0.5 | 0.5 |
| weight_decay | 5e-4 | 1e-3 | 1e-3 |
| use_vsn | False | False | True |
| use_feature_interaction | False | False | True |
| use_mixup | False | True | True |
| learning_rate | 5e-5 | 5e-5 | 8e-5 |
| warmup_epochs | 5 | 5 | 3 |
| label_smoothing_alpha | 0.03 | 0.05 | 0.05 |
| val_months | 3 | 2 | 2 |
| Params | 3.1M | ~1.5M | ~1.7M |

### Expected Outcome

Phase 1: score 0.0146 → 0.017-0.019, training time -20%
Phase 2: score → 0.020-0.023, training time similar to Phase 1 or +10%

### Files to Modify

- `code/src/config.py` — parameter changes (both phases)
- `code/src/train.py` — label construction logic in `_build_label_and_clean()`
- `code/src/utils.py` — `LazyRankingDataset` feature subset support
- `code/src/feature_selection.py` — NEW, mutual information feature ranking

### Unchanged

- `feature_num='39'` mapping to `engineer_features_158` (preserved)
- MultiScale, GRU dual path, cross_sectional_features remain OFF (heavy compute)
- EMA, SWA, AMP, gradient_checkpointing remain ON
- SmoothNDCG loss backbone unchanged
- `predict_weight_mode='rank_decay'` preserved
