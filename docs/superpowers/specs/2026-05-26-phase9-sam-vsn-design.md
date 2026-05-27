# Phase 9: SAM + VSN + Feature Interaction + Cosine Restarts

**Date**: 2026-05-26
**Goal**: 收益率得分跨越式稳定提升（目标 > 0.056），不依赖 seed，仅运行 train.py + predict.py
**Baseline**: Phase 8 — 5-model ensemble, 2 epochs, AdamW, self-score 0.056

## Root Cause Analysis

After exhaustive testing across 8 phases, the fundamental bottleneck is identified:

**AdamW converges to sharp minima on small data (422 samples).** Any increase in model capacity (VSN: +300K, feature_interaction: +12K), feature dimensionality (158+39, cross-sectional), or training epochs (>2) amplifies overfitting at these sharp minima. The model doesn't learn more — it memorizes more.

Evidence: 2-epoch training + ensemble achieves 5.6%, while 30-epoch training collapses to 1.4%. All architecture features (VSN, feature_interaction, cross-sectional) degraded under AdamW.

## Solution

Replace AdamW with SAM (Sharpness-Aware Minimization) — an optimizer that finds flat minima where the loss is low across a neighborhood of parameter space. At flat minima, additional parameters and training epochs enhance signal capture without amplifying noise.

Then re-enable previously-degraded features (VSN, feature_interaction, MSE aux loss) under SAM, and add cosine warm restarts for periodic escape from suboptimal regions.

## Design

### 1. SAM Optimizer (`code/src/sam.py` — NEW)

Two-step update per batch:
1. Compute gradient g, perturb parameters w' = w + rho * g/||g||
2. Forward+backward at w', compute SAM gradient g_sam
3. Restore parameters, update with g_sam

Parameters: rho=0.05, adaptive=False
Training time: 2x (two forward+backward per batch)

### 2. VSN Variable Selection Network (model.py — enable existing)

- use_vsn: False → True
- vsn_num_groups=10, vsn_hidden_size=64, vsn_temperature=1.0
- +300K params (503K → 803K)

### 3. Feature Interaction (model.py — enable existing)

- use_feature_interaction: False → True
- interaction_hidden_dim=64
- +12K params

### 4. MSE Auxiliary Loss (train.py — enable existing)

- use_mse_aux_loss: False → True
- mse_aux_weight=0.05

### 5. Cosine Warm Restarts (train.py — new scheduler)

- T_0=10, T_mult=2, eta_min=1e-6
- Replaces single cosine decay

### Training Config

- epochs: 30, warmup: 5, early_stopping_patience: 20
- ensemble_size: 5
- All other ON features preserved: EMA, SWA, mixup, label_smoothing, AMP

### Files Changed

| File | Action |
|------|--------|
| code/src/sam.py | CREATE |
| code/src/train.py | MODIFY (optimizer, scheduler, MSE aux) |
| code/src/config.py | MODIFY (feature flags, SAM params) |
| code/src/model.py | UNCHANGED (VSN/FI already implemented) |
