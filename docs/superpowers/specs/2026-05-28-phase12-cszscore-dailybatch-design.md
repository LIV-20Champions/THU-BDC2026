# Phase 12: CSZScoreNorm + Per-Day Batch Training

**Date**: 2026-05-28
**Goal**: Enable full training (15+ epochs) without overfitting by cleaning label signal and restoring cross-sectional batch structure
**Baseline**: Phase 8 — AdamW, 2 epochs, ensemble=5, self-score 0.0673

## Problem Diagnosis

After verifying that SAM causes score collapse (std 0.33→0.017), the core training problem is:

1. **Raw labels contain extreme outliers**: 5-day returns can range from -50% to +50%. These extreme values explode gradients and make SmoothNDCG's soft rank approximation unstable.
2. **Random batching destroys cross-sectional signal**: Ranking is defined "among all stocks on the same day". Training on random mini-batches of 2 samples breaks this structure.
3. **Label scale varies across days**: High-volatility days produce larger label values, dominating the loss.

MASTER (AAAI-2024) solves all three with simple preprocessing and daily batching.

## Solution

### 1. CSZScoreNorm + DropExtremeLabel

In `_build_label_and_clean`, after computing labels, for each trading day:
- **DropExtremeLabel**: Drop top 2.5% and bottom 2.5% of labels (remove 5% most extreme)
- **CSZScoreNorm**: `label = (label - mean_day) / std_day`

No architecture changes. Labels become well-behaved in [-3, 3] range with consistent scale across days.

### 2. Per-Day Batch

Replace random shuffle sampler with `DailyBatchSampler` that groups all stock samples from the same trading day into one batch. Each batch = 1 day × ~300 stocks = full cross-sectional ranking.

Implementation: New `DailyBatchSampler` class in train.py, based on MASTER's design.

### 3. Training Policy

- AdamW optimizer (no SAM — verified to cause collapse)
- 15 epochs (up from 2, enabled by clean labels)
- ensemble_size=5 (preserved)
- val_months=2 (preserved)
- early_stopping_patience=10

### Files Changed

| File | Action | Content |
|------|--------|---------|
| code/src/train.py | MODIFY | Add CSZScoreNorm + DropExtremeLabel to `_build_label_and_clean`, add `DailyBatchSampler`, update DataLoader |
| code/src/config.py | MODIFY | num_epochs default, output_dir |

### Unchanged

- model.py (no architecture changes)
- utils.py (no feature changes)
- loss functions (SmoothNDCG + pairwise preserved)
- All regularization (mixup, label_smoothing, dropout, wd)
