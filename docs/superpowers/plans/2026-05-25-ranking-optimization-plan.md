# Ranking Model Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 稳定提升 ranking model 收益率得分 (official_score_eq)，从 0.014562 提升到 0.020+，训练时间控制在 1.5 分钟以内

**Architecture:** Two-phase progressive optimization. Phase 1 reduces overfitting via feature selection, label fix, model downsizing, and regularization — no architecture changes. Phase 2 enables lightweight structured modules (VSN, feature interaction) on the stabilized foundation.

**Tech Stack:** Python 3.12, PyTorch, TA-Lib, pandas, scikit-learn, uv

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `code/src/feature_selection.py` | CREATE | Mutual information feature ranking utility |
| `code/src/config.py` | MODIFY | Phase 1 parameter changes, then Phase 2 |
| `code/src/train.py` | MODIFY | Label construction (`_build_label_and_clean`) |
| `code/src/utils.py` | MODIFY | `LazyRankingDataset` selected_features support |

---

## Phase 1 — Data & Regularization

### Task 1: Feature Selection Utility

**Files:**
- Create: `code/src/feature_selection.py`

- [ ] **Step 1: Create feature_selection.py**

```python
"""Feature selection via mutual information with label."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def select_features_by_mi(df, feature_cols, label_col='label', top_k=80, random_state=42):
    """Rank features by mutual information with label, return top_k feature names and full ranking.

    Args:
        df: DataFrame with features and label column
        feature_cols: list of feature column names to rank
        label_col: name of label column
        top_k: number of top features to retain
        random_state: seed for MI estimator

    Returns:
        selected: list of top_k feature names in descending MI order
        ranking: list of (feature_name, mi_score) tuples, all features ranked
    """
    X = df[feature_cols].fillna(0.0)
    y = df[label_col]

    mi_scores = mutual_info_regression(X.values, y.values, random_state=random_state)
    ranked = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ranked[:top_k]]
    return selected, ranked


def print_feature_ranking(ranking):
    """Print ranked features with MI scores."""
    print(f"\n{'Rank':<6}{'Feature':<30}{'MI Score':<12}")
    print("-" * 48)
    for i, (name, score) in enumerate(ranking, 1):
        print(f"{i:<6}{name:<30}{score:<12.6f}")


def save_feature_ranking(ranking, output_path):
    """Save feature ranking to CSV."""
    df = pd.DataFrame(ranking, columns=['feature', 'mi_score'])
    df.index.name = 'rank'
    df.index += 1
    df.to_csv(output_path)
    print(f"Feature ranking saved to {output_path}")
```

- [ ] **Step 2: Commit**

```bash
git add code/src/feature_selection.py
git commit -m "feat: add mutual information feature selection utility"
```

---

### Task 2: Label Construction Fix

**Files:**
- Modify: `code/src/train.py:45-59`

- [ ] **Step 1: Replace `_build_label_and_clean` in train.py**

Old (lines 45-59):
```python
def _build_label_and_clean(processed, drop_small_open=True):
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t3'] = processed.groupby('股票代码')['开盘'].shift(-3)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    ret_t1t3 = (processed['open_t3'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    ret_t1t5 = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['label'] = 0.3 * ret_t1t3 + 0.7 * ret_t1t5
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t3', 'open_t5'], inplace=True)
    return processed
```

New:
```python
def _build_label_and_clean(processed, drop_small_open=True, label_alpha=0.3):
    """Build label from future open returns, using close as base price.

    label = alpha * ret(close -> open_t3) + (1-alpha) * ret(close -> open_t5)
    where ret(A -> B) = (B - A) / A

    Using close (observable at prediction time) instead of open_t1 (unobservable).
    Default alpha=0.3 matches original weighting (70% weight on t5 horizon).
    """
    processed['open_t3'] = processed.groupby('股票代码')['开盘'].shift(-3)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    ret_close_t3 = (processed['open_t3'] - processed['收盘']) / (processed['收盘'] + 1e-12)
    ret_close_t5 = (processed['open_t5'] - processed['收盘']) / (processed['收盘'] + 1e-12)
    processed['label'] = label_alpha * ret_close_t3 + (1.0 - label_alpha) * ret_close_t5
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t3', 'open_t5'], inplace=True)
    return processed
```

- [ ] **Step 2: Update call sites in `_preprocess_common`**

In `train.py:87`, the call `_build_label_and_clean(processed, drop_small_open=drop_small_open)` needs to pass `label_alpha`.

Read the relevant lines and verify the change is at line 87:

```python
# Old (line 87):
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)

# New:
    label_alpha = float(config.get('label_alpha', 0.3))
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open, label_alpha=label_alpha)
```

- [ ] **Step 3: Add `label_alpha` to config.py**

Add to `config` dict in `config.py`:
```python
    'label_alpha': 0.3,
```

- [ ] **Step 4: Commit**

```bash
git add code/src/train.py code/src/config.py
git commit -m "fix: change label base price from open_t1 to close, add label_alpha config"
```

---

### Task 3: LazyRankingDataset Selected Features Support

**Files:**
- Modify: `code/src/utils.py:391-395`

- [ ] **Step 1: Add `selected_features` parameter to `LazyRankingDataset.__init__`**

In `utils.py`, change the `__init__` signature and feature filtering logic.

Old (line 391-395):
```python
    def __init__(self, data, features, sequence_length, min_window_end_date=None,
                 use_per_stock_norm=True, use_cs_features=False,
                 cs_feature_types=None):
        super().__init__()
        self.features = list(features)
```

New:
```python
    def __init__(self, data, features, sequence_length, min_window_end_date=None,
                 use_per_stock_norm=True, use_cs_features=False,
                 cs_feature_types=None, selected_features=None):
        super().__init__()
        full_features = list(features)
        if selected_features is not None:
            # keep only selected features that exist in the full feature list
            selected_set = set(selected_features)
            self.features = [f for f in full_features if f in selected_set]
            if 'instrument' in full_features and 'instrument' not in selected_set:
                self.features = ['instrument'] + self.features
            dropped = len(full_features) - len(self.features)
            print(f"Feature selection: {len(full_features)} -> {len(self.features)} ({dropped} dropped)")
        else:
            self.features = full_features
```

- [ ] **Step 2: Commit**

```bash
git add code/src/utils.py
git commit -m "feat: add selected_features parameter to LazyRankingDataset"
```

---

### Task 4: Phase 1 Config Changes

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Apply all Phase 1 config changes**

In `config.py`, change the following lines in the `config` dict:

```python
# Model downsizing:
    'd_model': 96,                # was 128
    'num_layers': 1,              # was 2
    'dim_feedforward': 192,       # was 256

# Stronger regularization:
    'dropout': 0.5,               # was 0.35
    'weight_decay': 1e-3,         # was 5e-4

# Enable mixup (data augmentation):
    'use_mixup': True,            # was False
    'mixup_alpha': 0.2,           # unchanged
    'mixup_prob': 0.3,            # unchanged

# Stronger label smoothing:
    'label_smoothing_alpha': 0.05, # was 0.03

# More training data:
    'val_months': 2,              # was 3

# New: label alpha (from Task 2)
    'label_alpha': 0.3,

# New: feature selection (top_k=0 means off, >0 means on)
    'selected_top_k_features': 80,
```

- [ ] **Step 2: Wire feature selection into train.py's `main()`**

In `train.py`, after the `feature_columns` are determined (around line 626-627 where `train_data, features = preprocess_data(...)` is called), add feature selection logic:

Find these lines in `main()`:
```python
    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_data(val_df, is_train=False, stockid2idx=stockid2idx)
```

Insert after them:
```python
    # Feature selection via mutual information (if configured)
    selected_top_k = int(config.get('selected_top_k_features', 0))
    selected_features_list = None
    if selected_top_k > 0 and selected_top_k < len(features):
        from feature_selection import select_features_by_mi, print_feature_ranking, save_feature_ranking
        selected_features_list, ranking = select_features_by_mi(
            train_data, [f for f in features if f != 'instrument'],
            label_col='label', top_k=selected_top_k
        )
        print_feature_ranking(ranking)
        save_feature_ranking(ranking, os.path.join(output_dir, 'feature_ranking.csv'))
        train_data = train_data[['instrument', 'label'] + selected_features_list]
        val_data = val_data[['instrument', 'label'] + selected_features_list]
        features = ['instrument'] + selected_features_list
```

Then update `LazyRankingDataset` calls to pass `selected_features`:
```python
    train_dataset = LazyRankingDataset(
        train_data, features, config['sequence_length'], min_window_end_date=None,
        use_per_stock_norm=use_per_stock_norm,
        use_cs_features=use_cs_features,
        cs_feature_types=cs_feature_types,
        selected_features=selected_features_list,
    )
    # ... same for val_dataset
    val_dataset = LazyRankingDataset(
        val_data, features, config['sequence_length'],
        min_window_end_date=val_start.strftime('%Y-%m-%d'),
        use_per_stock_norm=use_per_stock_norm,
        use_cs_features=use_cs_features,
        cs_feature_types=cs_feature_types,
        selected_features=selected_features_list,
    )
```

- [ ] **Step 3: Commit**

```bash
git add code/src/config.py code/src/train.py
git commit -m "feat: Phase 1 config — model downsizing, mixup, feature selection, label_alpha"
```

---

### Task 5: Phase 1 Verification — Label Alpha Sweep

**Files:**
- No code changes, run training with different `label_alpha` values

- [ ] **Step 1: Run training with label_alpha=0.7 (t3-heavy, 70% short-term)**

```bash
cd code/src && python train.py --seed 42 --output_dir ../../model/phase1_alpha07_s42
```
Expected: training completes, record the `official_score_eq` from `final_score.txt`

- [ ] **Step 2: Run training with label_alpha=0.5**

Edit `config.py` temporarily: `'label_alpha': 0.5`

```bash
cd code/src && python train.py --seed 42 --output_dir ../../model/phase1_alpha05_s42
```

- [ ] **Step 3: Run training with label_alpha=0.3**

Edit `config.py` temporarily: `'label_alpha': 0.3`

```bash
cd code/src && python train.py --seed 42 --output_dir ../../model/phase1_alpha03_s42
```

- [ ] **Step 4: Compare scores and select best label_alpha**

```bash
for d in ../../model/phase1_alpha*/; do echo "$d: $(cat $d/final_score.txt | grep 'Best score')"; done
```

Set the best `label_alpha` back in `config.py` and commit:
```bash
git add code/src/config.py
git commit -m "feat: set label_alpha to best value from sweep"
```

---

### Phase 1 Checkpoint

Stop here and verify: Phase 1 score should be ≥ 0.017 (improvement over baseline 0.0146). If not, investigate before proceeding to Phase 2.

---

## Phase 2 — Model Upgrade

### Task 6: Enable VSN + Feature Interaction

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Toggle Phase 2 flags in config.py**

```python
# Enable VSN:
    'use_vsn': True,              # was False
    'vsn_num_groups': 10,
    'vsn_hidden_size': 64,
    'vsn_temperature': 1.0,

# Enable feature interaction:
    'use_feature_interaction': True,  # was False
    'interaction_hidden_dim': 64,

# Adjust learning rate (smaller model tolerates higher LR):
    'learning_rate': 8e-5,        # was 5e-5
    'warmup_epochs': 3,           # was 5
```

- [ ] **Step 2: Commit**

```bash
git add code/src/config.py
git commit -m "feat: Phase 2 — enable VSN, feature interaction, adjust LR"
```

---

### Task 7: Phase 2 Verification — Loss Weight Sweep

**Files:**
- No code changes, run training with different loss weight combos

- [ ] **Step 1: Run with default loss weights (smooth_ndcg=0.7, pairwise=0.3)**

```bash
cd code/src && python train.py --seed 42 --output_dir ../../model/phase2_loss73_s42
```
Record `official_score_eq`.

- [ ] **Step 2: Run with balanced weights (smooth_ndcg=0.6, pairwise=0.4)**

```bash
# Edit config.py: smooth_ndcg_weight=0.6, lambda_pairwise_weight=0.4
cd code/src && python train.py --seed 42 --output_dir ../../model/phase2_loss64_s42
```

- [ ] **Step 3: Run with equal weights (smooth_ndcg=0.5, pairwise=0.5)**

```bash
# Edit config.py: smooth_ndcg_weight=0.5, lambda_pairwise_weight=0.5
cd code/src && python train.py --seed 42 --output_dir ../../model/phase2_loss55_s42
```

- [ ] **Step 4: Compare and pick best**

```bash
for d in ../../model/phase2_loss*/; do echo "$d: $(cat $d/final_score.txt | grep 'Best score')"; done
```

Set best weights in `config.py` and commit.

---

### Task 8: Prediction Weight Tuning

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Test steeper decay**

In `config.py`, temporarily change:
```python
    'predict_rank_weights': [0.40, 0.25, 0.18, 0.11, 0.06],
    'predict_rank_alpha': 0.65,
```

Run:
```bash
cd code/src && python train.py --seed 42 --output_dir ../../model/phase2_wt_steep_s42
```

- [ ] **Step 2: Test flatter decay**

```python
    'predict_rank_weights': [0.25, 0.22, 0.20, 0.18, 0.15],
    'predict_rank_alpha': 0.9,
```

Run:
```bash
cd code/src && python train.py --seed 42 --output_dir ../../model/phase2_wt_flat_s42
```

- [ ] **Step 3: Compare and pick best**

Set best weights in `config.py` and commit.

---

### Task 9: Final Verification — Multi-Seed Ensemble

**Files:**
- No code changes

- [ ] **Step 1: Train with 3 different seeds**

```bash
for seed in 42 123 7; do
    cd code/src && python train.py --seed $seed --output_dir ../../model/phase2_final_s${seed}
done
```

- [ ] **Step 2: Compare results across seeds**

```bash
for d in ../../model/phase2_final_s*/; do echo "$d: $(cat $d/final_score.txt | grep 'Best score')"; done
```

- [ ] **Step 3: Run prediction with best model**

```bash
cd code/src && python predict.py
```

Verify `output/result.csv` format is correct (5 stock_ids, weights sum to 1).

---

### Task 10: Final Commit

```bash
git add -A
git commit -m "feat: complete Phase 2 optimization — VSN, feature interaction, tuned loss weights and predict weights"
```
