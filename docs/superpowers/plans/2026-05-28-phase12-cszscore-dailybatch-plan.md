# Phase 12: CSZScoreNorm + Per-Day Batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable 15+ epoch training without overfitting via label preprocessing (CSZScoreNorm + DropExtremeLabel) and per-day cross-sectional batching

**Architecture:** Two independent changes in train.py: (1) label preprocessing in `_build_label_and_clean`, (2) `DailyBatchSampler` replaces random shuffle. No model.py changes. AdamW optimizer (SAM disabled).

**Tech Stack:** Python 3.12, PyTorch 2.6, pandas, numpy

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `code/src/train.py` | MODIFY | CSZScoreNorm + DropExtremeLabel in label builder, DailyBatchSampler class, DataLoader changes |
| `code/src/config.py` | MODIFY | num_epochs, output_dir, disable SAM |
| No other files touched | | |

---

### Task 1: Label Preprocessing — CSZScoreNorm + DropExtremeLabel

**Files:**
- Modify: `code/src/train.py:46-67`

- [ ] **Step 1: Add CSZScoreNorm + DropExtremeLabel to `_build_label_and_clean`**

Replace the function with:

```python
def _build_label_and_clean(processed, drop_small_open=True, label_alpha=0.3):
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t3'] = processed.groupby('股票代码')['开盘'].shift(-3)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    ret_t1t3 = (processed['open_t3'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    ret_t1t5 = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['label'] = label_alpha * ret_t1t3 + (1.0 - label_alpha) * ret_t1t5
    processed['score_target'] = ret_t1t5

    # --- CSZScoreNorm + DropExtremeLabel ---
    # Per trading day: drop top/bottom 2.5% extreme labels, then z-score normalize
    processed['日期_temp'] = processed['日期'].copy()
    for date, group in processed.groupby('日期_temp'):
        day_idx = group.index
        labels = group['label'].values
        n = len(labels)
        if n < 20:  # skip days with too few stocks
            continue
        # DropExtremeLabel: mask top 2.5% and bottom 2.5%
        sorted_idx = np.argsort(labels)
        drop_n = int(0.025 * n)
        if drop_n > 0:
            drop_indices = np.concatenate([sorted_idx[:drop_n], sorted_idx[-drop_n:]])
            processed.loc[day_idx[drop_indices], 'label'] = np.nan
            processed.loc[day_idx[drop_indices], 'score_target'] = np.nan
        # CSZScoreNorm: z-score remaining labels
        valid = processed.loc[day_idx, 'label'].notna()
        if valid.sum() < 10:
            continue
        mean_val = processed.loc[day_idx[valid], 'label'].mean()
        std_val = processed.loc[day_idx[valid], 'label'].std()
        if std_val > 1e-8:
            processed.loc[day_idx[valid], 'label'] = (
                (processed.loc[day_idx[valid], 'label'] - mean_val) / std_val
            )
    processed.drop(columns=['日期_temp'], inplace=True)
    # --- End CSZScoreNorm ---

    processed = processed.dropna(subset=['label', 'score_target'])
    processed.drop(columns=['open_t1', 'open_t3', 'open_t5'], inplace=True)
    return processed
```

- [ ] **Step 2: Verify label distribution changed**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0,'code/src')
import pandas as pd
from train import _build_label_and_clean, _preprocess_common
df = pd.read_csv('data/train.csv', dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
sid2idx = {s:i for i,s in enumerate(sorted(df['股票代码'].unique()))}
raw = df.copy()
# Process one stock group to get labels
raw['open_t1'] = raw.groupby('股票代码')['开盘'].shift(-1)
raw['open_t3'] = raw.groupby('股票代码')['开盘'].shift(-3)
raw['open_t5'] = raw.groupby('股票代码')['开盘'].shift(-5)
raw = raw[raw['open_t1'] > 1e-4]
raw['label_old'] = 0.3*(raw['open_t3']-raw['open_t1'])/raw['open_t1'] + 0.7*(raw['open_t5']-raw['open_t1'])/raw['open_t1']
# New label
raw['label_new'] = raw['label_old'].copy()
raw['score_target'] = (raw['open_t5']-raw['open_t1'])/raw['open_t1']
# Apply per-day CSZScoreNorm
for date, group in raw.groupby('日期'):
    labels = group['label_new'].values; n=len(labels)
    if n<20: continue
    si=np.argsort(labels); dn=int(0.025*n)
    if dn>0:
        raw.loc[group.index[si[:dn]],'label_new']=np.nan
        raw.loc[group.index[si[-dn:]],'label_new']=np.nan
    valid=raw.loc[group.index,'label_new'].notna()
    if valid.sum()>=10:
        m=raw.loc[group.index[valid],'label_new'].mean()
        s=raw.loc[group.index[valid],'label_new'].std()
        if s>1e-8: raw.loc[group.index[valid],'label_new']=(raw.loc[group.index[valid],'label_new']-m)/s
old=raw['label_old'].dropna()
new=raw['label_new'].dropna()
print(f'Old label: mean={old.mean():.3f}, std={old.std():.3f}, min={old.min():.3f}, max={old.max():.3f}')
print(f'New label: mean={new.mean():.3f}, std={new.std():.3f}, min={new.min():.3f}, max={new.max():.3f}')
print('CSZScoreNorm applied OK')
"
```

Expected: new label has std ≈ 1.0, min/max ≈ [-3, 3]. Old label has much larger range.

- [ ] **Step 3: Commit**

```bash
git add code/src/train.py
git commit -m "feat: add CSZScoreNorm + DropExtremeLabel to label preprocessing"
```

---

### Task 2: Daily Batch Sampler

**Files:**
- Modify: `code/src/train.py`

- [ ] **Step 1: Add DailyBatchSampler class**

Insert after imports (or before the `main()` function, after line 30 or near the sampler/collate section):

```python
class DailyBatchSampler(torch.utils.data.Sampler):
    """Groups samples by trading date. Each batch = one full day of stocks.

    Ensures cross-sectional ranking signal is preserved — the model sees
    the full stock universe for each date in a single forward pass.
    """
    def __init__(self, dataset, shuffle=False):
        from collections import defaultdict
        self.shuffle = shuffle
        daily_groups = defaultdict(list)
        for i in range(len(dataset)):
            date = dataset.samples[i]['date']
            daily_groups[date].append(i)
        self.batches = list(daily_groups.values())

    def __iter__(self):
        order = torch.randperm(len(self.batches)).tolist() if self.shuffle else range(len(self.batches))
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)
```

- [ ] **Step 2: Replace DataLoader creation in `main()`**

Find `train_loader = DataLoader(train_dataset, ...)` and replace with:

```python
    train_sampler = DailyBatchSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
```

Find `val_loader = DataLoader(val_dataset, ...)` and replace with:

```python
    val_sampler = DailyBatchSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(
        val_dataset, batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
```

- [ ] **Step 3: Verify training starts with daily batches**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0,'code/src')
from config import config
config['ensemble_size'] = 1
config['output_dir'] = './model/phase12_test'
from train import main
# Just test imports and sampler construction
print('DailyBatchSampler OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add code/src/train.py
git commit -m "feat: add DailyBatchSampler for cross-sectional per-day batching"
```

---

### Task 3: Config Changes

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Set Phase 12 config values**

```python
    'use_sam': False,                    # Disable SAM (verified harmful)
    'num_epochs': 15,                    # Enough for full training
    'output_dir': './model/phase12_daily',
    'val_months': 2,
    'ensemble_size': 5,
    'use_cnn_features': False,           # Not needed
    'use_cosine_restarts': False,        # Standard cosine decay
    'gradient_accumulation_steps': 1,
    
    # Keep from Phase 8:
    'smooth_ndcg_weight': 0.6,
    'lambda_pairwise_weight': 0.4,
    'dropout': 0.35,
    'weight_decay': 5e-4,
    'use_mixup': True,
    'use_label_smoothing': True,
    'use_ema': True,
    'use_swa': True,
```

- [ ] **Step 2: Verify config**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0,'code/src')
from config import config
for k in ['use_sam','num_epochs','ensemble_size','val_months']:
    print(f'{k}={config[k]}')
print('Config OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add code/src/config.py
git commit -m "feat: Phase 12 config — AdamW, 15 epochs, daily batch, CSZScoreNorm"
```

---

### Task 4: Verification — Single Model Quick Test

**Files:**
- No code changes

- [ ] **Step 1: Train single model (5 epochs quick test)**

```bash
rm -rf model/phase12_daily
source .venv/bin/activate && python code/src/train.py --seed 42 --output_dir ./model/phase12_test --num_epochs_override 5 2>&1 | tail -5
```
Expected: training completes without error, loss is not NaN

- [ ] **Step 2: Check score distribution**

```bash
source .venv/bin/activate && python temp/check_model_dist.py 2>&1 | grep -E "(phase12|std=)"
```
Expected: model std > 0.05 (healthy, not collapsed)

- [ ] **Step 3: Commit test model**

```bash
git add model/phase12_test/
git commit -m "test: Phase 12 single-model quick verification passes"
```

---

### Task 5: Full Ensemble Training + Scoring

**Files:**
- No code changes

- [ ] **Step 1: Full 5-model ensemble × 15 epochs**

```bash
rm -rf model/phase12_daily
source .venv/bin/activate && python code/src/train.py 2>&1 | tail -3
```

- [ ] **Step 2: Predict + score**

```bash
source .venv/bin/activate && python code/src/predict.py 2>&1 | tail -6
python test/score_self.py | tail -1
```

- [ ] **Step 3: Compare with Phase 8 baseline**

```bash
echo "Phase 8 (AdamW, 2ep, random batch): 0.0673"
echo "Phase 12 (AdamW, 15ep, daily batch, CSZScoreNorm): [INSERT_SCORE]"
```

- [ ] **Step 4: Commit results**

```bash
git add model/phase12_daily/
git commit -m "results: Phase 12 CSZScoreNorm + daily batch ensemble — score [INSERT_SCORE]"
```
