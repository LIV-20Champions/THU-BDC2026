# Phase 3: Robustness Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 降低跨 seed 方差，通过扩大验证集 + 时间序列 CV 找到稳健的 best epoch，最终 self-score ≥ 0.020

**Architecture:** Phase 3a 单独测试 val_months=3，Phase 3b 新增 cross_validate.py 做 3-fold expanding window CV，用 CV 确定 best epoch 后全量训练最终模型

**Tech Stack:** Python 3.12, PyTorch, pandas

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `code/src/config.py` | MODIFY | val_months 调整 |
| `code/src/train.py` | MODIFY | 新增 --train_end_date / --val_start_date CLI 参数 |
| `code/src/cross_validate.py` | CREATE | 3-fold 时间序列 CV 脚本 |

---

### Task 1: Phase 3a — 测试 val_months=3

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: 设置 val_months=3 并训练**

```bash
# Edit config.py: val_months: 2 → 3
source .venv/bin/activate && python code/src/train.py --seed 42 --output_dir ./model/phase3a_val3_s42
```

- [ ] **Step 2: 记录结果**

查看 `model/phase3a_val3_s42/final_score.txt`，与 val_months=2 的 0.02783 对比。若 ≥ 0.025 则采纳，否则保持 val_months=2。

- [ ] **Step 3: 提交最终 val_months 值**

```bash
git add code/src/config.py
git commit -m "feat: Phase 3a — val_months adjusted based on test result"
```

---

### Task 2: 为 train.py 添加 CLI 参数支持自定义切分日期

**Files:**
- Modify: `code/src/train.py:869-880`

- [ ] **Step 1: 在 argparse 和 main() 中添加参数**

当前 `main()` 函数签名和 argparse 位置（~line 869-880）：

```python
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        config['seed'] = args.seed
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    best_score = main()
```

新增参数后：

```python
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--train_end_date', type=str, default=None)
    parser.add_argument('--val_start_date', type=str, default=None)
    parser.add_argument('--num_epochs_override', type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        config['seed'] = args.seed
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.train_end_date is not None:
        config['_train_end_date'] = args.train_end_date
    if args.val_start_date is not None:
        config['_val_start_date'] = args.val_start_date
    if args.num_epochs_override is not None:
        config['_num_epochs_override'] = args.num_epochs_override
    best_score = main()
```

- [ ] **Step 2: 修改 `main()` 中的数据切分逻辑**

在 `main()` 中，`split_train_val_by_last_n_months` 调用处（约 line 613），添加对自定义日期的支持：

```python
    # 支持自定义切分日期（用于 CV）
    train_end_date_override = config.get('_train_end_date')
    val_start_date_override = config.get('_val_start_date')
    if train_end_date_override and val_start_date_override:
        train_df = full_df[full_df['日期'] <= train_end_date_override].copy()
        val_start = pd.to_datetime(val_start_date_override)
        # 验证集需要 sequence_length 天上下文用于特征归一化和窗口构建
        val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)
        val_df = full_df[
            (full_df['日期'] >= val_context_start.strftime('%Y-%m-%d'))
        ].copy()
        print(f"自定义切分: train <= {train_end_date_override}, val >= {val_context_start.date()}")
    else:
        train_df, val_df, val_start = split_train_val_by_last_n_months(
            full_df, config['sequence_length'], val_months
        )
```

**注意**: 当使用自定义切分时，需要确认 train_df 和 val_df 不为空。

- [ ] **Step 3: 支持 num_epochs_override（关闭 early stopping）**

在 `main()` 的训练循环中（约 line 776），使用 override：

```python
    num_epochs = int(config.get('_num_epochs_override', config['num_epochs']))
    early_stopping_patience = int(config.get('early_stopping_patience', 15))
    if config.get('_num_epochs_override') is not None:
        early_stopping_patience = num_epochs  # 禁用 early stopping
```

- [ ] **Step 4: 提交**

```bash
git add code/src/train.py
git commit -m "feat: add --train_end_date, --val_start_date, --num_epochs_override CLI args for CV"
```

---

### Task 3: 创建 cross_validate.py

**Files:**
- Create: `code/src/cross_validate.py`

- [ ] **Step 1: 创建 CV 脚本**

```python
"""3-fold expanding window cross-validation for ranking model.

Usage: python code/src/cross_validate.py

Determines the best epoch by averaging val scores across 3 time-based folds,
then trains a final model on all data for that many epochs.
"""

import subprocess
import sys
import os
import json
import numpy as np
import pandas as pd

def run_training(seed, output_dir, train_end, val_start, num_epochs):
    """Run one training fold. Returns list of per-epoch val official_score_eq."""
    cmd = [
        sys.executable, 'code/src/train.py',
        '--seed', str(seed),
        '--output_dir', output_dir,
        '--train_end_date', train_end,
        '--val_start_date', val_start,
        '--num_epochs_override', str(num_epochs),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    scores = []
    for line in result.stdout.split('\n'):
        if 'Eval official_score_eq:' in line and '保存' not in line:
            try:
                score = float(line.split(':')[-1].strip())
                scores.append(score)
            except ValueError:
                continue
    return scores

def main():
    # Load data to determine date ranges
    train_file = 'data/train.csv'
    df = pd.read_csv(train_file, dtype={'股票代码': str})
    df['日期'] = pd.to_datetime(df['日期'])
    all_dates = sorted(df['日期'].unique())
    
    # 3-fold expanding window
    # Fold 1: train up to 2025-10, val 2025-11 ~ 2025-12
    # Fold 2: train up to 2025-12, val 2026-01 ~ 2026-01
    # Fold 3: train up to 2026-01, val 2026-02 ~ 2026-03
    
    fold_splits = [
        ('2025-10-31', '2025-11-01'),
        ('2025-12-31', '2026-01-01'),
        ('2026-01-31', '2026-02-01'),
    ]
    
    num_epochs = 60
    seed = 42
    all_fold_scores = []
    
    for i, (train_end, val_start) in enumerate(fold_splits):
        output_dir = f'./model/phase3_cv_fold{i+1}_s{seed}'
        print(f"\n=== Fold {i+1}: train <= {train_end}, val >= {val_start} ===")
        scores = run_training(seed, output_dir, train_end, val_start, num_epochs)
        print(f"Fold {i+1} scores: {[f'{s:.4f}' for s in scores]}")
        all_fold_scores.append(scores)
    
    # Align by epoch and compute average
    min_len = min(len(s) for s in all_fold_scores)
    avg_scores = []
    for epoch in range(min_len):
        avg = np.mean([s[epoch] for s in all_fold_scores])
        avg_scores.append(avg)
    
    best_epoch = int(np.argmax(avg_scores))
    best_avg_score = avg_scores[best_epoch]
    
    print(f"\n=== CV Results ===")
    for epoch in range(min_len):
        fold_str = '  '.join([f'{all_fold_scores[f][epoch]:.4f}' for f in range(3)])
        marker = ' <-- BEST' if epoch == best_epoch else ''
        print(f"Epoch {epoch+1:2d}: avg={avg_scores[epoch]:.4f}  folds=[{fold_str}]{marker}")
    
    print(f"\nBest epoch: {best_epoch+1}, avg score: {best_avg_score:.4f}")
    
    # Train final model on all data
    print(f"\n=== Training final model ({best_epoch+1} epochs, all data) ===")
    final_output_dir = f'./model/phase3_final_s{seed}'
    cmd = [
        sys.executable, 'code/src/train.py',
        '--seed', str(seed),
        '--output_dir', final_output_dir,
        '--num_epochs_override', str(best_epoch + 1),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print final score
    final_score_path = os.path.join(final_output_dir, 'final_score.txt')
    if os.path.exists(final_score_path):
        with open(final_score_path) as f:
            print(f.read())
    
    # Save CV results
    cv_results = {
        'seed': seed,
        'num_folds': 3,
        'fold_splits': fold_splits,
        'best_epoch': int(best_epoch + 1),
        'best_avg_score': float(best_avg_score),
        'per_epoch_avg_scores': [float(s) for s in avg_scores],
    }
    with open(os.path.join(final_output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCV results saved to {final_output_dir}/cv_results.json")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交**

```bash
git add code/src/cross_validate.py
git commit -m "feat: add 3-fold time series cross-validation script"
```

---

### Task 4: 运行 3-fold CV 并确定 best epoch

**Files:**
- No code changes

- [ ] **Step 1: 运行 cross_validate.py**

```bash
source .venv/bin/activate && python code/src/cross_validate.py
```

- [ ] **Step 2: 记录结果**

查看输出中的 best epoch 和 avg score。记录 `model/phase3_final_s42/final_score.txt` 和 `cv_results.json`。

- [ ] **Step 3: 提交 CV 结果**

```bash
git add model/phase3_cv_fold*/ model/phase3_final_s42/
git commit -m "results: Phase 3 CV — best epoch determined by 3-fold expanding window"
```

---

### Task 5: 用 best epoch 训练最终模型 + 预测

**Files:**
- Modify: `code/src/config.py`（设置 output_dir 指向最终模型）

- [ ] **Step 1: 更新 config 的 output_dir**

```python
# config.py:
    'output_dir': './model/phase3_final_s42',
```

- [ ] **Step 2: 确保模型 predict_rank_alpha=0.2**

```bash
python3 -c "
import json
c = json.load(open('model/phase3_final_s42/config.json'))
c['predict_rank_alpha'] = 0.2
json.dump(c, open('model/phase3_final_s42/config.json', 'w'), indent=4)
"
```

- [ ] **Step 3: 运行 predict + score**

```bash
source .venv/bin/activate && python code/src/predict.py && python test/score_self.py
```

- [ ] **Step 4: 提交**

```bash
git add code/src/config.py && git commit -m "feat: Phase 3 final — CV-tuned model with predict_alpha=0.2"
```

---

### Task 6: 多 Seed 验证

**Files:**
- No code changes

- [ ] **Step 1: 用 CV 确定的 best epoch 训练 seed=7 和 seed=123**

```bash
N_BEST=$(python3 -c "import json; print(json.load(open('model/phase3_final_s42/cv_results.json'))['best_epoch'])")
source .venv/bin/activate
python code/src/train.py --seed 7 --output_dir ./model/phase3_final_s7 --num_epochs_override $N_BEST
python code/src/train.py --seed 123 --output_dir ./model/phase3_final_s123 --num_epochs_override $N_BEST
```

- [ ] **Step 2: 对比三 seed 分数**

```bash
for d in model/phase3_final_s*/; do echo "$d: $(cat $d/final_score.txt | grep 'Best score')"; done
```

预期：三 seed 分数差异应显著小于 Phase 2（0.02783 vs 0.01052 vs -0.00363）。

- [ ] **Step 3: 提交**

```bash
git add model/phase3_final_s7/ model/phase3_final_s123/
git commit -m "results: Phase 3 multi-seed validation"
```
