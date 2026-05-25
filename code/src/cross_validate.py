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

    # 3-fold expanding window
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
        print(f"Fold {i+1} scores ({len(scores)} epochs): "
              f"{[f'{s:.4f}' for s in scores[:5]]}...{[f'{s:.4f}' for s in scores[-3:]]}")
        all_fold_scores.append(scores)

    # Align by epoch and compute average
    min_len = min(len(s) for s in all_fold_scores)
    avg_scores = []
    for epoch in range(min_len):
        avg = np.mean([s[epoch] for s in all_fold_scores])
        avg_scores.append(avg)

    best_epoch = int(np.argmax(avg_scores))
    best_avg_score = avg_scores[best_epoch]

    print(f"\n=== CV Summary ===")
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
    if result.returncode != 0:
        print("Final training failed!")
        print(result.stderr[-500:])
        return

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
    os.makedirs(final_output_dir, exist_ok=True)
    with open(os.path.join(final_output_dir, 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)

    print(f"\nCV results saved to {final_output_dir}/cv_results.json")


if __name__ == '__main__':
    main()
