"""Ensemble only positive-scoring seeds."""
import sys; sys.path.insert(0, 'code/src')
import os, json, numpy as np, pandas as pd, torch
from config import config, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

seeds = [42, 123, 99]  # positive self-scorers
model_dirs = [f'./model/phase4_warm0_s{s}' for s in seeds]

data_path = config.get('data_path', './data')
data_file = 'stock_data.csv' if os.path.exists(os.path.join(data_path, 'stock_data.csv')) else 'train.csv'
df = pd.read_csv(os.path.join(data_path, data_file), dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
all_stock_ids = sorted(df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
num_stocks = len(stockid2idx)

processed, feature_columns = preprocess_predict_data(df, stockid2idx)
features = feature_columns
seq_len = config['sequence_length']
latest_date = processed['日期'].max()
stock_ids = sorted(processed['股票代码'].unique())

use_pn = config.get('use_per_stock_normalize', True)
sequences, seq_stock_ids = build_inference_sequences(
    processed, features, seq_len, stock_ids, latest_date)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_scores = []

for model_dir in model_dirs:
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved_config = json.load(f)
        for k, v in saved_config.items():
            if k in config: config[k] = v

    eff_dim = get_eff_input_dim(len(features))
    model = StockTransformer(input_dim=eff_dim, config=config, num_stocks=num_stocks)
    model.to(device); model.eval()
    state = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(state)

    if use_pn:
        from utils import per_stock_sliding_zscore
        seq_norm = np.empty_like(sequences)
        for i in range(len(sequences)):
            seq_norm[i] = per_stock_sliding_zscore(sequences[i].copy(), seq_len, clip_range=5.0)
    else:
        seq_norm = sequences

    s_t = torch.from_numpy(seq_norm).float().unsqueeze(0).to(device)
    si = torch.tensor([[stockid2idx.get(s, 0) for s in seq_stock_ids]], device=device)
    with torch.no_grad():
        scores = model(s_t, stock_indices=si).squeeze().cpu().numpy()
    all_scores.append(scores)

# Average then normalize (z-score per model before averaging for fair comparison)
all_scores_norm = []
for s in all_scores:
    all_scores_norm.append((s - s.mean()) / (s.std() + 1e-8))

avg_scores = np.mean(all_scores_norm, axis=0)
ranked = np.argsort(avg_scores)[::-1]

alpha = 0.2; k = 5
top_idx = ranked[:k]
top_stocks = [seq_stock_ids[i] for i in top_idx]
ranks = np.arange(1, k+1, dtype=np.float64)
weights = (1.0 / (ranks ** alpha))
weights /= weights.sum()

print("=== Ensemble (z-score normalized) ===")
for s, w in zip(top_stocks, weights):
    print(f"  {s}: {w:.4f}")

result_df = pd.DataFrame({'stock_id': top_stocks, 'weight': weights})
result_df.to_csv('./output/result.csv', index=False)

import subprocess
r = subprocess.run([sys.executable, 'test/score_self.py'], capture_output=True, text=True)
for line in r.stdout.split('\n'):
    if '加权收益率' in line: print(line)
