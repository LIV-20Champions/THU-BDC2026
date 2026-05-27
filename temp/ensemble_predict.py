"""Ensemble prediction: average scores across multiple seed models."""
import sys
sys.path.insert(0, 'code/src')
import os
import json
import numpy as np
import pandas as pd
import torch
from config import config, feature_columns_map, feature_engineer_func_map, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

seeds = [42, 7, 123]
model_dirs = [f'./model/phase3_final_s{s}' for s in seeds]

# Load data (same as predict.py)
data_path = config.get('data_path', './data')
data_file = 'stock_data.csv' if os.path.exists(os.path.join(data_path, 'stock_data.csv')) else 'train.csv'
df = pd.read_csv(os.path.join(data_path, data_file), dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
all_stock_ids = sorted(df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
num_stocks = len(stockid2idx)

# Preprocess (same as predict.py)
processed, feature_columns = preprocess_predict_data(df, stockid2idx)
features = feature_columns
sequence_length = config['sequence_length']
latest_date = processed['日期'].max()
stock_ids = sorted(processed['股票代码'].unique())

use_per_stock_norm = config.get('use_per_stock_normalize', True)
sequences, sequence_stock_ids = build_inference_sequences(
    processed, features, sequence_length, stock_ids, latest_date,
    use_cs_features=config.get('use_cross_sectional_features', False),
    cs_feature_types=config.get('cs_feature_types', ['rank_pct', 'zscore']),
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_scores = []

for model_dir in model_dirs:
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            saved_config = json.load(f)
        for k, v in saved_config.items():
            if k in config:
                config[k] = v

    eff_input_dim = get_eff_input_dim(len(features))
    model = StockTransformer(input_dim=eff_input_dim, config=config, num_stocks=num_stocks)
    model.to(device)
    model.eval()

    # Load best model
    model_path = os.path.join(model_dir, 'best_model.pth')
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # Normalize and predict
    if use_per_stock_norm:
        from utils import per_stock_sliding_zscore
        sequences_norm = np.empty_like(sequences)
        for i in range(len(sequences)):
            sequences_norm[i] = per_stock_sliding_zscore(sequences[i].copy(), sequence_length, clip_range=5.0)
    else:
        sequences_norm = sequences

    seq_tensor = torch.from_numpy(sequences_norm).float().unsqueeze(0).to(device)
    stock_indices = torch.tensor([[stockid2idx.get(sid, 0) for sid in sequence_stock_ids]], device=device)

    with torch.no_grad():
        scores = model(seq_tensor, stock_indices=stock_indices).squeeze().cpu().numpy()

    all_scores.append(scores)
    print(f"Model {model_dir}: top-5 stocks = {[sequence_stock_ids[i] for i in np.argsort(scores)[-5:][::-1]]}")

# Ensemble: average scores
avg_scores = np.mean(all_scores, axis=0)
ranked_idx = np.argsort(avg_scores)[::-1]

# Top-5 with rank_decay weights (alpha=0.2)
alpha = 0.2
k = 5
top_idx = ranked_idx[:k]
top_stocks = [sequence_stock_ids[i] for i in top_idx]
ranks = np.arange(1, k + 1, dtype=np.float64)
base_weights = 1.0 / (ranks ** alpha)
weights = base_weights / base_weights.sum()

print(f"\n=== Ensemble Top-5 ===")
result_df = pd.DataFrame({'stock_id': top_stocks, 'weight': weights})
print(result_df.to_string(index=False))

# Save result
result_df.to_csv('./output/result.csv', index=False)
print(f"\nEnsemble result saved to ./output/result.csv")

# Run score
import subprocess
result = subprocess.run([sys.executable, 'test/score_self.py'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if '加权收益率' in line:
        print(line)
