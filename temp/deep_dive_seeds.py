"""Deep dive: analyze seed differences in stock scoring."""
import sys
sys.path.insert(0, 'code/src')
import os, json
import numpy as np
import pandas as pd
import torch
from config import config, feature_columns_map, feature_engineer_func_map, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

seeds = [42, 7, 123]

# Load and preprocess data
data_path = config.get('data_path', './data')
data_file = 'stock_data.csv' if os.path.exists(os.path.join(data_path, 'stock_data.csv')) else 'train.csv'
df = pd.read_csv(os.path.join(data_path, data_file), dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
all_stock_ids = sorted(df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}
num_stocks = len(stockid2idx)

processed, feature_columns = preprocess_predict_data(df, stockid2idx)
features = feature_columns
sequence_length = config['sequence_length']
latest_date = processed['日期'].max()
stock_ids = sorted(processed['股票代码'].unique())

use_per_stock_norm = config.get('use_per_stock_normalize', True)
sequences, sequence_stock_ids = build_inference_sequences(
    processed, features, sequence_length, stock_ids, latest_date,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Collect all model scores
all_model_scores = {}
for seed in seeds:
    model_dir = f'./model/phase3_final_s{seed}'
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

    model_path = os.path.join(model_dir, 'best_model.pth')
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    if use_per_stock_norm:
        from utils import per_stock_sliding_zscore
        sequences_norm = np.empty_like(sequences)
        for i in range(len(sequences)):
            sequences_norm[i] = per_stock_sliding_zscore(sequences[i].copy(), sequence_length, clip_range=5.0)
    else:
        sequences_norm = sequences

    seq_tensor = torch.from_numpy(sequences_norm).float().unsqueeze(0).to(device)
    stock_indices_t = torch.tensor([[stockid2idx.get(sid, 0) for sid in sequence_stock_ids]], device=device)

    with torch.no_grad():
        scores = model(seq_tensor, stock_indices=stock_indices_t).squeeze().cpu().numpy()

    all_model_scores[seed] = dict(zip(sequence_stock_ids, scores))

# Create comparison dataframe
rows = []
for stock in sequence_stock_ids:
    s42 = all_model_scores[42].get(stock, np.nan)
    s7 = all_model_scores[7].get(stock, np.nan)
    s123 = all_model_scores[123].get(stock, np.nan)
    rows.append({'stock': stock, 's42': s42, 's7': s7, 's123': s123,
                 's42_rank': 0, 's7_rank': 0, 's123_rank': 0})

df_scores = pd.DataFrame(rows)

# Compute ranks
for s in seeds:
    col = f's{s}'
    df_scores[f'{col}_rank'] = df_scores[col].rank(ascending=False)

df_scores['rank_spread'] = df_scores[['s42_rank', 's7_rank', 's123_rank']].std(axis=1)
df_scores['score_std'] = df_scores[['s42', 's7', 's123']].std(axis=1)

# Top stocks for each seed
print("=== Top-10 stocks per seed ===")
for s in seeds:
    top10 = df_scores.nlargest(10, f's{s}')[['stock', f's{s}']]
    print(f"\nSeed {s}:")
    for _, r in top10.iterrows():
        print(f"  {r['stock']}: {r[f's{s}']:.4f}")

# Overlap analysis
print("\n=== Overlap between seed top-20 ===")
for s1, s2 in [(42,7), (42,123), (7,123)]:
    top_s1 = set(df_scores.nlargest(20, f's{s1}')['stock'])
    top_s2 = set(df_scores.nlargest(20, f's{s2}')['stock'])
    overlap = top_s1 & top_s2
    print(f"Seed {s1} vs {s2}: {len(overlap)}/20 overlap: {overlap}")

# Score correlation
print("\n=== Score correlations ===")
for s1, s2 in [(42,7), (42,123), (7,123)]:
    corr = df_scores[f's{s1}'].corr(df_scores[f's{s2}'])
    print(f"Seed {s1} vs {s2}: r = {corr:.4f}")

# Most disagreed stocks (high rank spread)
print("\n=== Top-10 most disagreed stocks (highest rank spread) ===")
top_disagreed = df_scores.nlargest(10, 'rank_spread')[['stock', 's42_rank', 's7_rank', 's123_rank', 'rank_spread']]
for _, r in top_disagreed.iterrows():
    print(f"  {r['stock']}: s42=#{int(r['s42_rank'])}, s7=#{int(r['s7_rank'])}, s123=#{int(r['s123_rank'])}, spread={r['rank_spread']:.1f}")

# Score distribution summary
print("\n=== Score distribution ===")
for s in seeds:
    col = f's{s}'
    print(f"Seed {s}: mean={df_scores[col].mean():.4f}, std={df_scores[col].std():.4f}, "
          f"min={df_scores[col].min():.4f}, max={df_scores[col].max():.4f}")
