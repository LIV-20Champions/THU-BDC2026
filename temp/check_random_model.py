"""Compare random-init model predictions vs trained model."""
import sys, os, json
sys.path.insert(0, 'code/src')
import numpy as np
import pandas as pd
import torch
from config import config, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

# Load and preprocess data (same as predict.py)
data_path = config.get('data_path', './data')
df = pd.read_csv(os.path.join(data_path, 'stock_data.csv'), dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
all_stock_ids = sorted(df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(all_stock_ids)}

processed, feature_columns = preprocess_predict_data(df, stockid2idx)
features = feature_columns
latest_date = processed['日期'].max()
stock_ids = sorted(processed['股票代码'].unique())

use_pn = config.get('use_per_stock_normalize', True)
sequences, seq_stock_ids = build_inference_sequences(
    processed, features, config['sequence_length'], stock_ids, latest_date)

# Normalize
if use_pn:
    from utils import per_stock_sliding_zscore
    seq_norm = np.empty_like(sequences)
    seq_len = config['sequence_length']
    for i in range(len(sequences)):
        seq_norm[i] = per_stock_sliding_zscore(sequences[i].copy(), seq_len, clip_range=5.0)
else:
    seq_norm = sequences

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seq_t = torch.from_numpy(seq_norm).float().unsqueeze(0).to(device)
si = torch.tensor([[stockid2idx.get(s, 0) for s in seq_stock_ids]], device=device)

print(f"\n=== Random-init model predictions (no training) ===")
for trial in range(3):
    eff_dim = get_eff_input_dim(len(features))
    model = StockTransformer(input_dim=eff_dim, config=config, num_stocks=len(all_stock_ids))
    model.to(device)
    model.eval()
    with torch.no_grad():
        scores = model(seq_t, stock_indices=si).squeeze().cpu().numpy()
    ranked = np.argsort(scores)[::-1]
    top5 = [seq_stock_ids[i] for i in ranked[:5]]
    print(f"  Trial {trial+1}: {top5}")

print(f"\n=== Trained model predictions ===")
# Load trained model
trained_dirs = [d for d in ['model/phase10_rankic_cnn/model_0'] if os.path.exists(d)]
model2 = None
if trained_dirs:
    model_dir = trained_dirs[0]
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            sc = json.load(f)
        for k, v in sc.items():
            if k in config: config[k] = v

    eff_dim = get_eff_input_dim(len(features))
    model2 = StockTransformer(input_dim=eff_dim, config=config, num_stocks=len(all_stock_ids))
    state = torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device)
    model2.load_state_dict(state)
    model2.to(device)
    model2.eval()
    with torch.no_grad():
        scores2 = model2(seq_t, stock_indices=si).squeeze().cpu().numpy()
    ranked2 = np.argsort(scores2)[::-1]
    top5_2 = [seq_stock_ids[i] for i in ranked2[:5]]
    print(f"  Trained ({model_dir}): {top5_2}")

# Compare score distributions
print(f"\n=== Score distribution comparison ===")
model.eval()
with torch.no_grad():
    random_scores = model(seq_t, stock_indices=si).squeeze().cpu().numpy()
model2.eval()
with torch.no_grad():
    trained_scores = model2(seq_t, stock_indices=si).squeeze().cpu().numpy()

print(f"  Random model: mean={random_scores.mean():.4f}, std={random_scores.std():.4f}, "
      f"min={random_scores.min():.4f}, max={random_scores.max():.4f}")
print(f"  Trained model: mean={trained_scores.mean():.4f}, std={trained_scores.std():.4f}, "
      f"min={trained_scores.min():.4f}, max={trained_scores.max():.4f}")

# Check if any stock consistently ranks high in the random model
from collections import Counter
top20_counter = Counter()
for trial in range(100):
    model3 = StockTransformer(input_dim=eff_dim, config=config, num_stocks=len(all_stock_ids))
    model3.to(device)
    model3.eval()
    with torch.no_grad():
        scores3 = model3(seq_t, stock_indices=si).squeeze().cpu().numpy()
    ranked3 = np.argsort(scores3)[::-1]
    top20_counter.update([seq_stock_ids[i] for i in ranked3[:20]])

print(f"\n=== Most frequent top-20 stocks across 100 random inits ===")
for stock, count in top20_counter.most_common(10):
    print(f"  {stock}: {count}/100")
