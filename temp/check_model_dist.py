"""Check score distributions from models trained with different configs."""
import sys, os, json, numpy as np, torch, pandas as pd
sys.path.insert(0, 'code/src')
from config import config, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

os.environ['CUDA_VISIBLE_DEVICES'] = ''

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
if use_pn:
    from utils import per_stock_sliding_zscore
    seq_norm = np.empty_like(sequences)
    for i in range(len(sequences)):
        seq_norm[i] = per_stock_sliding_zscore(sequences[i].copy(), config['sequence_length'], clip_range=5.0)
else:
    seq_norm = sequences

device = torch.device('cpu')
seq_t = torch.from_numpy(seq_norm).float().unsqueeze(0).to(device)
si = torch.tensor([[stockid2idx.get(s, 0) for s in seq_stock_ids]], device=device)

print("=== Score distributions across models ===\n")
for label, model_dirs, cnn_ok in [
    ("Phase 8 (AdamW,2ep)", ['model/phase8_ensemble/model_0'], False),
    ("Phase 10 CNN (SAM,NaN)", ['model/phase10_rankic_cnn/model_0'], True),
    ("Phase 11 (SAM,score_target)", ['model/phase11_soft_topk/model_0'], True),
]:
    for md in model_dirs:
        if not os.path.exists(md): continue
        cfg_path = os.path.join(md, 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f: sc = json.load(f)
        run_cfg = dict(config)
        for k, v in sc.items():
            if k in run_cfg: run_cfg[k] = v
        # Force CNN off if model doesn't have it
        if not cnn_ok:
            run_cfg['use_cnn_features'] = False

        eff_dim = get_eff_input_dim(len(features))
        model = StockTransformer(input_dim=eff_dim, config=run_cfg, num_stocks=len(all_stock_ids))
        state = torch.load(os.path.join(md, 'best_model.pth'), map_location=device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            scores = model(seq_t, stock_indices=si).squeeze().cpu().numpy()
        ranked = np.argsort(scores)[::-1]
        print(f"  {label}: mean={scores.mean():.4f}, std={scores.std():.4f}, "
              f"min={scores.min():.4f}, max={scores.max():.4f}")
        print(f"    Top-5: {[seq_stock_ids[i] for i in ranked[:5]]}")
        print(f"    Top-5 scores: {[f'{scores[i]:.4f}' for i in ranked[:5]]}")

# Also compare with random init
print(f"\n  Random init: mean={np.random.randn(300).mean():.4f}, std=1.0 (reference)")
print(f"\n  Key metric: std > 0.05 means model is distinguishing stocks")
print(f"  std < 0.02 means model has collapsed (all scores nearly equal)")
