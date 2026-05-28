import sys, os, json, numpy as np, torch, pandas as pd
sys.path.insert(0, 'code/src')
from config import config, get_eff_input_dim
from model import StockTransformer
from predict import preprocess_predict_data, build_inference_sequences

df = pd.read_csv('data/stock_data.csv', dtype={'股票代码': str})
df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
all_ids = sorted(df['股票代码'].unique())
sid2idx = {s: i for i, s in enumerate(all_ids)}
processed, fcols = preprocess_predict_data(df, sid2idx)
features = fcols
latest = processed['日期'].max()
stocks = sorted(processed['股票代码'].unique())
seqs, seq_stocks = build_inference_sequences(processed, features, config['sequence_length'], stocks, latest)
from utils import per_stock_sliding_zscore
for i in range(len(seqs)):
    seqs[i] = per_stock_sliding_zscore(seqs[i].copy(), 60, clip_range=5.0)
device = torch.device('cpu')
st = torch.from_numpy(seqs).float().unsqueeze(0)
si = torch.tensor([[sid2idx.get(s, 0) for s in seq_stocks]])

print(f"Individual model distributions:")
for md in ['model/phase8_orig/model_0', 'model/phase8_orig/model_1',
           'model/phase8_orig/model_2']:
    cfg_p = os.path.join(md, 'config.json')
    with open(cfg_p) as f: sc = json.load(f)
    for k, v in sc.items():
        if k in config: config[k] = v
    # Ensure architecture matches — Phase 8 models use CNN but we can load them
    # by just using the model's own settings from config.json
    eff = get_eff_input_dim(len(features))
    m = StockTransformer(input_dim=eff, config=config, num_stocks=len(all_ids))
    m.load_state_dict(torch.load(os.path.join(md, 'best_model.pth'), map_location='cpu'))
    m.eval()
    with torch.no_grad():
        scores = m(st, stock_indices=si).squeeze().numpy()
    ranked = np.argsort(scores)[::-1]
    top5 = [seq_stocks[i] for i in ranked[:5]]
    print(f"  {md}: std={scores.std():.4f} top5={top5}")
