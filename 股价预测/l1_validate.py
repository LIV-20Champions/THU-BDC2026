"""L1 Quick validation script: 5-epoch training, check loss health and score std.

Usage: python 股价预测/l1_validate.py [config overrides as JSON string]

Example: python 股价预测/l1_validate.py '{"num_epochs":5, "ensemble_size":1}'
"""

import sys
import json
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

from config import config, feature_columns_map, feature_engineer_func_map, get_scale_features, get_eff_input_dim

# Apply overrides from command line
if len(sys.argv) > 1:
    overrides = json.loads(sys.argv[1])
    for k, v in overrides.items():
        config[k] = v

# L1-specific overrides (always apply for speed)
config['output_dir'] = './model/_l1_test'
config['num_epochs'] = int(config.get('num_epochs', 5))
config['ensemble_size'] = 1
config['use_amp'] = True
config['use_gradient_checkpointing'] = False
config['use_ema'] = False
config['use_swa'] = False
config['use_mixup'] = False
config['use_label_smoothing'] = False
config['use_cnn_features'] = False
config['use_vsn'] = False
config['use_multi_scale'] = False
config['use_gru_dual_path'] = False
config['use_cross_sectional_features'] = False
config['use_feature_interaction'] = False
config['val_months'] = 0  # no val split

print("=== L1 Quick Validation ===")
print(f"Config overrides: {overrides}")
print(f"Key config: num_epochs={config['num_epochs']}, "
      f"seq_len={config['sequence_length']}, dropout={config['dropout']}, "
      f"d_model={config['d_model']}")
print(f"Loss: smooth_ndcg={config.get('use_smooth_ndcg_loss')}, "
      f"w={config.get('smooth_ndcg_weight')}, pw={config.get('lambda_pairwise_weight')}, "
      f"mse_aux={config.get('use_mse_aux_loss')}, mse_w={config.get('mse_aux_weight')}")

# Clean old output
import shutil
if os.path.exists(config['output_dir']):
    shutil.rmtree(config['output_dir'])

# Run training (import train module directly)
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from train import set_seed, preprocess_data, collate_fn, DailyBatchSampler, train_ranking_model, SmoothNDCGLoss, EMAWrapper
from torch.utils.data import DataLoader
from model import StockTransformer
import pandas as pd

set_seed(42)

# Load data
data_file = os.path.join(config['data_path'], 'train.csv')
raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
raw_df['日期'] = pd.to_datetime(raw_df['日期'])

stock_ids = sorted(raw_df['股票代码'].unique())
stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}
num_stocks = len(stock_ids)

# Preprocess
processed, features = preprocess_data(raw_df, is_train=True, stockid2idx=stockid2idx)
scale_features = get_scale_features(features)
processed[scale_features] = processed[scale_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Build dataset
from utils import LazyRankingDataset
seq_len = config['sequence_length']
dataset = LazyRankingDataset(
    processed, features=features, targets=['label', 'score_target'],
    sequence_length=seq_len, stockid2idx=stockid2idx,
    use_per_stock_norm=config.get('use_per_stock_normalize', True),
    selected_features=config.get('selected_top_k_features', 0),
)

sampler = DailyBatchSampler(dataset, shuffle=True)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0, pin_memory=False)

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Model
eff_input_dim = get_eff_input_dim(len(features))
model = StockTransformer(input_dim=eff_input_dim, config=config, num_stocks=num_stocks)
model.to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Criterion
criterion = SmoothNDCGLoss(
    top_k=int(config.get("ndcg_top_k", 5)),
    temperature=float(config.get("soft_sort_temperature", 0.5)),
    delta_clip=float(config.get("lambda_delta_clip", 10.0)),
    smooth_ndcg_weight=float(config.get("smooth_ndcg_weight", 0.7)),
    lambda_pairwise_weight=float(config.get("lambda_pairwise_weight", 0.3)),
    use_mse_aux=bool(config.get("use_mse_aux_loss", False)),
    mse_aux_weight=float(config.get("mse_aux_weight", 0.05)),
    temperature_start=float(config.get("temperature_anneal_start", 2.0)),
    temperature_target=float(config.get("temperature_anneal_target", 0.5)),
    temperature_anneal_epochs=int(config.get("temperature_anneal_epochs", 30)),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
use_amp = config.get('use_amp', True)
scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

# Train for num_epochs
all_losses = []
all_final_scores = []
try:
    for epoch in range(config['num_epochs']):
        loss, metrics = train_ranking_model(
            model, loader, criterion, optimizer, device, epoch, writer=None,
            accumulation_steps=config.get('gradient_accumulation_steps', 1),
            ema_wrapper=None, use_mixup=False, use_label_smoothing=False,
            scaler=scaler, use_sam=False,
        )
        all_losses.append(loss)
        all_final_scores.append(metrics.get('final_score', 0))
        print(f"  Epoch {epoch+1}: loss={loss:.6f}, final_score={metrics.get('final_score', 0):.4f}")

    # Test prediction
    model.eval()
    with torch.no_grad():
        pred_stds = []
        for batch in loader:
            sequences = batch['sequences'].to(device)
            stock_indices = batch['stock_indices'].to(device)
            masks = batch['masks'].to(device)
            outputs = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())
            # Get valid predictions per batch
            for i in range(outputs.size(0)):
                valid = masks[i].bool()
                pred = outputs[i, valid]
                pred_stds.append(pred.std().item())

    avg_std = np.mean(pred_stds) if pred_stds else 0.0
    print(f"\n=== L1 Results ===")
    print(f"Epochs trained: {len(all_losses)}")
    print(f"Final loss: {all_losses[-1]:.6f}")
    print(f"Loss trend: {all_losses}")
    print(f"Avg pred std: {avg_std:.6f}")
    loss_nan = any(np.isnan(l) for l in all_losses)
    print(f"NaN detected: {loss_nan}")

    # Gate check
    l1_pass = not loss_nan and avg_std > 0.05
    print(f"L1 GATE: {'PASS' if l1_pass else 'FAIL'}")
    if l1_pass:
        sys.exit(0)
    else:
        sys.exit(1)

except Exception as e:
    print(f"L1 FAILED with exception: {e}")
    sys.exit(1)
