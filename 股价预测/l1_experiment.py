"""L1 quick validation with preprocessed data caching.

First run:
  python 股价预测/l1_experiment.py --preprocess  # ~20s, saves cache
Then run experiments:
  python 股价预测/l1_experiment.py '{"num_epochs":5, "mse_aux_weight":1.0}'
"""

import sys, json, os, pickle, shutil
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code', 'src'))

from config import config
from model import StockTransformer
from train import (
    set_seed, SmoothNDCGLoss, EMAWrapper, DailyBatchSampler,
    train_ranking_model, collate_fn, calculate_ranking_metrics,
    get_feature_engineering_workers, _build_label_and_clean, add_market_features,
)
from utils import LazyRankingDataset
from config import get_eff_input_dim, get_scale_features
from torch.utils.data import DataLoader
import multiprocessing as mp

CACHE_DIR = './股价预测/.cache'
PREPROC_CACHE = f'{CACHE_DIR}/preprocessed.pkl'
META_CACHE = f'{CACHE_DIR}/meta.pkl'


def do_preprocess():
    """Run feature engineering once, save result."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    mp.set_start_method('spawn', force=True)

    from config import feature_engineer_func_map, feature_columns_map
    from tqdm import tqdm

    data_file = os.path.join(config['data_path'], 'train.csv')
    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    assert config['feature_num'] in feature_engineer_func_map
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_columns_map[config['feature_num']]

    df = raw_df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]

    num_proc = min(config.get('feature_engineering_workers', 2), mp.cpu_count())
    with mp.Pool(processes=num_proc) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    label_alpha = float(config.get('label_alpha', 0.3))
    processed = _build_label_and_clean(processed, drop_small_open=True, label_alpha=label_alpha)
    processed, feature_columns = add_market_features(processed, feature_columns)

    # Normalize
    scale_features = get_scale_features(feature_columns)
    processed[scale_features] = processed[scale_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Per-stock sliding z-score
    seq_len = config['sequence_length']
    processed = processed.sort_values(['股票代码', '日期']).reset_index(drop=True)
    for feat in scale_features:
        processed[feat] = processed[feat].astype(np.float32)

    from utils import per_stock_sliding_zscore
    normalized_list = []
    for stock_code, group in processed.groupby('股票代码', sort=False):
        if len(group) < seq_len:
            continue
        group = group.copy()
        vals = group[scale_features].values.astype(np.float32)
        vals = per_stock_sliding_zscore(vals, seq_len, clip_range=5.0)
        group[scale_features] = vals
        normalized_list.append(group)

    processed = pd.concat(normalized_list).reset_index(drop=True)

    # Cache
    processed.to_pickle(PREPROC_CACHE)
    meta = {
        'features': feature_columns,
        'scale_features': scale_features,
        'stockid2idx': stockid2idx,
        'stock_ids': stock_ids,
        'num_stocks': len(stock_ids),
    }
    with open(META_CACHE, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Preprocessed {len(processed)} rows, {len(feature_columns)} features. Cached.")


def run_l1(overrides_json):
    """Run 5-epoch training with given config overrides."""
    overrides = json.loads(overrides_json)

    # Apply overrides
    for k, v in overrides.items():
        config[k] = v

    # L1-specific speed overrides
    config['output_dir'] = './model/_l1_test'
    config['num_epochs'] = 5
    config['ensemble_size'] = 1
    config['val_months'] = 0
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
    config['use_amp'] = True
    config['use_gradient_checkpointing'] = False

    if os.path.exists(config['output_dir']):
        shutil.rmtree(config['output_dir'])

    # Load cached data
    processed = pd.read_pickle(PREPROC_CACHE)
    with open(META_CACHE, 'rb') as f:
        meta = pickle.load(f)
    features = meta['features']
    stockid2idx = meta['stockid2idx']
    num_stocks = meta['num_stocks']

    # Build dataset
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
    else:
        device = torch.device('cpu')

    set_seed(42)

    eff_input_dim = get_eff_input_dim(len(features))
    model = StockTransformer(input_dim=eff_input_dim, config=config, num_stocks=num_stocks)
    model.to(device)

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
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # Train
    all_losses = []
    all_scores = []
    try:
        for epoch in range(config['num_epochs']):
            loss, metrics = train_ranking_model(
                model, loader, criterion, optimizer, device, epoch, writer=None,
                accumulation_steps=1, ema_wrapper=None,
                use_mixup=False, use_label_smoothing=False,
                scaler=scaler, use_sam=False,
            )
            all_losses.append(loss)
            all_scores.append(metrics.get('final_score', 0))

        # Check prediction std
        model.eval()
        pred_stds = []
        with torch.no_grad():
            for batch in loader:
                sequences = batch['sequences'].to(device)
                stock_indices = batch['stock_indices'].to(device)
                masks = batch['masks'].to(device)
                outputs = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())
                for i in range(outputs.size(0)):
                    valid = masks[i].bool()
                    pred = outputs[i, valid]
                    if pred.numel() > 0:
                        pred_stds.append(pred.std().item())

        avg_std = np.mean(pred_stds) if pred_stds else 0.0

        print(f"Losses: {[f'{l:.4f}' for l in all_losses]}")
        print(f"Scores: {[f'{s:.4f}' for s in all_scores]}")
        print(f"Avg pred std: {avg_std:.6f}")

        loss_nan = any(np.isnan(l) for l in all_losses)
        l1_pass = not loss_nan and avg_std > 0.05

        print(f"L1 GATE: {'PASS' if l1_pass else 'FAIL'} (NaN={loss_nan}, std={avg_std:.4f})")
        return 0 if l1_pass else 1

    except Exception as e:
        print(f"L1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--preprocess':
        do_preprocess()
    elif len(sys.argv) > 1:
        sys.exit(run_l1(sys.argv[1]))
    else:
        print("Usage: python l1_experiment.py --preprocess | python l1_experiment.py '{\"key\": value}'")
        sys.exit(1)
