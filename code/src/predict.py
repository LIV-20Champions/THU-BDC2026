import os
import json
import multiprocessing as mp
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config, feature_columns_map, feature_engineer_func_map, get_scale_features, get_eff_input_dim
from model import StockTransformer
from utils import per_stock_sliding_zscore, build_cross_sectional_features, add_market_features, cross_sectional_rank_normalize


def preprocess_predict_data(df, stockid2idx):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_columns_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError('输入数据为空，无法预测')

    num_processes = min(int(config.get('feature_engineering_workers', 2)), mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

    processed = pd.concat(processed_list).reset_index(drop=True)
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)
    processed['日期'] = pd.to_datetime(processed['日期'])

    if config.get('use_cross_sectional_rank', False):
        processed = cross_sectional_rank_normalize(
            processed,
            date_col='日期',
            feature_cols=[f for f in feature_columns if f != 'instrument'],
            skip_cols=set(config.get('cross_sectional_rank_skip_cols', ['instrument'])),
        )
        print("推理: 特征截面Rank归一化已完成")

    processed, feature_columns = add_market_features(processed, feature_columns)

    return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date,
                               use_cs_features=False, cs_feature_types=None):
    sequences, sequence_stock_ids = [], []
    for stock_id in stock_ids:
        stock_history = data[
            (data['股票代码'] == stock_id) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            sequences.append(stock_history[features].values.astype(np.float32))
            sequence_stock_ids.append(stock_id)

    if len(sequences) == 0:
        raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

    sequences_np = np.asarray(sequences, dtype=np.float32)

    if use_cs_features and cs_feature_types:
        sequences_np = build_cross_sectional_features(sequences_np, cs_feature_types)

    return sequences_np, sequence_stock_ids


def _normalize_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError('weights 必须是一维且非空')
    if not np.all(np.isfinite(weights)):
        raise ValueError('weights 中存在非有限值')
    if weights.sum() <= 0:
        raise ValueError('weights 权重和必须大于 0')

    weights = weights / weights.sum()
    weights = np.round(weights, 4)

    diff = round(1.0 - float(weights.sum()), 4)
    weights[-1] = round(float(weights[-1]) + diff, 4)

    if weights[-1] < 0:
        weights[0] = round(float(weights[0]) + float(weights[-1]), 4)
        weights[-1] = 0.0

    return weights.tolist()


def _softmax_weights(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    temperature = max(float(temperature), 1e-8)
    shifted = scores / temperature
    shifted = shifted - np.max(shifted)
    weights = np.exp(shifted)
    return _normalize_weights(weights)


def select_and_allocate_weights(ranked_stock_ids, ranked_scores):
    mode = config.get('predict_weight_mode', 'equal')
    max_k = min(int(config.get('predict_top_k', 5)), 5, len(ranked_stock_ids))

    if max_k <= 0:
        raise ValueError('没有可用于输出的股票')

    ranked_scores = np.asarray(ranked_scores, dtype=np.float64)

    if mode == 'equal':
        k = max_k
        selected_ids = ranked_stock_ids[:k]
        weights = _normalize_weights(np.ones(k, dtype=np.float64))

    elif mode == 'rank_decay':
        k = max_k
        selected_ids = ranked_stock_ids[:k]
        alpha = config.get('predict_rank_alpha', None)
        if alpha is not None:
            ranks = np.arange(1, k + 1, dtype=np.float64)
            base_weights = 1.0 / (ranks ** float(alpha))
        else:
            base_weights = np.asarray(
                config.get('predict_rank_weights', [0.30, 0.25, 0.20, 0.15, 0.10]),
                dtype=np.float64,
            )[:k]
        weights = _normalize_weights(base_weights)

    elif mode == 'top3':
        k = min(3, len(ranked_stock_ids))
        selected_ids = ranked_stock_ids[:k]
        base_weights = np.asarray(
            config.get('predict_top3_weights', [0.45, 0.35, 0.20]),
            dtype=np.float64,
        )[:k]
        weights = _normalize_weights(base_weights)

    elif mode == 'top1':
        selected_ids = ranked_stock_ids[:1]
        weights = [1.0]

    elif mode == 'softmax':
        k = max_k
        selected_ids = ranked_stock_ids[:k]
        temperature = float(config.get('predict_weight_temperature', 0.5))
        weights = _softmax_weights(ranked_scores[:k], temperature)

    elif mode == 'hybrid':
        k = max_k
        selected_ids = ranked_stock_ids[:k]
        temperature = float(config.get('predict_weight_temperature', 0.3))
        softmax_w = _softmax_weights(ranked_scores[:k], temperature)
        decay_w = np.asarray(
            config.get('predict_rank_weights', [0.30, 0.25, 0.20, 0.15, 0.10]),
            dtype=np.float64,
        )[:k]
        combined = np.asarray(softmax_w, dtype=np.float64) * np.asarray(decay_w, dtype=np.float64)
        weights = _normalize_weights(combined)

    else:
        raise ValueError(f'未知 predict_weight_mode: {mode}')

    return selected_ids, weights


def _load_model_weights(model_dir_str):
    model_dir = Path(model_dir_str)
    config_path = model_dir / 'config.json'
    saved_config = {}
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)

    model_path = os.path.join(model_dir_str, 'best_model.pth')
    ema_path = os.path.join(model_dir_str, 'best_model_ema.pth')
    swa_path = os.path.join(model_dir_str, 'best_model_swa.pth')

    state = torch.load(model_path, map_location='cpu')
    if os.path.exists(swa_path):
        swa_state = torch.load(swa_path, map_location='cpu')
        state.update(swa_state)
        print(f"  [{model_dir_str}] SWA模型")
    elif os.path.exists(ema_path):
        ema_state = torch.load(ema_path, map_location='cpu')
        state.update(ema_state)
        print(f"  [{model_dir_str}] EMA模型")
    else:
        print(f"  [{model_dir_str}] 标准模型")
    return state, saved_config


def _predict_with_model(model, sequences_np, stockid2idx, sequence_stock_ids, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)
        if config.get('use_stock_embedding', True):
            sequence_stock_indices = [stockid2idx[sid] for sid in sequence_stock_ids]
            stock_indices = torch.LongTensor(sequence_stock_indices).unsqueeze(0).to(device)
            stock_mask = torch.ones_like(stock_indices, dtype=torch.bool, device=device)
            scores = model(
                x, stock_indices=stock_indices, stock_mask=stock_mask,
            ).squeeze(0).detach().cpu().numpy()
        else:
            scores = model(x).squeeze(0).detach().cpu().numpy()
    return scores


def main():
    ensemble_dirs = config.get('ensemble_model_dirs', None)
    # Auto-detect ensemble from ensemble_config.json
    output_dir = config['output_dir']
    ensemble_config_path = Path(output_dir) / 'ensemble_config.json'
    if not ensemble_dirs and ensemble_config_path.exists():
        with open(ensemble_config_path, 'r') as f:
            ens_meta = json.load(f)
        ensemble_size = ens_meta.get('ensemble_size', 0)
        if ensemble_size > 0:
            ensemble_dirs = [f'{output_dir}/model_{i}' for i in range(ensemble_size)]
            print(f"自动检测到集成模型: {ensemble_size}个")
    if ensemble_dirs:
        print(f"Ensemble模式: {len(ensemble_dirs)}个模型")
        primary_dir = ensemble_dirs[0]
    else:
        primary_dir = config['output_dir']

    config_path = Path(primary_dir) / 'config.json'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            primary_saved_config = json.load(f)
        for k, v in primary_saved_config.items():
            if k in config and k not in ('ensemble_size',):  # 保留当前集成设置
                config[k] = v
        print(f"已加载训练配置: {config_path}")

    data_file = os.path.join(config['data_path'], 'train.csv')
    output_dir = './output/'
    output_path = os.path.join(output_dir, 'result.csv')
    ranked_scores_path = os.path.join(output_dir, 'ranked_scores.csv')

    raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    latest_date = raw_df['日期'].max()

    stock_ids = sorted(raw_df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    processed, features = preprocess_predict_data(raw_df, stockid2idx)
    scale_features = get_scale_features(features)
    processed[scale_features] = processed[scale_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    use_per_stock_norm = config.get('use_per_stock_normalize', True)
    if use_per_stock_norm:
        print("使用逐股票时序Z-Score标准化（推理）")
        seq_len = config['sequence_length']
        processed = processed.sort_values(['股票代码', '日期']).reset_index(drop=True)
        for feat in scale_features:
            processed[feat] = processed[feat].astype(np.float32)
        normalized_list = []
        for stock_code, group in processed.groupby('股票代码', sort=False):
            if len(group) < seq_len:
                continue
            group = group.copy()
            vals = group[scale_features].values.astype(np.float32)
            vals = per_stock_sliding_zscore(vals, seq_len, clip_range=5.0)
            group[scale_features] = vals
            normalized_list.append(group)
        if normalized_list:
            processed = pd.concat(normalized_list).reset_index(drop=True)
        else:
            raise ValueError('标准化后无可用数据')
    else:
        scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')
        scaler = joblib.load(scaler_path)
        processed[scale_features] = scaler.transform(processed[scale_features])

    sequence_length = config['sequence_length']
    use_cs_features = config.get('use_cross_sectional_features', False)
    cs_feature_types = config.get('cs_feature_types', ['rank_pct', 'zscore'])
    sequences_np, sequence_stock_ids = build_inference_sequences(
        processed, features, sequence_length, stock_ids, latest_date,
        use_cs_features=use_cs_features, cs_feature_types=cs_feature_types,
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    eff_input_dim = get_eff_input_dim(len(features))

    if ensemble_dirs:
        all_scores = []
        for md in ensemble_dirs:
            state, saved_cfg = _load_model_weights(md)
            # Merge saved config WITHOUT mutating global config (use local copy)
            run_config = dict(config)
            for k, v in saved_cfg.items():
                if k in run_config:
                    run_config[k] = v
            model = StockTransformer(input_dim=eff_input_dim, config=run_config, num_stocks=len(stock_ids))
            model.load_state_dict(state)
            model.to(device)
            scores = _predict_with_model(model, sequences_np, stockid2idx, sequence_stock_ids, device)
            all_scores.append(scores)
            del model
        rank_scores = []
        for s in all_scores:
            order = np.argsort(s)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
            rank_scores.append(ranks / len(s))
        scores = np.mean(rank_scores, axis=0)
        print(f"Ensemble: {len(all_scores)}个模型rank归一化取均值")
    else:
        state, saved_cfg = _load_model_weights(config['output_dir'])
        run_config = dict(config)
        for k, v in saved_cfg.items():
            if k in run_config:
                run_config[k] = v
        model = StockTransformer(input_dim=eff_input_dim, config=run_config, num_stocks=len(stock_ids))
        model.load_state_dict(state)
        model.to(device)
        scores = _predict_with_model(model, sequences_np, stockid2idx, sequence_stock_ids, device)

    order = np.argsort(scores)[::-1]
    ranked_stock_ids = [sequence_stock_ids[i] for i in order]
    ranked_scores = scores[order]

    if len(ranked_stock_ids) < 1:
        raise ValueError('没有可预测股票')

    os.makedirs(output_dir, exist_ok=True)

    ranked_df = pd.DataFrame({
        'rank': np.arange(1, len(ranked_stock_ids) + 1),
        'stock_id': ranked_stock_ids,
        'score': ranked_scores,
    })
    ranked_df.to_csv(ranked_scores_path, index=False)

    selected_ids, final_weights = select_and_allocate_weights(ranked_stock_ids, ranked_scores)

    output_df = pd.DataFrame({
        'stock_id': selected_ids,
        'weight': final_weights,
    })
    output_df.to_csv(output_path, index=False)

    print(f'预测日期: {latest_date.date()}')
    print(f'参与排序股票数: {len(ranked_stock_ids)}')
    print(f'完整排序已写入: {ranked_scores_path}')
    print(f'最终结果已写入: {output_path}')
    print(f'权重模式: {config.get("predict_weight_mode", "equal")}')
    print(output_df.to_string(index=False))


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()