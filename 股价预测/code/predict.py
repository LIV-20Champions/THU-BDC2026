import os
import multiprocessing as mp
import json

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from allocation import allocate_by_mode
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39, engineer_features_100plus39, maybe_cross_sectional_rank_normalize


feature_cloums_map = {
	'39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'158+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC5', 'ROC10', 'ROC20', 'ROC30', 'ROC60', 'MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'STD5',
		'STD10', 'STD20', 'STD30', 'STD60', 'BETA5', 'BETA10', 'BETA20', 'BETA30', 'BETA60', 'RSQR5', 'RSQR10',
		'RSQR20', 'RSQR30', 'RSQR60', 'RESI5', 'RESI10', 'RESI20', 'RESI30', 'RESI60', 'MAX5', 'MAX10', 'MAX20',
		'MAX30', 'MAX60', 'MIN5', 'MIN10', 'MIN20', 'MIN30', 'MIN60', 'QTLU5', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLU60', 'QTLD5', 'QTLD10', 'QTLD20', 'QTLD30', 'QTLD60', 'RANK5', 'RANK10', 'RANK20', 'RANK30',
		'RANK60', 'RSV5', 'RSV10', 'RSV20', 'RSV30', 'RSV60', 'IMAX5', 'IMAX10', 'IMAX20', 'IMAX30', 'IMAX60',
		'IMIN5', 'IMIN10', 'IMIN20', 'IMIN30', 'IMIN60', 'IMXD5', 'IMXD10', 'IMXD20', 'IMXD30', 'IMXD60',
		'CORR5', 'CORR10', 'CORR20', 'CORR30', 'CORR60', 'CORD5', 'CORD10', 'CORD20', 'CORD30', 'CORD60',
		'CNTP5', 'CNTP10', 'CNTP20', 'CNTP30', 'CNTP60', 'CNTN5', 'CNTN10', 'CNTN20', 'CNTN30', 'CNTN60',
		'CNTD5', 'CNTD10', 'CNTD20', 'CNTD30', 'CNTD60', 'SUMP5', 'SUMP10', 'SUMP20', 'SUMP30', 'SUMP60',
		'SUMN5', 'SUMN10', 'SUMN20', 'SUMN30', 'SUMN60', 'SUMD5', 'SUMD10', 'SUMD20', 'SUMD30', 'SUMD60',
		'VMA5', 'VMA10', 'VMA20', 'VMA30', 'VMA60', 'VSTD5', 'VSTD10', 'VSTD20', 'VSTD30', 'VSTD60', 'WVMA5',
		'WVMA10', 'WVMA20', 'WVMA30', 'WVMA60', 'VSUMP5', 'VSUMP10', 'VSUMP20', 'VSUMP30', 'VSUMP60', 'VSUMN5',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMN60', 'VSUMD5', 'VSUMD10', 'VSUMD20', 'VSUMD30', 'VSUMD60',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	],
	'100+39': [
		'instrument', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅',
		'KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'OPEN0', 'HIGH0', 'LOW0',
		'VWAP0', 'ROC10', 'ROC20', 'ROC30', 'MA10', 'MA20', 'MA30', 'STD10', 'STD20', 'STD30',
		'BETA10', 'BETA20', 'BETA30', 'RSQR10', 'RSQR20', 'RSQR30', 'RESI10', 'RESI20', 'RESI30',
		'MAX10', 'MAX20', 'MAX30', 'MIN10', 'MIN20', 'MIN30', 'QTLU10', 'QTLU20', 'QTLU30',
		'QTLD10', 'QTLD20', 'QTLD30', 'RANK10', 'RANK20', 'RANK30', 'RSV10', 'RSV20', 'RSV30',
		'IMAX10', 'IMAX20', 'IMAX30', 'IMIN10', 'IMIN20', 'IMIN30', 'IMXD10', 'IMXD20', 'IMXD30',
		'CORR10', 'CORR20', 'CORR30', 'CORD10', 'CORD20', 'CORD30', 'CNTP10', 'CNTP20', 'CNTP30',
		'CNTN10', 'CNTN20', 'CNTN30', 'CNTD10', 'CNTD20', 'CNTD30', 'SUMP10', 'SUMP20', 'SUMP30',
		'SUMN10', 'SUMN20', 'SUMN30', 'SUMD10', 'SUMD20', 'SUMD30', 'VMA10', 'VMA20', 'VMA30',
		'VSTD10', 'VSTD20', 'VSTD30', 'WVMA10', 'WVMA20', 'WVMA30', 'VSUMP10', 'VSUMP20', 'VSUMP30',
		'VSUMN10', 'VSUMN20', 'VSUMN30', 'VSUMD10', 'VSUMD20', 'VSUMD30',
		'sma_5', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'volume_change', 'obv',
		'volume_ma_5', 'volume_ma_20', 'volume_ratio', 'kdj_k', 'kdj_d', 'kdj_j', 'boll_mid', 'boll_std',
		'atr_14', 'ema_60', 'volatility_10', 'volatility_20', 'return_1', 'return_5', 'return_10',
		'high_low_spread', 'open_close_spread', 'high_close_spread', 'low_close_spread'
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
	'100+39': engineer_features_100plus39,
}


RUNTIME_INFERENCE_CONFIG_KEYS = {
	'output_dir',
	'data_path',
	'predict_weight_mode',
	'predict_top_k',
	'predict_rank_weights',
	'predict_top3_weights',
	'predict_weight_temperature',
	'confidence_top1_threshold',
	'confidence_top3_threshold',
	'confidence_min_margin12',
	'confidence_margin_scale',
	'confidence_temperature',
	'confidence_fallback_mode',
}


def resolve_inference_config(runtime_config):
	"""
	预测时优先使用 checkpoint 目录中的训练配置，避免源码超参变化后无法加载旧模型。
	路径和权重分配参数仍允许由当前 config.py 覆盖，因为它们不改变模型结构。
	"""
	output_dir = runtime_config.get('output_dir')
	config_path = os.path.join(output_dir, 'config.json') if output_dir else None

	if config_path and os.path.exists(config_path):
		with open(config_path, 'r', encoding='utf-8') as f:
			resolved = json.load(f)

		for key in RUNTIME_INFERENCE_CONFIG_KEYS:
			if key in runtime_config:
				resolved[key] = runtime_config[key]

		return resolved, config_path

	return dict(runtime_config), 'runtime config.py'


def get_scale_features(features):
	"""
	instrument 是股票离散ID，若启用 stock embedding，就不能参与 StandardScaler。
	这里必须与 train.py 保持一致，否则训练和预测的特征分布会不一致。
	"""
	if config.get('use_stock_embedding', True):
		return [f for f in features if f != 'instrument']
	return list(features)


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])
	processed = maybe_cross_sectional_rank_normalize(
		processed,
		config,
		feature_columns,
		date_col='日期',
	)

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
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

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids



def select_and_allocate_weights(ranked_stock_ids, ranked_scores):
	"""
	根据 config 中的 predict_weight_mode 选择股票数量和分配权重。

	支持：
	- equal: 前 predict_top_k 只等权，默认前5只
	- rank_decay: 前 predict_top_k 只按固定递减权重
	- top3: 只买前三
	- top1: 只买第一
	- softmax: 前 predict_top_k 只按模型分数 softmax 分配权重
	- confidence: 先计算置信率，高置信单押，否则回退到 Top3/Top5
	"""
	mode = config.get('predict_weight_mode', 'equal')
	return allocate_by_mode(mode, ranked_stock_ids, ranked_scores, config)


def build_allocation_diagnostics(latest_date, ranked_stock_ids, ranked_scores, decision):
	weights_by_stock = dict(zip(decision.selected_ids, decision.weights))
	diagnostic_k = min(max(5, len(decision.selected_ids)), len(ranked_stock_ids))
	metrics = decision.metrics

	rows = []
	for rank, (stock_id, score) in enumerate(zip(ranked_stock_ids[:diagnostic_k], ranked_scores[:diagnostic_k]), start=1):
		rows.append({
			'date': latest_date.date(),
			'mode': config.get('predict_weight_mode', 'equal'),
			'selected_mode': decision.selected_mode,
			'reason': decision.reason,
			'confidence_rate': metrics.get('confidence_rate'),
			'softmax_p1': metrics.get('softmax_p1'),
			'entropy_confidence': metrics.get('entropy_confidence'),
			'gap_confidence': metrics.get('gap_confidence'),
			'margin12': metrics.get('margin12'),
			'margin1mean': metrics.get('margin1mean'),
			'z_gap': metrics.get('z_gap'),
			'rank': rank,
			'stock_id': stock_id,
			'score': float(score),
			'weight': float(weights_by_stock.get(stock_id, 0.0)),
			'selected': stock_id in weights_by_stock,
		})

	return pd.DataFrame(rows)


def main():
	global config
	config, config_source = resolve_inference_config(config)

	data_file = os.path.join(config['data_path'], 'train.csv')
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_dir = './output/'
	output_path = os.path.join(output_dir, 'result.csv')
	ranked_scores_path = os.path.join(output_dir, 'ranked_scores.csv')
	diagnostics_path = os.path.join(output_dir, 'allocation_diagnostics.csv')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, features = preprocess_predict_data(raw_df, stockid2idx)
	scale_features = get_scale_features(features)
	processed[scale_features] = processed[scale_features].replace([np.inf, -np.inf], np.nan)
	processed = processed.dropna(subset=scale_features)

	if processed.empty:
		raise ValueError('特征处理后无有效数据，请检查数据与特征工程')

	scaler = joblib.load(scaler_path)
	processed[scale_features] = scaler.transform(processed[scale_features])

	sequence_length = config['sequence_length']
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		stock_ids,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = StockTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))

	num_ensemble = int(config.get('num_ensemble_models', 1))
	model_paths = [model_path]
	for i in range(2, num_ensemble + 1):
		alt_path = os.path.join(config['output_dir'], f'best_model_{i}.pth')
		if os.path.exists(alt_path):
			model_paths.append(alt_path)

	if len(model_paths) > 1:
		print(f'使用 {len(model_paths)} 个模型进行集成预测')

	all_scores = []
	for mp_idx, mp in enumerate(model_paths):
		model.load_state_dict(torch.load(mp, map_location=device))
		model.to(device)
		model.eval()

		with torch.no_grad():
			x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]

			if config.get('use_stock_embedding', True):
				sequence_stock_indices = [stockid2idx[sid] for sid in sequence_stock_ids]
				stock_indices = torch.LongTensor(sequence_stock_indices).unsqueeze(0).to(device)
				stock_mask = torch.ones_like(stock_indices, dtype=torch.bool, device=device)
				scores = model(
					x,
					stock_indices=stock_indices,
					stock_mask=stock_mask,
				).squeeze(0).detach().cpu().numpy()
			else:
				scores = model(x).squeeze(0).detach().cpu().numpy()

		all_scores.append(scores)

	scores = np.mean(all_scores, axis=0)

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]
	ranked_scores = scores[order]

	if len(ranked_stock_ids) < 1:
		raise ValueError('没有可预测股票')

	os.makedirs(output_dir, exist_ok=True)

	# 保存完整排序。之后只搜索权重时，不需要重新运行模型。
	ranked_df = pd.DataFrame({
		'rank': np.arange(1, len(ranked_stock_ids) + 1),
		'stock_id': ranked_stock_ids,
		'score': ranked_scores,
	})
	ranked_df.to_csv(ranked_scores_path, index=False)

	decision = select_and_allocate_weights(ranked_stock_ids, ranked_scores)

	output_df = pd.DataFrame({
		'stock_id': decision.selected_ids,
		'weight': decision.weights,
	})
	output_df.to_csv(output_path, index=False)
	build_allocation_diagnostics(latest_date, ranked_stock_ids, ranked_scores, decision).to_csv(
		diagnostics_path,
		index=False,
	)

	print(f'预测日期: {latest_date.date()}')
	print(f'推理配置来源: {config_source}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	print(f'完整排序已写入: {ranked_scores_path}')
	print(f'最终结果已写入: {output_path}')
	print(f'权重诊断已写入: {diagnostics_path}')
	print(f'权重模式: {config.get("predict_weight_mode", "equal")}')
	print(f'实际分配模式: {decision.selected_mode}')
	print(f'分配理由: {decision.reason}')
	print(output_df.to_string(index=False))


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
