import os
import multiprocessing as mp

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39


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
	]
}

feature_engineer_func_map = {
	'39': engineer_features_39,
	'158+39': engineer_features_158plus39,
}


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

	# 避免四舍五入后权重和不是 1.0。
	diff = round(1.0 - float(weights.sum()), 4)
	weights[-1] = round(float(weights[-1]) + diff, 4)

	# 再做一次保护，防止最后一项因为校准变成负数。
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
	"""
	根据 config 中的 predict_weight_mode 选择股票数量和分配权重。

	支持：
	- equal: 前 predict_top_k 只等权，默认前5只
	- rank_decay: 前 predict_top_k 只按固定递减权重
	- top3: 只买前三
	- top1: 只买第一
	- softmax: 前 predict_top_k 只按模型分数 softmax 分配权重
	"""
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

	else:
		raise ValueError(f'未知 predict_weight_mode: {mode}')

	return selected_ids, weights


def main():
	data_file = os.path.join(config['data_path'], 'train.csv')
	model_path = os.path.join(config['output_dir'], 'best_model.pth')
	scaler_path = os.path.join(config['output_dir'], 'scaler.pkl')
	output_dir = './output/'
	output_path = os.path.join(output_dir, 'result.csv')
	ranked_scores_path = os.path.join(output_dir, 'ranked_scores.csv')

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
	processed[scale_features] = processed[scale_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
	model.load_state_dict(torch.load(model_path, map_location=device))
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
