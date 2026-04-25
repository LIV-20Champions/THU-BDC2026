import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import StockTransformer
from utils import engineer_features_39, engineer_features_158plus39
from utils import create_ranking_dataset_vectorized
import joblib
import os
import json
import multiprocessing as mp
import random
from collections import defaultdict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
    '158+39': engineer_features_158plus39
}


def get_feature_engineering_workers(num_groups):
    configured = config.get('feature_engineering_workers')
    if configured is not None:
        try:
            configured = int(configured)
        except (TypeError, ValueError):
            configured = None
    if configured is None:
        configured = min(4, mp.cpu_count())
    return max(1, min(int(configured), mp.cpu_count(), max(1, int(num_groups))))


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = get_feature_engineering_workers(len(groups))
    print(f"{desc}进程数: {num_processes}")
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


def get_scale_features(features):
    """
    instrument 是股票离散ID，若启用 stock embedding，就绝不能参与 StandardScaler。

    StockTransformer 默认启用 use_stock_embedding，因此这里默认也按 True 处理，
    避免 config.py 漏写该字段时把 instrument 标准化，导致 embedding 下标越界。
    """
    if config.get('use_stock_embedding', True):
        return [f for f in features if f != 'instrument']
    return list(features)


class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数。
    y_true 直接使用真实未来收益率，而不是 relevance 排名分数。
    """
    def __init__(self, temperature=1.0, k=5, weight_factor=2.0, pairwise_weight=1, base_weight=1.0):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight

    def listwise_loss(self, y_pred, y_true, weights):
        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)

        weighted_ce = -(target_probs * torch.log(pred_probs + 1e-12) * weights)
        ce_loss = (weighted_ce.sum(dim=1) / (weights.sum(dim=1) + 1e-12)).mean()
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)

        mask = (true_diff != 0).float()
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)

        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff))
        weighted_loss = pairwise_loss * mask * weight_matrix

        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        return loss

    def forward(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items]，真实未来收益率
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        _, top_indices = torch.topk(y_true, k, dim=1)

        weights = torch.full_like(y_true, fill_value=self.base_weight)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor

        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)

        total_loss = listwise + self.pairwise_weight * pairwise
        return total_loss


def _build_eval_weights(pred_scores, mode='equal', temperature=0.25):
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    if pred_scores.ndim != 1 or pred_scores.size == 0:
        raise ValueError('pred_scores 必须是一维且非空')

    if mode == 'equal':
        weights = np.ones_like(pred_scores, dtype=np.float64)
    elif mode == 'softmax':
        temp = max(float(temperature), 1e-8)
        shifted = pred_scores / temp
        shifted = shifted - np.max(shifted)
        weights = np.exp(shifted)
    else:
        raise ValueError(f'不支持的评估权重模式: {mode}')

    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        raise ValueError(f'评估权重非法，当前和为: {weights_sum}')
    return weights / weights_sum


def _resolve_model_selection_metric():
    configured = config.get('model_selection_metric')
    if configured:
        return configured

    predict_selection_mode = config.get('predict_selection_mode', 'topk')
    predict_weight_mode = config.get('predict_weight_mode', 'equal')

    if predict_selection_mode == 'topk':
        if predict_weight_mode == 'softmax':
            return 'official_score_weighted_softmax'
        return 'official_score_eq'

    return 'official_score_eq'


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """
    计算排序评估指标。
    除了原有等权 TopK 指标外，新增与预测阶段更一致的加权 TopK 收益指标。
    """
    batch_size = y_pred.size(0)

    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    official_score_eq_list = []
    official_score_weighted_eq_list = []
    official_score_weighted_softmax_list = []
    oracle_score_eq_list = []
    oracle_score_weighted_softmax_list = []

    eval_temperature = float(config.get('predict_weight_temperature', 0.25))

    for i in range(batch_size):
        valid_indices = masks[i].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_top_scores = valid_pred[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        pred_top_returns_np = pred_top_returns.detach().cpu().numpy()
        pred_top_scores_np = pred_top_scores.detach().cpu().numpy()
        official_score_eq = pred_return_sum / k
        official_score_weighted_eq = float(np.mean(pred_top_returns_np))
        softmax_weights = _build_eval_weights(pred_top_scores_np, mode='softmax', temperature=eval_temperature)
        official_score_weighted_softmax = float(np.dot(pred_top_returns_np, softmax_weights))

        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        true_top_scores = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        true_top_returns_np = true_top_returns.detach().cpu().numpy()
        true_top_scores_np = true_top_scores.detach().cpu().numpy()
        oracle_score_eq = float(np.mean(true_top_returns_np))
        oracle_softmax_weights = _build_eval_weights(true_top_scores_np, mode='softmax', temperature=eval_temperature)
        oracle_score_weighted_softmax = float(np.dot(true_top_returns_np, oracle_softmax_weights))

        random_return_sum = k * valid_true.mean().item()

        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        official_score_eq_list.append(official_score_eq)
        official_score_weighted_eq_list.append(official_score_weighted_eq)
        official_score_weighted_softmax_list.append(official_score_weighted_softmax)
        oracle_score_eq_list.append(oracle_score_eq)
        oracle_score_weighted_softmax_list.append(oracle_score_weighted_softmax)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        'ratio_pred': np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        'ratio_random': np.mean(ratio_random_list) if ratio_random_list else 0.0,
        'final_score': np.mean(final_score_list) if final_score_list else 0.0,
        'official_score_eq': np.mean(official_score_eq_list) if official_score_eq_list else 0.0,
        'official_score_weighted_eq': np.mean(official_score_weighted_eq_list) if official_score_weighted_eq_list else 0.0,
        'official_score_weighted_softmax': np.mean(official_score_weighted_softmax_list) if official_score_weighted_softmax_list else 0.0,
        'oracle_score_eq': np.mean(oracle_score_eq_list) if oracle_score_eq_list else 0.0,
        'oracle_score_weighted_softmax': np.mean(oracle_score_weighted_softmax_list) if oracle_score_weighted_softmax_list else 0.0,
    }

    return metrics


def _sanitize_stock_indices_array(stock_indices):
    """
    允许 stock_indices 在中间流程里是整数值的 float，
    只要数值本身是整数，就统一转成 int64。
    """
    arr = np.asarray(stock_indices).reshape(-1)

    if arr.size == 0:
        raise ValueError("stock_indices 不能为空")
    if not np.all(np.isfinite(arr)):
        raise ValueError("stock_indices 中存在非有限值")

    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, atol=1e-6):
        raise TypeError(f"stock_indices 必须是整数值，当前样例为: {arr[:10]}")

    return rounded.astype(np.int64)


class LazyRankingDataset(torch.utils.data.Dataset):
    """按日期懒加载排序样本，避免一次性把全部滑窗序列堆进内存。"""
    def __init__(self, data, features, sequence_length, min_window_end_date=None):
        super().__init__()
        self.features = list(features)
        self.sequence_length = int(sequence_length)
        self.stock_feature_store = {}
        self.stock_label_store = {}
        self.samples = []

        dataset_name = '验证集' if min_window_end_date is not None else '训练集'
        print(f"正在构建{dataset_name}懒加载索引...")

        df = data.copy()
        df = df.rename(columns={'日期': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
        df = df.dropna(subset=['label'])

        if min_window_end_date is not None:
            min_window_end_date = pd.to_datetime(min_window_end_date).to_datetime64()

        sample_buckets = defaultdict(lambda: {'entries': [], 'targets': []})

        for stock_code, group in tqdm(df.groupby('instrument'), desc=f"构建{dataset_name}索引"):
            if len(group) < self.sequence_length:
                continue

            feature_values = group[self.features].values.astype(np.float32)
            labels = group['label'].values.astype(np.float32)
            dates = pd.to_datetime(group['datetime']).values

            self.stock_feature_store[int(stock_code)] = feature_values
            self.stock_label_store[int(stock_code)] = labels

            for end_idx in range(self.sequence_length - 1, len(group)):
                target = labels[end_idx]
                if not np.isfinite(target):
                    continue

                end_date = dates[end_idx]
                if min_window_end_date is not None and end_date < min_window_end_date:
                    continue

                bucket = sample_buckets[end_date]
                bucket['entries'].append((int(stock_code), int(end_idx)))
                bucket['targets'].append(float(target))

        for date in tqdm(sorted(sample_buckets.keys()), desc=f"整理{dataset_name}样本"):
            bucket = sample_buckets[date]
            if len(bucket['entries']) < 10:
                continue

            day_targets = np.asarray(bucket['targets'], dtype=np.float32)
            sorted_indices = np.argsort(day_targets)[::-1]
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank

            stock_indices = np.asarray([entry[0] for entry in bucket['entries']], dtype=np.int64)
            self.samples.append({
                'date': date,
                'entries': bucket['entries'],
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': stock_indices,
            })

        print(f"成功构建 {len(self.samples)} 个懒加载样本")
        if self.samples:
            avg_stocks = float(np.mean([len(sample['entries']) for sample in self.samples]))
            print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        day_sequences = []
        for stock_code, end_idx in sample['entries']:
            feature_values = self.stock_feature_store[stock_code]
            start_idx = end_idx - self.sequence_length + 1
            seq = feature_values[start_idx:end_idx + 1]
            day_sequences.append(seq)

        sequences = np.stack(day_sequences).astype(np.float32)
        return {
            'sequences': torch.from_numpy(sequences),
            'targets': torch.from_numpy(sample['targets'].astype(np.float32)),
            'relevance': torch.from_numpy(sample['relevance'].astype(np.float32)),
            'stock_indices': torch.from_numpy(sample['stock_indices'].astype(np.int64)),
        }


class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = [_sanitize_stock_indices_array(x) for x in stock_indices]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),
            'targets': torch.FloatTensor(self.targets[idx]),
            'relevance': torch.FloatTensor(self.relevance_scores[idx]),
            'stock_indices': torch.LongTensor(self.stock_indices[idx]),
        }


def collate_fn(batch):
    """自定义 collate 函数，处理每日股票数量不一致问题。"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'].long() for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    result = {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices).long(),
        'masks': torch.stack(masks),
    }
    return result


def _get_batch_stock_indices(batch, device):
    if 'stock_indices' not in batch:
        raise KeyError("batch 中缺少 stock_indices，当前模型需要该字段支持股票 embedding")

    stock_indices = batch['stock_indices']
    if not isinstance(stock_indices, torch.Tensor):
        raise TypeError(f"stock_indices 必须是 torch.Tensor，当前为: {type(stock_indices)}")

    return stock_indices.long().to(device)


def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    total_metrics = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)
        stock_indices = _get_batch_stock_indices(batch, device)

        optimizer.zero_grad()
        outputs = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())

        masked_outputs = outputs * masks + (1 - masks) * (-1e9)
        masked_targets = targets * masks

        batch_loss = None
        batch_size = sequences.size(0)

        for i in range(batch_size):
            valid_indices = masks[i].nonzero(as_tuple=False).flatten()

            if valid_indices.numel() == 0:
                continue

            valid_pred = masked_outputs[i][valid_indices]
            valid_true = masked_targets[i][valid_indices]

            if len(valid_pred) > 1:
                loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()

            if not config.get('drop_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch * len(dataloader) + local_step)

            optimizer.step()
            total_loss += batch_loss.item()

            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch * len(dataloader) + local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch * len(dataloader) + local_step)

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            stock_indices = _get_batch_stock_indices(batch, device)

            outputs = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())

            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            batch_size = sequences.size(0)

            for i in range(batch_size):
                valid_indices = masks[i].nonzero(as_tuple=False).flatten()

                if valid_indices.numel() == 0:
                    continue

                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]

                if len(valid_pred) > 1:
                    loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()

            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for k in total_metrics:
        total_metrics[k] /= num_batches if num_batches > 0 else 1

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    训练脚本中的辅助预测函数。
    当前模型需要 stock_indices，因此这里也一起传入。
    """
    model.eval()

    latest_date = data['日期'].max()

    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []

    for stock_code in data['股票代码'].unique():
        stock_history = data[
            (data['股票代码'] == stock_code) &
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)

        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])

    if len(day_sequences) == 0:
        return []

    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)
    stock_indices_tensor = torch.LongTensor(np.array(day_stock_indices)).unsqueeze(0).to(device)

    with torch.no_grad():
        stock_mask = torch.ones_like(stock_indices_tensor, dtype=torch.bool, device=device)
        outputs = model(sequences, stock_indices=stock_indices_tensor, stock_mask=stock_mask)
        scores = outputs.squeeze().cpu().numpy()

        top_indices = np.argsort(scores)[::-1][:top_k]

        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })

    return top_stocks


def save_predictions(top_stocks, output_path):
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_n_months(df, sequence_length, val_months):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()

    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    if train_df.empty:
        raise ValueError("训练集为空，请检查 train.csv 的日期范围或 val_months 配置。")
    if val_df.empty:
        raise ValueError("验证集为空，请检查 train.csv 的日期范围或 val_months 配置。")

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后{val_months}个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def main():
    set_seed(config.get('seed', 42))

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    train_file = config.get('train_file', os.path.join(config.get('data_path', './data'), 'train.csv'))
    val_months = int(config.get('val_months', 2))

    full_df = pd.read_csv(train_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)

    train_df, val_df, val_start = split_train_val_by_last_n_months(
        full_df,
        config['sequence_length'],
        val_months
    )

    print(f"训练数据文件: {train_file}")
    print(f"验证区间(月): {val_months}")
    print(f"训练集原始行数: {len(train_df)}")
    print(f"验证集原始行数: {len(val_df)}")

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    val_data, _ = preprocess_val_data(val_df, stockid2idx=stockid2idx)

    scaler = StandardScaler()
    scale_features = get_scale_features(features)

    train_data[scale_features] = train_data[scale_features].replace([np.inf, -np.inf], np.nan)
    val_data[scale_features] = val_data[scale_features].replace([np.inf, -np.inf], np.nan)

    train_data = train_data.dropna(subset=scale_features)
    val_data = val_data.dropna(subset=scale_features)

    if train_data.empty:
        raise ValueError("训练特征处理后为空，请检查数据范围、特征工程或 sequence_length。")
    if val_data.empty:
        raise ValueError("验证特征处理后为空，请检查数据范围、特征工程或 sequence_length。")

    train_data[scale_features] = scaler.fit_transform(train_data[scale_features])
    val_data[scale_features] = scaler.transform(val_data[scale_features])

    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    if config.get('use_lazy_dataset', True):
        train_dataset = LazyRankingDataset(
            train_data,
            features,
            config['sequence_length'],
            min_window_end_date=None,
        )
        val_dataset = LazyRankingDataset(
            val_data,
            features,
            config['sequence_length'],
            min_window_end_date=val_start.strftime('%Y-%m-%d'),
        )
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(val_dataset)}")
    else:
        train_sequences, train_targets, train_relevance, train_stock_indices = create_ranking_dataset_vectorized(
            train_data,
            features,
            config['sequence_length'],
            ranking_data_path=config.get('train_ranking_data_path')
        )
        val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
            val_data,
            features,
            config['sequence_length'],
            ranking_data_path=config.get('val_ranking_data_path'),
            min_window_end_date=val_start.strftime('%Y-%m-%d')
        )

        print(f"训练集样本数: {len(train_sequences)}")
        print(f"验证集样本数: {len(val_sequences)}")

        if len(train_sequences) == 0:
            raise ValueError("训练排序样本数为 0，请检查数据范围、sequence_length 或样本构造逻辑。")
        if len(val_sequences) == 0:
            raise ValueError("验证排序样本数为 0，请检查 val_months、sequence_length 或样本构造逻辑。")

        train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
        val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)

    if len(train_dataset) == 0:
        raise ValueError("训练排序样本数为 0，请检查数据范围、sequence_length 或样本构造逻辑。")
    if len(val_dataset) == 0:
        raise ValueError("验证排序样本数为 0，请检查 val_months、sequence_length 或样本构造逻辑。")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    model = StockTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = WeightedRankingLoss(
        k=5,
        temperature=config.get('ranking_temperature', 0.05),
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=float(config.get('weight_decay', 1e-5))
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=config['num_epochs']
    )

    selection_metric_name = _resolve_model_selection_metric()
    print(f"模型选择指标: {selection_metric_name}")

    best_score = -float('inf')
    best_epoch = -1

    try:
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")

            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )

            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )

            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.4f}")

            writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            scheduler.step()

            current_selection_score = float(eval_metrics.get(selection_metric_name, 0.0))
            if current_selection_score > best_score:
                best_score = current_selection_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                print(f"保存最佳模型 - {selection_metric_name}: {best_score:.6f}")

    finally:
        writer.close()

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 {selection_metric_name}: {best_score:.6f}")
    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Selection metric: {selection_metric_name}\n")
        f.write(f"Best score: {best_score:.6f}\n")

    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    best_score = main()
    print(f"\n########## 训练完成！最佳 official_score_eq: {best_score:.6f} ##########")