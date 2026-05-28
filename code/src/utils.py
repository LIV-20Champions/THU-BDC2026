import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


def engineer_features_158plus39(df):
    df_copy = df.copy()
    df_158 = engineer_features_158(df_copy)
    df_39 = engineer_features_39(df_copy)

    orig_cols = list(df.columns)
    feature_cols_158 = [c for c in df_158.columns if c not in orig_cols]
    feature_cols_39 = [c for c in df_39.columns if c not in orig_cols]

    df_final = pd.concat([df, df_158[feature_cols_158], df_39[feature_cols_39]], axis=1)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final.fillna(method='ffill', inplace=True)
    df_final.fillna(0, inplace=True)

    return df_final


def engineer_features_158(df):
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)

    df['sma_5'] = talib.SMA(close, timeperiod=5)
    df['sma_20'] = talib.SMA(close, timeperiod=20)
    df['ema_12'] = talib.EMA(close, timeperiod=12)
    df['ema_26'] = talib.EMA(close, timeperiod=26)
    df['ema_60'] = talib.EMA(close, timeperiod=60)

    macd_line, macd_signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal_line

    df['rsi'] = talib.RSI(close, timeperiod=14)

    df['kdj_k'], df['kdj_d'] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

    df['boll_mid'], df['boll_upper'], df['boll_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['boll_std'] = (df['boll_upper'] - df['boll_mid']) / 2
    df.drop(columns=['boll_upper', 'boll_lower'], inplace=True)

    df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    df['obv'] = talib.OBV(close, volume)

    df['volume_change'] = volume.pct_change()
    df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
    df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)
    df['volume_ratio'] = df['volume_ma_5'] / df['volume_ma_20']

    df['return_1'] = close.pct_change(1)
    df['return_5'] = close.pct_change(5)
    df['return_10'] = close.pct_change(10)
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_20'] = df['return_1'].rolling(20).std()

    df['high_low_spread'] = high - low
    df['open_close_spread'] = open_ - close
    df['high_close_spread'] = high - close
    df['low_close_spread'] = low - close

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    return df


def engineer_features_39(df):
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    df = df.copy()

    open_ = df['开盘'].astype(float)
    high = df['最高'].astype(float)
    low = df['最低'].astype(float)
    close = df['收盘'].astype(float)
    volume = df['成交量'].astype(float)
    vwap = df['成交额'] / (volume + 1e-12)

    features = []
    feature_names = []

    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12)
    ])
    feature_names.extend(['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2', 'KLOW', 'KLOW2', 'KSFT', 'KSFT2'])

    features.extend([
        open_ / (close + 1e-12),
        high / (close + 1e-12),
        low / (close + 1e-12),
        vwap / (close + 1e-12)
    ])
    feature_names.extend(['OPEN0', 'HIGH0', 'LOW0', 'VWAP0'])

    windows = [5, 10, 20, 30, 60]

    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f'ROC{w}')

    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MA{w}')

    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'STD{w}')

    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f'BETA{w}')

        eff_w = min(w, max(len(close) - 1, 2))
        time_period_series = pd.Series(np.arange(len(close)), index=close.index)
        rolling_corr = close.rolling(w).corr(time_period_series)
        rsquare = rolling_corr ** 2
        features.append(rsquare)
        feature_names.append(f'RSQR{w}')

        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f'RESI{w}')

    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MAX{w}')
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f'MIN{w}')

    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f'QTLU{w}')
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f'QTLD{w}')

    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f'RANK{w}')

    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f'RSV{w}')

    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f'IMAX{w}')
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f'IMIN{w}')
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f'IMXD{w}')

    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f'CORR{w}')

    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f'CORD{w}')

    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f'CNTP{w}')
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f'CNTN{w}')
    for w in windows:
        cntp = close_diff_pos.rolling(w).mean()
        cntn = close_diff_neg.rolling(w).mean()
        features.append(cntp - cntn)
        feature_names.append(f'CNTD{w}')

    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'SUMP{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'SUMN{w}')
    for w in windows:
        sum_abs = close_diff_abs.rolling(w).sum()
        sum_up = close_diff_up.rolling(w).sum()
        sum_down = close_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'SUMD{w}')

    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VMA{w}')
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f'VSTD{w}')

    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vol_w_ret = vol_weighted_ret.rolling(w).mean()
        std_vol_w_ret = vol_weighted_ret.rolling(w).std()
        features.append(std_vol_w_ret / (mean_vol_w_ret + 1e-12))
        feature_names.append(f'WVMA{w}')

    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f'VSUMP{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f'VSUMN{w}')
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f'VSUMD{w}')

    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names

    df = pd.concat([df, feature_df], axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)
    return df


def create_ranking_dataset_vectorized(data, features, sequence_length, ranking_data_path=None, min_window_end_date=None):
    print("正在创建排序数据集（向量化加速版本）...")
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    data = data.dropna(subset=['label'])

    all_windows = []

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')

    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue

        feature_values = group[features].values.astype(np.float32)
        labels = group['label'].values.astype(np.float32)
        dates = group['datetime'].values

        num_windows = len(group) - sequence_length + 1
        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            seq = feature_values[i: i + sequence_length]
            target = labels[end_idx]
            end_date = dates[end_idx]
            all_windows.append((end_date, stock_code, seq, target))

    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target'])

    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)

    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue

        if len(group) < 10:
            continue

        day_seqs = np.stack(group['seq'].values)
        day_targets = group['target'].values
        day_stocks = group['stock_code'].tolist()

        sorted_indices = np.argsort(day_targets)[::-1]
        relevance = np.zeros_like(day_targets, dtype=np.float32)
        for rank, idx in enumerate(sorted_indices):
            relevance[idx] = len(day_targets) - rank

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    return sequences, targets, relevance_scores, stock_indices


def per_stock_sliding_zscore(feature_values, sequence_length, clip_range=5.0):
    T, F = feature_values.shape
    df = pd.DataFrame(feature_values)
    rolling = df.rolling(window=sequence_length, min_periods=1)
    mean = rolling.mean().values
    std = rolling.std().values
    std = np.where(std < 1e-8, 1e-8, std)
    norm = (feature_values - mean) / std
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(norm, -clip_range, clip_range)


def build_cross_sectional_features(sequences, cs_feature_types):
    N, L, F = sequences.shape
    seq_numeric = sequences[..., 1:]
    cs_list = []
    for cs_type in cs_feature_types:
        if cs_type == 'rank_pct':
            ranks = seq_numeric.argsort(axis=0).argsort(axis=0).astype(np.float32)
            cs = ranks / max(N - 1, 1)
        elif cs_type == 'zscore':
            mean = seq_numeric.mean(axis=0, keepdims=True)
            std = seq_numeric.std(axis=0, keepdims=True) + 1e-8
            cs = (seq_numeric - mean) / std
            cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            continue
        cs_list.append(cs)
    return np.concatenate([sequences, np.concatenate(cs_list, axis=-1)], axis=-1)


class LazyRankingDataset(torch.utils.data.Dataset):
    def __init__(self, data, features, sequence_length, min_window_end_date=None,
                 use_per_stock_norm=True, use_cs_features=False,
                 cs_feature_types=None, selected_features=None):
        super().__init__()
        full_features = list(features)
        if selected_features is not None:
            selected_set = set(selected_features)
            self.features = [f for f in full_features if f in selected_set]
            if 'instrument' in full_features and 'instrument' not in selected_set:
                self.features = ['instrument'] + self.features
            dropped = len(full_features) - len(self.features)
            print(f"Feature selection: {len(full_features)} -> {len(self.features)} ({dropped} dropped)")
        else:
            self.features = full_features
        self.sequence_length = int(sequence_length)
        self.stock_feature_store = {}
        self.stock_feature_store_norm = {}
        self.stock_label_store = {}
        self.samples = []
        self.use_per_stock_norm = use_per_stock_norm
        self.use_cs_features = use_cs_features
        self.cs_feature_types = cs_feature_types if cs_feature_types else []
        self.cs_multiplier = 1 + len(self.cs_feature_types) if self.use_cs_features else 1

        dataset_name = '验证集' if min_window_end_date is not None else '训练集'
        print(f"正在构建{dataset_name}懒加载索引...")

        df = data
        df = df.rename(columns={'日期': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values(['instrument', 'datetime']).reset_index(drop=True)
        df = df.dropna(subset=['label', 'score_target'])

        if min_window_end_date is not None:
            min_window_end_date = pd.to_datetime(min_window_end_date).to_datetime64()

        sample_buckets = defaultdict(lambda: {'entries': [], 'targets': [], 'score_targets': []})

        for stock_code, group in tqdm(df.groupby('instrument'), desc=f"构建{dataset_name}索引"):
            if len(group) < self.sequence_length:
                continue

            feature_values = group[self.features].values.astype(np.float32)
            labels = group['label'].values.astype(np.float32)
            score_targets = group['score_target'].values.astype(np.float32)
            dates = pd.to_datetime(group['datetime']).values

            if self.use_per_stock_norm:
                norm_features = per_stock_sliding_zscore(
                    feature_values.copy(), self.sequence_length, clip_range=5.0
                )
                self.stock_feature_store_norm[int(stock_code)] = norm_features.astype(np.float16)
            else:
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
                bucket['score_targets'].append(float(score_targets[end_idx]))

        for date in tqdm(sorted(sample_buckets.keys()), desc=f"整理{dataset_name}样本"):
            bucket = sample_buckets[date]
            if len(bucket['entries']) < 10:
                continue

            day_targets = np.asarray(bucket['targets'], dtype=np.float32)
            score_targets_day = np.asarray(bucket['score_targets'], dtype=np.float32)
            sorted_indices = np.argsort(day_targets)[::-1]
            relevance = np.zeros_like(day_targets, dtype=np.float32)
            for rank, idx in enumerate(sorted_indices):
                relevance[idx] = len(day_targets) - rank

            stock_indices = np.asarray([entry[0] for entry in bucket['entries']], dtype=np.int64)

            self.samples.append({
                'date': date,
                'entries': bucket['entries'],
                'targets': day_targets,
                'score_targets': score_targets_day,
                'relevance': relevance,
                'stock_indices': stock_indices,
            })

        print(f"成功构建 {len(self.samples)} 个懒加载样本")
        if self.samples:
            avg_stocks = float(np.mean([len(sample['entries']) for sample in self.samples]))
            print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

        if self.use_per_stock_norm:
            self.stock_feature_store = {}
        self.stock_label_store = {}

    def __len__(self):
        return len(self.samples)

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        entries = sample['entries']
        max_stocks = getattr(self, '_max_stocks', 0)

        if max_stocks > 0 and len(entries) > max_stocks:
            perm = np.random.permutation(len(entries))[:max_stocks]
            entries = [entries[i] for i in perm]
            targets_sub = sample['targets'][perm]
            score_targets_sub = sample['score_targets'][perm]
            relevance_sub = sample['relevance'][perm]
            stock_indices_sub = sample['stock_indices'][perm]
        else:
            targets_sub = sample['targets']
            score_targets_sub = sample['score_targets']
            relevance_sub = sample['relevance']
            stock_indices_sub = sample['stock_indices']

        feat_store = self.stock_feature_store_norm if self.use_per_stock_norm else self.stock_feature_store
        seq_len = self.sequence_length
        n_stocks = len(entries)
        n_features = feat_store[entries[0][0]].shape[1]

        sequences = np.empty((n_stocks, seq_len, n_features), dtype=np.float32)
        for i, (stock_code, end_idx) in enumerate(entries):
            start_idx = end_idx - seq_len + 1
            sequences[i] = feat_store[stock_code][start_idx:end_idx + 1]

        if self.use_cs_features:
            sequences = build_cross_sectional_features(sequences, self.cs_feature_types)
        return {
            'sequences': torch.from_numpy(sequences),
            'targets': torch.from_numpy(targets_sub.astype(np.float32)),
            'score_targets': torch.from_numpy(score_targets_sub.astype(np.float32)),
            'relevance': torch.from_numpy(relevance_sub.astype(np.float32)),
            'stock_indices': torch.from_numpy(stock_indices_sub.astype(np.int64)),
        }
