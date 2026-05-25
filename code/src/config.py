from utils import engineer_features_39, engineer_features_158plus39, engineer_features_158

sequence_length = 60
feature_num = '39'
config = {
    'sequence_length': sequence_length,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'batch_size': 2,
    'max_stocks_per_sample': 100,
    'num_epochs': 60,
    'learning_rate': 5e-5,
    'dropout': 0.35,
    'feature_num': feature_num,

    'use_stock_embedding': True,
    'stock_emb_dim': 16,
    'use_lazy_dataset': True,

    'max_grad_norm': 5.0,
    'enable_grad_clip': True,
    'num_workers': 2,
    'pin_memory': False,
    'feature_engineering_workers': 2,

    'predict_weight_mode': 'rank_decay',
    'predict_top_k': 5,
    'predict_rank_weights': [0.40, 0.25, 0.18, 0.11, 0.06],
    'predict_rank_alpha': 0.2,
    'predict_top3_weights': [0.45, 0.35, 0.20],
    'predict_weight_temperature': 0.5,

    'label_alpha': 0.3,
    'selected_top_k_features': 0,
    'output_dir': './model/phase8_ensemble',
    'ensemble_model_dirs': [],
    'data_path': './data',
    'val_months': 2,

    'warmup_epochs': 5,
    'cosine_min_lr_ratio': 0.01,
    'early_stopping_patience': 15,
    'gradient_accumulation_steps': 4,
    'weight_decay': 5e-4,

    'use_gru_dual_path': False,
    'gru_hidden_dim': 128,
    'gru_num_layers': 1,
    'cross_stock_layers': 2,
    'layer_scale_init': 1e-5,
    'drop_path_rate': 0.1,
    'use_feature_interaction': False,
    'interaction_hidden_dim': 64,

    'use_per_stock_normalize': True,

    'use_cross_sectional_features': False,
    'cs_feature_types': ['rank_pct', 'zscore'],

    'use_vsn': False,
    'vsn_num_groups': 10,
    'vsn_hidden_size': 64,
    'vsn_temperature': 1.0,
    'grn_hidden_size': 64,
    'grn_dropout': 0.1,
    'use_grn_in_temporal': False,

    'use_multi_scale': False,
    'ms_short_patch': 5,
    'ms_medium_segment': 15,
    'ms_short_layers': 1,
    'ms_medium_layers': 1,
    'ms_long_layers': 1,
    'ms_fusion_mode': 'linear',

    'use_smooth_ndcg_loss': True,
    'smooth_ndcg_weight': 0.6,
    'lambda_pairwise_weight': 0.4,
    'ndcg_top_k': 5,
    'soft_sort_temperature': 0.2,
    'temperature_anneal_start': 2.0,
    'temperature_anneal_target': 0.5,
    'temperature_anneal_epochs': 30,
    'lambda_delta_clip': 10.0,
    'use_mse_aux_loss': False,
    'mse_aux_weight': 0.05,

    'use_ema': True,
    'ema_decay': 0.999,
    'use_swa': True,
    'swa_start_epoch': 15,
    'swa_lookahead': 5,
    'swa_lr': 1e-5,
    'swa_anneal_epochs': 5,
    'use_mixup': True,
    'mixup_alpha': 0.2,
    'mixup_prob': 0.3,
    'use_label_smoothing': True,
    'label_smoothing_alpha': 0.03,
    'ensemble_size': 5,

    'use_amp': True,
    'use_gradient_checkpointing': True,
}

# ===== 共享数据结构：train.py 与 predict.py 共用的映射 =====
feature_columns_map = {
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
    '39': engineer_features_158,
    '158': engineer_features_39,
    '158+39': engineer_features_158plus39,
}


def get_scale_features(features):
    if config.get('use_stock_embedding', False):
        return [f for f in features if f != 'instrument']
    return list(features)


def get_eff_input_dim(num_features):
    use_stock_emb = config.get('use_stock_embedding', True)
    use_cs = config.get('use_cross_sectional_features', False)
    cs_types = config.get('cs_feature_types', ['rank_pct', 'zscore'])
    cs_mult = (1 + len(cs_types)) if use_cs else 1
    if use_stock_emb:
        return 1 + (num_features - 1) * cs_mult
    return num_features * cs_mult