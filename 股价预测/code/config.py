# 配置参数
sequence_length = 60
feature_num = '100+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 128,           # 标准容量
    'nhead': 8,              # 注意力头数量
    'num_layers': 2,         # 减少层数防过拟合
    'dim_feedforward': 512,  # 标准前馈网络
    'batch_size': 4,         # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 40,        # 排序任务可能需要更多epochs；调试时可临时改成 3 或 5
    'learning_rate': 5e-5,   # 标准学习率
    'dropout': 0.35,         # 微增dropout防过拟合（相对原0.3）
    'weight_decay': 5e-4,    # 标准L2正则化
    'val_months': 4,         # 验证窗口
    'early_stop_patience': 8, # 验证分连续8轮不创新高即早停
    'model_selection_metric': 'official_score_eq', # 稳定优先：用等权Top5验证分选择模型
    'feature_num': feature_num,
    'ranking_temperature': 0.07,  # 轻微提高温度，平滑损失面

    # 股票代码是离散ID：启用 embedding 后，instrument 不能参与 StandardScaler。
    'use_stock_embedding': True,
    'stock_emb_dim': 16,         # 标准嵌入维度
    'num_ensemble_models': 3,   # 保存top-K个模型用于推理时集成平均
    'use_lazy_dataset': True,

    'max_grad_norm': 5.0,
    'clip_grad': True,
    'warmup_ratio': 0.05,
    'lr_min_factor': 0.2,
    'feature_attention_activation': 'gelu',
    'use_cross_sectional_rank': True,   # 截面rank消除市场牛熊噪声，关键预处理
    'cross_sectional_rank_skip_cols': ['instrument'],

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 3.0, # top-5样本权重（应大于base_weight）

    # 第一阶段：预测权重策略。
    # 只改这些参数不需要重新训练，只需要重新运行 test.sh，或使用 tools/quick_weight_search.py 快速搜索。
    # 稳定优先默认使用 equal。confidence/confidence_flex 保留用于诊断，但不作为默认提交策略。
    # 可选：equal / rank_decay / top3 / top1 / softmax / confidence / confidence_flex
    'predict_weight_mode': 'equal',
    'predict_top_k': 5,
    'predict_rank_weights': [0.30, 0.25, 0.20, 0.15, 0.10],
    'predict_top3_weights': [0.45, 0.35, 0.20],
    'predict_weight_temperature': 0.5,

    # confidence 模式：先量化第一名优势，再决定是否单押。
    # 分数不是概率，因此这里用分数差距、softmax集中度、分布熵组合成置信率。
    'confidence_top1_threshold': 0.70,
    'confidence_top3_threshold': 0.40,
    'confidence_min_margin12': 0.01,
    'confidence_margin_scale': 0.05,
    'confidence_temperature': 0.05,
    'confidence_fallback_mode': 'equal',

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
