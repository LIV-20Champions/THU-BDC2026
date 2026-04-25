# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 80,       # 排序任务可能需要更多epochs；调试时可临时改成 3 或 5
    'learning_rate': 5e-5,  # 稍微降低学习率
    'dropout': 0.2,
    'feature_num': feature_num,

    # 股票代码是离散ID：启用 embedding 后，instrument 不能参与 StandardScaler。
    'use_stock_embedding': True,
    'stock_emb_dim': 16,
    'use_lazy_dataset': True,

    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 5.0, # top-5样本权重（应大于base_weight）

    # 第一阶段：预测权重策略。
    # 只改这些参数不需要重新训练，只需要重新运行 test.sh，或使用 tools/quick_weight_search.py 快速搜索。
    # 可选：equal / rank_decay / top3 / top1 / softmax
    'predict_weight_mode': 'top1',
    'predict_top_k': 5,
    'predict_rank_weights': [0.30, 0.25, 0.20, 0.15, 0.10],
    'predict_top3_weights': [0.45, 0.35, 0.20],
    'predict_weight_temperature': 0.5,

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}
