# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # Transformer输入维度
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 8,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 30,       # 排序任务可能需要更多epochs
    'learning_rate': 5e-5,  # 稍微降低学习率
    'dropout': 0.1,
    'feature_num': feature_num,

    # 股票代码是离散ID：启用 embedding 后，instrument 不能参与 StandardScaler。
    'use_stock_embedding': True,
    'stock_emb_dim': 16,
    'use_lazy_dataset': True,
    'val_months': 2,
    'max_grad_norm': 5.0,

    'pairwise_weight': 1, # 配对损失权重
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}