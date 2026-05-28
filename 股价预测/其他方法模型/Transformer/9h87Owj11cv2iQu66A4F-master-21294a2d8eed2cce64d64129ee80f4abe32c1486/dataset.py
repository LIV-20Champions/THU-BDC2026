#!/usr/bin/env python3
"""
数据预处理与加载器模块
实现数据分割、归一化、滑窗生成样本等功能
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler

# 配置参数
elements = ['open', 'close', 'high', 'low', 'volume', 'money']  # 6个特征
data_path = 'data/601398.XSHG.csv'

class StockDataset(Dataset):
    """股票数据集类"""
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        return self.X_data[idx], self.y_data[idx]

def split_data(batch_size=64, seq_length=5, pred_length=1, train_ratio=0.8):
    """
    数据分割函数 - 用N天数据预测下一天
    参数:
        batch_size: 批处理大小
        seq_length: 时间序列长度 (用多少天预测)
        pred_length: 预测长度 (预测未来几天)
        train_ratio: 训练集比例
    返回:
        train_loader, val_loader, min_val, max_val
    """
    print(f"开始处理数据: seq_length={seq_length}, batch_size={batch_size}")
    
    # 读取数据
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件 {data_path} 不存在，请先运行 data_download.py")
    
    data_all = pd.read_csv(data_path)
    print(f"原始数据条数: {len(data_all)}")
    
    # 数据清洗 - 移除缺失值
    data_all = data_all.dropna()
    print(f"清洗后数据条数: {len(data_all)}")
    
    # 提取需要的特征列
    data_ha = []
    length = len(data_all)
    
    for element in elements:
        if element in data_all.columns:
            data_element = data_all[element].values.astype(np.float32)
        else:
            # 如果列不存在，创建模拟数据
            print(f"警告: 列 {element} 不存在，使用模拟数据")
            data_element = np.random.normal(0, 1, length).astype(np.float32)
        
        data_element = data_element.reshape(length, 1)
        data_ha.append(data_element)
    
    # 合并所有特征
    X_hat = np.concatenate(data_ha, axis=1)  # shape: (length, 6)
    print(f"特征数据形状: {X_hat.shape}")
    
    # 数据归一化到0-1范围
    min_val = np.min(X_hat, axis=0)
    max_val = np.max(X_hat, axis=0)
    
    # 避免除零
    range_val = max_val - min_val
    range_val[range_val == 0] = 1.0
    
    X_normalized = (X_hat - min_val) / range_val
    
    # 转换为PyTorch张量
    X_CONVERT = torch.from_numpy(X_normalized).float()
    
    # 数据翻转 - 使时间顺序正确 (最新的数据在最后)
    X_CONVERT = X_CONVERT.flip(dims=[0])
    
    print(f"归一化后数据范围: [{X_CONVERT.min():.4f}, {X_CONVERT.max():.4f}]")
    
    # 创建滑窗样本
    # 用seq_length天的数据预测第seq_length+1天的close价格
    X_data = []
    y_data = []
    
    for i in range(len(X_CONVERT) - seq_length):
        # X: 连续的seq_length天所有特征
        X_data.append(X_CONVERT[i:i+seq_length, :])
        # y: 第seq_length+1天的close价格 (索引1对应close列)
        y_data.append(X_CONVERT[i+seq_length, 1])  # 1是close列的索引
    
    # 堆叠数据
    X_data = torch.stack(X_data)  # shape: (samples, seq_length, features)
    y_data = torch.stack(y_data).squeeze(-1)  # shape: (samples,)
    
    print(f"X_data形状: {X_data.shape}")
    print(f"y_data形状: {y_data.shape}")
    print(f"样本总数: {len(X_data)}")
    
    # 创建数据集
    dataset = TensorDataset(X_data, y_data)
    
    # 划分训练集和验证集
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子
    )
    
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, min_val, max_val

def inverse_transform_predictions(predictions, min_val, max_val):
    """
    将归一化的预测值转换回原始尺度
    参数:
        predictions: 归一化的预测值
        min_val: close列的最小值
        max_val: close列的最大值
    返回:
        反归一化后的预测值
    """
    # 只对close列(索引1)进行反归一化
    close_min = min_val[1]
    close_max = max_val[1]
    
    # 反归一化: y = y_norm * (max - min) + min
    original_predictions = predictions * (close_max - close_min) + close_min
    
    return original_predictions

if __name__ == "__main__":
    # 测试数据加载
    try:
        train_loader, val_loader, min_val, max_val = split_data(
            batch_size=32, 
            seq_length=5, 
            train_ratio=0.8
        )
        
        print("\n数据加载测试成功!")
        print(f"训练集批次数量: {len(train_loader)}")
        print(f"验证集批次数量: {len(val_loader)}")
        
        # 检查一个批次
        for X_batch, y_batch in train_loader:
            print(f"一个训练批次:")
            print(f"  X形状: {X_batch.shape}")  # (batch_size, seq_length, features)
            print(f"  y形状: {y_batch.shape}")  # (batch_size,)
            break
            
    except Exception as e:
        print(f"数据加载测试失败: {e}")
        print("请先运行: python data_download.py")