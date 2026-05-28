#!/usr/bin/env python3
"""
预测脚本 - 使用训练好的Transformer模型进行股价预测
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入自定义模块
from model import Transformer
from dataset import split_data
from evaluate import inverse_transform, plot_predictions

def load_model(model_path='models/best_model.pt', device='cpu'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 设备
        
    Returns:
        加载好的模型
    """
    model = Transformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_future(model, recent_data, min_val, max_val, device='cpu'):
    """
    使用最近的数据预测未来股价
    
    Args:
        model: 训练好的模型
        recent_data: 最近seq_length天的数据
        min_val: 归一化最小值
        max_val: 归一化最大值
        device: 设备
        
    Returns:
        预测结果
    """
    model.eval()
    
    with torch.no_grad():
        # 准备输入数据
        if isinstance(recent_data, np.ndarray):
            recent_data = torch.from_numpy(recent_data).float()
        
        recent_data = recent_data.unsqueeze(0).to(device)  # 添加batch维度
        
        # 预测
        prediction, attention_weights = model(recent_data)
        
        # 反归一化
        prediction_orig = inverse_transform(prediction.cpu().numpy(), min_val, max_val)
        
        return prediction_orig[0], attention_weights

def demo_prediction():
    """演示预测功能"""
    print("=== Transformer股价预测演示 ===\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件
    model_path = 'models/best_model.pt'
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        print("请先运行: python train.py")
        return
    
    # 加载数据（用于获取归一化参数）
    print("加载数据...")
    try:
        _, _, min_val, max_val = split_data(batch_size=32, seq_length=5, pred_length=1, train_ratio=0.8)
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请先运行: python data_download.py")
        return
    
    # 加载模型
    print("加载模型...")
    model = load_model(model_path, device)
    print("模型加载成功!")
    
    # 生成模拟的最近5天数据（实际应用中应该使用真实数据）
    print("\n生成测试数据...")
    np.random.seed(42)
    
    # 模拟6个特征: open, close, high, low, volume, money
    # 这里使用随机数据，实际应用中应该使用真实的股票数据
    recent_5_days = np.random.randn(5, 6).astype(np.float32)
    
    # 为了演示，我们让数据看起来更像真实的股价数据
    base_price = 4.5  # 基于工商银行的大致股价
    recent_5_days[:, 0] = base_price + np.random.randn(5) * 0.1  # open
    recent_5_days[:, 1] = base_price + np.random.randn(5) * 0.1  # close
    recent_5_days[:, 2] = base_price + 0.1 + np.random.randn(5) * 0.05  # high
    recent_5_days[:, 3] = base_price - 0.1 + np.random.randn(5) * 0.05  # low
    recent_5_days[:, 4] = np.random.randn(5) * 1000000 + 50000000  # volume
    recent_5_days[:, 5] = recent_5_days[:, 4] * recent_5_days[:, 1]  # money
    
    print("最近5天数据（归一化前）:")
    print("Day | Open  | Close | High  | Low   | Volume    | Money")
    print("-" * 55)
    for i, day_data in enumerate(recent_5_days):
        print(f"{i+1:3d} | {day_data[0]:5.2f} | {day_data[1]:5.2f} | {day_data[2]:5.2f} | {day_data[3]:5.2f} | {day_data[4]:9.0f} | {day_data[5]:11.0f}")
    
    # 预测
    print("\n进行预测...")
    prediction, attention_weights = predict_future(model, recent_5_days, min_val, max_val, device)
    
    print(f"\n预测结果:")
    print(f"预测的下一天收盘价: {prediction:.4f} 元")
    
    # 可视化注意力权重（如果有）
    if attention_weights and len(attention_weights) > 0:
        print(f"\n注意力权重信息:")
        print(f"注意力层数: {len(attention_weights)}")
        print(f"最后一层注意力形状: {attention_weights[-1].shape}")
        
        # 绘制注意力热力图
        try:
            plt.figure(figsize=(10, 8))
            
            # 使用最后一层的注意力权重
            attn_matrix = attention_weights[-1][0].cpu().numpy()  # 取第一个样本
            plt.imshow(attn_matrix, cmap='hot', interpolation='nearest')
            plt.colorbar(label='注意力权重')
            plt.title('Transformer注意力权重热力图', fontsize=14, fontweight='bold')
            plt.xlabel('Key位置')
            plt.ylabel('Query位置')
            
            # 保存图表
            os.makedirs('picture', exist_ok=True)
            plt.savefig('picture/attention_heatmap.png', dpi=300, bbox_inches='tight')
            print("注意力权重热力图已保存: picture/attention_heatmap.png")
            plt.show()
            
        except Exception as e:
            print(f"绘制注意力图失败: {e}")
    
    print("\n预测演示完成!")
    
    # 提供一些使用建议
    print("\n" + "="*50)
    print("使用建议:")
    print("1. 在实际应用中，请使用真实的股票数据")
    print("2. 模型预测仅供参考，不构成投资建议")
    print("3. 可以通过调整模型参数来提高预测精度")
    print("4. 建议定期重新训练模型以适应市场变化")
    print("="*50)

def predict_with_real_data(data_path, model_path='models/best_model.pt'):
    """
    使用真实数据进行预测
    
    Args:
        data_path: 数据文件路径
        model_path: 模型文件路径
    """
    print(f"使用真实数据预测: {data_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = load_model(model_path, device)
    
    # 加载数据
    try:
        data = pd.read_csv(data_path)
        print(f"数据形状: {data.shape}")
        print(f"列名: {list(data.columns)}")
        
        # 这里需要实现具体的数据预处理逻辑
        # 根据dataset.py中的逻辑处理数据
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        # 使用真实数据预测
        if len(sys.argv) > 2:
            predict_with_real_data(sys.argv[2])
        else:
            print("请提供数据文件路径: python predict.py --real <data_path>")
    else:
        # 演示预测
        demo_prediction()