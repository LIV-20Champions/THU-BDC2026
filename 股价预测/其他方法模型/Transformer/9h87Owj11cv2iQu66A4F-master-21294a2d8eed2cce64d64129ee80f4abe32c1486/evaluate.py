#!/usr/bin/env python3
"""
评估与可视化模块
计算MAE、RMSE、PCC等指标，并绘制预测结果图表
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import torch

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        dict: 包含MAE、RMSE、PCC的字典
    """
    # 确保是numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 处理PCC计算（需要至少两个样本）
    if len(y_true) > 1:
        pcc, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    else:
        pcc = 0.0
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'PCC': pcc
    }

def inverse_transform(y_normalized, min_val, max_val):
    """
    反归一化
    
    Args:
        y_normalized: 归一化后的值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        反归一化后的值
    """
    return y_normalized * (max_val - min_val) + min_val

def plot_predictions(y_true, y_pred, min_val, max_val, save_path=None, title="股价预测结果"):
    """
    绘制预测结果图表
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        min_val: 最小值（用于反归一化）
        max_val: 最大值（用于反归一化）
        save_path: 保存路径
        title: 图表标题
    """
    # 反归一化
    y_true_orig = inverse_transform(y_true, min_val, max_val)
    y_pred_orig = inverse_transform(y_pred, min_val, max_val)
    
    # 计算指标
    metrics = calculate_metrics(y_true_orig, y_pred_orig)
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制真实值和预测值
    plt.subplot(2, 1, 1)
    plt.plot(y_true_orig, label='真实值', color='blue', linewidth=2)
    plt.plot(y_pred_orig, label='预测值', color='red', linewidth=2, alpha=0.8)
    plt.title(f'{title} - 真实值 vs 预测值', fontsize=14, fontweight='bold')
    plt.xlabel('时间步')
    plt.ylabel('股价 (元)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加指标文本
    metrics_text = f"MAE: {metrics['MAE']:.4f} | RMSE: {metrics['RMSE']:.4f} | PCC: {metrics['PCC']:.4f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 绘制残差
    plt.subplot(2, 1, 2)
    residuals = y_true_orig - y_pred_orig
    plt.plot(residuals, label='残差', color='green', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('预测残差', fontsize=14, fontweight='bold')
    plt.xlabel('时间步')
    plt.ylabel('残差 (元)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    
    # 显示图表
    plt.show()
    
    return metrics

def evaluate_model(model, data_loader, min_val, max_val, device='cpu'):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        min_val: 最小值
        max_val: 最大值
        device: 设备
        
    Returns:
        dict: 评估指标
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            
            predictions, _ = model(X_batch)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(y_batch.cpu())
    
    # 合并所有批次
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 计算指标
    metrics = calculate_metrics(all_targets, all_predictions)
    
    # 反归一化后的指标
    all_targets_orig = inverse_transform(all_targets.numpy(), min_val, max_val)
    all_predictions_orig = inverse_transform(all_predictions.numpy(), min_val, max_val)
    metrics_orig = calculate_metrics(all_targets_orig, all_predictions_orig)
    
    return {
        'normalized': metrics,
        'original': metrics_orig,
        'predictions': all_predictions.numpy(),
        'targets': all_targets.numpy(),
        'predictions_orig': all_predictions_orig,
        'targets_orig': all_targets_orig
    }

if __name__ == "__main__":
    # 测试评估函数
    print("测试评估函数...")
    
    # 生成测试数据
    np.random.seed(42)
    y_true_test = np.random.randn(100) * 10 + 100  # 模拟股价数据
    y_pred_test = y_true_test + np.random.randn(100) * 2  # 添加噪声
    
    min_val, max_val = 80, 120
    
    # 测试绘图函数
    test_metrics = plot_predictions(y_true_test, y_pred_test, min_val, max_val, 
                                   save_path="picture/test_plot.png", 
                                   title="测试预测结果")
    
    print("测试指标:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n评估模块测试完成!")