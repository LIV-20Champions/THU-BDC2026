#!/usr/bin/env python3
"""
主训练脚本 - Transformer股价预测模型训练
整合所有模块，实现完整的训练循环
"""
import os
import sys
import copy
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入自定义模块
from dataset import split_data
from model import Transformer
from evaluate import evaluate_model, plot_predictions

# 训练参数
batch_size = 64
seq_length = 5  # 用5天数据预测下一天
pred_length = 1
train_ratio = 0.8
epochs = 50
lr = 0.001
png_save_path = "picture"  # 图片保存路径

# 模型配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def train_model():
    """训练主函数"""
    # 创建保存目录
    os.makedirs(png_save_path, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 数据准备
    print("准备数据...")
    train_loader, val_loader, min_val, max_val = split_data(
        batch_size=batch_size,
        seq_length=seq_length,
        pred_length=pred_length,
        train_ratio=train_ratio
    )
    
    print(f"训练集批次: {len(train_loader)}, 验证集批次: {len(val_loader)}")
    print(f"数据范围 - 最小值: {min_val}, 最大值: {max_val}")
    
    # 模型初始化
    print("初始化模型...")
    model = Transformer().to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 训练记录
    loss_history = []
    best_loss = float('inf')
    best_epoch = 0
    best_model_wts = None
    
    print("\n开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        y_pre = []
        y_true = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - 训练')
        for X_batch, y_batch in train_pbar:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            
            # 前向传播
            outputs, enc_self_attns = model(X_batch)
            
            # 计算损失
            loss = criterion(outputs, y_batch)
            epoch_loss += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # 记录预测值和真实值
            y_pre.append(outputs.detach().cpu())
            y_true.append(y_batch.detach().cpu())
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # 合并所有预测值和真实值
        y_pre_concat = torch.cat(y_pre, dim=0)
        y_true_concat = torch.cat(y_true, dim=0)
        
        # 计算训练集指标
        train_metrics = evaluate_model(model, train_loader, min_val, max_val, device)
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, min_val, max_val, device)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'models/best_model.pt')
        
        # 打印进度
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  训练损失: {avg_loss:.6f}')
        print(f'  训练指标 - MAE: {train_metrics["original"]["MAE"]:.4f}, RMSE: {train_metrics["original"]["RMSE"]:.4f}, PCC: {train_metrics["original"]["PCC"]:.4f}')
        print(f'  验证指标 - MAE: {val_metrics["original"]["MAE"]:.4f}, RMSE: {val_metrics["original"]["RMSE"]:.4f}, PCC: {val_metrics["original"]["PCC"]:.4f}')
        
        # 每10个epoch绘制一次图表
        if (epoch + 1) % 10 == 0:
            plot_path = os.path.join(png_save_path, f'epoch_{epoch+1}_predictions.png')
            plot_predictions(
                y_true_concat.numpy(), 
                y_pre_concat.numpy(), 
                min_val, 
                max_val, 
                save_path=plot_path,
                title=f'Epoch {epoch+1} 预测结果'
            )
    
    # 训练完成
    print(f"\n训练完成! 最佳模型在第 {best_epoch+1} 个epoch，损失: {best_loss:.6f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('models/best_model.pt'))
    
    # 最终评估
    print("\n最终模型评估:")
    final_train_metrics = evaluate_model(model, train_loader, min_val, max_val, device)
    final_val_metrics = evaluate_model(model, val_loader, min_val, max_val, device)
    
    print("训练集最终指标:")
    for key, value in final_train_metrics['original'].items():
        print(f"  {key}: {value:.4f}")
    
    print("验证集最终指标:")
    for key, value in final_val_metrics['original'].items():
        print(f"  {key}: {value:.4f}")
    
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', linewidth=2)
    plt.title('训练损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(png_save_path, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 绘制最终预测结果
    final_plot_path = os.path.join(png_save_path, 'final_predictions.png')
    plot_predictions(
        final_val_metrics['targets'],
        final_val_metrics['predictions'],
        min_val,
        max_val,
        save_path=final_plot_path,
        title='最终验证集预测结果'
    )
    
    return model, final_train_metrics, final_val_metrics

if __name__ == "__main__":
    try:
        # 检查数据文件是否存在
        if not os.path.exists('data/601398.XSHG.csv'):
            print("数据文件不存在，请先运行: python data_download.py")
            sys.exit(1)
        
        # 开始训练
        model, train_results, val_results = train_model()
        
        print("\n训练脚本执行完成!")
        print(f"模型已保存到: models/best_model.pt")
        print(f"图表已保存到: {png_save_path}/")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()