# 优化计划

## 目标
两个数据集 self-score 均 >= 0.04，Phase 12 基线：原始 0.0504, April 0.0018。

## 核心假设
1. **训练时间太长是主要问题** — 60 epoch 导致严重过拟合，MASTER 只用 1 epoch
2. **Loss 选择不当** — SmoothNDCG 复杂且慢，Margin Ranking Loss 论文实证最优
3. **窗口太长** — 60 天回看引入过多噪声，MASTER 8 天就够
4. **缺少市场引导** — 市场特征 gate 零成本引入宏观背景

## 优化方向（按优先级）

### 方向 1：MASTER 风格精简训练 [高优先]
- 缩短 sequence_length: 60 → 8 或 15
- 减少 epochs: 60 → 5~10
- 增大 dropout: 0.35 → 0.5
- 增大 d_model: 128 → 256
- Loss 从 SmoothNDCG 切换到 MSE（配合 CSZScoreNorm）
- **预期**: April 泛化改善明显（减少过拟合）

### 方向 2：Margin Ranking Loss [高优先]
- 实现 margin ranking loss: max(0, -(pred_i - pred_j) * sign(return_i - return_j) + margin)
- 论文实证年化收益提升 1.45% vs MSE
- 配合 CSZScoreNorm 使用
- **预期**: 两个数据集排名质量都提升

### 方向 3：Market Feature Gate [中优先]
- 从 300 只股票重建市场信息（等权收益、波动率统计）
- 用 Gate 机制对股票特征加权
- 参考 MASTER 的 gate 实现
- **预期**: 提供市场背景，帮助跨 regime 泛化

### 方向 4：Multi-Loss 合成 [中优先]
- MSE + Margin + (可选 BPR) 等权组合
- 国金报告实证有效
- **预期**: 提升鲁棒性

### 方向 5：缩短窗口 + 减少特征 [低优先]
- 尝试用更短窗口（5/8/15 天）
- 如果短窗口有效，进一步减少参数、加速训练
- **预期**: 进一步减少过拟合

## 执行顺序
```
广度扫描：
  Step 1: MSE + 10 epoch (30s) → L1 验证
  Step 2: MSE + 5 epoch (20s) → L1 验证  
  Step 3: Margin Loss + 10 epoch (30s) → L1 验证
  Step 4: 窗口 15 天 + 10 epoch (25s) → L1 验证
  Step 5: Dropout 0.5 + 10 epoch (30s) → L1 验证

评估 & 锁定方向 → 深度挖掘：
  最佳方向 + 叠加其他改动 → L2 完整训练 → L3 跨数据集
```
