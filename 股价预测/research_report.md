# 综合搜索报告：股票排序模型优化

## 信息来源
- C4/THU BDC 竞赛方案、Kaggle 方案
- AAAI 2024/2025 论文（MASTER、DHMoE、CI-STHPAN 等）
- arXiv 论文（Loss Functions for Stock Ranking）
- 开源项目（MASTER、TSPRank、AlphaNet、Qlib）
- 国金证券 Alpha 掘金系列研报

---

## 1. MASTER (AAAI 2024) — 市场引导的股票 Transformer

**最直接可借鉴的参考，架构相似度极高。**

### 架构
```
Input [N stocks, T=8, F=158+63 market]
  → Feature Gate (market info → softmax weights over features)
  → Linear(d_feat→d_model=256)
  → PositionalEncoding
  → TAttention (temporal, multi-head=4)
  → SAttention (spatial/cross-stock, multi-head=2)
  → TemporalAttention (learnable query over time)
  → Linear(d_model→1)
```

### 关键设计
- **只用 1 epoch 训练** — 最惊人的发现。MSE + CSZScoreNorm + DailyBatch 下，模型 1 epoch 就收敛
- **8 天回看窗口** — 比我们的 60 天短很多
- **Market Feature Gate** — 用当天市场信息（63 维指数特征）通过 softmax 门控股票特征
- **Dropout 0.5** — 极高，对抗小样本过拟合
- **d_model=256** — 比我们的 128 大一倍
- **MSE loss + CSZScoreNorm** — 简单的均方误差，配合截面 z-score 归一化
- **5-seed ensemble** — 和我们一样

### 与我们的差异
| 维度 | MASTER | Ours (Phase 12) |
|------|--------|-----------------|
| 回看窗口 | 8 天 | 60 天 |
| 特征数 | 158+63 | 39 |
| d_model | 256 | 128 |
| Dropout | 0.5 | 0.35 |
| Epochs | 1 | 60 |
| Loss | MSE | SmoothNDCG |
| Market Gate | 有 | 无 |
| 参数量 | ~400K | ~503K |

---

## 2. Loss Function 研究（关键发现）

### arXiv 2510.14156 系统对比结果

| Loss | 年化收益 | 夏普比率 | 最大回撤 |
|------|---------|---------|---------|
| **Margin Ranking** | **16.23%** | **0.753** | -18.33% |
| **ListNet** | 16.00% | 0.741 | -18.36% |
| BPR | 15.74% | 0.720 | **-15.77%** |
| RankNet | 15.5% | 0.72 | -18.5% |
| Weighted Hinge | 15.4% | 0.72 | -18.5% |
| MSE (baseline) | 14.78% | 0.664 | -19.58% |

**核心结论**：
1. 所有 ranking loss 都优于 MSE
2. Margin Ranking Loss 表现最优（年化 +1.45% vs MSE）
3. **IC Spearman 与投资组合收益不完全正相关** — 预测质量指标不一定反映真实收益

### 国金证券 A 股实证
- PairWise + ListWise 多损失等权合成，A 股 IC 均值 13.82%
- 沪深 300 年化超额 16.60%
- 核心方法：AGRU + 排序损失 + 多 epoch 参数取均值

---

## 3. 正则化/防过拟合方法

### 扩散模型数据增强 (DiffsFormer)
- 用 Diffusion Model 生成合成股票因子
- CSI300 年化收益提升 7.2%
- 实现复杂，性价比低

### 元学习 (MASSER, SDM 2024)
- 自监督学习检测时序域漂移
- 重要性采样平衡时序域差异
- 准确率提升 5-9.5%
- 实现复杂度中等

### 多模型参数平均 (国金方法)
- 多 epoch 模型参数取均值对抗过拟合
- 我们已有 EMA + SWA，效果类似

---

## 4. TSPRank (KDD 2025)

用旅行商问题建模 Listwise 排序，需要 Gurobi/CPLEX 求解器（商用），不实用。但核心思路——将 pairwise 比较矩阵组合为 listwise 排序——可以参考。

---

## 5. AlphaNet

端到端因子挖掘，用自定义运算符函数替代卷积核。v4 版引入 Transformer。核心思想是用 operators（ts_corr, ts_stddev 等）遍历提取特征。我们的特征工程已经是固定的 39 维 TA-Lib，不需要这种灵活性。

---

## 总结：Top 5 最有价值的优化方向

1. **MASTER 风格精简训练** — 缩短窗口、减少 epoch、加大 dropout、用简单 loss
2. **Margin Ranking Loss** — 论文实证最优，替换当前的 SmoothNDCG
3. **Market Feature Gate** — MASTER 的核心创新，几乎零成本引入市场信息
4. **多 Loss 等权合成** — 国金方法，MSE + Margin + ListNet 组合
5. **缩短训练周期** — MASTER 1 epoch 就收敛，我们 60 epoch 可能过度训练
