# 补充素材: 演进历程、学术参照与实验数据

> 本文件是 `presentation_materials.md` 的补充，提供项目演进叙事、学术对比和真实实验数据。三者配合使用，AI可生成更有故事性和说服力的PPT。

---

## 1. 项目演进历程 (Phase 1 → Phase 12)

> 这段叙事线是PPT"故事性"的核心——从基线搭建到突破瓶颈的完整探索过程。

### 1.1 演进时间线

```
Phase 1-3: 基线搭建 (2026-05-25)
    ↓  初版StockTransformer + 158维Alpha因子 + SmoothNDCG
    ↓  问题: 训练2-3 epoch后loss爆炸
Phase 4-7: 正则化探索
    ↓  尝试: Dropout调整、权重衰减、EMA、SWA
    ↓  问题: 最多训练5-8 epoch仍崩溃
Phase 8: 首个稳定基线
    ↓  AdamW优化器 + 2 epoch + 5模型集成
    ↓  原始数据集: 0.0402 | April: -0.0095
    ↓  问题: 只能训练2 epoch, 无法充分学习
Phase 9: SAM优化器实验
    ↓  尝试Sharpness-Aware Minimization寻找平坦极小值
    ↓  结果: Score崩塌 (std 0.33→0.017) — SAM有害!
    ↓  关键教训: SAM与极小样本(422)不兼容
Phase 10: 横截面特征实验
    ↓  添加截面排名百分位 + Z-Score特征
    ↓  结果: 未改善Phase 8基线
Phase 11: VSN + MultiScale实验
    ↓  引入Variable Selection Network + 多尺度时序编码
    ↓  结果: 参数量增加+300K/200K, 在422样本上严重退化
Phase 12: 突破! CSZScoreNorm + DailyBatchSampler (2026-05-28)
    ↓  诊断: 标签极端值 + 随机batch破坏截面信号 + 标签尺度跨日不一致
    ↓  方案: CSZScoreNorm(截面Z-Score标准化标签) + DropExtremeLabel(剔除5%极端标签)
    ↓        + DailyBatchSampler(按日分组batch)
    ↓  结果: 训练14+ epoch不崩溃!
    ↓  原始数据集: 0.0504 | April: 0.0018
    ↓  当前最优配置, self-score稳定>0.05
Phase 13: 去Z-Score实验
    ↓  验证逐股票Z-Score的必要性
    ↓  结果: 去除后score大幅下降(0.0052 vs 0.0504)
    ↓  证明: 逐股票滑动Z-Score是关键组件
Phase 14: 158特征 + 市场特征实验
    ↓  尝试更大特征集
    ↓  结果: 均未超越Phase 12基线
```

### 1.2 关键突破时刻: Phase 12

Phase 12是整个项目的转折点。在此之前，模型最多训练5-8个epoch就会崩溃。突破来自对问题根因的精准诊断:

**三个根因**:
1. **标签极端值**: 5日收益率范围从-50%到+50%，极端值引爆SmoothNDCG的软排名梯度
2. **随机batch破坏截面信号**: 排序定义在"同日所有股票"之间，随机batch打乱了这个结构
3. **标签尺度跨日不一致**: 高波动日标签值远大于低波动日，主导损失函数

**三个解法** (灵感来自MASTER论文):
1. **DropExtremeLabel**: 每日剔除top/bottom 2.5%极端标签
2. **CSZScoreNorm**: 每日对标签做Z-Score标准化，使标签范围归一至[-3, 3]
3. **DailyBatchSampler**: 每个batch = 一个完整交易日，保留截面排序结构

**效果**: 训练从"2 epoch就崩"变为"14+ epoch稳定"，self-score从0.0402提升至0.0504 (+25.4%)

### 1.3 各Phase真实分数对比

| Phase | 关键配置 | 原始数据集 Score | April数据集 Score | 备注 |
|-------|---------|-----------------|------------------|------|
| Phase 8 | AdamW, 2ep, random batch | **0.0402** | -0.0095 | 首个稳定基线 |
| Phase 9 | SAM优化器 | 崩塌 | 崩塌 | SAM有害 |
| Phase 12 | CSZScoreNorm + DailyBatch + AdamW | **0.0504** | 0.0018 | 当前最优 |
| Phase 13 | 去除Z-Score | 0.0052 | -0.0015 | Z-Score不可或缺 |
| Phase 14 | 158特征/市场特征 | <0.0504 | — | 特征增量不增效 |

---

## 2. 学术参照与行业对比

> 这部分让PPT具备学术深度，展示项目在行业研究脉络中的定位。

### 2.1 MASTER (AAAI 2024) — 最直接的参照

MASTER (Market-guided Stock Transformer) 是本项目架构的主要灵感来源，架构相似度极高。

**MASTER架构**:
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

**核心差异对比**:

| 维度 | MASTER | 本项目 (Phase 12) | 分析 |
|------|--------|-------------------|------|
| 回看窗口 | 8天 | 60天 | 我们窗口更长，捕获更多历史但引入更多噪声 |
| 特征数 | 158+63 | 39 | 我们特征更精简，避免小样本过拟合 |
| d_model | 256 | 128 | 我们更小，适配422样本 |
| Dropout | 0.5 | 0.35 | MASTER更激进的正则化 |
| 训练Epochs | 1 | 60 | 最惊人差异! MASTER 1 epoch即收敛 |
| 损失函数 | MSE | SmoothNDCG + 0.05×MSE | 我们用排序损失+微量MSE |
| Market Gate | 有 | 无 | MASTER用市场信息门控特征 |
| 参数量 | ~400K | ~200K | 我们参数量更小 |
| 标签处理 | CSZScoreNorm | CSZScoreNorm + DropExtremeLabel | 我们额外剔除极端标签 |

**MASTER的核心启示**:
- MSE + CSZScoreNorm + DailyBatch = 1 epoch收敛 — 简单方案在小样本下可能更优
- Market Feature Gate几乎零成本引入宏观信息
- 8天窗口足以捕获短期排序信号

### 2.2 排序损失函数研究 (arXiv 2510.14156)

系统性对比了6种损失函数在S&P 500股票排序上的表现:

| 损失函数 | 年化收益 | 夏普比率 | 最大回撤 |
|---------|---------|---------|---------|
| **Margin Ranking** | **16.23%** | **0.753** | -18.33% |
| ListNet | 16.00% | 0.741 | -18.36% |
| BPR | 15.74% | 0.720 | **-15.77%** |
| RankNet | 15.5% | 0.72 | -18.5% |
| Weighted Hinge | 15.4% | 0.72 | -18.5% |
| MSE (baseline) | 14.78% | 0.664 | -19.58% |

**关键发现**: 在大规模数据上，所有排序损失都优于MSE。但在我们的422样本极小数据集上，排序损失反而不如微量MSE — 这是小样本与大样本的根本差异。

### 2.3 国金证券A股实证

- 方法: AGRU + PairWise + ListWise多损失等权合成
- A股IC均值: 13.82%
- 沪深300年化超额: 16.60%
- 核心方法: 多epoch参数取均值(类似SWA)

### 2.4 TSPRank (KDD 2025)

- 用旅行商问题建模Listwise排序
- 需要Gurobi/CPLEX商用求解器
- 核心思路: 将pairwise比较矩阵组合为listwise排序
- 实用性低但理论价值高

### 2.5 AlphaNet

- 端到端因子挖掘，用自定义运算符函数替代卷积核
- v4版引入Transformer
- 核心思想: 用operators(ts_corr, ts_stddev等)遍历提取特征
- 与我们固定39维TA-Lib特征不同，AlphaNet自动发现特征

### 2.6 本项目在研究脉络中的定位

```
传统量化 (手工因子 + 线性模型)
    ↓
AlphaNet (自动因子挖掘 + CNN)
    ↓
MASTER (市场引导 + Transformer + 截面排序) ← 我们的直接参照
    ↓
本项目 (跨股票注意力 + 极小样本正则化 + 排序损失探索)
    ↑
    独特贡献: 在422样本极限下验证了深度学习排序的可行性边界
```

---

## 3. 真实实验数据

> 这部分提供PPT中可引用的具体数字，增强可信度。

### 3.1 Phase 12 三次独立运行 (原始数据集)

| 运行 | Final Score |
|------|------------|
| Run 1 | 0.0504 |
| Run 2 | 0.0504 |
| Run 3 | 0.0504 |

**结论**: 5模型集成 + 固定种子下，分数完全可复现，零方差。

### 3.2 Phase 12 三次独立运行 (April数据集)

| 运行 | Final Score |
|------|------------|
| Run 1 | 0.0018 |
| Run 2 | 0.0018 |
| Run 3 | 0.0018 |

**结论**: April数据集分数同样完全可复现，但远低于原始数据集。

### 3.3 Phase 8 vs Phase 12 对比

| 指标 | Phase 8 | Phase 12 | 变化 |
|------|---------|----------|------|
| 原始数据集 Score | 0.0402 | 0.0504 | **+25.4%** |
| April数据集 Score | -0.0095 | 0.0018 | **从负转正** |
| 最大训练Epochs | 2 | 14+ | **7倍提升** |
| 标签处理 | 无 | CSZScoreNorm + DropExtremeLabel | 新增 |
| 采样策略 | Random Shuffle | DailyBatchSampler | 改变 |
| 训练稳定性 | 2ep后崩溃 | 稳定 | 根本改善 |

### 3.4 Phase 13: Z-Score消融实验

| 指标 | 有Z-Score (Phase 12) | 无Z-Score (Phase 13) | 变化 |
|------|---------------------|---------------------|------|
| 原始数据集 Score | 0.0504 | 0.0052 | **-89.7%** |
| April数据集 Score | 0.0018 | -0.0015 | 从正转负 |

**结论**: 逐股票滑动Z-Score是模型性能的关键组件，去除后性能下降近90%。

### 3.5 消融实验完整数据

| # | 实验 | 配置 | 原始Score | vs Phase 12 |
|---|------|------|----------|-------------|
| 1 | Phase 12 基线 | CSZScoreNorm + 0.05×MSE + DailyBatch + 5模型 | **0.0504** | baseline |
| 2 | 纯MSE | mse_weight=1.0, 10ep, 1模型 | ~0.0026 | -94.8% |
| 3 | SmoothNDCG+Margin | 排序损失+成对损失 | ~0.0017 | -96.6% |
| 4 | 强正则化 | dropout=0.5, val_months=2 | ~0.0070 | -86.1% |
| 5 | 短窗口 | seq_len=15, val=0, 1模型×5ep | 0.0467 | -7.3% |
| 6 | 短窗口+验证 | seq_len=15, val=2, ensemble=5 | ~0.0106 | -79.0% |
| 7 | 同学方案 | 100+39特征+截面排名+WeightedRanking | -0.0126 | 负值 |
| 8 | 无Z-Score | Phase 12配置去除逐股票Z-Score | 0.0052 | -89.7% |

### 3.6 数据集规模

| 数据集 | 时间范围 | 交易日数(约) | 股票数 | 训练样本数 |
|--------|---------|------------|--------|-----------|
| 原始 (stock_data.csv) | 2024.01 - 2026.03 | ~422 | ~300 | ~422 |
| April (stock_data_April.csv) | 2024.01 - 2026.04 | ~440 | ~300 | ~440 |

**关键约束**: 422个训练样本是极小样本场景。作为对比，ImageNet有128万样本，CIFAR-10有6万样本。我们的样本量仅为典型深度学习场景的万分之一。

### 3.7 模型参数量分析

| 组件 | 参数量 | 占比 |
|------|--------|------|
| 输入投影 (Linear) | ~5K | ~2.5% |
| 股票嵌入 (300×16) | ~4.8K | ~2.4% |
| 股票嵌入投影 | ~2.2K | ~1.1% |
| Transformer Encoder (2层) | ~100K | ~50% |
| FeatureAttention (CLS) | ~66K | ~33% |
| CrossStockAttention (2层) | ~13K | ~6.5% |
| 排序头 | ~9K | ~4.5% |
| **总计** | **~200K** | 100% |

**对比**: MASTER ~400K参数，VSN版本+300K，MultiScale版本+200K。我们严格控制参数量以适配422样本。

---

## 4. 竞赛评分机制详解

### 4.1 官方评分公式

竞赛评分基于**加权收益率**:

```
Final Score = Σ(weight_i × return_i)
```

其中:
- `weight_i`: 模型分配给第i只股票的权重 (0 ≤ weight ≤ 1, Σweight ≤ 1)
- `return_i`: 第i只股票在未来5个交易日的开盘价收益率
- `return = (open_day5 - open_day1) / open_day1`
- 最多选5只股票

### 4.2 自评机制

项目实现了`test/score_self.py`进行本地自评:
1. 读取`output/result.csv`(模型预测)和`data/test.csv`(测试集标签)
2. 验证预测合法性: 最多5只股票, 权重和∈[0,1]
3. 在测试集中找到预测股票的后5日开盘价数据
4. 计算加权收益率作为Final Score

### 4.3 提交方式

- 最终提交为Docker镜像(.tar文件)
- 容器内执行`data/run.sh` → 运行训练+预测 → 生成`output/result.csv`
- 竞赛平台在隐藏测试集上评分

---

## 5. PPT叙事建议

### 5.1 推荐叙事结构 (10分钟)

```
[0:00-1:30] 开场: 问题定义
    - 沪深300选股 = 排序问题, 不是回归
    - 422样本的极限挑战
    - 从300只股票中选出Top-5

[1:30-3:30] 方法: StockTransformer架构
    - 核心创新: Cross-Stock Attention
    - DailyBatchSampler: 截面感知采样
    - 逐股票滑动Z-Score: 消除量级差异

[3:30-5:30] 演进: Phase 1→12的探索历程
    - Phase 8: 首个稳定基线(2ep就崩)
    - Phase 9: SAM实验失败
    - Phase 12: 突破! CSZScoreNorm + DailyBatch
    - 关键转折: 从"训练崩溃"到"14+ epoch稳定"

[5:30-7:30] 实验: 消融与发现
    - 6组消融实验数据
    - 4个反直觉发现
    - 与MASTER论文的对比

[7:30-9:00] 结果: 性能指标
    - Self-Score稳定>0.05
    - 三次运行零方差(完全可复现)
    - Z-Score消融: 去除后下降90%

[9:00-10:00] 总结: 贡献与展望
    - 小样本深度学习排序的可行性验证
    - 跨股票注意力 + 截面感知采样的范式
    - 未来: Market Gate、更短窗口、泛化提升
```

### 5.2 关键"钩子"(Hooks)

PPT中需要设置几个吸引注意力的"钩子":

1. **"422样本"** — 强调这是典型深度学习样本量的万分之一
2. **"1 epoch vs 60 epochs"** — MASTER 1 epoch收敛，我们60 epochs，为什么?
3. **"验证集反相关"** — 用验证集选模型反而更差，违反直觉
4. **"排序损失不如微量MSE"** — 行业共识是排序损失更优，但422样本下相反
5. **"SAM有害"** — 顶级会议方法在此场景下崩塌
6. **"特征越多越差"** — 158+39特征不如39特征

### 5.3 视觉风格建议

- **配色**: 深蓝(科技感) + 金色(金融感) + 红色(警示/负值)
- **图表风格**: 简洁、数据驱动、避免3D效果
- **动画**: 仅用于展示Cross-Stock Attention的信息流动
- **字体**: 标题用粗体无衬线，正文用细体无衬线

---

## 6. 可引用的参考论文

1. **MASTER** — Wu et al., "MASTER: Market-guided Stock Transformer for Stock Selection", AAAI 2024
2. **TSPRank** — "TSPRank: Listwise Learning-to-Rank via Traveling Salesman Problem", KDD 2025
3. **AlphaNet** — "AlphaNet: Learning to Generate New Factors", 2020
4. **Loss Functions for Stock Ranking** — arXiv 2510.14156, 2024
5. **SAM** — Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021
6. **国金证券** — "Alpha掘金系列: 深度学习排序模型在A股的实证", 2024
