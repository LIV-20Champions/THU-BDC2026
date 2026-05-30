# THU-BDC2026 Phase 12 全流程详解 — 10分钟演讲素材

> 本文档以Phase 12配置为唯一线索，追踪数据从原始CSV到最终选股结果的完整路径。创新点深入阐述设计思路与实现方法，常规步骤简明概括。
>
> **配套文件**: `presentation_supplementary.md` — 项目演进历程、学术参照、PPT叙事建议。

---

## 1. 问题定义

每个交易日从沪深300成分股中选出5只股票，分配权重（和为1），最大化未来5日加权开盘价收益率。本质是**排序问题**——关键不是预测绝对收益，而是识别相对排名前5的股票。

评分公式：`final_score = Σ(weight_i × (open_day5_i - open_day1_i) / open_day1_i)`

---

## 2. Phase 12 全流程

### 2.1 数据加载与切分（简述）

读入`data/train.csv`（~300只股票×~527个交易日），股票代码左补零至6位。取最后2个月作为验证集（约40个交易日），其余为训练集（约422个交易日）。建立`stockid2idx`映射（300条），训练和推理必须共用同一映射。

### 2.2 特征工程（简述）

Phase 12使用`feature_num='39'`，经`feature_engineer_func_map['39']`映射到`engineer_features_158`函数（注意：函数名与实际输出维度不一致）。对每只股票独立计算28个TA-Lib技术指标（SMA/EMA/MACD/RSI/KDJ/布林带/ATR/OBV等），加上原始11列共39列，再追加3个市场特征（全市场等权日收益率及其5日/20日均值），最终**42列**。多进程并行（`Pool(processes=2)`），后处理：inf→nan→ffill→fillna(0)。

### 2.3 标签构建（简述）

`label = 0.3 × ret(t+1→t+3) + 0.7 × ret(t+1→t+5)`，收益基于开盘价计算。`score_target = ret(t+1→t+5)`用于评估。剔除开盘价<1e-4的异常行。

### 2.4 逐股票滑动Z-Score标准化 ★ 核心创新

**设计动机**: 沪深300成分股价格差异极大——贵州茅台~1800元，低价股~3元，原始特征值相差600倍。全局StandardScaler无法处理这种跨股票量级差异，导致训练在5-8个epoch后崩溃。

**实现方法**: 对每只股票独立、逐特征计算滚动窗口Z-Score：

```python
def per_stock_sliding_zscore(feature_values, sequence_length, clip_range=5.0):
    rolling = pd.DataFrame(feature_values).rolling(window=sequence_length, min_periods=1)
    norm = (feature_values - rolling.mean().values) / np.maximum(rolling.std().values, 1e-8)
    return np.clip(np.nan_to_num(norm), -clip_range, clip_range)
```

**技术优势**:
- 窗口=60与模型输入长度一致，`min_periods=1`保证前期也能计算
- 仅使用历史数据，无未来信息泄露
- 裁剪至[-5,5]截断极端值，结果存为float16节省内存
- **每只股票独立标准化**，消除量级差异的同时保留时序动态

**消融验证**: 去除Z-Score（Phase 13）后self-score从0.0504降至0.0052（**-89.7%**），证明此组件不可或缺。

### 2.5 数据集构建（简述）

`LazyRankingDataset`按日期聚合样本：同一交易日的所有股票归入同一个bucket（至少10只），滑动窗口生成60日序列。训练集约422个样本，每个样本包含当日~250只股票的`[60, 42]`序列。`__getitem__`时懒加载，超过100只股票则随机采样。

### 2.6 DailyBatchSampler ★ 核心创新

**设计动机**: 排序定义在"同日所有股票"之间——模型需要看到完整截面才能学习相对强弱。传统随机batch将不同日期的股票混在一起，截面排序信号被彻底破坏。

**实现方法**:
```python
class DailyBatchSampler(Sampler):
    def __init__(self, dataset, shuffle=False):
        daily_groups = defaultdict(list)
        for i in range(len(dataset)):
            daily_groups[dataset.samples[i]['date']].append(i)
        self.batches = list(daily_groups.values())

    def __iter__(self):
        order = torch.randperm(len(self.batches)).tolist() if self.shuffle else range(len(self.batches))
        for i in order:
            yield self.batches[i]
```

**技术优势**:
- 每个batch = 一个完整交易日的全部股票（~250只），截面信号完整保留
- 训练时日期顺序随机打乱（防时间泄露），验证时按时间顺序
- 与Cross-Stock Attention形成闭环：Sampler保证截面完整，Attention利用截面信息

**关键效果**: 这是Phase 12的核心突破——使训练从"5-8 epoch崩溃"变为"14+ epoch稳定"，self-score从0.0402提升至0.0504（+25.4%）。

### 2.7 StockTransformer前向传播 ★ 核心架构

输入：`[1, ~250, 60, 42]`（1个batch×~250只股票×60日×42特征）

**维度流**:
```
[1, 250, 60, 42] → 去instrument列 → [250×60, 41]
  → Linear(41,128) → [250×60, 128]           # 特征投影
  → + StockEmbedding(300,16)→Linear(16,128)   # 股票嵌入叠加
  → + SinusoidalPosEnc                        # 位置编码
  → TransformerEncoder×2(d=128,nh=4,ff=256)   # 时序编码
  → CLS Attention → [250, 128]                # 时序聚合
  → reshape → [1, 250, 128]
  → CrossStockAttention×2 → [1, 250, 128]     # 跨股票交互
  → Interaction(128→64→128) → [1, 250, 128]   # 非线性交互
  → RankingHead(128→128→64→32→1) → [1, 250]   # 排序评分
```

**Cross-Stock Attention深入** ★:

这是架构的核心创新——让同一交易日的所有股票在模型内部相互交互，从"独立评分"进化为"相对排序"。

```python
# 每层CrossStockAttention
attended, _ = self.cross_attention(stock_features, stock_features, stock_features,
                                    key_padding_mask=mask)
output = self.norm(stock_features + self.drop_path(self.layer_scale(attended)))
```

设计要点：
- **2层Self-Attention**：股票间信息经两轮交互，逐步精炼相对排序
- **LayerScale(init=1e-5)**：初始化为极小值，训练初期几乎不改变残差，避免不稳定
- **DropPath率线性递增**（0.0→0.1）：第一层不丢弃，第二层轻量正则化
- **key_padding_mask**：padding的无效股票位置不参与注意力计算

**参数分布**（总计520,705）：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| temporal_encoder | 264,960 | 50.9% |
| cross_stock_layers | 132,864 | 25.5% |
| _ms_feature_attn | 66,560 | 12.8% |
| ranking_layers + score_head | 27,265 | 5.2% |
| 其余 | 29,056 | 5.6% |

时序编码器占一半参数，跨股票注意力占四分之一——模型的核心容量集中在"理解时序"和"比较股票"两个关键任务上。

### 2.8 损失函数 ★ 反直觉发现

**Phase 12实际损失 = 0.05 × MSE(pred, label)**

这是最反直觉的发现：配置中`SmoothNDCGLoss`的`smooth_ndcg_weight=0.0`，`lambda_pairwise_weight=0.0`，只有`mse_aux_weight=0.05`生效。所有排序损失（SmoothNDCG、MarginRanking、WeightedRanking）在422样本上均不如微量MSE。

**原因分析**: 排序损失（如SoftNDCG）通过sigmoid软化排名操作，梯度信号高度非线性。在422个样本上，这种非线性导致严重过拟合——模型在训练集上排名正确，但泛化到测试集时排序崩塌。0.05×MSE提供温和、线性的梯度信号，让模型学到"大致方向"而非"精确排名"，反而泛化更好。

**消融数据**:

| 损失配置 | Self-Score |
|---------|-----------|
| 0.05×MSE（Phase 12） | **0.0504** |
| 纯MSE(weight=1.0) | 0.0026 |
| SmoothNDCG+LambdaPairwise | 0.0017 |

损失按batch内每组（=每个交易日）独立计算后取均值，确保不同股票数量的日期贡献均等。

### 2.9 训练循环（简述关键正则化）

优化器AdamW(lr=5e-5, wd=5e-4)，Warmup(5ep)+CosineAnnealing，梯度裁剪(max_norm=5.0)。每batch：

1. **Mixup**（30%概率）：同一batch内随机两只股票的特征和标签做线性插值，λ~Beta(0.2,0.2)裁剪至[0.05,0.95]
2. **Label Smoothing**：`target = 0.97×target + 0.03×mean(targets)`，标签向当日均值收缩3%
3. 前向+AMP混合精度反向
4. **EMA**(decay=0.999)：shadow参数每步平滑更新
5. epoch≥5时用EMA参数评估；epoch≥15时启动**SWA**，对后续checkpoint取参数均值

**5模型集成**：seeds=[42,49,56,63,70]，5次独立训练。各子模型验证集最佳分数：

| Seed | Best Epoch | Val Score |
|------|-----------|-----------|
| 42 | 2 | 0.006260 |
| 49 | 2 | 0.000781 |
| 56 | 30 | 0.001420 |
| 63 | 4 | 0.006410 |
| 70 | 13 | 0.007181 |

**验证集反相关**：验证集分数与测试集self-score方向相反——验证集分数最高的seed=70(0.007181)并非集成中贡献最大的。Phase 12采用固定epoch+集成来规避此问题。

### 2.10 推理与输出（简述）

对最新交易日，每只股票取60日特征序列，5个模型各自评分后**Rank归一化融合**——不是对评分取均值，而是对排名百分位取均值，避免不同模型评分尺度差异。Top-5股票按`rank_decay`分配权重：`[0.40, 0.25, 0.18, 0.11, 0.06]`。

### 2.11 自评结果

Phase 12在原始数据集上self-score = **0.0504**，三次独立运行完全一致（零方差）。

---

## 3. 技术方案局限性分析

### 3.1 April数据集泛化不足

| 数据集 | Self-Score | 差距 |
|--------|-----------|------|
| 原始(2024.01-2026.03) | 0.0504 | — |
| April(2024.01-2026.04) | 0.0018 | -96.4% |

April数据集仅多出约18个交易日，score却近乎归零。说明模型对训练时间窗口内的数据严重过拟合，对新增数据缺乏泛化能力。根因可能是422样本不足以学到跨时间窗口稳定的排序模式。

### 3.2 验证集不可信

验证集分数与测试集self-score反相关，无法用于模型选择。这意味着：
- 无法通过早停(early stopping)防止过拟合
- 无法通过验证集调参来优化泛化
- 当前方案依赖"固定epoch+集成"这种粗暴策略，缺乏理论指导

### 3.3 配方脆弱性

Phase 12的任何配置偏离都会导致score崩塌：
- 去除Z-Score → -89.7%
- 换排序损失 → -96.6%
- 增大参数量(VSN +300K) → 退化
- 增加特征(158+39) → 不增效
- SAM优化器 → 崩塌

这说明当前方案是在422样本边界上的**精细平衡**，而非鲁棒的最优解。任何"改进"都可能打破这个平衡。

### 3.4 参数量天花板

422样本下模型参数量被严格限制在~520K。VSN(+300K)、MultiScale(+200K)等理论上更优的架构均导致退化。模型容量与样本量的矛盾无法通过正则化解决——这是深度学习在小样本场景下的根本限制。

### 3.5 特征工程天花板

在Phase 12配置下，添加更多特征（158维Alpha因子、横截面特征、市场特征）均未提升基线。39维TA-Lib指标可能已接近422样本能支撑的特征信息上限。更多特征引入的噪声超过了额外信息收益。

### 3.6 适用边界

本方案适用于：
- 样本量与参数量匹配的场景（~500样本配~500K参数）
- 训练/测试分布相近的场景
- 截面排序信号稳定的市场环境

本方案不适用于：
- 市场结构发生重大变化（如牛熊转换）
- 需要跨时间窗口强泛化的场景
- 样本量显著增加的场景（此时排序损失可能重新优于MSE）

---

## 4. 视觉内容指引

### 4.1 Phase 12全流程图
```
train.csv → 特征工程(42列) → 标签(0.3×ret3+0.7×ret5)
  → 逐股票Z-Score(window=60,clip=5) → LazyRankingDataset(422样本)
  → DailyBatchSampler(1天=1batch) → StockTransformer(520K参数)
  → Loss=0.05×MSE → 5模型集成 → Rank归一化融合
  → rank_decay权重[0.40,0.25,0.18,0.11,0.06] → self-score=0.0504
```

### 4.2 Cross-Stock Attention信息流
同一交易日5只股票的表示向量，通过Self-Attention相互交互。对比：左侧"独立评估"(无连线) vs 右侧"交互排序"(密集连线)。

### 4.3 DailyBatch vs RandomBatch
左：随机batch混合不同日期，截面信号丢失；右：DailyBatch保留完整截面。

### 4.4 消融实验柱状图
Phase 12(0.0504) vs 纯MSE(0.0026) vs SmoothNDCG(0.0017) vs 无Z-Score(0.0052)

### 4.5 局限性：双数据集对比
原始0.0504 vs April 0.0018，突出泛化差距

---

## 5. 目标受众适配

### 5.1 面向教师/评委
侧重"为什么"和"局限"：为什么0.05×MSE优于排序损失？为什么验证集反相关？泛化不足的根因是什么？方案的天花板在哪里？

### 5.2 面向同学/技术受众
侧重"怎么做"：`per_stock_sliding_zscore`的实现、`DailyBatchSampler`的分组逻辑、Cross-Stock Attention的LayerScale+DropPath设计、Rank归一化集成的代码

### 5.3 内容平衡
| 幻灯片区域 | 教师侧重 | 同学侧重 |
|-----------|---------|---------|
| 开场 | 422样本的极限挑战 | 技术栈(PyTorch+TA-Lib) |
| 流程 | 三个核心创新的设计动机 | 代码级实现细节 |
| 实验 | 反直觉发现+局限性分析 | 配置参数+复现命令 |
| 结尾 | 适用边界+未来方向 | 代码可复现性 |
