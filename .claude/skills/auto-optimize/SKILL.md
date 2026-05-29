---
name: auto-optimize
description: 全自动模型优化流水线——搜索行业方案→分析代码热点→制定计划→广度扫描→深度挖掘→报告。3小时时间预算，目标是两个数据集 self-score 均 >= 0.04。
---

# Auto-Optimize: 全自动模型优化流水线

## 总览

对 THU-BDC2026 股票排序模型进行闭环优化。入口调用一次，自动完成搜索→分析→优化→验证全流程，中间仅在必须批准时暂停。

**目标**: 两个数据集 self-score 均 >= 0.04，模型泛化能力强，不做针对性优化。

**时间预算**: 3 小时总时长。

**基准**: Phase 12 — 原始 0.0504, April 0.0018。

## 流程总图

```
Phase 1: 搜索 (不限时)           → 股价预测/ 目录
    ↓
Phase 2: 代码分析 (~15 min)      → 热点报告
    ↓
Phase 3: 制定计划 (~5 min)       → 优化计划 (通知用户)
    ↓
Phase 4: 执行循环 (剩余时间)
    ├── 广度扫描 (前 30-40 min) → 每个方向 1 min 快验
    ├── 方向评估                 → 锁定 2-3 个有效方向
    └── 深度挖掘 (剩余时间)      → 2 min/轮, 叠加优化
    ↓
Phase 5: 最终报告 (~5 min)       → 汇总 + commit
```

## Phase 1: 广泛搜索

**目标**: 搜集一切与股票排序/预测相关的行业方案、竞赛思路、开源代码、学术论文。

**搜索来源**（全覆盖）:

1. **C4/THU BDC 竞赛方案** — 搜索 "C4大数据挑战赛"、"THU BDC 股票预测"、"大数据挑战赛 往届方案"
2. **Kaggle 竞赛** — "Kaggle stock ranking top solution"、"Jane Street Market Prediction"、"Two Sigma"、"Optiver"
3. **学术论文** — "learning to rank stocks"、"stock selection transformer"、"cross-sectional stock prediction"、MASTER (AAAI-2024)、AlphaNet、GBM for stock ranking
4. **GitHub 开源项目** — Qlib、FinRL、qlib-based ranking models、stock-prediction-models
5. **技术博客** — "量化选股 Transformer"、"截面排序模型"、"股票预测深度学习"

**搜索手段**:
- `WebSearch` — 广泛搜索以上关键词
- `WebFetch` — 抓取具体文章/论文/代码 README
- `deep-research` skill — 对复杂主题做多源交叉验证搜索
- `gh` CLI — 搜索 GitHub 仓库
- 如果发现论文 PDF，用 `WebFetch` 下载

**材料组织** — 全部存入 `股价预测/`:

```
股价预测/
├── papers/           # 论文摘要/PDF
├── competition/      # C4/THU BDC 往年方案
├── kaggle/           # Kaggle solution writeups
├── github_repos/     # 克隆的开源项目 (git clone)
└── research_report.md  # 综合搜索报告（思路汇总 + 可借鉴点）
```

**搜索完毕标志**: 不再发现新的独立思路，3 个以上来源交叉印证同一方向。

## Phase 2: 代码热点分析

**目标**: 深入阅读当前模型代码，找出最大优化热点。

**分析清单**:

1. **train.py 核心训练循环** — loss 计算、梯度流、batch 构建
2. **model.py StockTransformer** — 注意力机制、cross-stock 交互、参数规模
3. **utils.py 数据处理** — LazyRankingDataset、normalization、label 构建
4. **config.py 参数空间** — 哪些参数未充分探索
5. **predict.py 推理** — ensemble 策略、权重分配

**输出**: 热点分析报告，列出 5-10 个优化方向，每个标注：
- 预期收益（高/中/低）
- 实现复杂度（高/中/低）
- 风险评估（是否会破坏现有稳定性）
- 时间估计

写入 `股价预测/hotspot_analysis.md`。

## Phase 3: 制定优化计划

**目标**: 基于搜索发现和热点分析，形成优先级排序的优化清单。

**原则**:
- 性价比优先 — 不引入过于复杂的优化
- 泛化第一 — 不针对单一数据集做特殊优化
- 先广度后深度 — 快速筛掉无效方向

**输出格式**: 优化清单 + 执行顺序。写入选 `股价预测/optimization_plan.md`。

**此阶段结束后通知用户确认计划**（唯一强制暂停点）。

## Phase 4: 执行循环

### 门禁定义

| 层级 | 触发 | 时长 | 通过标准 | 失败处理 |
|------|------|------|---------|---------|
| **L1 快验** | 每次代码改动 | ~1 min | 5 epoch 训练，loss 不炸，score std > 0.05 | `git checkout -- .` 回滚 |
| **L2 完整** | L1 通过 | ~2 min | 原始数据集 self-score >= 0.04 | 标记无效，回滚 |
| **L3 跨集** | L2 通过 | ~3 min | 原始 >= 0.04 **且** April >= 0.04 | 保留代码，标记为部分有效 |

### L1 快验脚本

```bash
# 修改 output_dir 到临时目录，跑 5 epoch
python -c "
import sys; sys.path.insert(0, 'code/src')
from config import config
config['output_dir'] = './model/_l1_test'
config['num_epochs'] = 5
config['ensemble_size'] = 1
# run train briefly
"
# 检查: loss 不 NaN/inf, 预测 std > 0.05
```

### L2 完整训练

```bash
python code/src/train.py   # 完整 60 epoch, 5-model ensemble
python code/src/predict.py # 用原始 train.csv
python test/score_self.py  # 读取 temp/tmp.csv 看 self-score
```

### L3 跨数据集验证

```bash
# 切换到 April 数据
python code/src/predict.py  # (用 April 产生的 train.csv)
python test/score_self.py
```

### 执行节奏

**广度扫描** (前 30-40 分钟):
- 从优化清单取方向，每个方向只跑 L1
- L1 通过的进入候选池，不通过的直接跳过
- 目标：2-3 个有效方向

**深度挖掘** (剩余时间):
- 候选池中取最佳方向深挖
- 跑完整 L2 → L3
- 有效改动保留（git commit），无效回滚
- 可叠加有效改动（A 通过后再试 A+B）

### 自主权限

| 等级 | 范围 | 操作 |
|------|------|------|
| 自主 | 参数调优、训练策略调整 | 直接改 config，跑 L1 |
| 通知 | 新 loss、新特征、模型结构修改 | 简要描述改动后跑 L1 |
| 批准 | 换数据、换架构范式 | 告知用户，等待确认 |

### 实验追踪

精简记录，每个实验一个 markdown 文件：

```markdown
# 实验 001: <标题>
- 改动: <一句话>
- L1: pass/fail
- L2 原始: 0.xxxx
- L3 April: 0.xxxx
- 结论: keep/rollback/partial
```

写入 `股价预测/experiments/001_xxx.md`，维护 `leaderboard.md` 排名。

## Phase 5: 最终报告

**输出**:
1. 汇总所有实验的 leaderboard
2. 最佳模型配置和分数
3. 未达标的差距分析
4. 后续建议方向
5. Git commit 所有有效改动

---

## 硬约束（不可违反）

1. **两个数据集都 >= 0.04** 才算成功——不是取其一
2. **不做针对性优化**——不针对特定数据集调参
3. **单轮 <= 2 分钟**——超时立即终止当前轮
4. **Phase 12 配置是保护基线**——任何改动 L2 < 0.04 立即回滚
5. **代码改动必须可逆**——用 git 管理，无效即回滚
6. **实验记录必须写**——哪怕只有一句话
