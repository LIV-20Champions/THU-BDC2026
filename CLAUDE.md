# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

THU Big Data Competition 2026 baseline — a **learning-to-rank stock selection** model for CSI 300 (沪深300) constituents. The model scores all candidate stocks for a given trading day, then outputs the top 5 with equal weights (0.2 each). Core model: `StockTransformer` (Transformer encoder + cross-stock attention + ranking head).

## Commands

```bash
# Setup
uv sync
source .venv/bin/activate

# Download data (modify start_date/end_date in get_stock_data.py first)
python get_stock_data.py

# Split into train/test (edit date ranges in data/split_train_test.py first)
python data/split_train_test.py

# Train
sh train.sh                # or: python code/src/train.py

# Predict (generates output/result.csv with top-5 stocks)
sh test.sh                 # or: python code/src/predict.py

# Self-scoring (compares predictions against test set labels)
python test/score_self.py  # output → temp/tmp.csv

# Docker build & verify
docker buildx build --platform linux/amd64 --build-arg IMAGE_NAME=nvidia/cuda -t bdc2026 .
docker compose up          # verify runnability → test/output/result.csv
docker save -o <team_name>.tar bdc2026:latest

# Final scoring simulation (put .tar in test/tars/, name in test/tar_files_list.txt)
python test/test.py         # linux
python test/test_windows.py # windows
```

Windows users can run `python code/src/train.py` / `python code/src/predict.py` directly instead of the shell scripts.

## Architecture

### Data flow

```
get_stock_data.py → data/stock_data.csv
                    ↓
          data/split_train_test.py
                    ↓
         data/train.csv, data/test.csv
                    ↓
        train.py (feature engineering + model training)
                    ↓
        model/<name>/best_model.pth, scaler.pkl, config.json
                    ↓
        predict.py (inference on latest date)
                    ↓
        output/result.csv (top 5 stock_id, weight)
```

### Key design decisions

- **Feature sets**: `39` (basic TA-Lib indicators), `158` (Alpha-style factors), or `158+39` (combined). Switched via `feature_num` in config. Despite naming, `feature_engineer_func_map['158']` maps to `engineer_features_39` and vice versa — the `config.py` map is authoritative.
- **Label**: `0.3 * ret_t1→t3 + 0.7 * ret_t1→t5`, where returns are based on opening price. Labels are relative future returns, not absolute.
- **Normalization**: Per-stock sliding z-score (window = `sequence_length`, default 60), applied before dataset construction. The scaler is built into the lazy dataset, not as a separate step.
- **Stock embedding**: Each stock has a learned embedding (dim=16) added to its feature projection. Stock IDs come from `stockid2idx` mapping built during training.
- **Multi-process feature engineering**: Feature computation runs via `multiprocessing.Pool` (configured by `feature_engineering_workers`). Entry points use `spawn` mode for compatibility.
- **GPU/CPU**: Auto-detects `CUDA → MPS → CPU`. The GPU config variant (`config_gpu.py`) enables VSN + MultiScale + cross-sectional features for higher scores.
- **Metric**: `final_score` combines `pred_return_sum / max_return_sum` ratio. Model selection is based on validation `final_score`, not loss.

### Config system

`config.py` is the single source of truth, shared by `train.py` and `predict.py`. It contains:
- Model hyperparameters (`d_model=128`, `nhead=4`, `num_layers=2`)
- Training params (`batch_size=2`, `num_epochs=60`, `lr=5e-5`)
- Loss weights (`smooth_ndcg_weight=0.7`, `lambda_pairwise_weight=0.3`)
- Feature flags (`use_ema`, `use_swa`, `use_amp`, `use_gradient_checkpointing` — many are togglable)
- `feature_columns_map`: column lists for each feature set
- `feature_engineer_func_map`: mapping from feature_num string to engineering function

`config_gpu.py` is a variant with `feature_num='158+39'` and more features enabled (VSN, MultiScale, cross-sectional). Swap it in for GPU training.

### TA-Lib dependency

Feature engineering requires the TA-Lib C library installed at the system level. The project bundles the source (`ta-lib-0.4.0-src.tar.gz`) and a pre-built copy in `ta-lib/`. The Dockerfile installs it from source.

### Submission format

Final submission is a Docker image exported as `.tar`. Inside the container, `data/run.sh` is executed — it should run training + prediction and produce `output/result.csv`. The `data/run.sh` file is mounted as a volume during Docker verification.

## 铁律

1. **TDD 是唯一编码方式** — 功能性变更必须先写失败测试，经过红-绿-重构循环。完成后确保无回归
2. **交互用中文，代码用英文** — 解释/询问/建议用中文；代码/变量/commit message 用英文

## 关键约束

- Worktree 隔离不可用，所有开发在主会话内联执行

## 当前基线 (Phase 12)

- **配置**: CSZScoreNorm + DropExtremeLabel + DailyBatchSampler, AdamW, SmoothNDCG loss, 60 epochs, 5-model ensemble (seeds 42,49,56,63,70)
- **原始数据集** (`stock_data.csv`, 2024.01-2026.03): self-score **0.0504**
- **April 数据集** (`stock_data_April.csv`, 2024.01-2026.04): self-score **0.0018**
- **训练耗时**: ~30 秒/轮 (GPU), 512 epochs

## 数据集约束

- `data/stock_data.csv` 和 `data/stock_data_April.csv` 是两套独立数据集，不做针对单一数据集的特殊优化
- 模型需在两个数据集上都 >= 0.04，泛化能力是核心目标
- 训练集本身信息量足够覆盖两个数据集的时间范围——问题在模型如何利用数据，不在数据不够

## 自动优化工作流

入口: `/auto-optimize` skill（详见 `.claude/skills/auto-optimize/SKILL.md`）

**阶段**: 搜索行业方案 → 分析代码热点 → 制定优化计划 → 执行（广度扫描→深度挖掘）→ 报告

**门禁标准**:
| 层级 | 触发 | 通过标准 | 失败处理 |
|------|------|---------|---------|
| L1 快验 | 每次改动 | 5 epoch, loss 不炸, score std > 0.05 | 回滚，不进入完整训练 |
| L2 完整 | L1 通过 | 原始 self-score >= 0.04 | 标记无效，回滚 |
| L3 跨集 | L2 通过 | 原始 >= 0.04 且 April >= 0.04 | 保留但不作主方向 |

**约束**:
- 单轮优化 <= 2 分钟，性价比优先，不引入过于复杂的优化
- 搜索策略：先广度扫描（每轮 1 分钟内）锁定方向，再深度挖掘
- 自主权限：参数调优全自主；新增特征/改模型结构通知后执行；换数据/换架构范式必须批准
- 实验记录：精简版，只记分数和结论（`股价预测/experiments/`）
- 总时间预算：3 小时

## 实验经验

- **Phase 12 是当前最优**: CSZScoreNorm + DailyBatch 使训练 14+ epoch 不崩，但 April 数据集几乎零收益
- **分数确定性**：5 模型集成 + 固定 seeds 下分数完全可复现。改特征集/embedding/架构参数在 422 样本上均不改善——需要从方法层面突破
- **验证集反相关**：val score 与 test self-score 负相关。不要用 val score 选模型——用固定 epoch 数或取末轮模型
- **SAM 有害**：SAM 导致 score 崩坏（std 0.33→0.017），与此模型/数据不兼容
- **特征函数命名**：engineer_features_158 实际生成 ~39 特征，engineer_features_39 实际生成 ~158 特征。feature_engineer_func_map 的 key 是权威入口
- **大参数均退化**：VSN (+300K)、MultiScale (+200K) 在 422 样本上无论如何都会退化。SAM 无法挽救参数量过大
- **特征增量不增效**：158+39 特征、横截面特征、市场特征均未提升 Phase 12 基线——任何特征添加都需在 Phase 12 配置下验证

## 调试注意事项

- Python 多行命令在 Bash 工具中会报 SyntaxError——使用 Write 工具创建临时脚本再运行
- 训练前必须 `rm -rf model/<output_dir>` 否则旧模型残留
- `__pycache__` 不会导致过期问题（pyc 随 py 同步更新）

## 关键文件位置

- 竞争入口：`python code/src/train.py`（训练）然后 `python code/src/predict.py`（预测）
- 自评：`python test/score_self.py`
- 数据集：`data/stock_data.csv`（主数据），`data/stock_data_April.csv`（新数据）
- 训练/测试分割：`data/split_train_test.py`