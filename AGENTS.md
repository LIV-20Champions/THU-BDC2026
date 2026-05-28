# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

## 实验经验

- **分数确定性**：5 模型集成 + 固定 seeds (42,49,56,63,70) 下分数 0.0673 完全可复现。改特征集/embedding/架构参数均不影响最终排序——这是当前数据的信息天花板
- **验证集反相关**：val score 与 test self-score 负相关。不要用 val score 选模型——用固定 epoch 数或取末轮模型
- **SAM 约束**：启用 SAM 时 gradient_accumulation_steps 必须为 1，代码未自动强制。SAM 的二次前向传播与梯度累积冲突
- **特征函数命名**：engineer_features_158 实际生成 ~39 特征，engineer_features_39 实际生成 ~158 特征。feature_engineer_func_map 的 key 是权威入口
- **大参数均退化**：VSN (+300K)、MultiScale (+200K) 在 422 样本上无论如何都会退化。SAM 无法挽救参数量过大
- **158+39/横截面特征不增效**：在 SAM 下无害但也不提升分数，说明 39 维特征已包含全部有效信号

## 调试注意事项

- Python 多行命令在 Bash 工具中会报 SyntaxError——使用 Write 工具创建临时脚本再运行
- 训练前必须 `rm -rf model/<output_dir>` 否则旧模型残留
- `__pycache__` 不会导致过期问题（pyc 随 py 同步更新）

## 关键文件位置

- 竞争入口：`python code/src/train.py`（训练）然后 `python code/src/predict.py`（预测）
- 自评：`python test/score_self.py`
- 数据集：`data/stock_data.csv`（主数据），`data/stock_data_April.csv`（新数据）
- 训练/测试分割：`data/split_train_test.py`