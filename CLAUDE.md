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