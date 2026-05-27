# Phase 3: Robustness Optimization Design

**Date**: 2026-05-25
**Goal**: 降低跨 seed 方差，在稳健基础上继续提升得分
**Baseline**: Phase 2 配置，seed=42 → 0.02783，seed=7 → 0.01052，seed=123 → -0.00363

## Problem Diagnosis

31 个验证样本导致模型选择（best epoch）受验证集随机性主导。单 seed 高分不可靠——seed=42 的 0.02783 可能包含大量运气成分。

## Solution: 扩大验证集 + 时间序列 CV

### Phase 3a — 扩大验证集

**改动**: `val_months: 2 → 3`

验证样本从 ~31 天增至 ~45 天。在完整 Phase 2 配置下单独测试 val_months=3，与 val_months=2 对比。

**验收标准**: val_months=3 单独得分 ≥ 0.025，则采纳；否则保持 val_months=2。

### Phase 3b — 时间序列交叉验证

**新增文件**: `code/src/cross_validate.py`

3-fold expanding window CV:
- Fold 1: train=[2024-01, 2025-10], val=[2025-11, 2025-12]
- Fold 2: train=[2024-01, 2025-12], val=[2026-01, 2026-01]
- Fold 3: train=[2024-01, 2026-01], val=[2026-02, 2026-03]

每个 fold 独立训练 60 epochs，记录每 epoch val score。三 fold 按 epoch 对齐取平均，选平均最高的 epoch N_best。

最终: 全量数据训练 N_best epochs（无 early stopping），保存模型。

**关键设计**:
- 复用现有 train.py 训练循环，通过参数传入 train_end_date / val_start_date
- 不在 fold 内做 early stopping——完整跑 60 epochs 以获取完整曲线
- 最终模型用全量数据，epochs=N_best，无 early stopping
- 训练时间: 3 folds × 1.5 min + 1 full × 1.5 min ≈ 6 min

### 可选 Phase 3c — 多 Seed 集成

用 N_best epochs 训练 3-5 个不同 seed，predict 时对多模型打分取平均。可最后做。

### Files to Modify

- `code/src/config.py` — val_months 调整
- `code/src/cross_validate.py` — NEW，时间序列 CV 脚本
- `code/src/train.py` — 可能需要支持 train_end_date/val_start_date 参数传入

### Expected Outcome

- 多 seed 方差显著降低（seed 间分数差异缩小 50%+）
- 稳健的 best epoch 选择（不再被单个验证窗口误导）
- 最终 self-score 目标 ≥ 0.020（稳健提升）
