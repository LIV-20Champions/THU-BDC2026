# 实验汇总

## Leaderboard (原始数据集 self-score)

| # | 实验 | 配置 | Score | vs Phase 12 |
|---|------|------|-------|-------------|
| 1 | Phase 12 Baseline | CSZScoreNorm, 0.05*MSE, 60ep, 5model | **0.0504** | baseline |
| 2 | MSE weight=1.0, 10ep, 1model | pure MSE, no rank loss | ~0.0026 | -94.8% |
| 3 | SmoothNDCG+MarginRanking | ranking loss + pairwise + margin | ~0.0017 | -96.6% |
| 4 | Dropout 0.5, val_months=2 | stronger regularization | ~0.0070 | -86.1% |
| 5 | seq_len=15, val=0, 1model×5ep | shorter window | 0.0467 | -7.3% |
| 6 | seq_len=15, val=2, ensemble=5 | val-based selection | ~0.0106 | anti-correlated val |
| 7 | 100+39 feat, cross-sec rank, WeightedRankingLoss | classmate's approach | **-0.0126** | negative |

## Phase 12 Best (reference)
- config: CSZScoreNorm + DropExtremeLabel + DailyBatch, mse_aux_weight=0.05, AdamW, 60ep, 5model ensemble (seeds 42,49,56,63,70)
- original: **0.0504**
- April: **0.0018**

## Key findings
1. Phase 12 recipe is fragile — any deviation causes score collapse
2. Ranking losses (SmoothNDCG, Margin, WeightedRanking) ALL worse than tiny MSE
3. Stronger regularization (dropout ↑, shorter window) degrades performance
4. Cross-sectional rank normalization + WeightedRankingLoss = negative score
5. Validation-based model selection is anti-correlated with test performance
6. Classmate's approach (cross-sectional rank + StandardScaler + batch shuffle) requires coupled design decisions — can't mix with our pipeline
