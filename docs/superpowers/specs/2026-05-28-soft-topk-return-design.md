# Phase 11: Soft Top-K Return Objective

**Date**: 2026-05-28
**Goal**: Improve the ceiling and stability of the stock-selection score by aligning training with the final top-5 weighted return metric.
**Baseline**: Phase 9/10 ensemble reaches the same top-5 basket and self-score around 0.0673, while architecture and feature expansion have not improved the result.

## Problem Diagnosis

The current system is already strong enough to find one stable basket, but the training objective is still not the same as the final scoring function.

The final scorer uses only:

1. The top at most 5 predicted stocks.
2. Their prediction-time allocation weights.
3. Each selected stock's true 5-day open-to-open return.

The current training path optimizes mostly ranking proxies:

1. `label` is a blended target: `0.3 * ret_t1_to_t3 + 0.7 * ret_t1_to_t5`.
2. `SmoothNDCG` and `SoftRankIC` reward broad rank quality, not the expected return of the selected top-5 basket.
3. Validation-based checkpoint selection is unreliable because previous experiments show validation score can be negatively correlated with the final self-score.

This mismatch can make the model train "better" according to internal metrics while not improving the actual output basket.

## Requirements

1. Keep the public workflow unchanged: users still run only `python code/src/train.py` and `python code/src/predict.py`.
2. Avoid relying on user-selectable seeds. Any ensemble seed logic must stay internal to `train.py`.
3. Keep training time practical. Do not add a large backbone or expensive feature pipeline in this phase.
4. Preserve existing working components: SAM, EMA/SWA, rank-decay prediction weights, ensemble inference, and the current Transformer backbone.
5. Make the loss mask-aware so padded stocks never affect training.

## Recommended Design

Introduce a scoring-aligned objective:

```text
total_loss =
    topk_return_weight * SoftTopKReturnLoss(pred, score_target, mask)
  + rankic_weight       * MaskedSoftRankICLoss(pred, label, mask)
  + optional_mse_weight  * MSE(pred, label)   # only if explicitly enabled
```

The main learning signal becomes "which predicted basket produces the highest expected 5-day return", while RankIC remains a small stabilizer.

## Data Targets

Keep the existing `label` for compatibility and auxiliary ranking stability.

Add a separate `score_target`:

```text
label        = label_alpha * ret_t1_to_t3 + (1 - label_alpha) * ret_t1_to_t5
score_target = ret_t1_to_t5
```

Rationale:

1. `score_target` matches `test/score_self.py`.
2. `label` can still regularize general rank ordering and preserve the medium-horizon signal that previously helped.
3. Splitting targets avoids overloading one column with two meanings.

Data flow changes:

1. `_build_label_and_clean()` computes both `label` and `score_target`.
2. `LazyRankingDataset` stores both per stock per day.
3. `__getitem__()` returns both `targets` and `score_targets`.
4. `collate_fn()` pads both tensors and returns both masks.

## SoftTopKReturnLoss

For each daily stock group:

1. Select only valid stocks from `mask`.
2. Compute differentiable ranks from predicted scores:

```text
soft_rank_i = 1 + sum_j sigmoid((score_j - score_i) / rank_temperature) - 0.5
```

3. Convert ranks into a soft top-k gate:

```text
gate_i = sigmoid((top_k + gate_margin - soft_rank_i) / gate_temperature)
```

4. Build differentiable basket weights:

```text
raw_weight_i = softmax(score_i / weight_temperature) * gate_i
basket_weight_i = raw_weight_i / sum(raw_weight)
```

5. Maximize expected basket return:

```text
expected_return = sum_i basket_weight_i * score_target_i
loss = -mean(expected_return)
```

This is not an exact discrete top-5 operation, but it gives gradients to near-top candidates and directly pushes the model toward profitable selected baskets.

## MaskedSoftRankICLoss

The existing `SoftRankICLoss` should be made mask-aware.

Instead of computing RankIC on padded full tensors, each daily group should be filtered to valid stocks first. This prevents padded zeros from creating artificial rank structure.

RankIC should use `label`, not `score_target`, because its job is broad ordering stability rather than exact scoring mimicry.

## Training And Checkpointing

The default Phase 11 training behavior should stop using validation score to choose the final checkpoint.

Recommended behavior:

1. Train for a fixed epoch budget.
2. Use `val_months=0` for the default competition run so checkpoint choice does not depend on validation score.
3. Keep validation metrics only as diagnostics when validation is enabled manually.
4. Save the final epoch model to `best_model.pth` for compatibility with `predict.py`.
5. Keep EMA/SWA outputs if enabled; `predict.py` can continue preferring SWA, then EMA, then standard weights.
6. Avoid early stopping by validation score for the default competition path.

This matches the project finding that validation score is not a trustworthy model-selection signal for the final self-score.

## Configuration

Add config switches so this phase can be tested and reverted cleanly:

```python
'use_soft_topk_return_loss': True,
'soft_topk_return_weight': 1.0,
'soft_topk_rankic_weight': 0.2,
'soft_topk_k': 5,
'soft_topk_rank_temperature': 0.5,
'soft_topk_gate_temperature': 0.5,
'soft_topk_weight_temperature': 0.5,
'soft_topk_gate_margin': 0.5,
'use_validation_checkpoint_selection': False,
'use_mse_aux_loss': False,
```

Default Phase 11 should disable or down-weight older proxy losses. They remain available for ablations, but the new default should be return-first.

## Files To Change

| File | Change |
| --- | --- |
| `code/src/train.py` | Add `SoftTopKReturnLoss`, mask-aware RankIC integration, and fixed-epoch checkpoint behavior. |
| `code/src/utils.py` | Carry `score_target` through `LazyRankingDataset`. |
| `code/src/config.py` | Add Phase 11 loss and checkpointing config flags. |
| `code/src/predict.py` | No required change unless checkpoint naming changes; the design keeps compatibility. |
| `test/` | Add targeted tests for target construction, dataset collation, masked loss behavior, and checkpoint selection mode. |

## Non-Goals

1. Do not add VSN, MultiScale, or a larger backbone in this phase.
2. Do not change the public output format.
3. Do not require the user to pass seed or epoch arguments.
4. Do not use `test.csv` inside `train.py`.
5. Do not optimize directly against `test/score_self.py` during training.

## Testing Plan

1. Unit-test `_build_label_and_clean()`:
   - `label` remains the configured blended target.
   - `score_target` equals pure `ret_t1_to_t5`.
2. Unit-test `LazyRankingDataset` and `collate_fn()`:
   - `score_targets` is present.
   - Padding preserves shape and mask semantics.
3. Unit-test `SoftTopKReturnLoss`:
   - Loss decreases when high-return stocks receive higher scores.
   - Padded entries do not change loss.
   - Gradients are finite.
4. Unit-test masked RankIC:
   - Padded entries do not affect correlation.
5. Integration-test `python code/src/train.py` and `python code/src/predict.py` on the normal path.
6. Run `python test/score_self.py` after prediction and compare against the current 0.0673 reference.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| The soft top-k loss over-focuses on noisy high-return outliers. | Keep RankIC as a stabilizer and use EMA/SWA. |
| The loss becomes too sharp and unstable. | Use temperature and gate margin config; start with moderate temperatures. |
| Training time increases. | Reuse the existing pairwise sigmoid style; no new large model branch. |
| The model learns high expected return but poorer discrete rank. | Monitor top-k metrics and self-score; keep older losses configurable for ablation. |
| Validation metrics look worse. | Treat validation as diagnostics only, not model selection. |

## Success Criteria

1. The default train/predict path completes without extra user arguments.
2. `output/result.csv` is regenerated by `predict.py`.
3. Across three clean runs, the median self-score should beat the current stable reference around 0.0673.
4. The minimum score across those runs should stay positive.
