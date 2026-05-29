from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationDecision:
    selected_ids: list
    weights: list
    selected_mode: str
    metrics: dict
    reason: str


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError('weights 必须是一维且非空')
    if not np.all(np.isfinite(weights)):
        raise ValueError('weights 中存在非有限值')
    if weights.sum() <= 0:
        raise ValueError('weights 权重和必须大于 0')

    weights = weights / weights.sum()
    weights = np.round(weights, 4)

    diff = round(1.0 - float(weights.sum()), 4)
    weights[-1] = round(float(weights[-1]) + diff, 4)

    if weights[-1] < 0:
        weights[0] = round(float(weights[0]) + float(weights[-1]), 4)
        weights[-1] = 0.0

    return weights.tolist()


def softmax_weights(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError('scores 必须是一维且非空')

    temperature = max(float(temperature), 1e-8)
    shifted = scores / temperature
    shifted = shifted - np.max(shifted)
    weights = np.exp(shifted)
    return normalize_weights(weights)


def calculate_confidence_metrics(scores, temperature=0.05, margin_scale=0.02):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError('scores 必须是一维且非空')
    if not np.all(np.isfinite(scores)):
        raise ValueError('scores 中存在非有限值')

    top_scores = scores[: min(5, scores.size)]
    if top_scores.size == 1:
        return {
            'confidence_rate': 1.0,
            'softmax_p1': 1.0,
            'entropy_confidence': 1.0,
            'gap_confidence': 1.0,
            'margin12': float('inf'),
            'margin1mean': float('inf'),
            'z_gap': float('inf'),
            'top_k': 1,
        }

    temperature = max(float(temperature), 1e-8)
    margin_scale = max(float(margin_scale), 1e-8)

    shifted = top_scores / temperature
    shifted = shifted - np.max(shifted)
    exp_scores = np.exp(shifted)
    probs = exp_scores / exp_scores.sum()

    margin12 = float(top_scores[0] - top_scores[1])
    margin1mean = float(top_scores[0] - np.mean(top_scores[1:]))
    std = float(np.std(top_scores))
    z_gap = margin12 / (std + 1e-12)

    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    max_entropy = float(np.log(top_scores.size))
    entropy_confidence = 1.0 - entropy / max_entropy if max_entropy > 0 else 1.0
    entropy_confidence = float(np.clip(entropy_confidence, 0.0, 1.0))

    gap_confidence = 1.0 - np.exp(-max(margin12, 0.0) / margin_scale)
    gap_confidence = float(np.clip(gap_confidence, 0.0, 1.0))

    softmax_p1 = float(np.clip(probs[0], 0.0, 1.0))

    confidence_rate = (
        0.40 * gap_confidence +
        0.50 * softmax_p1 +
        0.10 * entropy_confidence
    )
    confidence_rate = float(np.clip(confidence_rate, 0.0, 1.0))

    return {
        'confidence_rate': confidence_rate,
        'softmax_p1': softmax_p1,
        'entropy_confidence': entropy_confidence,
        'gap_confidence': gap_confidence,
        'margin12': margin12,
        'margin1mean': margin1mean,
        'z_gap': float(z_gap),
        'top_k': int(top_scores.size),
    }


def _max_output_k(ranked_stock_ids, ranked_scores, config):
    return min(
        int(config.get('predict_top_k', 5)),
        5,
        len(ranked_stock_ids),
        len(ranked_scores),
    )


def allocate_by_confidence(ranked_stock_ids, ranked_scores, config):
    ranked_scores = np.asarray(ranked_scores, dtype=np.float64)
    max_k = _max_output_k(ranked_stock_ids, ranked_scores, config)
    if max_k <= 0:
        raise ValueError('没有可用于输出的股票')

    top_scores = ranked_scores[:max_k]
    metrics = calculate_confidence_metrics(
        top_scores,
        temperature=config.get('confidence_temperature', config.get('predict_weight_temperature', 0.05)),
        margin_scale=config.get('confidence_margin_scale', 0.02),
    )

    top1_threshold = float(config.get('confidence_top1_threshold', 0.60))
    top3_threshold = float(config.get('confidence_top3_threshold', 0.40))
    min_margin12 = float(config.get('confidence_min_margin12', 0.01))
    confidence_rate = float(metrics['confidence_rate'])
    margin12 = float(metrics['margin12'])

    if confidence_rate >= top1_threshold and margin12 >= min_margin12:
        selected_ids = ranked_stock_ids[:1]
        weights = [1.0]
        selected_mode = 'confidence_top1'
        reason = (
            f'置信率 {confidence_rate:.4f} >= {top1_threshold:.4f}，'
            f'且第一二名分差 {margin12:.6f} >= {min_margin12:.6f}，采用单押。'
        )
    elif confidence_rate >= top3_threshold and max_k >= 3:
        k = 3
        selected_ids = ranked_stock_ids[:k]
        weights = normalize_weights(config.get('predict_top3_weights', [0.45, 0.35, 0.20])[:k])
        selected_mode = 'confidence_top3'
        reason = (
            f'置信率 {confidence_rate:.4f} 未达到单押阈值 {top1_threshold:.4f}，'
            f'但达到 Top3 阈值 {top3_threshold:.4f}，采用前三分散。'
        )
    else:
        fallback_mode = config.get('confidence_fallback_mode', 'rank_decay')
        if fallback_mode == 'equal':
            selected_ids = ranked_stock_ids[:max_k]
            weights = normalize_weights(np.ones(max_k, dtype=np.float64))
            selected_mode = 'confidence_equal'
        elif fallback_mode == 'softmax':
            selected_ids = ranked_stock_ids[:max_k]
            weights = softmax_weights(
                top_scores,
                config.get('predict_weight_temperature', config.get('confidence_temperature', 0.05)),
            )
            selected_mode = 'confidence_softmax'
        else:
            selected_ids = ranked_stock_ids[:max_k]
            base_weights = np.asarray(
                config.get('predict_rank_weights', [0.30, 0.25, 0.20, 0.15, 0.10]),
                dtype=np.float64,
            )[:max_k]
            weights = normalize_weights(base_weights)
            selected_mode = 'confidence_rank_decay'
        reason = (
            f'置信率 {confidence_rate:.4f} 低于 Top3 阈值 {top3_threshold:.4f}，'
            f'或第一二名分差 {margin12:.6f} 不足，采用 {fallback_mode} 回退。'
        )

    return AllocationDecision(
        selected_ids=list(selected_ids),
        weights=weights,
        selected_mode=selected_mode,
        metrics=metrics,
        reason=reason,
    )


def allocate_confidence_flex(ranked_stock_ids, ranked_scores, config):
    ranked_scores = np.asarray(ranked_scores, dtype=np.float64)
    max_k = _max_output_k(ranked_stock_ids, ranked_scores, config)
    if max_k <= 0:
        raise ValueError('没有可用于输出的股票')

    top_scores = ranked_scores[:max_k]
    metrics = calculate_confidence_metrics(
        top_scores,
        temperature=config.get('confidence_temperature', config.get('predict_weight_temperature', 0.05)),
        margin_scale=float(config.get('confidence_margin_scale', 0.05)),
    )

    cr = float(np.clip(metrics['confidence_rate'], 0.0, 1.0))
    margin12 = float(metrics['margin12'])
    min_margin = float(config.get('confidence_min_margin12', 0.01))

    # — 维度1：持仓数 k —  cr=1.0→1只, cr=0.0→5只, 连续映射 —
    raw_k = int(round(5.0 - 4.0 * cr))
    k = max(1, min(max_k, raw_k))

    # 即使置信率高，但一二名分差太小也不单押
    if k == 1 and margin12 < min_margin:
        k = 2

    # — 维度2：自适应温度 —  cr=1.0→超集中, cr=0.0→近等权 —
    base_temp = float(config.get('predict_weight_temperature', 0.5))
    temp = min(2.0, 0.05 / max(0.001, cr + 0.05))
    temp = max(base_temp * 0.1, temp)

    # — 计算权重 —
    if k == 1:
        weights = [1.0]
    else:
        weights = softmax_weights(top_scores[:k], temperature=temp)

    return AllocationDecision(
        selected_ids=list(ranked_stock_ids[:k]),
        weights=weights,
        selected_mode=f'conf_flex_k{k}',
        metrics=metrics,
        reason=(
            f'置信率{cr:.4f}→k={k}只(k_等权={max_k})，'
            f'温度{temp:.4f}，margin={margin12:.4f}'
        ),
    )


def allocate_by_mode(mode, ranked_stock_ids, ranked_scores, config):
    ranked_scores = np.asarray(ranked_scores, dtype=np.float64)
    max_k = _max_output_k(ranked_stock_ids, ranked_scores, config)
    if max_k <= 0:
        raise ValueError('没有可用于输出的股票')

    if mode == 'confidence':
        return allocate_by_confidence(ranked_stock_ids, ranked_scores, config)

    if mode == 'confidence_flex':
        return allocate_confidence_flex(ranked_stock_ids, ranked_scores, config)

    metrics = calculate_confidence_metrics(
        ranked_scores[:max_k],
        temperature=config.get('confidence_temperature', config.get('predict_weight_temperature', 0.05)),
        margin_scale=config.get('confidence_margin_scale', 0.02),
    )

    if mode == 'equal':
        selected_ids = ranked_stock_ids[:max_k]
        weights = normalize_weights(np.ones(max_k, dtype=np.float64))
    elif mode == 'rank_decay':
        selected_ids = ranked_stock_ids[:max_k]
        base_weights = np.asarray(
            config.get('predict_rank_weights', [0.30, 0.25, 0.20, 0.15, 0.10]),
            dtype=np.float64,
        )[:max_k]
        weights = normalize_weights(base_weights)
    elif mode == 'top3':
        k = min(3, len(ranked_stock_ids), len(ranked_scores))
        selected_ids = ranked_stock_ids[:k]
        weights = normalize_weights(config.get('predict_top3_weights', [0.45, 0.35, 0.20])[:k])
    elif mode == 'top1':
        selected_ids = ranked_stock_ids[:1]
        weights = [1.0]
    elif mode == 'softmax':
        selected_ids = ranked_stock_ids[:max_k]
        weights = softmax_weights(ranked_scores[:max_k], config.get('predict_weight_temperature', 0.5))
    else:
        raise ValueError(f'未知 predict_weight_mode: {mode}')

    return AllocationDecision(
        selected_ids=list(selected_ids),
        weights=weights,
        selected_mode=mode,
        metrics=metrics,
        reason=f'使用固定权重模式 {mode}。',
    )
