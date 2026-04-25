"""
快速搜索预测权重方案。

使用前提：已经运行过一次 test.sh，并生成 output/ranked_scores.csv。
该脚本不会重新训练，也不会重新跑模型，只根据完整排序文件和 data/test.csv 测试不同权重。

常用命令：
python tools/quick_weight_search.py
python tools/quick_weight_search.py --write-best
"""

import argparse
import os
import numpy as np
import pandas as pd


def calculate_return(group):
    group = group.sort_values('日期')
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']


def build_returns(test_data, stock_ids):
    test_data = test_data.copy()
    test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)
    test_data['日期'] = pd.to_datetime(test_data['日期'])

    selected = test_data[test_data['股票代码'].isin(stock_ids)].copy()
    selected = selected.groupby('股票代码', group_keys=False).tail(5)

    rows = []
    for stock_id, group in selected.groupby('股票代码'):
        rows.append((stock_id, float(calculate_return(group))))

    return dict(rows)


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

    return np.round(weights, 4)


def softmax_weights(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    temperature = max(float(temperature), 1e-8)
    shifted = scores / temperature
    shifted = shifted - np.max(shifted)
    weights = np.exp(shifted)
    return normalize_weights(weights)


def evaluate_scheme(name, ranked_ids, ranked_scores, returns_map, weights):
    weights = normalize_weights(weights)
    k = len(weights)
    stock_ids = ranked_ids[:k]

    score = 0.0
    rows = []
    for sid, weight in zip(stock_ids, weights):
        stock_return = float(returns_map.get(sid, 0.0))
        score += stock_return * float(weight)
        rows.append((sid, float(weight), stock_return))

    return {
        'scheme': name,
        'k': k,
        'score': score,
        'stocks': ','.join([row[0] for row in rows]),
        'weights': ','.join([f'{row[1]:.4f}' for row in rows]),
        'returns': ','.join([f'{row[2]:.6f}' for row in rows]),
    }


def main():
    parser = argparse.ArgumentParser(description='快速搜索 Top 股票权重方案')
    parser.add_argument('--ranked', default='output/ranked_scores.csv', help='predict.py 生成的完整排序文件')
    parser.add_argument('--test', default='data/test.csv', help='本地测试集文件')
    parser.add_argument('--out', default='output/weight_search_results.csv', help='搜索结果保存路径')
    parser.add_argument('--write-best', action='store_true', help='把最高分方案写入 output/result.csv')
    parser.add_argument('--top-n-return-cache', type=int, default=20, help='只为排序前 N 的股票计算真实收益，默认20')
    args = parser.parse_args()

    if not os.path.exists(args.ranked):
        raise FileNotFoundError(f'未找到完整排序文件: {args.ranked}。请先运行 sh test.sh。')
    if not os.path.exists(args.test):
        raise FileNotFoundError(f'未找到测试集: {args.test}')

    ranked = pd.read_csv(args.ranked, dtype={'stock_id': str})
    ranked['stock_id'] = ranked['stock_id'].astype(str).str.zfill(6)
    ranked = ranked.sort_values('rank')

    ranked_ids = ranked['stock_id'].tolist()
    ranked_scores = ranked['score'].astype(float).values

    test_data = pd.read_csv(args.test, dtype={'股票代码': str})
    returns_map = build_returns(test_data, ranked_ids[:args.top_n_return_cache])

    schemes = []

    # 固定前5权重
    schemes.append(('equal5', [0.20, 0.20, 0.20, 0.20, 0.20]))
    schemes.append(('decay_30_25_20_15_10', [0.30, 0.25, 0.20, 0.15, 0.10]))
    schemes.append(('decay_35_25_18_12_10', [0.35, 0.25, 0.18, 0.12, 0.10]))
    schemes.append(('decay_40_25_20_10_05', [0.40, 0.25, 0.20, 0.10, 0.05]))
    schemes.append(('decay_50_20_15_10_05', [0.50, 0.20, 0.15, 0.10, 0.05]))

    # 只买前几名
    schemes.append(('top1_100', [1.00]))
    schemes.append(('top2_60_40', [0.60, 0.40]))
    schemes.append(('top2_70_30', [0.70, 0.30]))
    schemes.append(('top3_45_35_20', [0.45, 0.35, 0.20]))
    schemes.append(('top3_50_30_20', [0.50, 0.30, 0.20]))
    schemes.append(('top3_60_25_15', [0.60, 0.25, 0.15]))

    # softmax 权重，只看前5
    for temp in [0.1, 0.2, 0.5, 1.0, 2.0]:
        weights = softmax_weights(ranked_scores[:5], temp)
        schemes.append((f'softmax_t{temp}', weights))

    results = []
    for name, weights in schemes:
        results.append(evaluate_scheme(name, ranked_ids, ranked_scores, returns_map, weights))

    result_df = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(args.out, index=False)

    print(result_df[['scheme', 'k', 'score', 'stocks', 'weights']].to_string(index=False))
    print(f'\n搜索结果已保存到: {args.out}')

    if args.write_best:
        best = result_df.iloc[0]
        best_stocks = best['stocks'].split(',')
        best_weights = [float(x) for x in best['weights'].split(',')]

        output_df = pd.DataFrame({
            'stock_id': best_stocks,
            'weight': best_weights,
        })
        os.makedirs('output', exist_ok=True)
        output_df.to_csv('output/result.csv', index=False)
        print(f"\n已将最佳方案写入 output/result.csv: {best['scheme']}, score={best['score']:.8f}")
        print(output_df.to_string(index=False))


if __name__ == '__main__':
    main()
