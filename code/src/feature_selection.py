"""Feature selection via mutual information with label."""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def select_features_by_mi(df, feature_cols, label_col='label', top_k=80, random_state=42):
    """Rank features by mutual information with label, return top_k feature names and full ranking.

    Args:
        df: DataFrame with features and label column
        feature_cols: list of feature column names to rank
        label_col: name of label column
        top_k: number of top features to retain
        random_state: seed for MI estimator

    Returns:
        selected: list of top_k feature names in descending MI order
        ranking: list of (feature_name, mi_score) tuples, all features ranked
    """
    X = df[feature_cols].fillna(0.0)
    y = df[label_col]

    mi_scores = mutual_info_regression(X.values, y.values, random_state=random_state)
    ranked = sorted(zip(feature_cols, mi_scores), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in ranked[:top_k]]
    return selected, ranked


def print_feature_ranking(ranking):
    """Print ranked features with MI scores."""
    print(f"\n{'Rank':<6}{'Feature':<30}{'MI Score':<12}")
    print("-" * 48)
    for i, (name, score) in enumerate(ranking, 1):
        print(f"{i:<6}{name:<30}{score:<12.6f}")


def save_feature_ranking(ranking, output_path):
    """Save feature ranking to CSV."""
    df = pd.DataFrame(ranking, columns=['feature', 'mi_score'])
    df.index.name = 'rank'
    df.index += 1
    df.to_csv(output_path)
    print(f"Feature ranking saved to {output_path}")
