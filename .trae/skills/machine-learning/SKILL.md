# Machine Learning Optimization Skill

## Key Strategies for ML Model Optimization

### 1. Feature Engineering
- Create interaction features between correlated variables
- Use rolling statistics (mean, std, skew, kurtosis) over multiple windows
- Add momentum and mean-reversion features
- Use rank-based features for robustness to outliers

### 2. Loss Function Design
- Use ranking-aware losses (ListMLE, Pairwise) for ranking tasks
- Combine multiple loss components with learnable or tuned weights
- Apply label smoothing to prevent overconfidence
- Use focal loss for imbalanced data

### 3. Regularization
- Use dropout with appropriate rates (0.1-0.3 for transformers)
- Apply weight decay (L2 regularization)
- Use early stopping based on validation metrics
- Consider stochastic depth for deep models

### 4. Training Strategy
- Use warm-up + cosine decay learning rate schedule
- Apply gradient clipping (max_norm=1.0)
- Use larger batch sizes when possible for stable gradients
- Train with multiple random seeds and ensemble

### 5. Evaluation & Validation
- Use walk-forward validation for time series
- Monitor both ranking metrics and return metrics
- Check for data leakage (future information in features)
- Use out-of-sample validation that matches competition setup
