# PyTorch Optimization Skill

## Key Optimization Strategies for PyTorch Training

### 1. Training Speed Optimization
- Use `torch.compile()` for automatic graph optimization (PyTorch 2.0+)
- Enable mixed precision training with `torch.cuda.amp.autocast` and `GradScaler`
- Use `DataLoader` with `num_workers > 0` and `pin_memory=True`
- Set `torch.backends.cudnn.benchmark = True` for fixed input sizes
- Use `torch.no_grad()` during evaluation

### 2. Memory Optimization
- Use gradient accumulation for large effective batch sizes
- Enable gradient checkpointing with `torch.utils.checkpoint`
- Use `.backward(retain_graph=False)` by default
- Clear unused tensors with `del` and `torch.cuda.empty_cache()`

### 3. Learning Rate Scheduling
- Use `OneCycleLR` for fast convergence
- Use `CosineAnnealingWarmRestarts` for periodic exploration
- Warm up learning rate for first few epochs

### 4. Model Architecture Optimization
- Use LayerNorm instead of BatchNorm for variable-size batches
- Use GELU activation for transformer models
- Apply dropout strategically (lower in deeper layers)
- Use residual connections for gradient flow

### 5. Data Pipeline Optimization
- Pre-compute and cache features when possible
- Use memory-mapped files for large datasets
- Batch padding to avoid variable-size overhead
- Use `prefetch_factor` in DataLoader

### 6. Numerical Stability
- Use `log_softmax` instead of `softmax` + `log`
- Use `torch.clamp` to prevent gradient explosion
- Apply gradient clipping with `torch.nn.utils.clip_grad_norm_`
- Use epsilon in normalization layers
