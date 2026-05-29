import math


def is_gradient_clipping_enabled(config):
    if 'clip_grad' in config:
        return bool(config.get('clip_grad'))
    return not bool(config.get('drop_clip', True))


def get_lr_factor(epoch, num_epochs, warmup_ratio=0.05, min_factor=0.2):
    num_epochs = max(int(num_epochs), 1)
    epoch = min(max(int(epoch), 0), num_epochs - 1)
    warmup_ratio = max(float(warmup_ratio), 0.0)
    min_factor = min(max(float(min_factor), 0.0), 1.0)

    warmup_epochs = int(math.ceil(num_epochs * warmup_ratio)) if warmup_ratio > 0 else 0
    warmup_epochs = min(warmup_epochs, num_epochs)

    if warmup_epochs > 0 and epoch < warmup_epochs:
        return round(float(epoch + 1) / float(warmup_epochs), 12)

    decay_epochs = num_epochs - warmup_epochs
    if decay_epochs <= 1:
        return 1.0 if epoch < num_epochs - 1 else min_factor

    progress = float(epoch - warmup_epochs) / float(decay_epochs - 1)
    factor = 1.0 - (1.0 - min_factor) * progress
    return round(max(min_factor, min(1.0, factor)), 12)


def set_optimizer_lr(optimizer, base_lr, factor):
    lr = float(base_lr) * float(factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
