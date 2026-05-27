import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config, feature_columns_map, feature_engineer_func_map, get_scale_features, get_eff_input_dim
from model import StockTransformer
from utils import LazyRankingDataset
import joblib
import os
import json
import gc
import multiprocessing as mp
import random
import math
from sam import SAM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_feature_engineering_workers(num_groups):
    configured = config.get('feature_engineering_workers')
    if configured is not None:
        try:
            configured = int(configured)
        except (TypeError, ValueError):
            configured = None
    if configured is None:
        configured = min(4, mp.cpu_count())
    return max(1, min(int(configured), mp.cpu_count(), max(1, int(num_groups))))


def _build_label_and_clean(processed, drop_small_open=True, label_alpha=0.3):
    """Build label from future open-to-open returns, using open_t1 as base price.

    label = alpha * ret(open_t1 -> open_t3) + (1-alpha) * ret(open_t1 -> open_t5)
    where ret(A -> B) = (B - A) / A

    Default alpha=0.3 means 30% weight on t1→t3, 70% on t1→t5 (matching original).
    """
    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t3'] = processed.groupby('股票代码')['开盘'].shift(-3)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    ret_t1t3 = (processed['open_t3'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    ret_t1t5 = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed['label'] = label_alpha * ret_t1t3 + (1.0 - label_alpha) * ret_t1t5
    processed = processed.dropna(subset=['label'])

    processed.drop(columns=['open_t1', 'open_t3', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_columns_map[config['feature_num']]

    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = get_feature_engineering_workers(len(groups))
    print(f"{desc}进程数: {num_processes}")
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)

    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    label_alpha = float(config.get('label_alpha', 0.3))
    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open, label_alpha=label_alpha)
    return processed, feature_columns


def preprocess_data(df, is_train=True, stockid2idx=None):
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=is_train)


def _soft_rank(scores, temperature=0.5):
    diff = scores.unsqueeze(1) - scores.unsqueeze(2)
    sig = torch.sigmoid(diff / max(temperature, 0.01))
    return 1.0 + sig.sum(dim=2) - 0.5


class SmoothNDCGLoss(nn.Module):
    def __init__(self, top_k=5, temperature=0.5, delta_clip=10.0,
                 smooth_ndcg_weight=0.7, lambda_pairwise_weight=0.3,
                 use_mse_aux=False, mse_aux_weight=0.05,
                 temperature_start=None, temperature_target=None,
                 temperature_anneal_epochs=None):
        super(SmoothNDCGLoss, self).__init__()
        self.top_k = top_k
        self.temperature = temperature
        self.delta_clip = delta_clip
        self.smooth_ndcg_weight = smooth_ndcg_weight
        self.lambda_pairwise_weight = lambda_pairwise_weight
        self.use_mse_aux = use_mse_aux
        self.mse_aux_weight = mse_aux_weight
        self.temperature_start = temperature_start if temperature_start is not None else temperature
        self.temperature_target = temperature_target if temperature_target is not None else temperature
        self.temperature_anneal_epochs = temperature_anneal_epochs if temperature_anneal_epochs is not None else 0
        self.current_temperature = float(self.temperature_start)

    def set_temperature(self, epoch, total_epochs):
        if self.temperature_anneal_epochs <= 0 or self.temperature_start == self.temperature_target:
            self.current_temperature = self.temperature
            return
        progress = min(float(epoch) / float(self.temperature_anneal_epochs), 1.0)
        self.current_temperature = self.temperature_start + (self.temperature_target - self.temperature_start) * progress

    def smooth_ndcg_loss(self, y_pred, y_true):
        B, N = y_true.size()
        k = min(self.top_k, N)
        temp = max(self.current_temperature, 0.01)
        eps = 1e-8

        relevance = y_true - y_true.mean(dim=1, keepdim=True)
        gains = torch.clamp(2.0 ** relevance - 1.0, min=0.0)

        y_pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        y_pred_norm = y_pred_centered / (y_pred_centered.std(dim=1, keepdim=True) + eps)
        soft_ranks = _soft_rank(y_pred_norm, temp)
        dcg_positions = soft_ranks
        discounts = 1.0 / torch.log2(dcg_positions + 1.0)
        soft_weight = torch.sigmoid((k + 1.5 - dcg_positions) / temp)
        soft_dcg = (gains * discounts * soft_weight).sum(dim=1)

        with torch.no_grad():
            ideal_gains_sorted, _ = torch.sort(gains, dim=1, descending=True)
            ideal_gains_k = ideal_gains_sorted[:, :k]
            ideal_discounts = 1.0 / torch.log2(torch.arange(1, k + 1, device=gains.device, dtype=gains.dtype) + 1.0)
            ideal_dcg = (ideal_gains_k * ideal_discounts.unsqueeze(0)).sum(dim=1)

        ndcg = soft_dcg / (ideal_dcg + eps)
        return (1.0 - ndcg).mean()

    def lambda_pairwise_loss(self, y_pred, y_true):
        B, N = y_true.size()
        relevance = y_true - y_true.mean(dim=1, keepdim=True)

        with torch.no_grad():
            gains = 2.0 ** relevance - 1.0
            ranks = y_pred.argsort(dim=1, descending=True).argsort(dim=1).float()
            discounts = 1.0 / torch.log2(ranks + 2.0)

            gain_i = gains.unsqueeze(2)
            gain_j = gains.unsqueeze(1)
            disc_i = discounts.unsqueeze(2)
            disc_j = discounts.unsqueeze(1)

            delta = torch.abs(gain_i * disc_i + gain_j * disc_j
                            - gain_i * disc_j - gain_j * disc_i)
            delta = torch.clamp(delta, max=self.delta_clip)

            rel_sign = torch.sign(relevance.unsqueeze(2) - relevance.unsqueeze(1))
            mask = (rel_sign != 0).float()

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        pair_loss = torch.log1p(torch.exp(-rel_sign * pred_diff))
        weighted = pair_loss * delta * mask
        n_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        return (weighted.sum(dim=[1, 2]) / n_pairs).mean()

    def forward(self, y_pred, y_true):
        loss = torch.tensor(0.0, device=y_pred.device, dtype=y_pred.dtype)
        if self.smooth_ndcg_weight > 0:
            loss = loss + self.smooth_ndcg_weight * self.smooth_ndcg_loss(y_pred, y_true)
        if self.lambda_pairwise_weight > 0:
            loss = loss + self.lambda_pairwise_weight * self.lambda_pairwise_loss(y_pred, y_true)
        if self.use_mse_aux:
            loss = loss + self.mse_aux_weight * F.mse_loss(y_pred, y_true)
        return loss


class EMAWrapper:
    """Exponential Moving Average of model parameters for smoother inference."""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_val = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                    self.shadow[name] = new_val.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def _build_eval_weights(pred_scores, mode='equal', temperature=0.25):
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    if pred_scores.ndim != 1 or pred_scores.size == 0:
        raise ValueError('pred_scores 必须是一维且非空')
    if not np.isfinite(pred_scores).all():
        pred_scores = np.nan_to_num(pred_scores, nan=0.0, posinf=1.0, neginf=-1.0)

    if mode == 'equal':
        weights = np.ones_like(pred_scores, dtype=np.float64)
    elif mode == 'softmax':
        temp = max(float(temperature), 1e-8)
        shifted = pred_scores / temp
        shifted = shifted - np.max(shifted)
        weights = np.exp(shifted)
    else:
        raise ValueError(f'不支持的评估权重模式: {mode}')

    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0:
        raise ValueError(f'评估权重非法，当前和为: {weights_sum}')
    return weights / weights_sum


def _resolve_model_selection_metric():
    configured = config.get('model_selection_metric')
    if configured:
        return configured
    predict_weight_mode = config.get('predict_weight_mode', 'equal')
    if predict_weight_mode == 'softmax':
        return 'official_score_weighted_softmax'
    return 'official_score_eq'


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    batch_size = y_pred.size(0)

    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    official_score_eq_list = []
    official_score_weighted_eq_list = []
    official_score_weighted_softmax_list = []
    oracle_score_eq_list = []
    oracle_score_weighted_softmax_list = []

    eval_temperature = float(config.get('predict_weight_temperature', 0.25))

    for i in range(batch_size):
        valid_indices = masks[i].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() < k:
            continue

        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices]

        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_top_scores = valid_pred[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()

        pred_top_returns_np = pred_top_returns.detach().cpu().numpy()
        pred_top_scores_np = pred_top_scores.detach().cpu().numpy()
        official_score_eq = pred_return_sum / k
        official_score_weighted_eq = float(np.mean(pred_top_returns_np))
        softmax_weights = _build_eval_weights(pred_top_scores_np, mode='softmax', temperature=eval_temperature)
        official_score_weighted_softmax = float(np.dot(pred_top_returns_np, softmax_weights))

        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        true_top_scores = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()

        true_top_returns_np = true_top_returns.detach().cpu().numpy()
        true_top_scores_np = true_top_scores.detach().cpu().numpy()
        oracle_score_eq = float(np.mean(true_top_returns_np))
        oracle_softmax_weights = _build_eval_weights(true_top_scores_np, mode='softmax', temperature=eval_temperature)
        oracle_score_weighted_softmax = float(np.dot(true_top_returns_np, oracle_softmax_weights))

        random_return_sum = k * valid_true.mean().item()

        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0

        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)
        official_score_eq_list.append(official_score_eq)
        official_score_weighted_eq_list.append(official_score_weighted_eq)
        official_score_weighted_softmax_list.append(official_score_weighted_softmax)
        oracle_score_eq_list.append(oracle_score_eq)
        oracle_score_weighted_softmax_list.append(oracle_score_weighted_softmax)

    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
        'ratio_pred': np.mean(ratio_pred_list) if ratio_pred_list else 0.0,
        'ratio_random': np.mean(ratio_random_list) if ratio_random_list else 0.0,
        'final_score': np.mean(final_score_list) if final_score_list else 0.0,
        'official_score_eq': np.mean(official_score_eq_list) if official_score_eq_list else 0.0,
        'official_score_weighted_eq': np.mean(official_score_weighted_eq_list) if official_score_weighted_eq_list else 0.0,
        'official_score_weighted_softmax': np.mean(official_score_weighted_softmax_list) if official_score_weighted_softmax_list else 0.0,
        'oracle_score_eq': np.mean(oracle_score_eq_list) if oracle_score_eq_list else 0.0,
        'oracle_score_weighted_softmax': np.mean(oracle_score_weighted_softmax_list) if oracle_score_weighted_softmax_list else 0.0,
    }

    return metrics


def collate_fn(batch):
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'].long() for item in batch]

    max_stocks = max(seq.size(0) for seq in sequences)

    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []

    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)

        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size)
            stock_pad = torch.full((pad_size,), -1, dtype=torch.long)

            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)

        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0

        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)

    result = {
        'sequences': torch.stack(padded_sequences),
        'targets': torch.stack(padded_targets),
        'relevance': torch.stack(padded_relevance),
        'stock_indices': torch.stack(padded_stock_indices).long(),
        'masks': torch.stack(masks),
    }
    return result


def _get_batch_stock_indices(batch, device):
    if 'stock_indices' not in batch:
        raise KeyError("batch 中缺少 stock_indices，当前模型需要该字段支持股票 embedding")
    stock_indices = batch['stock_indices']
    if not isinstance(stock_indices, torch.Tensor):
        raise TypeError(f"stock_indices 必须是 torch.Tensor，当前为: {type(stock_indices)}")
    return stock_indices.long().to(device)


def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer,
                        accumulation_steps=1, ema_wrapper=None,
                        use_mixup=False, mixup_alpha=0.2, mixup_prob=0.3,
                        use_label_smoothing=False, ls_alpha=0.05, scaler=None,
                        use_sam=False):
    model.train()
    total_loss = 0.0
    total_metrics = {}
    local_step = 0
    optimizer.zero_grad()
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        sequences = batch['sequences'].to(device)
        targets = batch['targets'].to(device)
        masks = batch['masks'].to(device)
        stock_indices = _get_batch_stock_indices(batch, device)

        if use_mixup and torch.rand(1).item() < mixup_prob:
            lbd = np.random.beta(mixup_alpha, mixup_alpha)
            lbd = max(min(lbd, 0.95), 0.05)
            for b in range(sequences.size(0)):
                valid_idx = masks[b].nonzero(as_tuple=False).flatten()
                n_valid = valid_idx.numel()
                if n_valid > 1:
                    perm_idx = valid_idx[torch.randperm(n_valid, device=device)]
                    inv_lbd = 1.0 - lbd
                    sequences[b, valid_idx] = lbd * sequences[b, valid_idx] + inv_lbd * sequences[b, perm_idx]
                    targets[b, valid_idx] = lbd * targets[b, valid_idx] + inv_lbd * targets[b, perm_idx]

        if use_label_smoothing:
            num = targets.size(1)
            targets = (1 - ls_alpha) * targets + ls_alpha * targets.mean(
                dim=1, keepdim=True
            ).expand_as(targets)

        amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)
        with amp_ctx:
            outputs = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())

            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks

            batch_loss = None
            for i in range(sequences.size(0)):
                valid_indices = masks[i].nonzero(as_tuple=False).flatten()
                if valid_indices.numel() <= 1:
                    continue
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss

            if batch_loss is not None:
                batch_loss = batch_loss / (sequences.size(0) * accumulation_steps)
            else:
                batch_loss = torch.tensor(0.0, device=sequences.device, requires_grad=True) / (sequences.size(0) * accumulation_steps)

        if use_amp:
            scaler.scale(batch_loss).backward()
        else:
            batch_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if config.get('enable_grad_clip', True):
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

            if use_sam:
                # SAM first step: perturb parameters in gradient direction
                optimizer.first_step(zero_grad=True)

                # Second forward-backward with perturbed parameters
                with amp_ctx:
                    outputs2 = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())
                    masked_outputs2 = outputs2 * masks + (1 - masks) * (-1e9)
                    batch_loss2 = None
                    for i in range(sequences.size(0)):
                        valid_indices = masks[i].nonzero(as_tuple=False).flatten()
                        if valid_indices.numel() <= 1:
                            continue
                        valid_pred = masked_outputs2[i][valid_indices]
                        valid_true = masked_targets[i][valid_indices]
                        loss2 = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                        batch_loss2 = batch_loss2 + loss2 if isinstance(batch_loss2, torch.Tensor) else loss2
                    if batch_loss2 is not None:
                        batch_loss2 = batch_loss2 / sequences.size(0)
                    else:
                        batch_loss2 = torch.tensor(0.0, device=sequences.device, requires_grad=True) / sequences.size(0)

                if use_amp:
                    scaler.scale(batch_loss2).backward()
                else:
                    batch_loss2.backward()

                # SAM second step: restore original params + update with SAM gradient
                optimizer.second_step(zero_grad=True)
                if use_amp:
                    scaler.update()
            else:
                # Standard optimizer step (no SAM)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if ema_wrapper is not None:
                ema_wrapper.update()

            total_loss += batch_loss.item() * accumulation_steps
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item() * accumulation_steps,
                                  global_step=epoch * len(dataloader) + local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v,
                                      global_step=epoch * len(dataloader) + local_step)

    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss, total_metrics


def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            stock_indices = _get_batch_stock_indices(batch, device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(sequences, stock_indices=stock_indices, stock_mask=masks.bool())

                masked_outputs = outputs * masks + (1 - masks) * (-1e9)
                masked_targets = targets * masks

                batch_loss = None
                for i in range(sequences.size(0)):
                    valid_indices = masks[i].nonzero(as_tuple=False).flatten()
                    if valid_indices.numel() <= 1:
                        continue
                    valid_pred = masked_outputs[i][valid_indices]
                    valid_true = masked_targets[i][valid_indices]
                    loss = criterion(valid_pred.unsqueeze(0), valid_true.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

                if batch_loss is not None:
                    batch_loss = batch_loss / sequences.size(0)
                    total_loss += batch_loss.item()

            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    for k in total_metrics:
        total_metrics[k] /= num_batches if num_batches > 0 else 1

    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)

    return avg_loss, total_metrics


def split_train_val_by_last_n_months(df, sequence_length, val_months):
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=val_months)).normalize()
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    if train_df.empty:
        raise ValueError("训练集为空，请检查 train.csv 的日期范围或 val_months 配置。")
    if val_df.empty:
        raise ValueError("验证集为空，请检查 train.csv 的日期范围或 val_months 配置。")

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后{val_months}个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def _create_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    base_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _create_warmup_cosine_restart_scheduler(optimizer, warmup_epochs, T_0=10, T_mult=2, min_lr_ratio=0.01):
    base_lr = optimizer.param_groups[0]['lr']

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        epoch_after_warmup = epoch - warmup_epochs
        T_cur = T_0
        while epoch_after_warmup >= T_cur:
            epoch_after_warmup -= T_cur
            T_cur = T_cur * T_mult
        cos_inner = math.cos(math.pi * float(epoch_after_warmup) / float(max(T_cur, 1)))
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + cos_inner)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    set_seed(config.get('seed', 42))

    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log'))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    train_file = config.get('train_file', os.path.join(config.get('data_path', './data'), 'train.csv'))
    val_months = int(config.get('val_months', 2))

    full_df = pd.read_csv(train_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)

    # 支持自定义切分日期（用于 CV）
    train_end_date_override = config.get('_train_end_date')
    val_start_date_override = config.get('_val_start_date')
    sequence_length = config['sequence_length']
    no_val = int(config.get('val_months', 2)) == 0
    if no_val:
        print("val_months=0: 全量数据训练，不拆分验证集")
        train_df = full_df.copy()
        val_df = pd.DataFrame()  # empty
        val_start = pd.to_datetime(full_df['日期'].max())  # dummy, not used
    elif train_end_date_override and val_start_date_override:
        train_df = full_df[full_df['日期'] <= train_end_date_override].copy()
        val_start = pd.to_datetime(val_start_date_override)
        val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)
        val_df = full_df[
            (full_df['日期'] >= val_context_start.strftime('%Y-%m-%d'))
        ].copy()
        print(f"自定义切分: train <= {train_end_date_override}, val >= {val_context_start.date()}")
    else:
        train_df, val_df, val_start = split_train_val_by_last_n_months(
            full_df, sequence_length, val_months
        )
        print(f"验证区间(月): {val_months}")

    print(f"训练数据文件: {train_file}")
    print(f"训练集原始行数: {len(train_df)}")
    print(f"验证集原始行数: {len(val_df) if not no_val else 0}")

    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    train_data, features = preprocess_data(train_df, is_train=True, stockid2idx=stockid2idx)
    if not no_val:
        val_data, _ = preprocess_data(val_df, is_train=False, stockid2idx=stockid2idx)
    else:
        val_data = pd.DataFrame()

    # Feature selection via mutual information (if configured)
    selected_top_k = int(config.get('selected_top_k_features', 0))
    selected_features_list = None
    if selected_top_k > 0 and selected_top_k < len(features):
        from feature_selection import select_features_by_mi, print_feature_ranking, save_feature_ranking
        selected_features_list, ranking = select_features_by_mi(
            train_data, [f for f in features if f != 'instrument'],
            label_col='label', top_k=selected_top_k
        )
        print_feature_ranking(ranking)
        save_feature_ranking(ranking, os.path.join(output_dir, 'feature_ranking.csv'))
        # Keep essential columns: instrument, label, 日期, and selected features
        keep_cols = ['instrument', 'label', '日期'] + selected_features_list
        train_data = train_data[[c for c in keep_cols if c in train_data.columns]]
        val_data = val_data[[c for c in keep_cols if c in val_data.columns]]
        features = ['instrument'] + selected_features_list

    scale_features = get_scale_features(features)
    use_per_stock_norm = config.get('use_per_stock_normalize', True)
    use_cs_features = config.get('use_cross_sectional_features', False)
    cs_feature_types = config.get('cs_feature_types', ['rank_pct', 'zscore'])
    if use_per_stock_norm:
        print("使用逐股票时序Z-Score标准化（训练），跳过全局StandardScaler")
        train_data[scale_features] = train_data[scale_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if not no_val:
            val_data[scale_features] = val_data[scale_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        scaler = StandardScaler()
        train_data[scale_features] = scaler.fit_transform(train_data[scale_features])
        if not no_val:
            val_data[scale_features] = scaler.transform(val_data[scale_features])
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    if train_data.empty:
        raise ValueError("训练特征处理后为空，请检查数据范围、特征工程或 sequence_length。")
    if not no_val and val_data.empty:
        raise ValueError("验证特征处理后为空，请检查数据范围、特征工程或 sequence_length。")

    train_dataset = LazyRankingDataset(
        train_data, features, config['sequence_length'], min_window_end_date=None,
        use_per_stock_norm=use_per_stock_norm,
        use_cs_features=use_cs_features,
        cs_feature_types=cs_feature_types,
        selected_features=selected_features_list,
    )
    train_dataset._max_stocks = config.get('max_stocks_per_sample', 0)
    if not no_val:
        val_dataset = LazyRankingDataset(
            val_data, features, config['sequence_length'],
            min_window_end_date=val_start.strftime('%Y-%m-%d'),
            use_per_stock_norm=use_per_stock_norm,
            use_cs_features=use_cs_features,
            cs_feature_types=cs_feature_types,
            selected_features=selected_features_list,
        )
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset) if not no_val else 0}")
    del train_data
    if not no_val:
        del val_data
    gc.collect()

    if len(train_dataset) == 0:
        raise ValueError("训练排序样本数为 0，请检查数据范围、sequence_length 或样本构造逻辑。")
    if not no_val and len(val_dataset) == 0:
        raise ValueError("验证排序样本数为 0，请检查 val_months、sequence_length 或样本构造逻辑。")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        collate_fn=collate_fn,
        num_workers=int(config.get('num_workers', 0)),
        pin_memory=bool(config.get('pin_memory', False)),
        persistent_workers=int(config.get('num_workers', 0)) > 0,
    )
    if not no_val:
        val_loader = DataLoader(
            val_dataset, batch_size=config['batch_size'], shuffle=False,
            collate_fn=collate_fn,
            num_workers=int(config.get('num_workers', 0)),
            pin_memory=bool(config.get('pin_memory', False)),
            persistent_workers=int(config.get('num_workers', 0)) > 0,
        )
    else:
        val_loader = None

    eff_input_dim = get_eff_input_dim(len(features))
    model = StockTransformer(input_dim=eff_input_dim, config=config, num_stocks=num_stocks)
    model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"原始特征维度: {len(features)}, 有效输入维度: {eff_input_dim}")

    use_smooth_ndcg = config.get("use_smooth_ndcg_loss", True)
    criterion = SmoothNDCGLoss(
        top_k=int(config.get("ndcg_top_k", 5)),
        temperature=float(config.get("soft_sort_temperature", 0.5)),
        delta_clip=float(config.get("lambda_delta_clip", 10.0)),
        smooth_ndcg_weight=float(config.get("smooth_ndcg_weight", 0.7)),
        lambda_pairwise_weight=float(config.get("lambda_pairwise_weight", 0.3)),
        use_mse_aux=bool(config.get("use_mse_aux_loss", False)),
        mse_aux_weight=float(config.get("mse_aux_weight", 0.05)),
        temperature_start=float(config.get("temperature_anneal_start", 2.0)),
        temperature_target=float(config.get("temperature_anneal_target", 0.5)),
        temperature_anneal_epochs=int(config.get("temperature_anneal_epochs", 30)),
    )
    print(f"Using SmoothNDCGLoss (top_k={config['ndcg_top_k']}, "
          f"temp={config['soft_sort_temperature']})")

    use_ema = bool(config.get("use_ema", False))
    ema_decay = float(config.get("ema_decay", 0.999))
    ema_wrapper = EMAWrapper(model, decay=ema_decay) if use_ema else None

    use_swa = bool(config.get("use_swa", False))
    swa_start_epoch = int(config.get("swa_start_epoch", 20))
    swa_lookahead = int(config.get("swa_lookahead", 5))
    swa_lr = float(config.get("swa_lr", 1e-5))
    swa_anneal = int(config.get("swa_anneal_epochs", 5))
    swa_checkpoints = []
    swa_n_ckpts = 0
    swa_active = False

    use_mixup = bool(config.get("use_mixup", False))
    mixup_alpha = float(config.get("mixup_alpha", 0.2))
    mixup_prob = float(config.get("mixup_prob", 0.3))
    use_label_smoothing = bool(config.get("use_label_smoothing", False))
    ls_alpha = float(config.get("label_smoothing_alpha", 0.05))

    print(f"VSN={config.get('use_vsn', False)}, "
          f"MultiScale={config.get('use_multi_scale', False)}, "
          f"SmoothNDCG={use_smooth_ndcg}")
    print(f"EMA={use_ema}, SWA={use_swa}, Mixup={use_mixup}, "
          f"LabelSmooth={use_label_smoothing}")

    use_amp = bool(config.get('use_amp', False)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    if use_amp:
        print("AMP混合精度训练已启用")
    else:
        print(f"AMP未启用 (config={config.get('use_amp', False)}, device={device.type})")

    base_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=float(config.get('weight_decay', 1e-5))
    )
    use_sam = bool(config.get('use_sam', False))
    sam_rho = float(config.get('sam_rho', 0.05))
    if use_sam:
        optimizer = SAM(model.parameters(), base_optimizer, rho=sam_rho)
        print(f"SAM优化器已启用 (rho={sam_rho})")
    else:
        optimizer = base_optimizer
        print("使用标准 AdamW 优化器")

    warmup_epochs = int(config.get('warmup_epochs', 5))
    use_cosine_restarts = bool(config.get('use_cosine_restarts', False))
    cosine_restart_T0 = int(config.get('cosine_restart_T0', 10))
    cosine_restart_T_mult = int(config.get('cosine_restart_T_mult', 2))
    if use_cosine_restarts:
        scheduler = _create_warmup_cosine_restart_scheduler(
            optimizer, warmup_epochs, T_0=cosine_restart_T0, T_mult=cosine_restart_T_mult,
            min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
        )
        print(f"Cosine热重启: T_0={cosine_restart_T0}, T_mult={cosine_restart_T_mult}")
    else:
        scheduler = _create_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=config['num_epochs'],
            min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
        )

    # 支持 num_epochs_override（用于 CV：固定 epoch 数，禁用 early stopping）
    num_epochs_override = config.get('_num_epochs_override')
    if num_epochs_override is not None:
        config['num_epochs'] = num_epochs_override
        if use_cosine_restarts:
            scheduler = _create_warmup_cosine_restart_scheduler(
                optimizer, warmup_epochs, T_0=cosine_restart_T0, T_mult=cosine_restart_T_mult,
                min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
            )
        else:
            scheduler = _create_warmup_cosine_scheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=num_epochs_override,
                min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
            )

    accumulation_steps = int(config.get('gradient_accumulation_steps', 1))
    early_stopping_patience = int(config.get('early_stopping_patience', 15))
    if num_epochs_override is not None:
        early_stopping_patience = num_epochs_override  # 禁用 early stopping

    selection_metric_name = _resolve_model_selection_metric()
    print(f"模型选择指标: {selection_metric_name}")
    print(f"梯度累积步数: {accumulation_steps}")
    print(f"早停耐心值: {early_stopping_patience}")

    best_score = -float('inf')
    best_epoch = -1
    epochs_without_improvement = 0

    gc.collect()
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass
    print("内存清理完成，开始训练...")

    try:
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")

            if use_smooth_ndcg and hasattr(criterion, 'set_temperature'):
                criterion.set_temperature(epoch, config['num_epochs'])
                if epoch == 0 or epoch == config['num_epochs'] - 1:
                    print(f"SmoothNDCG temperature: {criterion.current_temperature:.3f}")

            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                accumulation_steps=accumulation_steps,
                ema_wrapper=ema_wrapper,
                use_mixup=use_mixup, mixup_alpha=mixup_alpha, mixup_prob=mixup_prob,
                use_label_smoothing=use_label_smoothing, ls_alpha=ls_alpha,
                scaler=scaler,
                use_sam=use_sam,
            )

            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.4f}")

            if no_val:
                # 无验证集模式：每 epoch 保存模型
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                if ema_wrapper is not None:
                    torch.save(ema_wrapper.shadow,
                               os.path.join(output_dir, 'best_model_ema.pth'))
                best_epoch = epoch + 1
                best_score = train_loss  # track train loss
                print(f"保存模型 (epoch {best_epoch}) — 无验证集模式")
            else:
                if ema_wrapper is not None and epoch >= 5:
                    ema_wrapper.apply_shadow()

                eval_loss, eval_metrics = evaluate_ranking_model(
                    model, val_loader, criterion, device, writer, epoch,
                    use_amp=use_amp,
                )

                if ema_wrapper is not None and epoch >= 5:
                    ema_wrapper.restore()

                print(f"Eval Loss: {eval_loss:.4f}")
                for k, v in eval_metrics.items():
                    print(f"Eval {k}: {v:.4f}")

                current_selection_score = float(eval_metrics.get(selection_metric_name, 0.0))
                if current_selection_score > best_score:
                    best_score = current_selection_score
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                    if ema_wrapper is not None:
                        torch.save(ema_wrapper.shadow,
                                   os.path.join(output_dir, 'best_model_ema.pth'))
                    print(f"保存最佳模型 - {selection_metric_name}: {best_score:.6f}")
                else:
                    epochs_without_improvement += 1
                    print(f"未改善 ({epochs_without_improvement}/{early_stopping_patience}), "
                          f"当前最佳: {best_score:.6f}")

            writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            scheduler.step()

            should_swa = use_swa and (epoch >= swa_start_epoch or epochs_without_improvement >= early_stopping_patience - swa_lookahead)
            if should_swa:
                if not swa_active:
                    swa_active = True
                    print(f"SWA 启动于 epoch {epoch+1}")
                swa_n_ckpts += 1
                if swa_n_ckpts % 2 == 0:
                    swa_checkpoints.append(
                        {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    )

            if not no_val and epochs_without_improvement >= early_stopping_patience:
                print(f"\n早停触发！连续 {early_stopping_patience} 个 epoch 未改善。")
                break

    finally:
        writer.close()

    if use_swa and len(swa_checkpoints) > 0:
        print(f"\n应用SWA（{len(swa_checkpoints)}个checkpoints）...")
        avg_dict = {}
        for ckpt in swa_checkpoints:
            for k, v in ckpt.items():
                avg_dict[k] = avg_dict.get(k, torch.zeros_like(v.float())) + v.float()
        for k in avg_dict:
            avg_dict[k] /= len(swa_checkpoints)
        model.load_state_dict(avg_dict)
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_swa.pth'))
        print("SWA模型已保存")

    print(f"\n训练完成！最佳 epoch: {best_epoch}, 最佳 {selection_metric_name}: {best_score:.6f}")
    with open(os.path.join(output_dir, 'final_score.txt'), 'w') as f:
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Selection metric: {selection_metric_name}\n")
        f.write(f"Best score: {best_score:.6f}\n")

    return best_score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--train_end_date', type=str, default=None)
    parser.add_argument('--val_start_date', type=str, default=None)
    parser.add_argument('--num_epochs_override', type=int, default=None)
    args = parser.parse_args()
    if args.seed is not None:
        config['seed'] = args.seed
    if args.output_dir is not None:
        config['output_dir'] = args.output_dir
    if args.train_end_date is not None:
        config['_train_end_date'] = args.train_end_date
    if args.val_start_date is not None:
        config['_val_start_date'] = args.val_start_date
    if args.num_epochs_override is not None:
        config['_num_epochs_override'] = args.num_epochs_override

    ensemble_size = int(config.get('ensemble_size', 1))
    base_output_dir = config['output_dir']

    if ensemble_size > 1:
        print(f"\n=== 集成训练模式：{ensemble_size} 个模型 ===")
        base_seed = config.get('seed', 42)
        # Preserve original config keys that main() mutates
        _orig_config_snapshot = {
            'seed': config.get('seed'),
            'output_dir': config['output_dir'],
        }
        for i in range(ensemble_size):
            seed_i = base_seed + i * 7  # different seeds
            config['seed'] = seed_i
            config['output_dir'] = f'{base_output_dir}/model_{i}'
            config['_num_epochs_override'] = config.get('_num_epochs_override', 30)
            config['_train_end_date'] = None
            config['_val_start_date'] = None
            print(f"\n--- 训练模型 {i+1}/{ensemble_size} (seed={seed_i}) ---")
            score = main()
        # Save ensemble metadata
        import json
        meta = {'ensemble_size': ensemble_size, 'base_seed': base_seed,
                'seeds': [base_seed + i * 7 for i in range(ensemble_size)]}
        os.makedirs(base_output_dir, exist_ok=True)
        with open(os.path.join(base_output_dir, 'ensemble_config.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"\n########## 集成训练完成！{ensemble_size} 个模型已保存 ##########")
        best_score = 0.0
    else:
        best_score = main()
        print(f"\n########## 训练完成！最佳 official_score_eq: {best_score:.6f} ##########")