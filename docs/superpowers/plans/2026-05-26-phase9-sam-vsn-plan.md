# Phase 9: SAM + VSN + Feature Interaction + Cosine Restarts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AdamW with SAM optimizer, enable VSN + feature_interaction + MSE aux loss, add cosine warm restarts — achieving stable self-score > 0.06 without seed dependency

**Architecture:** SAM wraps base AdamW optimizer with two-step perturbation update per batch. Existing VSN/FI modules enabled via config. Cosine warm restart scheduler replaces single cosine decay. Gradient accumulation disabled when SAM is active (redundant — SAM already has 2x compute per batch).

**Tech Stack:** Python 3.12, PyTorch 2.6, torch.optim, torch.amp

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `code/src/sam.py` | CREATE | SAM optimizer implementation |
| `code/src/train.py` | MODIFY | SAM integration, cosine restart scheduler, MSE aux, gradient accumulation bypass |
| `code/src/config.py` | MODIFY | Enable VSN, FI, MSE aux; SAM rho; scheduler params; epochs=30; gradient_accumulation=1 |

---

### Task 1: Create SAM Optimizer

**Files:**
- Create: `code/src/sam.py`

- [ ] **Step 1: Implement SAM optimizer**

```python
"""Sharpness-Aware Minimization (SAM) optimizer.

Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization", ICLR 2021.
https://arxiv.org/abs/2010.01412

Usage:
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer = SAM(model.parameters(), base_optimizer, rho=0.05)

    # Training loop:
    loss1 = criterion(model(x), y)
    loss1.backward()
    optimizer.first_step(zero_grad=True)

    loss2 = criterion(model(x), y)
    loss2.backward()
    optimizer.second_step(zero_grad=True)
"""

import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"rho must be non-negative, got {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # Single-step mode (no SAM, falls back to base optimizer)
        assert closure is not None, "SAM single-step requires closure"
        closure = torch.enable_grad()(closure)
        self.base_optimizer.step(closure)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
```

- [ ] **Step 2: Verify imports work**

```bash
source .venv/bin/activate && python -c "from code.src.sam import SAM; print('SAM imported OK')"
```
Expected output: `SAM imported OK`

- [ ] **Step 3: Commit**

```bash
git add code/src/sam.py
git commit -m "feat: add SAM (Sharpness-Aware Minimization) optimizer"
```

---

### Task 2: SAM Integration + Cosine Restarts in train.py

**Files:**
- Modify: `code/src/train.py`

This task has 4 sub-parts executed together: (A) SAM wrapper creation, (B) SAM training loop, (C) cosine restart scheduler, (D) gradient accumulation bypass.

- [ ] **Step 1: Add SAM import**

At the top of train.py (after line 19 `import math`):
```python
from sam import SAM
```

- [ ] **Step 2: Replace optimizer creation (lines 798-802)**

Old:
```python
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=float(config.get('weight_decay', 1e-5))
    )
```

New:
```python
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
```

- [ ] **Step 3: Disable gradient accumulation when SAM active**

After `accumulation_steps` line (after our num_epochs_override section):
```python
    if use_sam and accumulation_steps > 1:
        print(f"SAM模式下梯度累积从 {accumulation_steps} 调整为 1（SAM已双重计算）")
        accumulation_steps = 1
```

- [ ] **Step 4: Replace training loop gradient update section (lines 465-485)**

Old:
```python
        if use_amp:
            scaler.scale(batch_loss).backward()
        else:
            batch_loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if config.get('enable_grad_clip', True):
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm,
                                      global_step=epoch * len(dataloader) + local_step)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if ema_wrapper is not None:
                ema_wrapper.update()
            optimizer.zero_grad()
```

New:
```python
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
                    scaler.unscale_(optimizer)
                    if config.get('enable_grad_clip', True):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])

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
```

- [ ] **Step 5: Replace scheduler creation (lines 804-810)**

Old:
```python
    warmup_epochs = int(config.get('warmup_epochs', 5))
    scheduler = _create_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=config['num_epochs'],
        min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
    )
```

New:
```python
    warmup_epochs = int(config.get('warmup_epochs', 5))
    use_cosine_restarts = bool(config.get('use_cosine_restarts', False))
    if use_cosine_restarts:
        T_0 = int(config.get('cosine_restart_T0', 10))
        T_mult = int(config.get('cosine_restart_T_mult', 2))
        scheduler = _create_warmup_cosine_restart_scheduler(
            optimizer, warmup_epochs, T_0=T_0, T_mult=T_mult,
            min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
        )
        print(f"Cosine热重启: T_0={T_0}, T_mult={T_mult}")
    else:
        scheduler = _create_warmup_cosine_scheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=config['num_epochs'],
            min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
        )
```

- [ ] **Step 6: Add cosine restart scheduler function**

After the existing `_create_warmup_cosine_scheduler` function (after line 594), add:

```python
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
```

- [ ] **Step 7: Fix num_epochs_override section for scheduler recreation**

In the `num_epochs_override` block (around line 813-822), when `use_cosine_restarts` is True, recreate the restart scheduler:

```python
    num_epochs_override = config.get('_num_epochs_override')
    if num_epochs_override is not None:
        config['num_epochs'] = num_epochs_override
        if use_cosine_restarts:
            scheduler = _create_warmup_cosine_restart_scheduler(
                optimizer, warmup_epochs, T_0=T_0, T_mult=T_mult,
                min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
            )
        else:
            scheduler = _create_warmup_cosine_scheduler(
                optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=num_epochs_override,
                min_lr_ratio=float(config.get('cosine_min_lr_ratio', 0.01))
            )
```

- [ ] **Step 8: Verify import of train.py**

```bash
source .venv/bin/activate && python -c "import sys; sys.path.insert(0,'code/src'); import train; print('train.py imports OK with SAM')"
```
Expected: `train.py imports OK with SAM`

- [ ] **Step 9: Commit**

```bash
git add code/src/train.py
git commit -m "feat: integrate SAM optimizer, cosine warm restarts, gradient accumulation bypass"
```

---

### Task 3: Config Changes

**Files:**
- Modify: `code/src/config.py`

- [ ] **Step 1: Apply all config changes**

The following values in the `config` dict of `config.py` need to be changed:

```python
# --- SAM optimizer ---
    'use_sam': True,
    'sam_rho': 0.05,

# --- Cosine warm restarts ---
    'use_cosine_restarts': True,
    'cosine_restart_T0': 10,
    'cosine_restart_T_mult': 2,

# --- Enable VSN ---
    'use_vsn': True,               # was False
    'vsn_num_groups': 10,
    'vsn_hidden_size': 64,
    'vsn_temperature': 1.0,

# --- Enable feature interaction ---
    'use_feature_interaction': True,  # was False
    'interaction_hidden_dim': 64,

# --- Enable MSE aux loss ---
    'use_mse_aux_loss': True,      # was False
    'mse_aux_weight': 0.05,

# --- Training ---
    'num_epochs': 30,              # was 60 (override for 30)
    'gradient_accumulation_steps': 1,  # was 4 (SAM handles it)

# --- Output ---
    'ensemble_size': 5,
    'output_dir': './model/phase9_sam',

# --- Keep all other values unchanged ---
```

- [ ] **Step 2: Verify config loads**

```bash
source .venv/bin/activate && python -c "
import sys; sys.path.insert(0,'code/src')
from config import config
checks = ['use_sam','use_vsn','use_feature_interaction','use_mse_aux_loss','use_cosine_restarts']
for k in checks:
    print(f'{k}={config[k]}')
print('Config OK')
"
```
Expected output: all True, then `Config OK`

- [ ] **Step 3: Commit**

```bash
git add code/src/config.py
git commit -m "feat: Phase 9 config — SAM, VSN, FI, MSE aux, cosine restarts, epochs=30"
```

---

### Task 4: Verification — Single Model Quick Test

**Files:**
- No code changes

- [ ] **Step 1: Train single model with SAM (2 epochs quick test)**

```bash
source .venv/bin/activate && python code/src/train.py --seed 42 --output_dir ./model/phase9_test_s42 --num_epochs_override 2 2>&1 | grep -E "(SAM|SmoothNDCG|VSN|MultiScale|训练完成|最佳 epoch)"
```
Expected: see "SAM优化器已启用 (rho=0.05)", training completes without error

- [ ] **Step 2: Verify predict works**

```bash
python3 -c "import json; c=json.load(open('model/phase9_test_s42/config.json')); c['predict_rank_alpha']=0.2; json.dump(c, open('model/phase9_test_s42/config.json','w'), indent=4)"
sed -i "s|'output_dir': *'[^']*'|'output_dir': './model/phase9_test_s42'|" code/src/config.py
source .venv/bin/activate && python code/src/predict.py 2>&1 | grep -A6 stock_id
python test/score_self.py 2>&1 | grep "加权收益率"
```
Expected: prediction completes, score is computed

- [ ] **Step 3: If test passes, commit test model**

```bash
git add model/phase9_test_s42/
git commit -m "test: Phase 9 single-model quick verification passes"
```

---

### Task 5: Full 5-Model Ensemble Training + Scoring

**Files:**
- No code changes

- [ ] **Step 1: Train 5-model ensemble (30 epochs)**

```bash
sed -i "s|'output_dir': *'[^']*'|'output_dir': './model/phase9_sam'|" code/src/config.py
source .venv/bin/activate && python code/src/train.py 2>&1 | grep -E "(SAM|VSN|MultiScale|集成训练完成|最佳 epoch|Cosine热重启)"
```
Expected: training completes with 5 models, SAM and VSN enabled, cosine restarts active

- [ ] **Step 2: Predict + score**

```bash
source .venv/bin/activate && python code/src/predict.py 2>&1 | grep -E "(Ensemble:|Ensemble模式)"
python test/score_self.py 2>&1 | grep "加权收益率"
```

- [ ] **Step 3: Record and commit results**

```bash
git add model/phase9_sam/
git commit -m "results: Phase 9 SAM+VSN+FI ensemble — score [INSERT_SCORE]"
```

---

### Task 6: Compare vs Phase 8 Baseline

**Files:**
- No code changes

- [ ] **Step 1: Compare scores**

```bash
echo "Phase 8 (2 epochs, AdamW, no VSN): 0.05595"
echo "Phase 9 (30 epochs, SAM, VSN, FI, MSE, restarts): [INSERT_SCORE]"
```

- [ ] **Step 2: If Phase 9 > 0.056, mark as new best. If not, analyze failure and iterate.**

