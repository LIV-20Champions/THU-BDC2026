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
