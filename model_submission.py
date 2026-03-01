"""
model_submission.py — Hackenza 2026
=====================================
RobustClassifier with self-contained TTA in forward().

Evaluation flow (Kaggle template — no changes needed):
    model = RobustClassifier()
    model.load_weights('weights.pth')
    model.eval()
    with torch.no_grad():
        preds = model(target_images).argmax(1)  # TTA runs automatically here

On the first eval-mode forward() call:
    1. BNStats reset  — recomputes BN running stats from the incoming batch
    2. EM estimation  — estimates target class prior, stores log(w_t) bias
    3. All subsequent forward() calls just run the network + apply the bias

Per-scenario isolation (generate_submission.py):
    Call model.reset() before each new domain to clear adaptation state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RobustClassifier(nn.Module):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone (random init, no pretrained weights) ─────────────────
        backbone = resnet18(weights=None)
        backbone.conv1   = nn.Conv2d(1, 64, kernel_size=3,
                                     stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()   # preserve 28x28 spatial res
        backbone.fc      = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

        # ── Adaptation state ──────────────────────────────────────────────
        self._adapted = False
        self.register_buffer('_log_w_t', torch.zeros(num_classes))

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    # Weight init
    # ─────────────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────
    # Weight I/O
    # ─────────────────────────────────────────────────────────────────────

    def load_weights(self, path: str):
        """Load weights and snapshot pristine BN buffers for later reset."""
        self.load_state_dict(torch.load(path, map_location='cpu'),
                             strict=False)
        self._save_pristine_bn()
        self._adapted = False

    def _save_pristine_bn(self):
        """Snapshot BN running stats right after loading trained weights."""
        self._pristine_bn = {
            k: v.clone()
            for k, v in self.state_dict().items()
            if any(s in k for s in
                   ('running_mean', 'running_var', 'num_batches_tracked'))
        }

    def _restore_bn(self):
        """Restore BN buffers to post-training state (undo any TTA changes)."""
        if not hasattr(self, '_pristine_bn'):
            return
        sd = self.state_dict()
        for k, v in self._pristine_bn.items():
            sd[k].copy_(v)

    # ─────────────────────────────────────────────────────────────────────
    # Public reset — call between domains in generate_submission.py
    # ─────────────────────────────────────────────────────────────────────

    def reset(self):
        """
        Clear adaptation state so the next forward() call re-adapts.
        Call this between scenarios instead of deepcopying the model.
        """
        self._restore_bn()
        self._adapted = False
        self._log_w_t.zero_()

    # ─────────────────────────────────────────────────────────────────────
    # BNStats reset
    # ─────────────────────────────────────────────────────────────────────

    def _bnstats_reset(self, images: torch.Tensor, batch_size: int = 512):
        """
        Recompute BN running mean/var from target images.
        Uses cumulative average (momentum=None) for exact statistics.
        One forward pass, no gradients.
        """
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = None
                m.reset_running_stats()

        with torch.no_grad():
            device = next(self.parameters()).device
            for i in range(0, len(images), batch_size):
                self.backbone(images[i:i + batch_size].to(device))

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
        self.eval()

    # ─────────────────────────────────────────────────────────────────────
    # EM label-shift estimator
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _em_estimate_w_t(self, probs: torch.Tensor,
                          max_iter: int = 50,
                          tol: float = 1e-6) -> torch.Tensor:
        """
        Black-box EM label shift estimation (Lipton et al., ICML 2018).
        Assumes uniform source prior p_s(y) = 1/C (balanced training set).
        Returns importance weights w_t = p_t(y) / p_s(y), mean-normalised.
        """
        C   = self.num_classes
        p_s = torch.ones(C) / C
        p_t = torch.ones(C) / C

        for _ in range(max_iter):
            old = p_t.clone()
            w   = p_t / p_s.clamp(min=1e-8)
            rw  = probs * w.unsqueeze(0)
            rw  = rw / rw.sum(1, keepdim=True).clamp(min=1e-8)
            p_t = rw.mean(0)
            p_t = p_t / p_t.sum()
            if (p_t - old).abs().max() < tol:
                break

        w_t = p_t / p_s.clamp(min=1e-8)
        w_t = w_t / w_t.mean()
        return w_t   # [C]

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training mode : pure forward, no side effects.

        Eval mode     : on the FIRST call, runs BNStats + EM on x,
                        stores log(w_t), sets _adapted=True.
                        All subsequent calls apply stored log(w_t) bias.

        This means the Kaggle template works as-is with zero changes:
            model.eval()
            with torch.no_grad():
                preds = model(target_images).argmax(1)
        """
        if self.training:
            return self.backbone(x)

        # ── First eval call: adapt to this domain ─────────────────────────
        if not self._adapted:
            # Normalise if needed
            x_norm = x.float()
            if x_norm.max() > 1.5:
                x_norm = x_norm / 255.0

            # Step 1: BNStats reset
            self._bnstats_reset(x_norm)

            # Step 2: EM label-shift estimation (batched to avoid OOM)
            device = next(self.parameters()).device
            probs_list = []
            with torch.no_grad():
                for i in range(0, len(x_norm), 512):
                    chunk = x_norm[i:i + 512].to(device)
                    probs_list.append(
                        F.softmax(self.backbone(chunk), dim=1).cpu())
            probs = torch.cat(probs_list)
            w_t = self._em_estimate_w_t(probs)
            self._log_w_t = torch.log(w_t.clamp(min=1e-8)).to(device)
            self._adapted = True

        # ── Every eval call: predict with label-shift correction ──────────
        logits = self.backbone(x)
        return logits + self._log_w_t.unsqueeze(0)


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = RobustClassifier()

    # Training mode — pure, no TTA
    model.train()
    out = model(torch.randn(8, 1, 28, 28))
    print(f'Train output : {out.shape}')        # [8, 10]

    # Eval mode — first call triggers BNStats + EM automatically
    model.eval()
    target = torch.rand(500, 1, 28, 28)
    with torch.no_grad():
        out2 = model(target)
    print(f'Eval output  : {out2.shape}')       # [500, 10]
    print(f'_adapted     : {model._adapted}')   # True
    print(f'log_w_t      : {model._log_w_t.numpy().round(3)}')

    # reset() clears state for next domain
    model.reset()
    print(f'After reset  : {model._adapted}')   # False

    print(f'Parameters   : {sum(p.numel() for p in model.parameters()):,}')