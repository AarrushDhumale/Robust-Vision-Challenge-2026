"""
model_submission.py — Hackenza 2026
=====================================
RobustClassifier encapsulates the FULL inference pipeline:
  - Pure forward() for training (no side effects)
  - adapt(images) method for TTA: BNStats reset + TENT + EM
  - forward() during inference applies stored log(w_t) logit bias
  - load_weights() restores weights AND saves pristine BN buffers

Evaluation flow expected by judges:
    model = RobustClassifier()
    model.load_weights('weights.pth')
    model.adapt(target_images)          # <-- call once per domain
    predictions = model(images).argmax(1)
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RobustClassifier(nn.Module):
    """
    ResNet18 adapted for 1-channel 28×28 input (Fashion-MNIST scale).

    Attributes set after adapt():
        self._log_w_t   : [C] log label-shift correction bias
        self._adapted   : bool flag so forward() knows to apply bias
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone (random init only) ───────────────────────────────────
        backbone = resnet18(weights=None)
        backbone.conv1  = nn.Conv2d(1, 64, kernel_size=3,
                                    stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()   # keep 28×28 spatial res
        backbone.fc      = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

        # ── Adaptation state ──────────────────────────────────────────────
        self._adapted = False
        self.register_buffer('_log_w_t',
                             torch.zeros(num_classes))  # log(uniform) = 0

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────
    # Initialisation
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
    # Forward — pure during training, bias-corrected after adapt()
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if self._adapted:
            # Apply label-shift correction: MAP under target prior
            logits = logits + self._log_w_t.to(logits.device).unsqueeze(0)
        return logits

    # ─────────────────────────────────────────────────────────────────────
    # Weight I/O
    # ─────────────────────────────────────────────────────────────────────

    def load_weights(self, path: str):
        """Load trained weights and store a pristine copy of BN buffers."""
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=False)
        # Pristine BN buffers used to reset before each new domain
        self._pristine_bn = {
            k: v.clone()
            for k, v in self.state_dict().items()
            if 'running_mean' in k or 'running_var' in k
            or 'num_batches_tracked' in k
        }
        self._adapted = False

    def _restore_bn(self):
        """Reset BN buffers to the post-training (pristine) state."""
        if not hasattr(self, '_pristine_bn'):
            return
        sd = self.state_dict()
        for k, v in self._pristine_bn.items():
            sd[k].copy_(v)

    # ─────────────────────────────────────────────────────────────────────
    # EM Label-Shift Estimator
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _em_estimate_w_t(self, probs: torch.Tensor,
                          max_iter: int = 50,
                          tol: float = 1e-6) -> torch.Tensor:
        """
        Black-box EM (Lipton et al., ICML 2018).
        Estimates importance weights w_t = p_t(y) / p_s(y).
        Assumes uniform source prior p_s(y) = 1/C.
        Returns w_t [C], mean-normalised.
        """
        C   = self.num_classes
        p_s = torch.ones(C) / C
        p_t = torch.ones(C) / C

        for _ in range(max_iter):
            old   = p_t.clone()
            w     = p_t / p_s.clamp(min=1e-8)
            rw    = probs * w.unsqueeze(0)
            rw    = rw / rw.sum(1, keepdim=True).clamp(min=1e-8)
            p_t   = rw.mean(0)
            p_t   = p_t / p_t.sum()
            if (p_t - old).abs().max() < tol:
                break

        w_t = p_t / p_s.clamp(min=1e-8)
        w_t = w_t / w_t.mean()   # normalize so mean = 1
        return w_t                # [C]

    # ─────────────────────────────────────────────────────────────────────
    # BNStats Reset
    # ─────────────────────────────────────────────────────────────────────

    def _bnstats_reset(self, images: torch.Tensor, batch_size: int = 256):
        """
        Recompute BN running stats from target images using cumulative avg.
        Runs multiple passes for stability when dataset is small.
        """
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = None          # cumulative moving average
                m.reset_running_stats()

        n_passes = 3 if len(images) < 2000 else 1
        with torch.no_grad():
            for _ in range(n_passes):
                perm = torch.randperm(len(images))
                for i in range(0, len(images), batch_size):
                    idx  = perm[i:i + batch_size]
                    imgs = images[idx].to(next(self.parameters()).device)
                    self.backbone(imgs)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
        self.eval()

    # ─────────────────────────────────────────────────────────────────────
    # TENT Entropy Minimization
    # ─────────────────────────────────────────────────────────────────────

    def _tent(self, images: torch.Tensor, w_t: torch.Tensor,
              num_steps: int = 3, lr: float = 5e-4, batch_size: int = 128):
        """
        Minimize prediction entropy by updating only BN affine params (γ, β).
        Uses log(w_t) logit bias so entropy is minimized under target prior.
        """
        device = next(self.parameters()).device
        log_bias = torch.log(w_t.clamp(min=1e-8)).to(device)

        # Collect BN affine params; freeze everything else
        bn_params = []
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.requires_grad_(True)
                m.train()
                bn_params += [m.weight, m.bias]
        for name, p in self.named_parameters():
            if not any(p is bp for bp in bn_params):
                p.requires_grad_(False)

        if not bn_params:
            self.eval()
            return

        opt     = torch.optim.Adam(bn_params, lr=lr)
        max_ent = math.log(self.num_classes)

        for _ in range(num_steps):
            perm = torch.randperm(len(images))
            for i in range(0, len(images), batch_size):
                idx  = perm[i:i + batch_size]
                imgs = images[idx].to(device)

                logits = self.backbone(imgs) + log_bias.unsqueeze(0)
                probs  = F.softmax(logits, dim=1)
                ent    = -(probs * (probs + 1e-8).log()).sum(1)

                # SAR-style filter: skip near-max-entropy (ambiguous) samples
                mask = ent < max_ent * 0.8
                if mask.sum() == 0:
                    continue

                opt.zero_grad()
                ent[mask].mean().backward()
                opt.step()

        # Freeze all params, back to eval
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC: adapt()  — call once per target domain before inference
    # ─────────────────────────────────────────────────────────────────────

    def adapt(self, images: torch.Tensor,
              tent_steps: int   = 3,
              tent_lr:    float = 5e-4,
              batch_size: int   = 256):
        """
        Full TTA pipeline for ONE target domain.
        Call this once with all available target images before predicting.

        Steps:
            1. Restore pristine BN buffers (isolation between domains)
            2. BNStats reset  (align feature distributions)
            3. EM estimation  (compute label-shift weights w_t)
            4. TENT           (entropy minimization guided by w_t)
            5. Store log(w_t) for use in forward()

        Args:
            images     : [N, 1, 28, 28] float tensor, target domain images
            tent_steps : number of TENT gradient steps  (default 3)
            tent_lr    : TENT learning rate              (default 5e-4)
            batch_size : batch size for BNStats + TENT   (default 256)
        """
        device = next(self.parameters()).device
        images = images.float()
        if images.max() > 1.5:
            images = images / 255.0

        # ── Step 1: Restore pristine BN buffers ───────────────────────────
        self._restore_bn()

        # ── Step 2: BNStats reset ─────────────────────────────────────────
        self._bnstats_reset(images, batch_size=batch_size)

        # ── Step 3: EM label-shift estimation ────────────────────────────
        self.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                imgs = images[i:i + batch_size].to(device)
                probs_list.append(F.softmax(self.backbone(imgs), dim=1).cpu())
        probs = torch.cat(probs_list)
        w_t   = self._em_estimate_w_t(probs)

        # ── Step 4: TENT ─────────────────────────────────────────────────
        self._tent(images, w_t, num_steps=tent_steps,
                   lr=tent_lr, batch_size=batch_size)

        # ── Step 5: Store log(w_t) for forward() ─────────────────────────
        self._log_w_t = torch.log(w_t.clamp(min=1e-8)).to(device)
        self._adapted = True


# ── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = RobustClassifier()
    x     = torch.randn(8, 1, 28, 28)
    out   = model(x)
    print(f'Output shape (no adapt): {out.shape}')   # [8, 10]

    # Simulate adapt + predict
    target = torch.rand(200, 1, 28, 28)
    model.adapt(target, tent_steps=1)
    out2 = model(x)
    print(f'Output shape (adapted) : {out2.shape}')  # [8, 10]

    total = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total:,}')