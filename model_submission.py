import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # ==========================================================
        # Backbone (Random Init Only)
        # ==========================================================
        self.backbone = resnet18(weights=None)

        # Modify for 1-channel input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

        # For scenario isolation
        self.pristine_state = None

    # ==========================================================
    # Controlled BN Recalibration (Single Pass)
    # ==========================================================
    def _bn_recalibration(self, x):
        """
        Perform a single forward pass with BN in train mode
        to update running statistics using target batch.
        No gradients, no weight updates.
        """
        self.train()
        with torch.no_grad():
            _ = self.backbone(x)
        self.eval()

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x):
        device = x.device
        n = x.size(0)

        # ----------------------------------------------------------
        # 0. Scenario Isolation (Reset BN buffers)
        # ----------------------------------------------------------
        if not self.training and self.pristine_state is not None:
            for name, buffer in self.named_buffers():
                if name in self.pristine_state:
                    buffer.data.copy_(self.pristine_state[name].to(device))

        # ----------------------------------------------------------
        # 1. Controlled Covariate Shift Adaptation (BNStats)
        # ----------------------------------------------------------
        if not self.training and n >= 32:
            self._bn_recalibration(x)

        # ----------------------------------------------------------
        # 2. Logit Extraction (OOM Safe)
        # ----------------------------------------------------------
        if self.training:
            logits = self.backbone(x)
        else:
            batch_size = 256
            chunks = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    chunks.append(self.backbone(x[i:i+batch_size]))
            logits = torch.cat(chunks, dim=0)

        # ----------------------------------------------------------
        # 3. Label Shift Correction (EM)
        # ----------------------------------------------------------
        if not self.training and n >= 16:
            num_classes = logits.size(1)

            # Assume uniform source prior
            p_s = torch.ones(num_classes, device=device) / num_classes
            p_t = p_s.clone()

            # Temperature smoothing for stability
            probs = F.softmax(logits / 1.5, dim=1)

            # EM iterations
            for _ in range(10):
                weighted = probs * (p_t / p_s)
                weighted = weighted / \
                    (weighted.sum(dim=1, keepdim=True) + 1e-8)
                p_t = weighted.mean(dim=0)

            # Logit prior correction
            log_adjust = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
            logits = logits + log_adjust

        return logits

    # ==========================================================
    # Weight Loading
    # ==========================================================
    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))

        # Save pristine BN buffers for reset
        self.pristine_state = {
            n: b.clone() for n, b in self.named_buffers()
        }
