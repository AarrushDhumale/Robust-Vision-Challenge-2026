import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # ----------------------------------------------------------
        # Backbone (random init only)
        # ----------------------------------------------------------
        self.backbone = resnet18(weights=None)

        # Modify for 1-channel 28x28 input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

        # Store pristine BN buffers for scenario reset
        self.pristine_buffers = None

    # ==========================================================
    # Controlled BN Recalibration (Single Pass, momentum=0.5)
    # ==========================================================
    def _bn_recalibration(self, x):
        """
        Perform one forward pass in train() mode to update
        BatchNorm running stats using target data.
        Temporarily set momentum=0.5 for balanced adaptation.
        """

        # Save original momentums
        original_momentums = {}
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                original_momentums[module] = module.momentum
                module.momentum = 0.2  # Balanced adaptation

        # Single recalibration pass
        self.train()
        with torch.no_grad():
            _ = self.backbone(x)
        self.eval()

        # Restore original momentum values
        for module, momentum in original_momentums.items():
            module.momentum = momentum

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, x):
        device = x.device
        n = x.size(0)

        # ----------------------------------------------------------
        # 0. Scenario Isolation (Reset BN buffers only)
        # ----------------------------------------------------------
        if not self.training and self.pristine_buffers is not None:
            for name, buffer in self.named_buffers():
                if name in self.pristine_buffers:
                    buffer.data.copy_(
                        self.pristine_buffers[name].to(device)
                    )

        # ----------------------------------------------------------
        # 1. Covariate Shift Adaptation (BNStats - controlled)
        # ----------------------------------------------------------
        if not self.training and n >= 64:
            self._bn_recalibration(x)

        # ----------------------------------------------------------
        # 2. Stable Logit Extraction (OOM-safe)
        # ----------------------------------------------------------
        if self.training:
            logits = self.backbone(x)
        else:
            batch_size = 256
            outputs = []
            with torch.no_grad():
                for i in range(0, n, batch_size):
                    chunk = x[i:i+batch_size]
                    outputs.append(self.backbone(chunk))
            logits = torch.cat(outputs, dim=0)

        # ----------------------------------------------------------
        # 3. Label Shift Correction (EM)
        # ----------------------------------------------------------
        if not self.training and n >= 16:
            num_classes = logits.size(1)

            # Assume uniform source prior
            p_s = torch.ones(num_classes, device=device) / num_classes
            p_t = p_s.clone()

            # Temperature smoothing improves EM stability
            probs = F.softmax(logits / 1.5, dim=1)

            # EM iterations
            for _ in range(10):
                weighted = probs * (p_t / p_s)
                weighted = weighted / (
                    weighted.sum(dim=1, keepdim=True) + 1e-8
                )
                p_t = weighted.mean(dim=0)

            # Logit prior correction
            log_adjust = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
            logits = logits + log_adjust

        return logits

    # ==========================================================
    # Load Weights
    # ==========================================================
    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

        # Save clean BN buffers for scenario resets
        self.pristine_buffers = {
            name: buffer.clone()
            for name, buffer in self.named_buffers()
        }
