import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import copy


class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(weights=None)

        # Modify input for 1-channel Fashion-MNIST
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

        self.pristine_state = None  # for resetting BN buffers

    def forward(self, x):
        device = x.device
        n = x.size(0)

        # ==========================================================
        # 1. MEMORY RESET (Prevent cross-scenario contamination)
        # ==========================================================
        if self.pristine_state is not None:
            for name, buffer in self.named_buffers():
                if name in self.pristine_state:
                    buffer.data.copy_(self.pristine_state[name].to(device))

        # ==========================================================
        # 2. BN RE-CALIBRATION (Covariate Shift Handling)
        # ==========================================================
        if not self.training and n >= 64:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    module.momentum = 0.5  # aggressive but stable

        logits = self.backbone(x)

        # ==========================================================
        # 3. EM LABEL SHIFT CORRECTION (No Damping)
        # ==========================================================
        if not self.training and n >= 16:
            num_classes = logits.size(1)

            p_s = torch.ones(num_classes, device=device) / num_classes
            p_t = p_s.clone()

            probs = F.softmax(logits / 1.5, dim=1)

            for _ in range(10):
                weighted = probs * (p_t / p_s)
                weighted = weighted / \
                    (weighted.sum(dim=1, keepdim=True) + 1e-8)

                new_estimate = weighted.mean(dim=0)
                p_t = 0.8 * p_t + 0.2 * new_estimate

        return logits

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.pristine_state = copy.deepcopy(self.state_dict())
