import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import copy


class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.backbone.maxpool = nn.Identity()

        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

        self.pristine_state = None

    def forward(self, x):
        device = x.device
        n = x.size(0)

        # Reset BN buffers
        if self.pristine_state is not None:
            for name, buffer in self.named_buffers():
                if name in self.pristine_state:
                    buffer.data.copy_(self.pristine_state[name].to(device))

        # BN recalibration
        if not self.training and n >= 64:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
                    module.momentum = 0.5

        logits = self.backbone(x)

        # EM label shift correction
        if not self.training and n >= 16:
            num_classes = logits.size(1)
            p_s = torch.ones(num_classes, device=device) / num_classes
            p_t = p_s.clone()

            probs = F.softmax(logits / 1.3, dim=1)  # slightly sharper

            for _ in range(10):
                weighted = probs * (p_t / p_s)
                weighted = weighted / (weighted.sum(dim=1, keepdim=True) + 1e-8)
                new_estimate = weighted.mean(dim=0)

                # EM damping
                p_t = 0.6 * p_t + 0.4 * new_estimate

            log_adjustment = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
            logits = logits + log_adjustment

        return logits

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.pristine_state = copy.deepcopy(self.state_dict())
