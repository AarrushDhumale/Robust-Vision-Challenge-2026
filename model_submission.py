import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Initialize ResNet-18 WITHOUT pre-trained weights (Kaggle strict rule)
        self.backbone = resnet18(weights=None)

        # 2. Modify input for 1-channel Fashion-MNIST [B, 1, 28, 28]
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # We keep standard nn.BatchNorm2d for Test-Time Statistic Alignment

        # 3. Modify the final FC layer for 10 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        """
        x shape: [B, 1, 28, 28]
        """
        device = x.device

        # ==========================================================
        # PHASE 3: ALIGNMENT (BNStats + OOM Chunking)
        # ==========================================================
        # Force BatchNorm layers to calculate stats on target data
        if not self.training:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()

        batch_size = 256
        all_logits = []

        # Chunking prevents VRAM explosion when evaluated on massive scenario tensors
        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i+batch_size]
            chunk_logits = self.backbone(chunk)
            all_logits.append(chunk_logits)

        base_logits = torch.cat(all_logits, dim=0)

        # ==========================================================
        # PHASE 2: RECONNAISSANCE (Expectation-Maximization)
        # ==========================================================
        # 1. Setup Source Priors (Uniform 10% for the 10 classes)
        num_classes = 10
        p_s = torch.ones(num_classes, device=device) / num_classes

        # 2. Initialize Target Priors estimate (uniform start)
        p_t = p_s.clone()

        # 3. Apply static Temperature Softening (T=1.5) to mimic BCTS calibration
        softened_logits = base_logits / 1.5
        base_probs = F.softmax(softened_logits, dim=1)

        # 4. EM Iterations (Runs rapidly on the GPU)
        em_iterations = 15
        for _ in range(em_iterations):
            # E-Step: Re-weight probabilities based on current target estimate
            adj_probs = base_probs * (p_t / p_s)
            adj_probs = adj_probs / (adj_probs.sum(dim=1, keepdim=True) + 1e-8)

            # M-Step: Update target estimate by averaging batch posteriors
            p_t = adj_probs.mean(dim=0)

        # 5. Apply discovered Target Priors to adjust the final true logits
        log_adjustments = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
        final_logits = base_logits + log_adjustments

        return final_logits

    def load_weights(self, path):
        # Strict interface requirement for evaluation
        self.load_state_dict(torch.load(path, map_location='cpu'))
