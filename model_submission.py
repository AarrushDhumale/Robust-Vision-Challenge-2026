import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import copy


class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=None)

        # Modify input for 1-channel Fashion-MNIST [B, 1, 28, 28]
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # Modify the final FC layer for 10 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

        self.pristine_state = None  # To store clean weights
        self.source_quantiles = None  # To store the golden 96% logit distributions

    def forward(self, x):
        device = x.device
        n_target = x.size(0)

        # ==========================================================
        # 0. THE AMNESIA PROTOCOL (Crucial for 24-Scenario Independence)
        # ==========================================================
        if not self.training and self.pristine_state is not None:
            # Wipe the memory clean before every batch to prevent cross-contamination
            for name, buffer in self.named_buffers():
                if name in self.pristine_state:
                    buffer.data.copy_(self.pristine_state[name].to(device))

        # HACKATHON SURVIVAL GUARD:
        # TTA requires a mathematically sound sample size.
        # If Kaggle feeds us tiny batches, adapting will destroy the predictions.
        safe_to_adapt = (not self.training) and (n_target >= 16)

        # ==========================================================
        # PHASE 1: ALIGNMENT (BNStats)
        # ==========================================================
        if safe_to_adapt:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Use the current corrupted batch's mean and variance
                    module.train()

        # ==========================================================
        # PHASE 2: FEATURE EXTRACTION (OOM Chunking)
        # ==========================================================
        # ==========================================================
        # PHASE 2: FEATURE EXTRACTION
        # ==========================================================
        if self.training:
            # 1. Training Mode (train.py) - Keep gradients flowing!
            base_logits = self.backbone(x)
        else:
            # 2. Test Mode (Kaggle submission) - OOM-safe chunking with no gradients
            batch_size = 256
            all_logits = []
            with torch.no_grad():
                for i in range(0, n_target, batch_size):
                    chunk = x[i:i+batch_size]
                    all_logits.append(self.backbone(chunk))
            base_logits = torch.cat(all_logits, dim=0)

        # ==========================================================
        # PHASE 3: GROUPED QUANTILE ALIGNMENT (GQA)
        # Fixes severe, non-linear distribution warping on the logits
        # ==========================================================
        if safe_to_adapt and self.source_quantiles is not None:
            aligned_logits = torch.empty_like(base_logits)

            # Fast mathematical interpolation (Zero VRAM overhead)
            grid = torch.linspace(0, 99, steps=n_target, device=device)
            idx_low = grid.floor().long()
            idx_high = grid.ceil().long()
            weight = grid - idx_low

            for c in range(10):  # Align each class distribution independently
                class_logits = base_logits[:, c]
                # Sort the corrupted target logits
                sorted_logits, indices = torch.sort(class_logits)

                # Fetch golden source quantiles for this class
                source_q_c = self.source_quantiles[:, c].to(device)

                # Interpolate to match the target batch size
                mapped_values = source_q_c[idx_low] * \
                    (1 - weight) + source_q_c[idx_high] * weight

                # Invert the sorting permutation to put corrected logits back in original order
                _, reverse_indices = torch.sort(indices)
                aligned_logits[:, c] = mapped_values[reverse_indices]

            base_logits = aligned_logits

        # ==========================================================
        # PHASE 4: RECONNAISSANCE (Expectation-Maximization)
        # Fixes Label Shift (Class Imbalance) using the perfectly aligned features
        # ==========================================================
        if safe_to_adapt:
            num_classes = 10
            p_s = torch.ones(num_classes, device=device) / num_classes
            p_t = p_s.clone()

            # Temperature scaling softens the peaks so EM doesn't get overconfident
            softened_probs = F.softmax(base_logits / 1.5, dim=1)

            em_iterations = 15
            for _ in range(em_iterations):
                adj_probs = softened_probs * (p_t / p_s)
                adj_probs = adj_probs / \
                    (adj_probs.sum(dim=1, keepdim=True) + 1e-8)
                p_t = adj_probs.mean(dim=0)

            # Apply the final logit adjustment
            log_adjustments = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
            base_logits = base_logits + log_adjustments

        return base_logits

    def load_weights(self, path):
        # 1. Load the Pure Shield weights
        self.load_state_dict(torch.load(path, map_location='cpu'))

        # 2. Capture the pristine baseline buffers for the Amnesia Protocol
        self.pristine_state = {n: b.clone() for n, b in self.named_buffers()}

        # 3. Safely load the golden quantiles from the same directory as the weights
        weight_dir = os.path.dirname(path)
        quantile_path = os.path.join(
            weight_dir, 'source_quantiles.pt') if weight_dir else 'source_quantiles.pt'

        if os.path.exists(quantile_path):
            self.source_quantiles = torch.load(
                quantile_path, map_location='cpu')
            print("Successfully loaded source_quantiles.pt for GQA.")
        else:
            print("WARNING: source_quantiles.pt not found. GQA will be disabled.")
