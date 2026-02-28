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
        
        self.pristine_state = None # To store clean weights

    def forward(self, x):
        device = x.device

        # ==========================================================
        # 0. THE MEMORY RESET (Crucial for 24-Scenario Independence)
        # ==========================================================
        if self.pristine_state is not None:
            # We only restore the BN buffers (running mean/var) to prevent 
            # cross-scenario contamination, keeping it fast.
            for name, buffer in self.named_buffers():
                if name in self.pristine_state:
                    buffer.data.copy_(self.pristine_state[name].to(device))

        # ==========================================================
        # PHASE 3: ALIGNMENT (BNStats + OOM Chunking)
        # ==========================================================
        if not self.training:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Force momentum to 1.0 so it ONLY uses the target batch stats,
                    # completely ignoring the source training stats for adaptation.
                    # module.momentum = 1.0 
                    module.train()

        batch_size = 256
        all_logits = []

        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i+batch_size]
            chunk_logits = self.backbone(chunk)
            all_logits.append(chunk_logits)

        base_logits = torch.cat(all_logits, dim=0)

        # ==========================================================
        # PHASE 2: RECONNAISSANCE (Expectation-Maximization)
        # ==========================================================
        num_classes = 10
        p_s = torch.ones(num_classes, device=device) / num_classes
        p_t = p_s.clone()

        softened_logits = base_logits / 1.5
        base_probs = F.softmax(softened_logits, dim=1)

        em_iterations = 15
        for _ in range(em_iterations):
            adj_probs = base_probs * (p_t / p_s)
            adj_probs = adj_probs / (adj_probs.sum(dim=1, keepdim=True) + 1e-8)
            p_t = adj_probs.mean(dim=0)

        log_adjustments = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
        final_logits = base_logits + log_adjustments

        return final_logits

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        # Save a pristine copy in memory immediately after loading
        self.pristine_state = copy.deepcopy(self.state_dict())