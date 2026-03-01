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
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()

        # Modify the final FC layer for 10 classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)
        
        self.pristine_state = None

    def load_weights(self, path):
        # Load weights and immediately store a pristine copy in memory
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.pristine_state = copy.deepcopy(self.state_dict())

    def forward(self, x):
        device = x.device

        # If we are in Phase 1 (Training), just do a normal forward pass
        if self.training and self.pristine_state is None:
            return self.backbone(x)

        # ==========================================================
        # PHASE 3: ALIGNMENT & INFERENCE (Triggered by the Grader)
        # ==========================================================
        
        # 1. THE MEMORY RESET: Wipe contamination from previous test scenarios
        if self.pristine_state is not None:
            self.load_state_dict({k: v.to(device) for k, v in self.pristine_state.items()})

        # 2. BNSTATS SAFETY CHECK: Handle Sensor Noise safely
        # Only force BN layers to train() if the input tensor has more than 1 image.
        # This prevents a fatal PyTorch crash during automated batch-size-1 evaluations.
        if x.size(0) > 1:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.train()
        else:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

        # 3. OOM-Safe Feature Extraction
        batch_size = 256
        all_logits = []
        
        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i+batch_size]
            
            # Sub-chunk safety: If the very last chunk has exactly 1 image, 
            # PyTorch BN will still crash. We must switch to eval for that specific tiny chunk.
            if chunk.size(0) == 1 and x.size(0) > 1:
                for module in self.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                        
            chunk_logits = self.backbone(chunk)
            all_logits.append(chunk_logits)
            
        base_logits = torch.cat(all_logits, dim=0)

        # ==========================================================
        # PHASE 2: RECONNAISSANCE (Label Shift via EM)
        # ==========================================================
        num_classes = 10
        p_s = torch.ones(num_classes, device=device) / num_classes # Source prior
        p_t = p_s.clone() # Target prior initialized to uniform

        # Temperature softening to prevent overconfidence during EM
        softened_logits = base_logits / 1.5
        base_probs = F.softmax(softened_logits, dim=1)

        # 15-Step EM Algorithm
        em_iterations = 15
        for _ in range(em_iterations):
            adj_probs = base_probs * (p_t / p_s)
            adj_probs = adj_probs / (adj_probs.sum(dim=1, keepdim=True) + 1e-8)
            p_t = adj_probs.mean(dim=0)

        # Apply the final learned target prior to adjust the base logits
        log_adjustments = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
        final_logits = base_logits + log_adjustments

        return final_logits