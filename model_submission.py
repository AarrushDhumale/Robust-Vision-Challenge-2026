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
        
        self.pristine_state = None

    def _reset_model(self):
        """Restores the Phase 1 weights before adapting to a new scenario."""
        if self.pristine_state is not None:
            self.load_state_dict(self.pristine_state)

    def forward(self, x):
        device = x.device

        if self.training:
            return self.backbone(x)

        # ==========================================================
        # STEP 1: MEMORY WIPE
        # ==========================================================
        self._reset_model()

        # ==========================================================
        # STEP 2: ACTIVE ADAPTATION (TENT / ENTROPY MINIMIZATION)
        # ==========================================================
        params_to_adapt = []
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() 
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
                params_to_adapt.extend([module.weight, module.bias])
            else:
                for param in module.parameters():
                    param.requires_grad_(False)

        # The internal optimizer for Test-Time Adaptation
        optimizer = torch.optim.Adam(params_to_adapt, lr=1e-3)

        batch_size = 128 
        all_adapted_logits = []

        # Adapt to the static chunk-by-chunk
        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i+batch_size]
            
            with torch.enable_grad():
                optimizer.zero_grad()
                logits = self.backbone(chunk)
                
                probs = F.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                # SAR Filter: Only learn from images where the model isn't completely blind
                filter_mask = entropy < 2.0 
                
                if filter_mask.sum() > 0:
                    loss = entropy[filter_mask].mean()
                    loss.backward()
                    optimizer.step()
            
            # Save the cleaned logits for this chunk
            with torch.no_grad():
                clean_logits = self.backbone(chunk)
                all_adapted_logits.append(clean_logits)

        # Reassemble the entire scenario dataset
        base_logits = torch.cat(all_adapted_logits, dim=0)

        # ==========================================================
        # STEP 3: RECONNAISSANCE (EXPECTATION-MAXIMIZATION)
        # ==========================================================
        # Now that the logits are clean from static, EM can safely run
        num_classes = 10
        p_s = torch.ones(num_classes, device=device) / num_classes
        p_t = p_s.clone()

        # Temperature softening keeps EM stable
        softened_logits = base_logits / 1.5
        base_probs = F.softmax(softened_logits, dim=1)

        em_iterations = 15
        for _ in range(em_iterations):
            adj_probs = base_probs * (p_t / p_s)
            adj_probs = adj_probs / (adj_probs.sum(dim=1, keepdim=True) + 1e-8)
            p_t = adj_probs.mean(dim=0)

        # Apply the discovered label shift to the final logits
        log_adjustments = torch.log(p_t + 1e-8) - torch.log(p_s + 1e-8)
        final_logits = base_logits + log_adjustments

        return final_logits

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.pristine_state = copy.deepcopy(self.state_dict())