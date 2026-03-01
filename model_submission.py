import torch
import torch.nn as nn

class RobustClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 4-Block CNN tailored for 1-channel 28x28 images
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        # x shape: [B, 1, 28, 28]
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits

    def load_weights(self, path):
        # Strict mapping to ensure no device mismatch errors
        self.load_state_dict(torch.load(path, map_location='cpu'))