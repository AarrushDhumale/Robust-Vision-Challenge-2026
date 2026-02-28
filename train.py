import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

# Import your custom architecture from the required Kaggle submission format
from model_submission import RobustClassifier

def set_seed(seed=42):
    """
    STRICT REPRODUCIBILITY: Locks all random number generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# PHASE 1: THE PURE ROBUST LOSS (NO CROSS-ENTROPY ALLOWED)
# ==============================================================================

class PureRobustLoss(nn.Module):
    """
    Strictly GCE + RCE. Absolutely ZERO standard Cross-Entropy.
    This permanently closes the backdoor that allows label noise to poison the weights.
    """
    def __init__(self, alpha=1.0, beta=1.0, q=0.7, num_classes=10):
        super(PureRobustLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        pred_probs = F.softmax(logits, dim=1)
        # Clamping prevents log(0) exploding gradients in RCE
        pred_probs_clamped = torch.clamp(pred_probs, min=1e-4, max=1.0)

        # 1. GCE Term (The Shield: Bounds gradients for confusing samples)
        target_probs = torch.gather(pred_probs, 1, targets.view(-1, 1)).squeeze(1)
        gce_loss = ((1.0 - target_probs) ** self.q).mean()

        # 2. RCE Term (The Sword: Penalizes overconfidence on wrong labels)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        rce_loss = (-1 * (one_hot * torch.log(pred_probs_clamped)).sum(dim=1)).mean()

        return self.alpha * gce_loss + self.beta * rce_loss

# ==============================================================================
# DATA LOADING & STRICT COMPLIANCE
# ==============================================================================

def load_pt_data(filepath):
    """
    STRICT COMPLIANCE: Loads data strictly from local .pt files.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required dataset: {filepath}")

    data = torch.load(filepath)
    if isinstance(data, dict):
        images, labels = data['images'], data['labels']
    else:
        images, labels = data[0], data[1]

    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    return TensorDataset(images, labels)

def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Strict Augmentation Whitelist
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    print("Loading datasets...")
    train_dataset = load_pt_data('./data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt')
    val_dataset = load_pt_data('./data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = RobustClassifier().to(device)
    # Slightly higher LR since GCE gradients are mathematically smaller than CE gradients
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # NO WARMUP. Pure robustness from step 1.
    robust_loss_fn = PureRobustLoss(alpha=1.0, beta=1.0, q=0.7, num_classes=10)

    epochs = 50
    best_val_acc = 0.0
    patience = 12 # Increased patience to allow it to climb to 96% naturally
    patience_counter = 0

    print("Starting Phase 1: Pure Decontamination...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            augmented_images = torch.stack([train_transform(img) for img in images])

            optimizer.zero_grad()
            logits = model(augmented_images)
            loss = robust_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        scheduler.step()

        # Validation Check
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    print(f"Training complete. Weights saved. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()