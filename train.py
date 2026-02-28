import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import copy

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


class GeneralizedCrossEntropy(nn.Module):
    """
    Generalized Cross Entropy (GCE) Loss.
    Robust to noisy labels.
    Formula: L_q(f(x), e_j) = (1 - f_j(x)^q) / q
    """
    def __init__(self, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        # Gather probabilities of the correct class
        p_true = torch.gather(p, 1, targets.view(-1, 1)).squeeze(1)
        # Clamp to avoid numerical instability
        p_true = torch.clamp(p_true, min=1e-7, max=1.0)
        
        # GCE Formula
        loss = (1.0 - (p_true ** self.q)) / self.q
        return loss.mean()


class SymmetricCrossEntropy(nn.Module):
    """
    Symmetric Cross Entropy (SCE) Loss.
    Combines Cross Entropy (CE) and Reverse Cross Entropy (RCE).
    Robust to noisy labels.
    """
    def __init__(self, alpha=0.1, beta=1.0, num_classes=10):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # CE
        ce_loss = self.cross_entropy(logits, targets)

        # RCE
        pred = F.softmax(logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(targets, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce_loss = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        return self.alpha * ce_loss + self.beta * rce_loss.mean()


class RobustHybridLoss(nn.Module):
    """
    Combines Generalized Cross Entropy (GCE) and Symmetric Cross Entropy (SCE).
    Provides maximum robustness against label noise and corruptions.
    """
    def __init__(self, q=0.7, alpha=0.1, beta=1.0, num_classes=10):
        super(RobustHybridLoss, self).__init__()
        self.gce = GeneralizedCrossEntropy(q=q)
        self.sce = SymmetricCrossEntropy(alpha=alpha, beta=beta, num_classes=num_classes)

    def forward(self, logits, targets):
        return 0.5 * self.gce(logits, targets) + 0.5 * self.sce(logits, targets)


def entropy_loss(logits):
    """
    Entropy Minimization for Unlabeled Target Data.
    Forces the model to be confident on target data.
    """
    p = F.softmax(logits, dim=1)
    log_p = F.log_softmax(logits, dim=1)
    loss = -torch.sum(p * log_p, dim=1)
    return loss.mean()


def load_pt_data(filepath, labeled=True):
    """
    STRICT COMPLIANCE: Loads data strictly from local .pt files.
    Handles both labeled (source) and unlabeled (static) data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing required dataset: {filepath}")

    data = torch.load(filepath)
    
    # Handle dictionary format
    if isinstance(data, dict):
        images = data['images']
        labels = data['labels'] if labeled else None
    else:
        # Fallback for older formats if necessary
        images = data[0]
        labels = data[1] if labeled else None

    if images.dtype == torch.uint8:
        images = images.float() / 255.0

    if labeled:
        return TensorDataset(images, labels)
    else:
        # Create dummy labels for unlabeled data to satisfy TensorDataset structure
        dummy_labels = torch.zeros(len(images), dtype=torch.long)
        return TensorDataset(images, dummy_labels)


def main():
    # 1. Lock seeds
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Strict Augmentation Whitelist
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    # 3. Load Datasets
    print("Loading datasets...")
    # Source: Toxic (Noisy Labels)
    train_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt', 
        labeled=True
    )
    # Target: Static (Unlabeled, Domain Shift)
    target_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/static.pt', 
        labeled=False
    )
    # Validation
    val_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt', 
        labeled=True
    )

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize Model
    model = RobustClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 5. Define Losses
    # Hybrid Robust Loss (SCE + GCE) for noisy source labels
    robust_loss_fn = RobustHybridLoss(num_classes=10)
    # Standard CE for warmup
    cce_loss_fn = nn.CrossEntropyLoss()

    epochs = 50
    warmup_epochs = 5
    best_val_acc = 0.0
    
    # Weight for Entropy Minimization (Unsupervised Adaptation)
    ent_weight = 0.1 

    print("Starting Training with Domain Adaptation...")
    
    target_iter = iter(target_loader)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_ent = 0.0

        for i, (source_images, source_labels) in enumerate(train_loader):
            source_images, source_labels = source_images.to(device), source_labels.to(device)

            # Get target batch (unlabeled)
            try:
                target_images, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_images, _ = next(target_iter)
            target_images = target_images.to(device)

            # Apply augmentations to BOTH source and target
            # Augmenting target helps entropy minimization be more robust
            aug_source = torch.stack([train_transform(img) for img in source_images])
            aug_target = torch.stack([train_transform(img) for img in target_images])

            optimizer.zero_grad()

            # 1. Supervised Loss on Source
            source_logits = model(aug_source)
            
            if epoch < warmup_epochs:
                # Warmup with standard CE to learn easy patterns first
                cls_loss = cce_loss_fn(source_logits, source_labels)
            else:
                # Switch to Robust Hybrid Loss to reject noisy labels
                cls_loss = robust_loss_fn(source_logits, source_labels)

            # 2. Unsupervised Entropy Loss on Target (Domain Adaptation)
            target_logits = model(aug_target)
            ent_loss = entropy_loss(target_logits)

            # Combine losses
            total_loss = cls_loss + (ent_weight * ent_loss)

            total_loss.backward()
            optimizer.step()
            
            running_loss += cls_loss.item()
            running_ent += ent_loss.item()

        scheduler.step()

        # Validation
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

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Src Loss: {running_loss/len(train_loader):.4f} | "
            f"Tgt Ent: {running_ent/len(train_loader):.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights.pth')

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
