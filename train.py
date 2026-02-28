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
    STRICT REPRODUCIBILITY: Locks all random number generators to ensure 
    the judges get the exact same weights.pth when they run this script.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RobustComboLoss(nn.Module):
    """
    The Ultimate Phase 1 Loss: GCE + RCE
    Combines the gradient throttling of GCE with the symmetric rejection of RCE.
    """

    def __init__(self, alpha=1.0, beta=1.0, q=0.7, num_classes=10):
        super(RobustComboLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # 1. Prediction Probabilities
        pred_probs = F.softmax(logits, dim=1)
        pred_probs_clamped = torch.clamp(pred_probs, min=1e-7, max=1.0)

        # 2. GCE Term (The Shield)
        target_probs = torch.gather(
            pred_probs, 1, targets.view(-1, 1)).squeeze(1)
        gce_loss = ((1.0 - target_probs) ** self.q).mean()

        # 3. RCE Term (The Sword)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        one_hot_clamped = torch.clamp(one_hot, min=1e-4, max=1.0)
        rce_loss = (-1 * (pred_probs_clamped *
                    torch.log(one_hot_clamped)).sum(dim=1)).mean()

        # 4. Combine
        return self.alpha * gce_loss + self.beta * rce_loss


def load_pt_data(filepath):
    """
    STRICT COMPLIANCE: Loads data strictly from local .pt files.
    No external data downloaded (e.g., standard Fashion-MNIST).
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
    # 1. Lock seeds for 50% Reproducibility score
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Strict Augmentation Whitelist (No AugMix, PixMix, or AutoAugment)
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        # No corruption-based augmentations applied here to avoid penalty.
    ])

    # 3. Load Datasets strictly from provided files
    print("Loading datasets...")
    train_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt')
    val_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt')

    # Apply transforms manually in a custom collate_fn or wrapper if needed,
    # but for TensorDatasets, we can apply them batch-wise in the loop.
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # 4. Initialize Model (Weights must be random, handled in RobustClassifier)
    model = RobustClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 5. Define Losses based on the Trainsmart proposal
    cce_loss_fn = nn.CrossEntropyLoss()
    combo_loss_fn = RobustComboLoss(alpha=1.0, beta=1.0, q=0.7)

    epochs = 50
    warmup_epochs = 5
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    print("Starting Phase 1: Decontamination...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # 5-epoch CCE warm-up, then transition to custom GCE loss
        current_loss_fn = cce_loss_fn if epoch < warmup_epochs else combo_loss_fn

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Apply whitelist augmentations batch-wise
            augmented_images = torch.stack(
                [train_transform(img) for img in images])

            optimizer.zero_grad()
            logits = model(augmented_images)
            loss = current_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation on val_sanity.pt
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
            f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Save weights.pth strictly
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= warmup_epochs:
                print("Early stopping triggered.")
                break

    print(
        f"Training complete. Weights saved to weights.pth. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
