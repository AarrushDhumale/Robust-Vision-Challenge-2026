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
        p = F.softmax(logits, dim=1)

        # 1. GCE Term (The Shield) - Clips gradients of extreme outliers
        p_true = torch.gather(p, 1, targets.view(-1, 1)).squeeze(1)
        gce_loss = (1.0 - torch.clamp(p_true, 1e-7, 1.0)**self.q) / self.q

        # 2. RCE Term (The Sword) - FIXED MATH
        # Reverse Cross Entropy: -sum(pred * log(true))
        pred_clamped = torch.clamp(p, 1e-7, 1.0)
        label_oh = torch.clamp(
            F.one_hot(targets, self.num_classes).float(), 1e-4, 1.0)

        # This penalizes the model heavily if it confidently predicts a wrong label
        rce_loss = -1 * torch.sum(pred_clamped * torch.log(label_oh), dim=1)

        # 3. Combine with no standard CE backdoor
        return self.alpha * gce_loss.mean() + self.beta * rce_loss.mean()


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


def generate_source_quantiles(model, dataloader, device, filepath='source_quantiles.pt'):
    """
    Extracts the 100 quantiles of the pristine feature distributions.
    This gives model_submission.py the "golden reference" it needs for GQA.
    """
    print("Generating Grouped Quantile Alignment (GQA) references...")
    model.eval()
    all_logits = []

    with torch.no_grad():
        for images, _ in dataloader:
            all_logits.append(model(images.to(device)))

    base_logits = torch.cat(all_logits, dim=0)

    # Calculate 100 quantiles (0th to 100th percentile) for all 10 classes
    quantiles = torch.empty((100, 10), device=device)
    q_steps = torch.linspace(0.0, 1.0, 100, device=device)

    for c in range(10):
        quantiles[:, c] = torch.quantile(base_logits[:, c], q_steps)

    torch.save(quantiles.cpu(), filepath)
    print(f"Successfully saved {filepath} for Test-Time Adaptation!")


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
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    # 5. Define Losses based on the Trainsmart proposal
    cce_loss_fn = nn.CrossEntropyLoss()
    combo_loss_fn = RobustComboLoss(alpha=1.0, beta=1.0, q=0.7)

    epochs = 50
    warmup_epochs = 2
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
