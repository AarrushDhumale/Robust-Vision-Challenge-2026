import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms

from model_submission import RobustClassifier


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# PURE GENERALIZED CROSS ENTROPY (Correct Formula)
# ==========================================================
class GCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        p_true = torch.gather(
            probs, 1, targets.view(-1, 1)
        ).squeeze(1)

        p_true = torch.clamp(p_true, 1e-7, 1.0)

        if self.q == 0:
            loss = -torch.log(p_true)
        else:
            loss = (1 - p_true.pow(self.q)) / self.q

        return loss.mean()


def load_pt_data(filepath):
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

    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    print("Loading datasets...")
    train_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt'
    )
    val_dataset = load_pt_data(
        './data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt'
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = RobustClassifier().to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    ce_loss_fn = nn.CrossEntropyLoss()
    gce_loss_fn = GCELoss(q=0.7)

    epochs = 50
    warmup_epochs = 5
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    print("Avaneesh: GCE+simplified model_sub")
    print("Starting Phase 1: Decontamination...")

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        confidence_sum = 0.0

        current_loss_fn = ce_loss_fn if epoch < warmup_epochs else gce_loss_fn

        for images, labels in train_loader:
            augmented_images = torch.stack(
                [train_transform(img) for img in images]
            )

            augmented_images = augmented_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(augmented_images)
            loss = current_loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item()

            # --------- Metrics ----------
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            confidence_sum += probs.max(dim=1)[0].sum().item()

        train_acc = 100 * correct_train / total_train
        avg_conf = confidence_sum / total_train

        # ----------------------------
        # Validation
        # ----------------------------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                preds = torch.argmax(logits, dim=1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = 100 * correct / total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {running_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Train Conf: {avg_conf:.3f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= warmup_epochs:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
