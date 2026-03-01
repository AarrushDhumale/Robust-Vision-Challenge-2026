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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RobustComboLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, q=0.7, num_classes=10):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        # 1. Prediction Probabilities (Clamped to prevent NaN)
        pred_probs = F.softmax(logits, dim=1)
        pred_probs = torch.clamp(pred_probs, min=1e-7, max=1.0 - 1e-7)
        
        # 2. GCE Term: (1 - p^q) / q
        target_probs = torch.gather(pred_probs, 1, targets.view(-1, 1)).squeeze(1)
        gce_loss = (1.0 - torch.pow(target_probs, self.q)) / self.q
        gce_loss = gce_loss.mean()

        # 3. RCE Term: - p * log(y)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        targets_clamped = torch.clamp(one_hot, min=1e-4, max=1.0)
        rce_loss = (-1 * (pred_probs * torch.log(targets_clamped)).sum(dim=1)).mean()

        return self.alpha * gce_loss + self.beta * rce_loss

def load_pt_data(filepath):
    data = torch.load(filepath, map_location='cpu')
    images, labels = data['images'], data['labels']
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

    train_dataset = load_pt_data('./data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt')
    val_dataset = load_pt_data('./data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt')

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = RobustClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

    cce_loss_fn = nn.CrossEntropyLoss()
    combo_loss_fn = RobustComboLoss(alpha=1.0, beta=0.1, q=0.7)

    # --- Training Hyperparameters ---
    epochs = 50
    warmup_epochs = 2
    
    # --- Tracking & Early Stopping Variables ---
    best_val_acc = 0.0
    best_epoch = 0
    patience = 7
    patience_counter = 0

    print("Running on branch - fresh")
    print("Starting Phase 1: Decontamination...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Switch from CCE Warmup to RobustComboLoss at epoch 3
        current_loss_fn = cce_loss_fn if epoch < warmup_epochs else combo_loss_fn

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            augmented_images = torch.stack([train_transform(img) for img in images])

            optimizer.zero_grad()
            logits = model(augmented_images)
            loss = current_loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- Validation on Clean Data ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                correct += (logits.argmax(1) == labels).sum().item()

        val_acc = 100 * correct / len(val_dataset)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # --- Early Stopping & Checkpoint Logic ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Lock in the weights
            torch.save(model.state_dict(), 'weights.pth')
            print(f"  --> Model improved! Weights saved. (Best Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  --> No improvement. Patience: {patience_counter}/{patience}")
            
            # Only trigger a cutoff AFTER the warmup phase is done
            if patience_counter >= patience and epoch >= warmup_epochs:
                print(f"\n[EARLY STOPPING] Triggered at epoch {epoch+1} due to {patience} consecutive epochs without improvement.")
                break

    # --- Final Output Summary ---
    print("\n" + "="*50)
    print("🚀 TRAINING COMPLETE: PHASE 1 SUMMARY")
    print("="*50)
    print(f"Best Epoch     : {best_epoch}")
    print(f"Best Val Acc   : {best_val_acc:.2f}%")
    print("Saved Weights  : 'weights.pth' has been successfully locked to this exact epoch.")
    print("="*50)

if __name__ == '__main__':
    main()