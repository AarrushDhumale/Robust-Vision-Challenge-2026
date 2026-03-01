import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
from model_submission import RobustClassifier

# --- 1. Generalized Cross Entropy (GCE) Loss ---
class GCELoss(nn.Module):
    def __init__(self, q=0.7, num_classes=10):
        super().__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_1hot = F.one_hot(targets, num_classes=self.num_classes)
        p_y = torch.sum(probs * targets_1hot, dim=1)
        # (1 - p^q) / q
        loss = (1.0 - torch.pow(p_y + 1e-8, self.q)) / self.q
        return loss.mean()

# --- 2. Dataset Loader ---
class ToxicDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path, map_location='cpu')
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Strictly legal geometric augmentations
    transform = T.Compose([
        T.RandomCrop(28, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
    ])
    
    # Load Training Data (Toxic)
    train_dataset = ToxicDataset('source_toxic.pt', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, drop_last=True)
    
    # Load Validation Data (Clean Sanity Check)
    val_data = torch.load('val_sanity.pt', map_location='cpu')
    val_dataset = TensorDataset(val_data['images'], val_data['labels'])
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    model = RobustClassifier().to(device)
    criterion = GCELoss(q=0.7)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    
    epochs = 40
    best_val_acc = 0.0
    
    print("Initiating Phase 1: Robust Training...")
    for epoch in range(epochs):
        # --- Training Loop ---
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        # --- Validation Loop (The Audit) ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for val_imgs, val_labels in val_loader:
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                val_logits = model(val_imgs)
                val_correct += (val_logits.argmax(1) == val_labels).sum().item()
                
        val_acc = val_correct / len(val_dataset)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Sanity Acc: {val_acc:.4f}")
        
        # Checkpoint the best model based strictly on clean data
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights.pth')
            print(f"  --> New best weights saved! (Sanity Acc: {best_val_acc:.4f})")

if __name__ == '__main__':
    train()