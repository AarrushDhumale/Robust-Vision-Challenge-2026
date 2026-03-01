"""
train.py — Hackenza 2026 Robust Training
==========================================
Phase 1 only: train a robust classifier on source_toxic.pt.
All TTA logic lives in model_submission.py::adapt().

Algorithm:
  - Warmup epochs : standard CE on all samples (model learns signal first)
  - Main epochs   : GCE(q=0.7) + small-loss filter (discards noisiest 30%)
  - Optimizer     : Adam + CosineAnnealingWarmRestarts
  - Augmentation  : crop, flip, affine ONLY (no corruption augmentations)

Usage:
    python train.py \
        --source source_toxic.pt \
        --val    val_sanity.pt   \
        --output weights.pth     \
        [--epochs 100] [--batch_size 128] [--lr 3e-4]

Compliance:
  ✓ weights=None  (random init only)
  ✓ No external data or pretrained weights
  ✓ No corruption augmentations (AugMix / noise / blur all absent)
  ✓ static.pt never touched during training
"""

import os
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from model_submission import RobustClassifier


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Loss — GCE (per-sample, for small-loss filter compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class GCELoss(nn.Module):
    """
    Generalized Cross-Entropy (Zhang et al., NeurIPS 2018).
    Returns per-sample losses so the small-loss filter can mask them.
    q=0.7 is robust against symmetric noise up to ~50%.
    """
    def __init__(self, q: float = 0.7):
        super().__init__()
        self.q = q

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        p_y   = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(min=1e-7)
        return (1.0 - p_y ** self.q) / self.q   # [B], per-sample


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Small-loss filter
# ─────────────────────────────────────────────────────────────────────────────

def small_loss_mask(losses: torch.Tensor,
                    forget_rate: float,
                    epoch: int,
                    warmup: int) -> torch.Tensor:
    """
    Keep the (1 - forget_rate) fraction of samples with lowest loss.
    All samples kept during warmup so the model first learns signal.
    """
    if epoch <= warmup:
        return torch.ones(len(losses), dtype=torch.bool, device=losses.device)
    n_keep = max(1, int(len(losses) * (1.0 - forget_rate)))
    _, idx  = losses.topk(n_keep, largest=False, sorted=False)
    mask    = torch.zeros(len(losses), dtype=torch.bool, device=losses.device)
    mask[idx] = True
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_tensors(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw  = torch.load(path, map_location="cpu")
    imgs = raw["images"] if isinstance(raw, dict) else raw[0]
    lbls = raw["labels"] if isinstance(raw, dict) else raw[1]
    imgs = imgs.float()
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    return imgs, lbls.long()


# Permitted augmentations — geometric / occlusion only, NO corruption
TRAIN_AUG = T.Compose([
    T.RandomCrop(28, padding=4),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
])


class AugDataset(Dataset):
    """
    Applies augmentation per sample inside DataLoader workers —
    fast, parallel, no Python loop bottleneck in the training loop.
    Set transform=None for the val set.
    """
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for imgs, lbls in loader:
        preds = model(imgs.to(device)).argmax(1)
        correct += (preds == lbls.to(device)).sum().item()
        total   += len(lbls)
    return correct / total


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 60)
    print('Hackenza 2026 — Phase 1: Robust Training')
    print('=' * 60)
    print(f'  Device      : {device}')
    print(f'  Epochs      : {args.epochs}')
    print(f'  Batch size  : {args.batch_size}')
    print(f'  LR          : {args.lr}')
    print(f'  GCE q       : {args.gce_q}')
    print(f'  Forget rate : {args.forget_rate}  (matches noise rate)')
    print(f'  Warmup      : {args.warmup} epochs')
    print(f'  Patience    : {args.patience} epochs')

    # ── Data ──────────────────────────────────────────────────────────────────────
    print('\n[DATA] Loading...')
    train_imgs, train_lbls = load_tensors(args.source)
    val_imgs,   val_lbls   = load_tensors(args.val)

    # Augmentation runs inside DataLoader workers - fast and parallel
    train_ds = AugDataset(train_imgs, train_lbls, transform=TRAIN_AUG)
    val_ds   = AugDataset(val_imgs,   val_lbls,   transform=None)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    print(f'  Train: {len(train_ds):,} samples  |  Val: {len(val_ds)} samples')
    # ── Model ──────────────────────────────────────────────────────────────
    model = RobustClassifier(num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[MODEL] Parameters: {n_params:,}')

    # ── Loss & optimizer ───────────────────────────────────────────────────
    gce = GCELoss(q=args.gce_q)
    ce  = nn.CrossEntropyLoss(reduction='none')  # warmup

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    # ── Loop ───────────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    print('\n[TRAIN] Starting…\n')
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = correct = total = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            logits = model(imgs)

            # Warmup: CE on all samples; afterwards: GCE + small-loss filter
            per_sample = ce(logits, lbls) if epoch <= args.warmup \
                         else gce(logits, lbls)

            mask = small_loss_mask(per_sample.detach(), args.forget_rate,
                                   epoch, args.warmup)
            if mask.sum() == 0:
                continue

            loss = per_sample[mask].mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (logits.detach().argmax(1) == lbls).sum().item()
            total      += len(lbls)

        scheduler.step()

        val_acc   = evaluate(model, val_loader, device)
        train_acc = correct / max(total, 1)
        lr_now    = scheduler.get_last_lr()[0]
        tag       = '  ← WARMUP' if epoch <= args.warmup else ''

        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Loss {epoch_loss/len(train_loader):.4f} | '
              f'Train {train_acc*100:5.2f}% | '
              f'Val {val_acc*100:5.2f}% | '
              f'LR {lr_now:.1e}{tag}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, args.output)
            no_improve   = 0
            print(f'         ✓ Best val acc {best_val_acc*100:.2f}% → {args.output}')
        else:
            no_improve += 1
            if no_improve >= args.patience and epoch > args.warmup:
                print(f'\nEarly stop: no improvement for {args.patience} epochs.')
                break

    # ── Restore best weights ───────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), args.output)
    print(f'\n[DONE] Best val acc : {best_val_acc*100:.2f}%')
    print(f'[DONE] Weights      : {args.output}')


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source',      default='./data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt')
    p.add_argument('--val',         default='./data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt')
    p.add_argument('--output',      default='weights.pth')
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=128)
    p.add_argument('--lr',          type=float, default=3e-4)
    p.add_argument('--gce_q',       type=float, default=0.7)
    p.add_argument('--forget_rate', type=float, default=0.30)
    p.add_argument('--warmup',      type=int,   default=10)
    p.add_argument('--patience',    type=int,   default=20)
    p.add_argument('--seed',        type=int,   default=42)
    train(p.parse_args())