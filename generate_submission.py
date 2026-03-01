"""
generate_submission.py — Hackenza 2026
========================================
Thin submission generator. All adaptation logic lives in
model_submission.py::RobustClassifier.adapt().

This file only handles:
  - Loading data
  - Calling model.adapt(images) once per domain
  - Collecting predictions into submission.csv

Usage:
    python generate_submission.py \
        --weights weights.pth        \
        --static  static.pt          \
        --suite   test_suite_public.pt \
        --output  submission.csv
"""

import os
import time
import argparse
import copy

import torch
import pandas as pd
from collections import Counter

from model_submission import RobustClassifier


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {DEVICE}')


def extract_images(raw) -> torch.Tensor:
    """Handle dict-with-'images', nested dict, or raw tensor."""
    if isinstance(raw, torch.Tensor):
        imgs = raw
    elif isinstance(raw, dict):
        imgs = raw.get('images', next(iter(raw.values())))
        if isinstance(imgs, dict):
            imgs = imgs.get('images', next(iter(imgs.values())))
    else:
        raise ValueError(f'Unknown data format: {type(raw)}')
    imgs = imgs.float()
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    return imgs


def load_base_model(weights_path: str) -> RobustClassifier:
    model = RobustClassifier(num_classes=10)
    model.load_weights(weights_path)   # also saves pristine BN buffers
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def predict(model: RobustClassifier,
            images: torch.Tensor,
            batch_size: int = 256) -> torch.Tensor:
    """Argmax predictions; forward() applies stored log(w_t) bias."""
    model.eval()
    preds = []
    for i in range(0, len(images), batch_size):
        imgs = images[i:i + batch_size].to(DEVICE)
        preds.append(model(imgs).argmax(1).cpu())
    return torch.cat(preds)


def run_domain(base_model:  RobustClassifier,
               images:      torch.Tensor,
               label:       str,
               tent_steps:  int,
               tent_lr:     float,
               batch_size:  int) -> list:
    """
    For one domain:
      1. Deep-copy base model (full isolation between domains)
      2. model.adapt(images)  — BNStats + EM + TENT inside the class
      3. Predict
    Returns list of {'ID': ..., 'Category': ...} dicts.
    """
    t0 = time.time()
    print(f'\n  [{label}] images: {images.shape}')

    # Full isolation: each domain gets a fresh copy of the base model
    model = copy.deepcopy(base_model)
    model = model.to(DEVICE)

    # ── ALL adaptation happens inside model.adapt() ───────────────────────
    model.adapt(images, tent_steps=tent_steps,
                tent_lr=tent_lr, batch_size=batch_size)

    preds = predict(model, images, batch_size=batch_size)

    dist  = dict(sorted(Counter(preds.numpy().tolist()).items()))
    print(f'  [{label}] predicted dist : {dist}')
    print(f'  [{label}] unique classes : {len(dist)}/10  |  '
          f'time: {time.time()-t0:.1f}s')

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    id_prefix = 'static' if label == 'static' else label
    return [{'ID': f'{id_prefix}_{i}', 'Category': int(p)}
            for i, p in enumerate(preds)]


def generate_submission(args):
    base_model = load_base_model(args.weights)
    print(f'[INFO] Loaded: {args.weights}')

    results = []
    t_total = time.time()

    # ── 1. static.pt (Public Leaderboard) ────────────────────────────────
    print('\n' + '='*60)
    print('[1/2] static.pt  →  Public Leaderboard')
    print('='*60)
    static_images = extract_images(torch.load(args.static, map_location='cpu'))
    results += run_domain(base_model, static_images, 'static',
                          args.tent_steps, args.tent_lr, args.batch_size)

    # ── 2. 24-scenario suite (Private Leaderboard) ────────────────────────
    if args.suite and os.path.exists(args.suite):
        print('\n' + '='*60)
        print('[2/2] test_suite  →  Private Leaderboard')
        print('='*60)
        suite = torch.load(args.suite, map_location='cpu')
        keys  = sorted(k for k in suite if k.lower().startswith('scenario'))
        print(f'  Found {len(keys)} scenarios')

        for idx, skey in enumerate(keys):
            images  = extract_images(suite[skey])
            results += run_domain(base_model, images, skey,
                                  args.tent_steps, args.tent_lr,
                                  args.batch_size)
    else:
        print(f'\n[WARN] Suite not found at "{args.suite}" — skipping.')

    # ── Write CSV ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'\n[DONE] {args.output}  |  rows: {len(df)}  |  '
          f'total time: {time.time()-t_total:.1f}s')
    print(df.head())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights',    default='weights.pth')
    p.add_argument('--static',     default='./data/hackenza-2026-test-time-adaptation-in-the-wild/static.pt')
    p.add_argument('--suite',      default='./data/hackenza-2026-test-time-adaptation-in-the-wild/test_suite_public.pt')
    p.add_argument('--output',     default='submission.csv')
    p.add_argument('--tent_steps', type=int,   default=0,
                   help='0=BNStats+EM only (fast). 1-3=add TENT (slower but may help)')
    p.add_argument('--tent_lr',    type=float, default=5e-4)
    p.add_argument('--batch_size', type=int,   default=256)
    p.parse_args().__dict__
    generate_submission(p.parse_args())