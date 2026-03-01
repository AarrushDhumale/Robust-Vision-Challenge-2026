"""
generate_submission.py — Hackenza 2026
========================================
Thin submission generator. All TTA logic is inside RobustClassifier.forward().

The model self-adapts on the first eval-mode forward() call per domain.
model.reset() clears adaptation state between domains.

Usage:
    python generate_submission.py \
        --weights weights.pth          \
        --static  static.pt            \
        --suite   test_suite_public.pt \
        --output  submission.csv
"""

import os
import time
import argparse

import torch
import pandas as pd
from collections import Counter

from model_submission import RobustClassifier

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[INFO] Device: {DEVICE}')


def extract_images(raw) -> torch.Tensor:
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


def run_domain(model, images, label):
    """
    Reset model state, run one forward pass (triggers BNStats + EM),
    collect predictions.
    """
    t0 = time.time()
    print(f'\n  [{label}] images: {images.shape}')

    # Reset adaptation state — next forward() call will re-adapt to this domain
    model.reset()

    with torch.no_grad():
        # First call: self-adapts (BNStats + EM) then predicts
        preds = model(images.to(DEVICE)).argmax(1).cpu()

    dist = dict(sorted(Counter(preds.numpy().tolist()).items()))
    print(f'  [{label}] distribution : {dist}')
    print(f'  [{label}] unique classes: {len(dist)}/10  |  '
          f'time: {time.time()-t0:.1f}s')

    id_prefix = 'static' if label == 'static' else label
    return [{'ID': f'{id_prefix}_{i}', 'Category': int(p)}
            for i, p in enumerate(preds)]


def generate_submission(args):
    # Load model once — reset() handles per-domain isolation
    model = RobustClassifier(num_classes=10)
    model.load_weights(args.weights)
    model = model.to(DEVICE)
    model.eval()
    print(f'[INFO] Loaded: {args.weights}')

    results = []
    t_total = time.time()

    # ── 1. static.pt (Public Leaderboard) ────────────────────────────────
    print('\n' + '='*60)
    print('[1/2] static.pt  ->  Public Leaderboard')
    print('='*60)
    static_images = extract_images(
        torch.load(args.static, map_location='cpu'))
    results += run_domain(model, static_images, 'static')

    # ── 2. 24-scenario suite (Private Leaderboard) ────────────────────────
    if args.suite and os.path.exists(args.suite):
        print('\n' + '='*60)
        print('[2/2] test_suite  ->  Private Leaderboard')
        print('='*60)
        suite = torch.load(args.suite, map_location='cpu')
        keys  = sorted(k for k in suite if k.lower().startswith('scenario'))
        print(f'  Found {len(keys)} scenarios')

        for skey in keys:
            images  = extract_images(suite[skey])
            results += run_domain(model, images, skey)
    else:
        print(f'\n[WARN] Suite not found at "{args.suite}" -- skipping.')

    # ── Write CSV ─────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f'\n[DONE] {args.output}  |  rows: {len(df)}  |  '
          f'total time: {time.time()-t_total:.1f}s')
    print(df.head())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='weights.pth')
    p.add_argument('--static',  default='static.pt')
    p.add_argument('--suite',   default='test_suite_public.pt')
    p.add_argument('--output',  default='submission.csv')
    generate_submission(p.parse_args())