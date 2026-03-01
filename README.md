# Robust-Vision-Challenge-2026
# Hackenza 2026: Test-Time Adaptation in the Wild

A robust vision classifier that survives noisy training data and adapts itself — without labels — to hostile target domains at inference time.

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Solution Architecture](#solution-architecture)
- [Core Logic](#core-logic)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Generating a Submission](#generating-a-submission)
- [Design Decisions & Compliance](#design-decisions--compliance)
- [Unit Testing & Sanity Checks](#unit-testing--sanity-checks)

---

## Problem Overview

The challenge simulates a robot deployed in a hostile environment where:

1. **Toxicity** — Training data (`source_toxic.pt`) contains **30% label noise**, meaning roughly 1 in 3 labels is intentionally wrong.
2. **Blind Adaptation** — The target domain (`static.pt`) has **no labels**, suffers from significant sensor noise, and exhibits heavy **class imbalance**.
3. **Stress Test** — The final model is evaluated across **24 hidden corruption scenarios** at multiple severity levels, requiring genuine generalisation rather than overfitting to a single distribution.

The evaluation metrics are **Robust Accuracy**, **Macro-F1** (critical due to class imbalance), and a weighted **Robustness Score** that penalises failures at high severity.

---

## Solution Architecture

The solution is split cleanly across two files:

| File | Responsibility |
|---|---|
| `train.py` | Phase 1: robust training on noisy source data |
| `model_submission.py` | Architecture + self-contained Test-Time Adaptation (TTA) |

### Model

The backbone is a **ResNet-18** modified for 28×28 greyscale images:

- `conv1` replaced with a 3×3 convolution accepting 1 input channel (greyscale).
- `maxpool` replaced with `nn.Identity()` to preserve spatial resolution at 28×28.
- Final `fc` layer outputs 10 class logits.
- All weights are **randomly initialised via Kaiming Normal** — no pretrained weights, as required by competition rules.

---

## Core Logic

The solution addresses the three challenge phases with distinct, principled techniques.

### Phase 1 — Surviving Toxicity (train.py)

**Problem:** 30% of training labels are wrong. Standard cross-entropy will memorise the noise.

**Solution: Generalised Cross-Entropy (GCE) + Small-Loss Filtering**

- **Warmup phase (epochs 1–10):** Standard cross-entropy loss is used on *all* samples. The model first learns the strong signal before noise filtering begins. Filtering too early risks discarding clean samples.
- **Main phase (epoch 11+):** GCE loss (`q=0.7`) is used, which is theoretically robust to symmetric noise up to ~50%. Loss is computed *per sample*.
- **Small-loss filter:** After computing per-sample GCE losses, only the **lowest-loss 70% of samples** in each batch are used for the gradient update. The noisiest 30% (matching the known noise rate) are silently discarded. The intuition is that noisy samples have higher loss because the model's prediction disagrees with the (wrong) label.

**Optimiser & Scheduler:**

- Adam with weight decay `1e-4`.
- `CosineAnnealingWarmRestarts` (`T_0=30, T_mult=2`) provides periodic learning rate resets, helping the model escape local minima caused by noisy gradients.
- Gradient clipping at norm 5.0 prevents exploding gradients.

**Augmentation (permitted only):**

- `RandomCrop(28, padding=4)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomAffine(degrees=10, translate=0.1, scale=0.9–1.1)`

No corruption augmentations (AugMix, blur, noise) are used, as prohibited by competition rules.

---

### Phase 2 — Blind Adaptation (model_submission.py)

**Problem:** At inference time, the target domain has no labels and a different class distribution (label shift) and different pixel statistics (covariate shift / sensor noise).

**Solution: Two-stage automatic TTA on the first `eval()` forward call**

The model detects when it is in `eval()` mode and has not yet adapted (`_adapted = False`). On the first call it runs two sequential adaptation steps, then sets `_adapted = True` so subsequent calls just run prediction.

**Step 1 — BN Statistics Reset (Covariate Shift)**

Batch Normalisation layers store running mean and variance computed on the source (clean) distribution. The target domain has different pixel statistics due to sensor noise. The BN stats reset:

1. Switches the model temporarily to `train()` mode (so BN layers accumulate statistics).
2. Sets `momentum=None` (cumulative average) and resets all BN running stats.
3. Passes all target images through the backbone in a single forward sweep with no gradients.
4. Restores `momentum=0.1` and switches back to `eval()` mode.

This recalibrates the normalisation layers to the target distribution without any labels.

**Step 2 — EM Label-Shift Estimation (Label Shift)**

Class imbalance means the prior `p_t(y)` in the target domain differs from the source `p_s(y)` (which is approximately uniform). The Black-Box EM algorithm (Lipton et al., ICML 2018) estimates importance weights `w_t = p_t(y) / p_s(y)` from the model's own soft predictions:

1. Collect softmax probabilities for all target images.
2. Run EM iterations: re-weight predictions by current `w_t`, re-estimate `p_t`, repeat until convergence (max 50 iterations, tolerance `1e-6`).
3. Store `log(w_t)` as a buffer on the model.

At prediction time, the stored `log(w_t)` is added directly to the output logits, which is equivalent to multiplying probabilities by the importance weights in probability space. This corrects the model's implicit class prior from uniform to the target distribution.

**Per-Domain Isolation:**

Calling `model.reset()` between domains restores pristine BN statistics and clears the EM weights, ensuring each scenario is adapted independently.

---

### Phase 3 — Stress Test Robustness

No explicit training is done for Phase 3. The design choices in Phases 1 and 2 are intended to generalise:

- GCE training produces a more calibrated model that is less over-confident on any one corruption.
- BN reset handles a wide variety of covariate shifts (blur, noise, contrast change) because it re-estimates statistics from whatever distribution is presented.
- EM correction handles class imbalance shifts without needing any labels.

---

## Project Structure

```
submission/
├── train.py               # Phase 1 robust training script
├── model_submission.py    # RobustClassifier architecture + TTA
├── weights.pth            # Trained model weights
└── submission.csv         # Final predictions
```

Data files (not included in submission):
```
data/
├── source_toxic.pt        # 60,000 training images with 30% label noise
├── static.pt              # 10,000 unlabelled target images
├── val_sanity.pt          # 100 clean validation images (10 per class)
└── test_suite_public.pt   # 24-scenario local test suite
```

---

## Setup & Installation

**Requirements:** Python 3.8+, CUDA (optional but recommended)

```bash
pip install torch torchvision pandas numpy
```

---

## Running the Project

### Train the model

```bash
python train.py \
    --source ./data/source_toxic.pt \
    --val    ./data/val_sanity.pt   \
    --output weights.pth
```

**All training arguments (with defaults):**

| Argument | Default | Description |
|---|---|---|
| `--source` | `./data/.../source_toxic.pt` | Path to noisy training data |
| `--val` | `./data/.../val_sanity.pt` | Path to sanity validation set |
| `--output` | `weights.pth` | Output path for saved weights |
| `--epochs` | `100` | Maximum training epochs |
| `--batch_size` | `128` | Batch size |
| `--lr` | `3e-4` | Initial learning rate |
| `--gce_q` | `0.7` | GCE robustness parameter |
| `--forget_rate` | `0.30` | Fraction of samples filtered per batch |
| `--warmup` | `10` | Warmup epochs (standard CE, no filtering) |
| `--patience` | `20` | Early stopping patience (post-warmup) |
| `--seed` | `42` | Random seed for reproducibility |

### Verify the model architecture

```bash
python model_submission.py
```

This runs a built-in sanity check that prints the output shapes, confirms `_adapted` flips to `True` after the first eval forward pass, and prints the estimated `log_w_t` weights.

---

## Generating a Submission

Use the official template provided by the competition:

```python
import torch
import pandas as pd
from model_submission import RobustClassifier

def generate_submission(model, static_path, suite_path):
    model.eval()
    results = []

    # Static set (Public Leaderboard)
    static = torch.load(static_path)
    # First call triggers BNStats reset + EM estimation automatically
    with torch.no_grad():
        preds = model(static['images']).argmax(1)
        for i, p in enumerate(preds):
            results.append({'ID': f'static_{i}', 'Category': int(p)})

    # 24-scenario suite (Private Leaderboard)
    suite = torch.load(suite_path)
    scenario_keys = sorted([k for k in suite.keys() if k.startswith('scenario')])

    for skey in scenario_keys:
        model.reset()  # Re-adapt independently for each scenario
        scenario_images = suite[skey]
        with torch.no_grad():
            preds = model(scenario_images).argmax(1)
            for i, p in enumerate(preds):
                results.append({'ID': f'{skey}_{i}', 'Category': int(p)})

    pd.DataFrame(results).to_csv('submission.csv', index=False)

model = RobustClassifier()
model.load_weights('weights.pth')
generate_submission(model, 'static.pt', 'test_suite_public.pt')
```

> **Important:** `model.reset()` must be called before each new scenario. This restores the trained BN statistics and clears the EM weights so each domain is adapted from the same clean baseline.

---

## Design Decisions & Compliance

### Competition Compliance Checklist

| Requirement | Status | How |
|---|---|---|
| Random weight initialisation only | ✓ | `resnet18(weights=None)` + Kaiming init |
| No pretrained / external weights | ✓ | No `torch.hub`, no external state dicts |
| No corruption augmentations | ✓ | Only crop, flip, affine in `TRAIN_AUG` |
| No external clean data | ✓ | `static.pt` never loaded in `train.py` |
| `static.pt` used only for adaptation | ✓ | TTA happens inside `forward()` at eval time |
| Fully reproducible via `train.py` | ✓ | Fixed seed, deterministic CUDA |

### Why ResNet-18 and not a custom CNN?

ResNet-18's residual connections provide stable gradient flow during noisy training — deep plain networks can have their gradients dominated by the high-loss noisy samples. The architecture is minimal (11M parameters) but well-studied.

### Why GCE over Co-teaching or DIVIDEMIX?

Co-teaching and DivideMix require two networks and are significantly more complex to implement and tune. GCE with a small-loss filter is a single-model approach that achieves comparable noise robustness with far less risk of implementation bugs, which matters for a reproducibility audit.

### Why add `log(w_t)` to logits rather than re-weighting probabilities?

Adding a constant offset to logits before `argmax` is numerically identical to multiplying softmax probabilities by the weights. The logit formulation is simpler to implement, avoids a redundant softmax-then-multiply, and integrates cleanly with the existing `forward()` return value.

---

## Unit Testing & Sanity Checks

### Built-in architecture test

```bash
python model_submission.py
```

Expected output:
```
Train output : torch.Size([8, 10])
Eval output  : torch.Size([500, 10])
_adapted     : True
log_w_t      : [...]          # non-zero after EM estimation
After reset  : False          # reset() correctly clears state
Parameters   : 11,181,642     # (approximate)
```

### Validating training logic

The `val_sanity.pt` set (100 clean images, 10 per class) is used exclusively to check that the model is learning real signal and to trigger early stopping. A healthy run should reach >90% val accuracy within 30–50 epochs.

### Checking submission format

```python
import pandas as pd
df = pd.read_csv('submission.csv')
assert list(df.columns) == ['ID', 'Category']
assert df['Category'].between(0, 9).all()
print(f"Total predictions: {len(df)}")  # Should be 10000 + sum of scenario sizes
```

### Manual BN adaptation check

```python
import torch
from model_submission import RobustClassifier

model = RobustClassifier()
model.load_weights('weights.pth')
model.eval()

# Confirm not adapted yet
assert not model._adapted

# Run one forward pass
with torch.no_grad():
    _ = model(torch.rand(500, 1, 28, 28))

# Confirm adaptation ran
assert model._adapted
assert model._log_w_t.abs().sum() > 0  # weights are non-trivial

# Confirm reset works
model.reset()
assert not model._adapted
assert model._log_w_t.abs().sum() == 0
```
