# Hackenza 2026: Test-Time Adaptation in the Wild

A robust vision classifier that survives poisoned training data and automatically self-adapts — without labels — to hostile target domains at inference time.

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

The challenge simulates a robot deployed in a hostile environment where three compounding problems must be solved simultaneously:

1. **Toxicity** — Training data (`source_toxic.pt`) contains **30% label noise**, meaning roughly 1 in 3 labels is intentionally wrong. Standard cross-entropy will memorise the noise and produce an overconfident, brittle classifier.

2. **Blind Adaptation** — The target domain (`static.pt`) has **no labels**, suffers from significant sensor noise (covariate shift), and exhibits heavy **class imbalance** (label shift). The model must correct for both without any supervision.

3. **Stress Test** — The final model is evaluated across **24 hidden corruption scenarios** at multiple severity levels, requiring genuine generalisation rather than overfitting to a single distribution.

Standard Cross-Entropy training + standard inference fails all three. This pipeline addresses each one explicitly.

---

## Solution Architecture

The solution is split across three files with a clean separation of responsibilities:

| File | Responsibility |
|---|---|
| `train.py` | Phase 1: robust training on noisy source data |
| `model_submission.py` | Model architecture + self-contained Test-Time Adaptation (TTA) |
| `generate_submission.py` | Thin submission generator that orchestrates per-domain adaptation |

### Model Backbone

The backbone is a **ResNet-18** modified for 28×28 greyscale images:

- `conv1` replaced with a 3×3 convolution accepting **1 input channel** (greyscale, no RGB).
- `maxpool` replaced with `nn.Identity()` to **preserve spatial resolution** at 28×28 — the standard ResNet maxpool would aggressively downsample these small images.
- Final `fc` layer outputs **10 class logits**.
- All weights are **randomly initialised via Kaiming Normal** — no pretrained weights, as required by competition rules.
- ResNet-18's residual connections provide stable gradient flow during noisy training, where plain deep networks would have their gradients dominated by high-loss noisy samples.

---

## Core Logic

The solution addresses each challenge phase with a distinct, principled technique.

### Phase 1 — Surviving Toxicity (`train.py`)

**Problem:** 30% of training labels are wrong. Standard cross-entropy will memorise the noise.

**Solution: Two-stage approach — warmup then GCE + small-loss filtering**

**Stage 1 — Warmup (epochs 1–10):** Standard `CrossEntropyLoss` is applied on *all* samples with no filtering. The model first learns the genuine signal present in the 70% clean labels before any sample rejection begins. Starting the filter too early risks mistakenly discarding clean samples whose loss happens to be high at initialisation.

**Stage 2 — Main training (epoch 11+):** Two mechanisms work in tandem:

- **Generalised Cross-Entropy (GCE, Zhang et al. NeurIPS 2018):** Replaces standard CE with a noise-robust loss. At `q=0.7`, GCE is theoretically robust to symmetric label noise up to ~50% — well above the 30% noise rate in this challenge. Unlike CE, GCE naturally down-weights the gradient contribution of confidently wrong predictions. Loss is computed *per sample* to enable the filter below.

- **Small-loss filter:** Within each batch, per-sample GCE losses are computed and only the **lowest-loss 70% of samples** are used for the gradient update. The noisiest 30% (matching the known noise rate) are silently discarded. The intuition is clean: a correctly labelled sample will tend to have a lower loss because the model's prediction aligns with the label, while a mislabelled sample forces the model to predict something it has learned is unlikely, producing a higher loss.

**Optimiser and stability:**

- Adam with `weight_decay=1e-4`.
- `CosineAnnealingWarmRestarts` (`T_0=30, T_mult=2`) provides periodic learning rate resets, helping the model escape local minima induced by noisy gradient batches.
- Gradient clipping at norm 5.0 prevents exploding gradients on high-loss noisy batches.
- Early stopping with 20-epoch patience (post-warmup only) saves the best checkpoint by validation accuracy on `val_sanity.pt`.

**Augmentation (permitted only — no corruption augmentations):**

- `RandomCrop(28, padding=4)`
- `RandomHorizontalFlip(p=0.5)`
- `RandomAffine(degrees=10, translate=0.1, scale=0.9–1.1)`

Augmentation runs inside DataLoader workers for parallel, non-blocking data loading.

---

### Phase 2 — Blind Adaptation (`model_submission.py`)

**Problem:** At inference time, the target domain has different pixel statistics (covariate shift from sensor noise) and a different class distribution (label shift from class imbalance). No labels are available.

**Solution: Two-stage automatic TTA, self-triggered on the first `eval()` forward call**

The `forward()` method detects when it is in `eval()` mode and `_adapted` is `False`. On the first call it runs two sequential adaptation steps, sets `_adapted = True`, and all subsequent calls simply predict with the stored corrections. This means the standard Kaggle evaluation template (`model.eval(); preds = model(images).argmax(1)`) works with **zero modifications**.

**Step 1 — BN Statistics Reset (handles covariate shift)**

Batch Normalisation layers store running mean and variance computed on the source distribution during training. When the target domain has different pixel statistics (e.g. added Gaussian noise shifts the mean and variance), these stored statistics cause all downstream feature maps to be incorrectly normalised, degrading accuracy significantly.

The BN reset procedure:
1. Switches the model temporarily to `train()` mode so BN layers accumulate new statistics.
2. Sets `momentum=None` (cumulative average mode) and resets all BN running stats to zero.
3. Passes all target images through the backbone in a single forward sweep with no gradients — this populates the running stats with exact target-domain statistics.
4. Restores `momentum=0.1` and switches back to `eval()` mode.

This re-fits the normalisation layers to the target distribution using only the unlabelled target images, with no parameter updates.

**Step 2 — EM Label-Shift Estimation (handles label shift)**

Class imbalance means the prior `p_t(y)` in the target domain differs from the source prior `p_s(y)` (approximately uniform due to balanced training). Without correction, the model's predictions are biased toward classes that were frequent at training time, hurting Macro-F1 on under-represented target classes.

The **Black-Box EM algorithm** (Lipton et al., ICML 2018) estimates importance weights `w_t = p_t(y) / p_s(y)` using only the model's own soft predictions:

1. Initialise `p_t = p_s = uniform`.
2. Re-weight each sample's predicted probability vector by current `w_t`.
3. Re-normalise to get a posterior, then re-estimate `p_t` as the mean posterior.
4. Repeat until convergence (max 50 iterations, tolerance `1e-6`).
5. Store `log(w_t)` as a buffer on the model.

At prediction time, the stored `log(w_t)` is **added directly to the output logits** before `argmax`. This is numerically equivalent to multiplying softmax probabilities by `w_t` in probability space, but avoids a redundant softmax-then-multiply and integrates cleanly with the existing `forward()` return value.

**Adaptation State Machine:**

| State | `_adapted` | `_log_w_t` | Next `forward()` action |
|---|---|---|---|
| Just loaded | `False` | zeros | Runs BNStats + EM, then predicts |
| After first eval call | `True` | estimated | Just predicts + adds bias |
| After `reset()` | `False` | zeros | Runs BNStats + EM again |

---

### Phase 3 — Stress Test Robustness (`generate_submission.py`)

No additional training is performed for Phase 3. The submission generator handles the 24-scenario suite by calling `model.reset()` before each scenario, which restores the pristine trained BN statistics and clears the EM weights. Each scenario is then adapted independently from the same clean baseline.

The generator also logs the prediction distribution and number of unique classes for each domain — useful for diagnosing degenerate scenarios where the model collapses to predicting a single class.

The robustness of the base approach across diverse corruptions comes from:
- GCE training producing a well-calibrated model that is not over-confident on any particular distribution.
- BN reset adapting to whatever pixel statistics are present in a given scenario.
- EM correction adapting to whatever class imbalance is present in that scenario.

---

## Project Structure

```
submission/
├── train.py                  # Phase 1: robust training script
├── model_submission.py       # RobustClassifier architecture + TTA
├── generate_submission.py    # Submission generator
├── weights.pth               # Trained model weights
└── submission.csv            # Final predictions (generated output)
```

Data files (not included in submission, expected in `./data/`):
```
data/hackenza-2026-test-time-adaptation-in-the-wild/
├── source_toxic.pt           # 60,000 training images with 30% label noise
├── static.pt                 # 10,000 unlabelled target images
├── val_sanity.pt             # 100 clean validation images (10 per class)
└── test_suite_public.pt      # 24-scenario local test suite
```

---

## Setup & Installation

**Requirements:** Python 3.8+, CUDA (optional but strongly recommended for training)

```bash
pip install torch torchvision pandas numpy
```

---

## Running the Project

### Step 1 — Train the model

```bash
python train.py \
    --source ./data/hackenza-2026-test-time-adaptation-in-the-wild/source_toxic.pt \
    --val    ./data/hackenza-2026-test-time-adaptation-in-the-wild/val_sanity.pt   \
    --output weights.pth
```

All training arguments with their defaults:

| Argument | Default | Description |
|---|---|---|
| `--source` | `./data/.../source_toxic.pt` | Path to noisy training data |
| `--val` | `./data/.../val_sanity.pt` | Path to clean sanity validation set |
| `--output` | `weights.pth` | Output path for saved weights |
| `--epochs` | `100` | Maximum training epochs |
| `--batch_size` | `128` | Batch size |
| `--lr` | `3e-4` | Initial learning rate |
| `--gce_q` | `0.7` | GCE robustness parameter (higher = more robust to noise) |
| `--forget_rate` | `0.30` | Fraction of noisy samples filtered per batch |
| `--warmup` | `10` | Warmup epochs using standard CE with no filtering |
| `--patience` | `20` | Early stopping patience (post-warmup only) |
| `--seed` | `42` | Random seed for full reproducibility |

A healthy training run should reach >90% validation accuracy on `val_sanity.pt` within 30–50 epochs. Training progress is printed each epoch including loss, train accuracy, val accuracy, and current learning rate.

### Step 2 — Verify the model architecture

```bash
python model_submission.py
```

This runs the built-in sanity check and should print:

```
Train output : torch.Size([8, 10])
Eval output  : torch.Size([500, 10])
_adapted     : True
log_w_t      : [...]      # non-zero after EM estimation
After reset  : False      # reset() correctly cleared state
Parameters   : 11,181,642
```

---

## Generating a Submission

```bash
python generate_submission.py \
    --weights weights.pth           \
    --static  static.pt             \
    --suite   test_suite_public.pt  \
    --output  submission.csv
```

This will:
1. Load the trained model
2. For each domain (static + 24 scenarios): call `model.reset()`, then `model(images)` which triggers BNStats + EM automatically
3. Write predictions to `submission.csv`

---

## How the Adaptation Works

The core of the solution is `RobustClassifier.forward()`:

```python
def forward(self, x):
    if self.training:
        return self.backbone(x)          # pure forward during training

    if not self._adapted:
        self._bnstats_reset(x)           # Step 1: recompute BN stats from target
        probs = get_softmax_probs(x)     
        w_t = self._em_estimate_w_t(probs)  # Step 2: estimate target class prior
        self._log_w_t = log(w_t)         # store logit bias
        self._adapted = True

    return self.backbone(x) + self._log_w_t   # predict with label-shift correction
```

**BNStats Reset** — BatchNorm layers store running mean and variance from training data. When target images have different statistics due to corruption, all BN normalizations use wrong reference values. `_bnstats_reset()` sets `momentum=None` (cumulative average mode), zeros the buffers, and runs one full forward pass over the target images to recompute correct statistics.

**EM Label Shift Estimation** — Iteratively estimates the target class distribution `p_t(y)` from the model's softmax predictions. Applies importance weights `w_t = p_t(y) / p_s(y)` as a logit bias `log(w_t)`. Classes more common in the target domain get their logits boosted; rare classes are penalized. Runs once on the full dataset for stable estimates.

**Per-scenario isolation** — `model.reset()` restores pristine BN buffers and clears `_adapted`, so each of the 24 scenarios gets a fresh adaptation from the base trained weights.

---

## Design Decisions & Compliance

### Competition Compliance Checklist

| Requirement | Status |
|---|---|
| Random weight initialisation only | ✅ `weights=None` throughout |
| No pretrained / external weights | ✅ Only `source_toxic.pt` used in training |
| No corruption augmentations | ✅ Only crop, flip, affine in `TRAIN_AUG` — no AugMix, blur, or noise |
| No external clean data | ✅ `static.pt` is never loaded or referenced in `train.py` |
| `static.pt` used only for adaptation | ✅ TTA runs entirely inside `forward()` at eval time |
| Fully reproducible via `train.py` | ✅ Fixed seed, `deterministic=True`, `benchmark=False` |
