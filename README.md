# SA Currency Classifier - Model Training Pipeline

A MobileNetV2-based image classifier for South African banknotes and coins, trained in Google Colab and exported to TFLite for on-device inference on Android.

---

## Overview

This repo contains the full ML pipeline - data augmentation scripts, the training notebook, and the exported TFLite models - for a classifier that identifies SA currency by denomination, era (old/new), and side (front/back).

Two separate models are produced from the same notebook:

| Model | Classes | Test Accuracy |
|---|---|---|
| Banknotes | 20 (R10–R200, old/new, front/back) | **92.8%** |
| Coins | 28 (5c–R5, old/new, front/back) | **87.9%** |

> The trained `.tflite` models are deployed in the companion Android app: [SA-Currency-Classifier-App](https://github.com/Shaista149/SA-Currency-Classifier-App)

---

## Repository Structure

```
SA-Currency-Classifier-Model/
│
├── SA_Currency_Classifier.ipynb      # Main training notebook (run in Google Colab)
│
├── banknote_augment.py               # Core augmentation - 15 variants per image
├── banknote_augment_extra.py         # Extra variants: perspective, shadow, motion blur, bg composite
├── banknote_bg_augment.py            # Background placement augmentation for banknotes
│
├── coin_augment.py                   # Core augmentation - 15 variants per image
├── coin_augment_extra.py             # Extra variants: perspective, shadow, motion blur, bg composite
├── coin_bg_augment.py                # Background placement augmentation for coins
│
└── README.md
```

---

## Model Architecture

Both models share the same architecture:

- **Backbone:** MobileNetV2 (pretrained on ImageNet, input 224×224)
- **Attention:** Squeeze-and-Excitation (SE) block inserted after the backbone
- **Head:** Global Average Pooling → Dense 256 (ReLU) → Dropout 0.4 → Softmax output
- **Export:** Float16 quantised TFLite (5.61 MB per model)

### Two-Phase Training

**Phase 1 - Classification Head**
Backbone frozen. Only the SE block and classification head are trained with Adam + `ReduceLROnPlateau`.

**Phase 2 - Fine-tuning**
Top N layers of the backbone unfrozen and trained with cosine annealing:
- Banknotes: top 40 layers at lr=1e-5
- Coins: top 60 layers at lr=5e-6 (coins are less ImageNet-like so more layers need adapting)

---

## Preprocessing Pipeline

Every image goes through the same steps before reaching the model:

1. Resize to 224×224
2. Softer unsharp mask (kernel centre 4.2) - preserves fine print detail
3. CLAHE on L channel in LAB space (clipLimit 1.2) - improves contrast without washing out colour
4. Bilateral filter (d=7, sigma=55) - smooths noise while keeping edges
5. Circular mask *(coins only)* - zeros corners outside the inscribed circle so the model ignores white padding
6. Scale to [−1, 1] - matches MobileNetV2's pretraining normalisation

MixUp augmentation is applied at batch level during training to smooth the decision boundary between visually similar classes (e.g. R50 new vs R200 old).

---

## Data Augmentation

The dataset was **self-collected** and is not publicly available. Augmentation was applied offline using the scripts in this repo before training.

### Core augmentation (`*_augment.py`) - 15 variants per original image
- 1 original
- 7 rotations (45° steps)
- 2 contrast boosts
- 2 brightness boosts
- 2 Gaussian blurs
- 1 Gaussian noise

### Extra augmentation (`*_augment_extra.py`) - 10 additional variants
- 2 perspective warps
- 2 shadow overlays
- 2 motion blurs
- 2 background composites (note/coin pasted onto real-world backgrounds)
- 2 combined (perspective warp + shadow)

### Background augmentation (`*_bg_augment.py`) - 2 additional variants
Places the currency as a scaled object on a real background photo, simulating it lying on a surface.

---

## Dataset Split

| Set | Size | Notes |
|---|---|---|
| Train | 7,208 | With oversampling applied |
| Validation | 600 | Stratified, no oversampling |
| Test | 1,000 | Held out until final evaluation |

Oversampling (3×) was applied to underperforming classes after the split so validation and test sets are never contaminated by duplicated rows.

---

## Results - Banknotes

**Test accuracy: 92.8%** (1,000 held-out samples)

| Denomination | Accuracy |
|---|---|
| R10 | 96.0% |
| R20 | 98.0% |
| R50 | 87.5% |
| R100 | 95.5% |
| R200 | 93.8% |

| Side | Accuracy |
|---|---|
| Front | 92.6% |
| Back | 93.0% |

Notable: R50 new front/back had the weakest recall (~68–72%) due to visual similarity with R200 old. These classes were given 3× oversampling exposure.

---

## Results - Coins

**Test accuracy: 87.9%** (1,475 held-out samples)

| Denomination | Best F1 | Weakest class |
|---|---|---|
| 5c | 0.90–0.97 | - |
| 10c | 0.87–0.90 | - |
| 20c | 0.86–0.91 | 20c new back (0.69) |
| 50c | 0.83–0.84 | 50c new back (0.71) |
| R1 | 0.85–0.89 | - |
| R2 | 0.81–0.91 | R2 old front (0.81) |
| R5 | 0.82–0.98 | R5 old back (0.82) |

| Side | Accuracy |
|---|---|
| Front | ~88% |
| Back | ~88% |

Notable: 20c new back and 50c new back were the weakest classes, likely due to smaller physical size and similar surface textures across denominations. R2 old back achieved perfect recall (1.00).

---

## How to Run

1. Open `SA_Currency_Classifier.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`)
3. Mount your Google Drive and update `BASE_PATH` in Cell 2 to point to your dataset folder
4. Set `MODE = 'notes'` or `MODE = 'coins'` in Cell 2
5. Run all cells - training is resumable (checkpoints and splits are cached to Drive)

### Expected Drive structure
```
Currency_Classifier/
├── banknote_labels.csv
├── coin_labels.csv
├── data/
│   ├── banknotes/augmented/   ← augmented banknote images
│   └── coins/augmented/       ← augmented coin images
├── models/                    ← checkpoints and exports written here
└── cache/                     ← split and history caches written here
```

### Dependencies
All installed automatically in Cell 1:
```
tensorflow  opencv-python-headless  scikit-learn  pandas  matplotlib  seaborn
```

---


## Related

- **Android App:** [SA-Currency-Classifier-App](https://github.com/Shaista149/SA-Currency-Classifier-App)
- **Dataset:** Self-collected, privately held. Available on request.
