# FashionMNIST Image Classification with a Convolutional Neural Network

**Student ID:** 220133
**Framework:** PyTorch
**Dataset:** FashionMNIST (standard test set) + 10 custom phone-camera photos

## 1. Overview

This project implements, trains, and evaluates a Convolutional Neural Network (CNN) for classifying clothing images into 10 categories from the **FashionMNIST** dataset. Beyond the standard benchmark evaluation, the model is additionally stress-tested on **10 real-world phone photographs** of clothing items to assess generalization from clean, centered, 28×28 grayscale training images to noisy, real-world inputs.

The pipeline covers the full deep learning workflow:

1. Data loading and preprocessing
2. CNN architecture design
3. Training with tracked loss/accuracy history
4. Model checkpointing
5. Quantitative evaluation (confusion matrix)
6. Qualitative evaluation on custom, out-of-distribution images
7. Error analysis on misclassified samples

## 2. Dataset

- **Source:** `torchvision.datasets.FashionMNIST`
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Training set:** 60,000 grayscale images (28×28)
- **Test set:** 10,000 grayscale images (28×28)
- **Custom set:** 10 phone photos of real clothing items, cloned from a companion GitHub repository (`220133_CNN/dataset`), used purely for out-of-sample qualitative testing.

### Preprocessing
- Convert to single-channel grayscale
- Resize to 28×28
- Convert to tensor
- Normalize with mean = 0.5, std = 0.5

For the custom photos, an additional **auto-invert** step is applied: if the average pixel intensity is greater than 0.5 (i.e., the background is light), the image colors are inverted so the input matches the dark-background/light-garment convention seen in FashionMNIST.

## 3. Model Architecture

A compact CNN with two convolutional blocks followed by a fully connected classifier head:

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| conv1 | Conv2d(1 → 32, 3×3, padding=1) | 32×28×28 | — |
| relu + pool | ReLU, MaxPool2d(2×2) | 32×14×14 | — |
| conv2 | Conv2d(32 → 64, 3×3, padding=1) | 64×14×14 | — |
| relu + pool | ReLU, MaxPool2d(2×2) | 64×7×7 | — |
| flatten | — | 3136 | — |
| fc1 | Linear(3136 → 128) | 128 | — |
| relu + dropout | ReLU, Dropout(p=0.25) | 128 | — |
| fc2 | Linear(128 → 10) | 10 (logits) | — |

**Loss function:** Cross-Entropy Loss
**Optimizer:** Adam (learning rate = 0.001)
**Batch size:** 64
**Epochs:** 10 (initial run) + 10 (continued fine-tuning run) = 20 total epochs
**Device:** CUDA GPU (with automatic CPU fallback)

## 4. Training Results

The model was trained for two consecutive 10-epoch runs (the second continuing from the weights of the first, for 20 epochs of cumulative training).

### Run 1 (Epochs 1–10)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.4807 | 82.60% | 0.3496 | 87.26% |
| 2 | 0.3149 | 88.59% | 0.2966 | 89.02% |
| 3 | 0.2637 | 90.35% | 0.2582 | 90.74% |
| 4 | 0.2323 | 91.43% | 0.2472 | 91.04% |
| 5 | 0.2039 | 92.46% | 0.2343 | 91.32% |
| 6 | 0.1842 | 93.25% | 0.2273 | 91.85% |
| 7 | 0.1651 | 93.84% | 0.2406 | 91.73% |
| 8 | 0.1506 | 94.38% | 0.2312 | 92.27% |
| 9 | 0.1346 | 95.03% | 0.2436 | 92.04% |
| 10 | 0.1206 | 95.54% | 0.2548 | 91.95% |

### Run 2 — Continued Training (Epochs 11–20)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 11 | 0.1062 | 95.91% | 0.2709 | 91.58% |
| 12 | 0.0991 | 96.22% | 0.2693 | 92.12% |
| 13 | 0.0905 | 96.54% | 0.2725 | 92.42% |
| 14 | 0.0818 | 96.85% | 0.2783 | 92.47% |
| 15 | 0.0753 | 97.04% | 0.3072 | **92.65% (best)** |
| 16 | 0.0706 | 97.38% | 0.2990 | 92.40% |
| 17 | 0.0672 | 97.50% | 0.3328 | 92.41% |
| 18 | 0.0607 | 97.70% | 0.3385 | 92.43% |
| 19 | 0.0608 | 97.75% | 0.3304 | 92.36% |
| 20 | 0.0550 | 97.90% | 0.3526 | 92.18% |

**Best validation accuracy: 92.65%** (epoch 15). Training accuracy continues climbing to 97.9% while validation loss rises after roughly epoch 13–15, indicating the model begins to **overfit** in the second run — validation accuracy plateaus/oscillates around 92% while the train/val loss gap widens.

Loss and accuracy curves are saved to `training_history.png`.

## 5. Evaluation on the Standard Test Set

A confusion matrix was computed over all 10,000 test images and visualized as a heatmap (`confusion_matrix.png`), showing per-class performance. As expected for FashionMNIST, the model performs strongly on visually distinct classes (Trouser, Bag, Sandal, Ankle boot) and shows more confusion among visually similar upper-body garments (Shirt vs. T-shirt/top vs. Pullover vs. Coat).

## 6. Evaluation on 10 Custom Real-World Photos

10 phone photographs of physical clothing items were preprocessed with the same pipeline (grayscale → resize → auto-invert → normalize) and passed through the trained model.

| Image | True Label | Predicted Label | Confidence | Correct? |
|---|---|---|---|---|
| 01_Shoe.jpg | Sneaker | Bag | 99.8% | ✗ |
| 02_Shirt.jpg | Shirt | T-shirt/top | 88.6% | ✗ |
| 03_Pant.jpg | Trouser | Trouser | 99.9% | ✓ |
| 04_Jacket.jpg | Coat | Coat | 98.9% | ✓ |
| 05_sandal.jpg | Sandal | Bag | 100.0% | ✗ |
| 06_sweater.jpg | Pullover | Pullover | 80.3% | ✓ |
| 07_Blazer.png | Coat | Trouser | 48.8% | ✗ |
| 08_Pholo_Shirt.jpg | T-shirt/top | T-shirt/top | 100.0% | ✓ |
| 09_Women_Dress.jpg | Dress | Bag | 85.7% | ✗ |
| 10_Bag.webp | Bag | Bag | 100.0% | ✓ |

**Custom photo accuracy: 5/10 = 50%**

This is dramatically lower than the ~92% test-set accuracy, and is a clear illustration of the **domain gap** between the clean, centered, low-resolution FashionMNIST training distribution and real-world phone photos (different lighting, backgrounds, framing, and object scale/orientation). Several errors are notably confident-but-wrong (e.g., the sandal and shoe are misclassified as "Bag" with very high confidence), suggesting the simplistic grayscale + auto-invert + resize preprocessing is not sufficient to bridge this gap — object silhouette/background cues dominate the model's decision rather than fine garment texture.

Visual outputs:
- `custom_prediction_gallery.png` — 10 custom images with true/predicted labels (green = correct, red = incorrect)
- `error_analysis.png` — 3 randomly sampled misclassified images from the standard test set

## 7. Repository / Project Structure

```
.
├── 220133.ipynb                  # Main notebook (this project)
├── 220133.pth                    # Saved trained model weights (state_dict)
├── training_history.png          # Loss & accuracy vs. epoch curves
├── confusion_matrix.png          # Confusion matrix heatmap on test set
├── custom_prediction_gallery.png # Predictions on 10 custom phone photos
├── error_analysis.png            # Sample misclassified test images
└── 220133_CNN/                   # Cloned repo containing dataset/ of 10 custom photos
    └── dataset/
```

## 8. How to Run

1. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib pillow scikit-learn seaborn
   ```
2. Open `220133.ipynb` in Jupyter/Google Colab.
3. Run all cells sequentially. FashionMNIST downloads automatically via `torchvision`. The 10 custom images are pulled from the `220133_CNN` GitHub repository via `git clone`.
4. Trained weights are saved as `220133.pth`.

## 9. Key Takeaways

- A simple 2-conv-layer CNN reaches **~92–93% validation accuracy** on FashionMNIST within 10–15 epochs.
- Continuing training beyond ~15 epochs increases training accuracy toward 98% but yields diminishing/negative returns on validation accuracy — a sign of **overfitting**, suggesting early stopping around epoch 13–15 or additional regularization (data augmentation, stronger dropout, weight decay) would improve generalization.
- Performance on genuinely out-of-distribution, real-world images (50% accuracy) is substantially weaker than on the curated test set, highlighting the gap between benchmark performance and real-world deployment.

## 10. Possible Improvements

- Add data augmentation (random crop, rotation, flips) for better generalization
- Add batch normalization to stabilize/accelerate training
- Use a learning-rate scheduler and/or early stopping based on validation loss
- Increase custom-image preprocessing robustness (e.g., background removal, contrast normalization) instead of a simple mean-based inversion heuristic
- Train a deeper architecture or fine-tune a pretrained backbone if higher accuracy is required
