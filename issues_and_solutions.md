# Issues and Solutions: Dog vs. Cat Classification with CNN and MobileNetV2

This document outlines the key challenges we encountered during the implementation of binary image classification (cats vs. dogs) using both custom CNNs and transfer learning via MobileNetV2 in TensorFlow and PyTorch. We also document the strategies and solutions applied to overcome these challenges.

---

## 1. Data Directory Not Structured for ImageFolder

**Issue**: PyTorch's `ImageFolder` and TensorFlow's `image_dataset_from_directory()` both expect images to be sorted into subdirectories per class.

**Solution**: Wrote a Python script to sort the images into `cat/` and `dog/` folders based on filename prefixes.

```python
import os, shutil
source_dir = "data/train/train"
target_base = "data/train/sorted"
os.makedirs(f"{target_base}/cat", exist_ok=True)
os.makedirs(f"{target_base}/dog", exist_ok=True)
for filename in os.listdir(source_dir):
    if filename.startswith("cat"):
        shutil.copy(os.path.join(source_dir, filename), os.path.join(target_base, "cat", filename))
    elif filename.startswith("dog"):
        shutil.copy(os.path.join(source_dir, filename), os.path.join(target_base, "dog", filename))
```

---

## 2. Validation Accuracy Not Improving / Overfitting

**Issue**: CNNs showed rising training accuracy, but validation accuracy plateaued or declined.

**Solution**:

- Applied **data augmentation**.
- Introduced **Dropout** (0.4) and **L2 regularization**.
- Switched to **GPU runtime** in Colab for faster experimentation.
- Later used **early stopping** (TensorFlow) and learning rate schedulers (PyTorch).

---

## 3. FileNotFoundError When Saving Images

**Issue**: Trying to save plots (e.g., `plt.savefig('results/loss_plot.png')`) without creating the folder first caused errors.

**Solution**:

```python
os.makedirs("results", exist_ok=True)
plt.savefig("results/loss_plot.png")
```

---

## 4. MobileNetV2 in PyTorch Underperformed vs TensorFlow

**Issue**: PyTorch's MobileNetV2 accuracy was lower than TensorFlow's.

**Analysis**:

- TensorFlow model used `include_top=False` and `GlobalAveragePooling2D`, making it efficient and effective.
- PyTorch initially used too few trainable layers.

**Solution**:

- **Unfroze** more layers in PyTorch (`model.features[10:].parameters()`).
- Used **ReduceLROnPlateau** to adaptively control learning rate.
- Applied **data augmentation** and better **image normalization**.
- Final model improved to 96.02% accuracy.

---

## 5. Training Loss/Validation Loss Misalignment in Plots

**Issue**: Loss curves looked suspicious (flat or mismatched).

**Root Cause**:

- Incorrect averaging of loss during validation.
- Reusing `running_loss` from training.

**Solution**:

- Separated training and validation loss computations with individual counters and resets.

---

## 6. Tensor Type Mismatch in PyTorch

**Issue**:

```
RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
```

**Solution**:

- Moved model to GPU: `model.to(device)`.
- Ensured data batches are also on the same device: `images, labels = images.to(device), labels.to(device)`.

---

## 7. Saving and Downloading Plots in Colab

**Issue**: Plots were not downloadable without saving.

**Solution**:

- Used `plt.savefig()` after creating the `results/` folder.
- Downloaded using:

```python
from google.colab import files
files.download("results/plot_name.png")
```

---

## 8. Visualizing Predictions

**Issue**: Some predictions appeared too dark or incorrectly classified.

**Solution**:

- Applied proper image denormalization before `imshow`.
- Selected random samples from the validation set.
- Manually validated prediction accuracy visually.

---

## 9. Accuracy Not Showing in Plot

**Issue**: Validation accuracy variable `correct` was not plottable.

**Solution**:

- Stored epoch-wise accuracy in a list `val_accuracies` and plotted that.

---

## 10. Training Logs Not Plottable Later

**Issue**: Wanted to plot training logs after training finished without rerunning.

**Solution**:

- Manually recorded logs per epoch.
- Created lists for accuracy/loss values and plotted them post-training.
