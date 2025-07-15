# 🐶🐱 Dog vs. Cat Classifier using PyTorch and MobileNetV2

This project implements binary classification (cats vs. dogs) using **MobileNetV2** with **transfer learning in PyTorch**. It builds upon earlier CNN models and compares results with TensorFlow-based implementations.

---

## 🔧 Model Details

- **Base model**: `torchvision.models.mobilenet_v2(pretrained=True)`
- **Modifications**:
  - Unfrozen layers: `features[10:]`
  - Custom classifier:
    ```python
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 1)
    )
    ```
  - Output activated via **Sigmoid**
- **Loss**: `BCEWithLogitsLoss`
- **Optimizer**: `Adam`, `lr=1e-3`
- **Scheduler**: `ReduceLROnPlateau` for adaptive learning
- **Epochs**: 5
- **Data augmentation**: Yes
- **Device**: GPU (Colab)

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | `96.02%` |
| **Final Validation Loss** | ~0.106 |
| **Training Loss (final)** | ~0.048 |
| **Epochs** | 5 |

---

## 📊 Training & Evaluation

- ✅ Used `ReduceLROnPlateau` for scheduler
- ✅ Applied **data augmentation** and normalization
- ✅ Unfroze deeper layers of MobileNetV2
- ✅ Used GPU in Google Colab for faster training
- ✅ Saved loss/accuracy plots and predictions

---

## ✅ Results

![Loss Plot](results/loss_plot.png)  
![Predictions](results/sample_predictions.png)

---

## ⚔️ Comparison with Other Models

| Model | Framework | Accuracy | Training Time | Notes |
|-------|-----------|----------|----------------|-------|
| CNN | TensorFlow | ~76% | Fast (GPU) | Overfitting without reg. |
| CNN | PyTorch | ~78% | Fast | Improved after L2/dropout |
| MobileNetV2 | TensorFlow | **96%** | Medium | Excellent generalization |
| MobileNetV2 | PyTorch | **96.02%** | Medium | Match after tuning layers & lr |

---

## 💡 Key Takeaways

- **Transfer learning** provides a huge boost over CNNs
- TensorFlow made setup easier via `include_top=False`
- PyTorch required more tuning (unfreezing, scheduler)
- Both frameworks ultimately achieved **comparable accuracy**
- **Learning rate scheduler** and **fine-tuning layers** were critical

---

## 🛠️ How to Run

1. Upload the dataset to `data/train/train`
2. Use the script to sort files into `cat/` and `dog/` folders
3. Train with:
    ```bash
    python train.py
    ```
4. Plot and visualize results

---

## 📁 Related Files

- `issues_and_solutions.md`: Full documentation of bugs & fixes
- `plots/`: Loss, accuracy, and prediction samples
- `models/`: Saved PyTorch models (optional)

---

