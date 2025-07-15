# 🧵 PyTorch MobileNetV2 - Cats vs. Dogs Classification

This project implements **transfer learning** using **MobileNetV2** in PyTorch for binary classification of cats and dogs. It also includes comparisons with the same architecture in TensorFlow and earlier CNN models built from scratch.

---

## 📊 Model Overview
- **Architecture**: Pretrained MobileNetV2 from `torchvision.models`
- **Modifications**:
  - Replaced final classifier layer to output 1 logit (for binary classification)
  - Used `BCEWithLogitsLoss` with sigmoid activation
  - Fine-tuned final layers only (last stages of feature extractor)

---

## 📈 Training Strategy
- Image size: **128x128**
- Batch size: **32**
- Optimizer: `Adam`, LR: `1e-3`
- Scheduler: `ReduceLROnPlateau` (ideal for fewer epochs)
- **Data Augmentation**: Applied via `torchvision.transforms`
  - RandomHorizontalFlip
  - RandomRotation
  - RandomResizedCrop

---

## 🌟 Results Summary (PyTorch MobileNetV2)
| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|---------------|
| 1     | 0.1668     | 0.1016   | 95.90%        |
| 2     | 0.1041     | 0.1091   | 95.82%        |
| 3     | 0.0833     | 0.1012   | 96.00%        |
| 4     | 0.0618     | 0.1121   | 95.94%        |
| 5     | 0.0480     | 0.1062   | 96.02%        |

### 📸 Prediction Samples
All shown predictions were correct. The model performed well even on darker or cluttered images.

---

## 📊 Comparison with TensorFlow MobileNetV2
| Framework     | Model         | Accuracy  | Notes                                     |
|---------------|---------------|-----------|-------------------------------------------|
| TensorFlow    | MobileNetV2   | ~97%      | Easier pipeline with `image_dataset_from_directory` |
| PyTorch       | MobileNetV2   | ~96%      | More control and customization            |

---

## 🔍 CNN vs MobileNetV2 (TensorFlow vs PyTorch)
| Framework     | Model         | Accuracy  | Notes                                     |
|---------------|---------------|-----------|-------------------------------------------|
| TensorFlow    | CNN (custom)  | ~76%      | Dropout and L2 helped generalization      |
| TensorFlow    | MobileNetV2   | ~97%      | Transfer learning significantly better    |
| PyTorch       | CNN (custom)  | ~74%      | Good baseline but limited by feature depth|
| PyTorch       | MobileNetV2   | ~96%      | Excellent tradeoff between speed & accuracy |

---

## 🚀 How to Run
1. Clone the repo and open `notebook.ipynb`
2. Upload dataset to: `data/train/train`
3. Run preprocessing to organize images into `cat/` and `dog/` folders
4. Train the model with MobileNetV2
5. Run prediction samples and evaluation

---

## 📊 GitHub Notes
- ✅ All training logs and figures are saved
- ✅ `show_predictions()` for qualitative analysis
- ✅ Compared MobileNetV2 between PyTorch and TensorFlow
- ✅ Reduced learning rate dynamically using `ReduceLROnPlateau`

> If you plan to train longer, consider using `StepLR` instead.

---

## 🔄 Next Steps
- Try **ResNet50** or **EfficientNet** in PyTorch
- Explore **early stopping** + **confidence calibration**
- Deploy model via **TorchServe** or **Flask API**

---

## 📅 Author
Project built and compared across both frameworks for learning, experimentation, and resume portfolio.

---

## 📊 Related Files
- `cnn_tensorflow.ipynb`
- `cnn_pytorch.ipynb`
- `mobilenet_tensorflow.ipynb`
- `mobilenet_pytorch.ipynb`
- `README.md`
- `issues_and_solutions.md`

