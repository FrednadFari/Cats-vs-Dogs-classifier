🧵 PyTorch MobileNetV2 - Cats vs. Dogs Classification

This project implements transfer learning using MobileNetV2 in PyTorch for binary classification of cats and dogs. It also includes comparisons with the same architecture in TensorFlow and earlier CNN models built from scratch.

📊 Model Overview

Architecture: Pretrained MobileNetV2 from torchvision.models

Modifications:

Replaced final classifier layer to output 1 logit (for binary classification)

Used BCEWithLogitsLoss with sigmoid activation

Fine-tuned final layers only (last stages of feature extractor)

📈 Training Strategy

Image size: 128x128

Batch size: 32

Optimizer: Adam, LR: 1e-3

Scheduler: ReduceLROnPlateau (ideal for fewer epochs)

Data Augmentation: Applied via torchvision.transforms

RandomHorizontalFlip

RandomRotation

RandomResizedCrop

🌟 Results Summary (PyTorch MobileNetV2)

Epoch

Train Loss

Val Loss

Val Accuracy

1

0.1668

0.1016

95.90%

2

0.1041

0.1091

95.82%

3

0.0833

0.1012

96.00%

4

0.0618

0.1121

95.94%

5

0.0480

0.1062

96.02%

📸 Prediction Samples

All shown predictions were correct. The model performed well even on darker or cluttered images.

📊 Comparison with TensorFlow MobileNetV2

Framework

Model

Accuracy

Notes

TensorFlow

MobileNetV2

~97%

Easier pipeline with image_dataset_from_directory

PyTorch

MobileNetV2

~96%

More control and customization
