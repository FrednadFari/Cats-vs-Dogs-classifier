# Cats-vs-Dogs-classifier
A deep learning project using PyTorch & TensorFlow to classify cat vs. dog images

🔍 Here’s a comparison based on your context (Cats vs. Dogs CNN):

Criteria	TensorFlow	PyTorch
Ease of Setup (Colab)	✅ Built-in data pipeline (image_dataset_from_directory)	Requires more manual setup
Model Summary / Layers	✅ Clear with .summary()	More verbose with print(model)
Training Loop	✅ High-level (model.fit())	Manual training loop
Built-in Augmentation	✅ Easy with Sequential() layers	Requires torchvision.transforms
Saving/Loading	✅ One-liner model.save()	Requires tracking both model and state_dict
Binary Classification	✅ Easier with sparse_categorical_crossentropy + Softmax	Needs careful loss/function selection


🧠 Verdict (for Binary Image Classification):
TensorFlow is often more convenient and faster to prototype, especially for binary classification using CNNs and working in Google Colab.

