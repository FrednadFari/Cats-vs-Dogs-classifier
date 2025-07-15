🐱🐶 TensorFlow CNN - Cats vs Dogs Classifier

This project uses a custom Convolutional Neural Network (CNN) built from scratch in TensorFlow to classify images of cats and dogs. It mirrors our earlier PyTorch implementation for comparative learning, performance evaluation, and GitHub portfolio presentation.

🧠 Model Architecture

A custom CNN built using tf.keras.Sequential, including:

3 Convolutional layers with ReLU + MaxPooling

Flatten + Dense (512 units)

Dropout (0.4) for regularization

L2 kernel regularization

Softmax output with 2 classes (cat/dog)

⚙️ Performance Optimization

Issue: Model training was slow on CPU with 25k images

✅ Fix: Switched to GPU runtime in Colab (Runtime > Change runtime type > GPU)

Result: 3–5x faster training with the same accuracy

📊 Results

Training Accuracy: ~74%

Validation Accuracy: ~76%

Validation Loss: Consistently decreased

Model generalizes reasonably with some edge-case confusion




🛠️ How to Run

Upload the raw dataset to /data/train/train (images named like cat.123.jpg)

Run the provided sorting script to organize images into /cat/ and /dog/ folders

Execute the notebook in Google Colab

Train the model (model_1)

Evaluate and save results

Make sure to switch to GPU for faster training

📥 Save and Load the Model

To save the trained model:

model_1.save("tensorflow_model/saved_model")

To load the model later:

from tensorflow.keras.models import load_model
model_loaded = load_model("tensorflow_model/saved_model")
model_loaded.evaluate(val_ds)

📚 Learnings

This project reinforced key deep learning practices:

Manual CNN design

GPU acceleration

Image augmentation

Regularization

Evaluation and visualization
