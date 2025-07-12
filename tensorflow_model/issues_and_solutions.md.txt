TensorFlow Cats vs Dogs Classifier

Documented list of challenges and how we resolved them while building and training the TensorFlow model.

🧐 Issue 1: Dataset not recognized as two classes

Problem: All images were in a single folder without cat/ and dog/ subdirectories

Fix: Used Python script to sort files into cat/ and dog/ folders based on filename prefix

🧵 Issue 2: Overfitting (training accuracy >> validation accuracy)

Problem: Model was memorizing training data and not generalizing

Fix:

Added data augmentation (flip, rotation, zoom)

Increased dropout to 0.4

Added L2 kernel regularization

Reduced epochs to 5

⚡ Issue 3: Slow training speed

Problem: CPU training was very slow with 25,000 images

Fix: Switched runtime to GPU in Colab (Runtime > Change runtime type > GPU)

Result: 3–5x faster model training

🚫 Issue 4: FileNotFoundError when saving plots

Problem: Tried saving to results/ folder that didn’t exist

Fix: Used os.makedirs("results", exist_ok=True) before saving with plt.savefig()

