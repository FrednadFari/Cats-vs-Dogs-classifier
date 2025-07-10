# Issues & Solutions – PyTorch Cat vs. Dog Classifier

### 🧩 Issue 1: ImageFolder only shows one class ['train']
- **Cause:** Images were not stored inside `cat/` and `dog/` folders.
- **Fix:** Created a script to move files into `data/train/sorted/cat/` and `data/train/sorted/dog/`.

---

### 🧩 Issue 2: Model overfit too fast (loss dropped to zero)
- **Cause:** High learning rate (0.001) + small dataset or simple task
- **Fix:** Lowered learning rate to `0.0001` and monitored with loss curve.

---

### 🧩 Issue 3: `NotImplementedError` in `forward()`
- **Cause:** Custom model class was missing `def forward(self, x):`
- **Fix:** Added correct `forward()` method returning `self.model(x)`

---

### 🧩 Issue 4: `plt.savefig(...)` raised FileNotFoundError
- **Cause:** Tried saving plot to a folder that didn’t exist yet
- **Fix:** Used `os.makedirs("pytorch_model/results", exist_ok=True)` before saving

---

### 🧩 Issue 5: Poor predictions despite low loss
- **Cause:** Not enough training (only 3 epochs) or insufficient augmentation
- **Fix:** Trained for 20 epochs; improved predictions seen in sample output

---

✅ Documenting these will show you're not just building models — you're solving problems like a real developer.
