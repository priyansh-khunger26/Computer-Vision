
## Assignment 2 – Image Classification on MNIST

**Course**: CS7.505 Computer Vision, IIIT Hyderabad (Spring 2024)  
**Author**: Priyansh Khunger  
**Roll Number**: 2020101056  

---

### 📖 Overview

This repository contains solutions to **Assignment 2: Image Classification**, exploring two paradigms for handwritten-digit recognition on MNIST:

1. **Q1: SIFT + Bag-of-Visual-Words + Linear SVM**  
2. **Q2: Deep Learning Approaches**  
   - **LeNet CNN** (with modular code and hyperparameter sweeps)  
   - **Extended CNN** (doubling convolutional layers, varying training set size)  
   - **Transformer Encoder** (ViT-style classification)

All code, results, plots, and observations are contained in the Jupyter notebook [2020101056_Assignment2.ipynb](./2020101056_Assignment2.ipynb).

---

### 📁 Repository Structure

```
.
├── Assign2.pdf                   # Assignment prompt
├── 2020101056_Assignment2.ipynb  # Solution notebook (code + outputs + discussion)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

### 🛠 Dependencies

- Python 3.8+  
- OpenCV (`opencv-python`)  
- scikit-learn  
- NumPy  
- Matplotlib  
- PyTorch & Torchvision  
- (Optional) Weights & Biases (`wandb`) for logging  

Install all required packages via:

```bash
pip install -r requirements.txt
```

---

### 🚀 How to Run

1. Clone this repo.  
2. Create & activate a virtual environment.  
3. `pip install -r requirements.txt`  
4. Launch Jupyter and open the notebook:

   ```bash
   jupyter notebook 2020101056_Assignment2.ipynb
   ```

5. Run the cells **in order**. All results (plots, tables, accuracy numbers) will appear inline.

---

### 🔍 Assignment Tasks & Highlights

#### Q1 – SIFT + BoVW + Linear SVM

1. **SIFT Feature Extraction**  
   - Implemented keypoint detection and descriptor computation using OpenCV’s SIFT.  
2. **Visual Vocabulary Construction**  
   - Performed k-means clustering on descriptors for visual words.  
   - Explored 6 cluster sizes: 5, 20, 50, 100, 200, 500.  
3. **Image Representation & SVM Training**  
   - Represented each image as a histogram over visual words.  
   - Trained a one-vs-rest linear SVM for 10-way classification.  
4. **Results & Observations**  
   - **Accuracy vs. Vocabulary Size**: Rises from ~70 % and plateaus around **88 %** at 500 clusters.  
   - **Hyperparameter Sweep**: Best SVM test accuracy of **89 %** with C=1.0 and 200 clusters.  

---

#### Q2 – CNNs and Transformers

1. **Modular LeNet CNN** – Data loader, model, and trainer separation with W&B logging.  
2. **Hyperparameter Sweep** – Batch size, learning rate (0.01, 0.001), optimizer (SGD/Adam).  
   - Best CNN test accuracy: **99.17 %**.  
3. **Extended CNN** – Two extra conv layers; test accuracy improved to **99.31 %**.  
4. **Training Set Size** – Subsets [0.6 K, 1.8 K, 6 K, 18 K, 60 K]; accuracy scales from **96.5 %** to **99.1 %**.  
5. **Transformer Encoder** – ViT-style model; **95.2 %** on 6K, **98.7 %** on 60K.  

---

### 📊 Summary of Results

| Method                     | Small Data (6K) | Full Data (60K) | Best Test Acc. |
|----------------------------|-----------------|-----------------|----------------|
| SIFT-BoVW + Linear SVM     | –               | –               | **89 %**       |
| LeNet CNN                  | 94.8 %          | 99.17 %         | **99.17 %**    |
| Extended CNN               | 95.5 %          | 99.31 %         | **99.31 %**    |
| Transformer Encoder        | 95.2 %          | 98.70 %         | **98.70 %**    |

*(Full plots and detailed analysis in the notebook.)*

---

### 📚 Learnings & Challenges

- Hand-crafted features vs. end-to-end learning differences.  
- Model complexity trade-offs and diminishing returns.  
- Importance of data volume for transformer models.  
- Sensitivity of convergence to hyperparameters.  

---
