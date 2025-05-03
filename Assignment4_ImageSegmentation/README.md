# Assignment 4 – Segmentation and CLIP

**Course**: CS7.505 Computer Vision, IIIT Hyderabad (Spring 2024)  
**Author**: Priyansh Khunger  
**Roll Number**: 2020101506  

---

## 📖 Overview

This repository contains solutions to **Assignment 4: Segmentation and CLIP**, demonstrating:

1. **Q1: Image Segmentation with UNet**  
   - Training and evaluating a standard UNet on a subset of Cityscapes.  
   - Investigating the importance of skip connections.  
   - Correctly implementing class-wise IoU and computing mean IoU (mIoU).

2. **Q2: Contrastive Language–Image Pretraining (CLIP) vs. ImageNet**  
   - Loading and comparing ResNet-50 pretrained on ImageNet and CLIP’s RN50 visual encoder.  
   - Zero-shot image classification on ImageNet classes using CLIP.  
   - Case studies contrasting CLIP and ImageNet models on selected ImageNet categories.  
   - Evaluating FP16 vs. FP32 inference: timing, output consistency, and GPU memory usage.

All code, results, and visualizations are in the notebook:  
**2020101506_Assignment4.ipynb**

---

## 📁 Repository Structure

```
Assignment4_Segmentation_CLIP/
├── Assign4.pdf                   # Assignment prompt
├── 2020101506_Assignment4.ipynb  # Solution notebook (code + outputs + discussion)
└── README.md                     # This file
```

---

## 🛠 Dependencies

- Python 3.8+  
- PyTorch & Torchvision  
- OpenCV (`opencv-python`)  
- NumPy  
- Matplotlib  
- CLIP (`git+https://github.com/openai/CLIP.git`)  
- scikit-learn  

Install via:

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-learn
pip install git+https://github.com/openai/CLIP.git
```

---

## 🚀 How to Run

1. **Clone the repository**  
   ```bash
   git clone <repo_url>
   cd Assignment4_Segmentation_CLIP
   ```

2. **Install dependencies** (see above).  

3. **Launch the notebook**  
   ```bash
   jupyter notebook 2020101506_Assignment4.ipynb
   ```

4. **Execute all cells** in order to reproduce training, evaluation, and plots.

---

## 🔍 Assignment Tasks & Findings

### Q1 – Image Segmentation with UNet

- **Data Preparation**: Loaded a Cityscapes subset, defined label clusters via K-means, and built a custom dataset.  
- **Standard UNet**: Trained for 1 epoch; initial IoU calculated over a validation batch.  
- **No-Skip UNet**: Removed encoder–decoder skip connections, halved channel dimensions, retrained for the same number of epochs.  
- **Metric Fix**: Identified bug in original IoU metric; implemented class-wise IoU and computed mIoU per image and over the set.  
- **Results**:  
  - *Quantitative*: The UNet **with skip connections** yielded a higher mIoU compared to the **no-skip** variant (see notebook for exact values).  
  - *Qualitative*: Visual overlays show sharper segment boundaries and more accurate object shapes when skip connections are used.

### Q2 – CLIP vs. ImageNet Pretraining

- **Model Setup**:  
  - **ImageNet RN50**: `torchvision.models.resnet50(pretrained=True)`  
  - **CLIP RN50**: `clip.load('RN50')` – compared architecture differences (stem, pooling, bottlenecks).  
- **Zero-Shot Classification**: Generated true-label scores for ImageNet classes using CLIP’s text prompts; validated on sample images.  
- **Case Studies**: For 10 diverse ImageNet classes:  
  - Identified 2 images per class where **CLIP** outperforms ImageNet RN50.  
  - Identified 1 image per class where **ImageNet RN50** outperforms CLIP.  
- **FP16 vs. FP32**:  
  - Measured inference time (100 runs) for both precisions; reported mean ± std.  
  - Compared top-5 probabilities on 5 test images; outputs remain consistent across precisions.  
  - Profiling GPU memory usage for forward pass shows reduced memory footprint with FP16.

---

## 📚 Learnings & Challenges

- **Skip Connections**: Crucial for preserving spatial detail in segmentation.  
- **Metric Implementation**: Importance of per-class IoU in evaluating dense predictions.  
- **CLIP Strengths**: Robust to visual variations and open-vocabulary; better zero-shot on many classes.  
- **Precision vs. Performance**: FP16 inference reduces memory usage with negligible accuracy drop.

---