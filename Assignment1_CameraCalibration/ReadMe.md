# Assignments 1 – Computer Vision Course (CS7.505, IIIT Hyderabad)

---

## Assignment 1 – Camera Calibration

This repository contains solutions to Assignment 1 for the Computer Vision course (CS7.505, IIIT Hyderabad, Spring 2024). The goal of this assignment is to understand and implement camera calibration using both manual and OpenCV-based methods, and analyze how real-world measurements map to images.

> **Author**: Priyansh Khunger  
> **Roll Number**: 2020101056  
> **Submitted as**: Jupyter Notebook

---

### 📝 Assignment Overview

The assignment is divided into three main parts:

#### ✅ Q1: Manual Camera Calibration (No External Libraries)
- **Chessboard Corner Detection**: Used OpenCV to detect internal corners.
- **Camera Calibration from Scratch**: 
  - Defined 3D world points assuming a known square size (2cm × 2cm).
  - Computed intrinsic and extrinsic parameters (translation vector, rotation matrix).
- **Wireframe Projection**: Used projection matrix to project a 3D wireframe onto the image.
- **Rotation Angles**: Extracted pan, tilt, and roll from the rotation matrix.

#### ✅ Q2: OpenCV-Based Calibration
- **Used `cv2.calibrateCamera()`** with appropriate flags to estimate intrinsic and extrinsic parameters.
- **Compared Results** with the manually implemented calibration in Q1.
- **Applied Calibration on a Second Image** (`assign1.jpg`) and analyzed distortions due to coplanar assumptions.
- **Computed Image of World Origin** from the calibration matrix and interpreted results.

#### ✅ Q3: Simulated Chessboard Movement
- **10cm Virtual Movement**: Simulated moving the chessboard rightwards along the ruler.
- **Wireframe Overlay**: Reconstructed wireframe at new location and analyzed realism.
- **Pattern Projection**: Reprojected the entire pattern inside the predicted region and verified consistency.

---

### 🔍 Key Takeaways

- Hands-on understanding of how real-world measurements and coordinate systems map into image space.
- Gained clarity on intrinsic matrix parameters and their real-world implications.
- Understood how calibration accuracy depends on scene geometry (coplanar points vs. diverse 3D structure).
- Learned challenges of noise, corner detection errors, and sensitivity of projection matrices.

---

### 🖼️ Outputs and Visuals

All major steps (corner detection, wireframe overlays, matrix outputs, etc.) are visualized in the Jupyter Notebook:
- `Camera_Calibration.ipynb`

Make sure to view it for:
- Detected corners and calibration outputs
- 3D-to-2D projections and visual overlays
- Detailed code explanations and observations

---

### 🛠️ Dependencies

- Python 3.8+  
- OpenCV (`opencv-python`)  
- NumPy  
- Matplotlib (for visualization)

Install requirements with:

```bash
pip install opencv-python numpy matplotlib
```

---

### 📁 File Structure

```
Assignment1_CameraCalibration/
├── Camera_Calibration.ipynb  # Main solution notebook
├── Assign1.pdf               # Assignment instructions
├── calib-object.jpg          # Image used in Q1 and Q2
├── assign1.jpg               # Image used in Q2 and Q3
└── README.md                 # This file
```

---

### 🚀 How to Run

Open the notebook:

```bash
jupyter notebook Camera_Calibration.ipynb
```

Follow the cells sequentially to run and visualize each task.

---

### 💡 Observations & Learnings

- Manual calibration reinforced matrix multiplication and projection theory.
- OpenCV calibration was more efficient but requires careful configuration (e.g., distortion flags).
- Visualization of projected 3D points onto 2D image revealed calibration quality vividly.

---
