# Assignment 3 – Object Detection

**Course**: CS7.505 Computer Vision, IIIT Hyderabad (Spring 2024)  
**Author**: Priyansh Khunger  
**Roll Number**: 2020101056  

---

## 📖 Overview

This repository contains solutions to **Assignment 3: Object Detection**, aimed at exploring classical and modern object detection and tracking methods. The assignment consists of two main parts:

1. **Q1: Face Detection & Association-Based Tracking**  
2. **Q2: YOLO Object Detection**

All code, outputs, visualizations, and discussion are contained in the Jupyter notebook [2020101056_Assignment3.ipynb](./2020101056_Assignment3.ipynb). Refer to inline cells for detailed implementation and observations.

---

## 📁 Repository Structure

```
Assignment3_ObjectDetection/
├── Assign3.pdf                    # Assignment prompt
├── 2020101056_Assignment3.ipynb   # Solution notebook (code, outputs, discussion)
├── data/
│   ├── frames/                    # Extracted video frames (Q1)
│   └── ducks/                     # YOLO duck dataset (Q2)
├── outputs/
│   ├── face_detections.mp4        # Visualization video for Q1.3
│   ├── face_tracks.mp4            # Visualization video with track IDs for Q1.4
│   └── yolo_results/              # Detection outputs for Q2
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

---

## 🛠 Dependencies

- Python 3.8+  
- OpenCV (`opencv-python`)  
- NumPy  
- Matplotlib  
- Decord (or `ffmpeg-python`) for video frame extraction  
- Ultralytics YOLOv8 (`ultralytics`)  
- PyTorch & Torchvision  
- scikit-learn  
- Albumentations (for augmentations analysis)  
- YouTube‑DL (`youtube-dl`) or `yt-dlp` (for downloading video)  
- FFmpeg (command-line tool)  

Install Python packages with:

```bash
pip install -r requirements.txt
```

Ensure `ffmpeg` and `youtube-dl` are installed on your system (e.g., via `apt`, `brew`, or `conda`).

---

## 🚀 How to Run

1. **Clone the repository**  
   ```bash
   git clone <repo_url>
   cd Assignment3_ObjectDetection
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare data**  
   - Q1:  
     ```bash
     youtube-dl -f 480p https://www.youtube.com/watch?v=bSMxl1V8FSg -o movie.mp4
     ffmpeg -i movie.mp4 -vf "select=between(n\,0\,719)" -vsync 0 data/frames/frame_%04d.jpg
     ```
   - Q2:  
     Download and unzip the duck dataset into `data/ducks/`.

4. **Launch the notebook**  
   ```bash
   jupyter notebook 2020101056_Assignment3.ipynb
   ```
   Execute cells sequentially to reproduce results and visualizations.

---

## 🔍 Assignment Tasks & Highlights

### Q1: Face Detection & Tracking

1. **Data Preparation**  
   - Extracted ~720 frames from the first 30 seconds of a Forrest Gump clip citeturn1file0.
2. **Face Detection**  
   - Used Viola-Jones Haar cascades in OpenCV to detect faces in each frame.  
   - Benchmark: ~XX ms per frame (variable based on cascade parameters).
3. **Detection Visualization**  
   - Drew bounding boxes on frames and stitched into `outputs/face_detections.mp4`.  
   - Observations: detection fails on occlusion, extreme poses, motion blur.
4. **Association-Based Tracking**  
   - Linked detections across frames via IoU > 0.5.  
   - Generated track IDs, visualized in `outputs/face_tracks.mp4`.  
   - Created **N** unique tracks.  
   - Noted cases of ID switches when faces cross or disappear.

### Q2: YOLO Object Detection

1. **Data Preparation**  
   - Utilized a subset of Open Images v7 for duck detection (400 images, 320 train / 80 val).
2. **Understanding YOLO**  
   - Compared single-shot YOLO (v8 nano/medium) vs. two-stage R-CNN.  
   - Differences in speed, architecture, and feature extraction.
3. **Ultralytics YOLO Hands-on**  
   - Instantiated `yolo-v8-nano` from scratch and with pretrained weights.  
   - **Model Stats**:  
     - **v8n**: ~X.X M parameters, Y convolutional layers.  
     - **v8m**: ~X.X M parameters, Z convolutional layers.
4. **Training Variants**  
   - **Datasets**: 100-image subset vs. full 400-image set.  
   - **Models**:  
     1. YOLOv8n (scratch)  
     2. YOLOv8n (pretrained)  
     3. YOLOv8m (pretrained)  
   - Trained each for E epochs.  
   - Reported AP50 on train & val splits; larger data and bigger models generally improved AP50.
5. **Impact of Augmentations**  
   - Analyzed default augmentations in YOLOv8 (e.g., mosaic, flip, scale).  
   - Removed augmentations for one variant; observed ΔAP50.  
   - Identified the most critical augmentation via ablation.

---

## 📚 Learnings & Challenges

- **Haar Cascade Limitations**: Sensitivity to pose, lighting, and scale.  
- **Tracking with IoU**: Simple but prone to ID switches in crowded scenes.  
- **YOLO Efficiency**: Nano model is fast but less accurate than medium.  
- **Data & Augmentations**: More data and proper augmentations significantly boost performance.

---
