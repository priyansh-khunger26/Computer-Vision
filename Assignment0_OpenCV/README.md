# Assignment 0 – OpenCV Fundamentals

**Course** : CS7.505 Computer Vision, IIIT Hyderabad (Spring 2024)  
**Author** : Priyansh Khunger  
**Roll No.** : 2020101056  

---

## 📑 Quick Description

The first assignment in the course is a gentle, hands-on dive into
OpenCV’s Python API.   
You will:

| Task | Topic | What you practise |
|------|-------|-------------------|
| `Q1` | **Image I/O** | Read → resize → save an image |
| `Q2` | **Video I/O** | Capture webcam stream & save to disk |
| `Q3` | **Drawing Primitives** | Lines, rectangles, circles, text |
| `Q4` | **Trackbars & Blending** | Real-time alpha–blending with sliders |
| `Q5` | **Mouse Callbacks** | Drag-to-draw interactive rectangles |
| `Q6` | **Extra (Edge & Contour Explorer)** | Live Canny/threshold sliders, contour overlay & bounding-box stats |

Every task is implemented twice:

* **Script version** (`Q1.py`, … `Q6.py`) – quick demo runnable from the terminal.  
* **Notebook version** (`Task1.ipynb`, … `Task6.ipynb`) – same logic with
  inline explanations, figures and discussion.

> **Why two versions?**  
> - Scripts are great for CLI automation.  
> - Notebooks are ideal for step-by-step learning and reports.

---

## 🗂️ Repository Tree

```
Assignment0_OpenCV/
├── data/
│   ├── input.jpg                # Sample test image for Q1, Q3, Q4
│   └── logo.png                 # 2nd image for alpha-blending demo
├── Q1.py                        # Task-1 script – image I/O
├── Q2.py                        # Task-2 script – video capture
├── Q3.py                        # Task-3 script – drawing primitives
├── Q4.py                        # Task-4 script – blending w/ trackbars
├── Q5.py                        # Task-5 script – mouse callbacks
├── Q6.py                        # Task-6 script – edge & contour explorer
├── Task1.ipynb                  # Notebook version of Q1
├── Task2.ipynb                  # Notebook version of Q2
├── Task3.ipynb                  # Notebook version of Q3
├── Task4.ipynb                  # Notebook version of Q4
├── Task5.ipynb                  # Notebook version of Q5
├── Task6.ipynb                  # Notebook version of Q6 (extra)
└── README.md                    # ← you are here
```

---

## 🔧 Setup

| Requirement | Version |
|-------------|---------|
| Python | 3.8 or newer |
| OpenCV-Python | ≥ 4.5 |
| NumPy | ≥ 1.19 |
| Matplotlib *(notebooks)* | ≥ 3.3 |
| Jupyter Notebook | any recent version |

**Install in one shot**

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 1. Plain-Python scripts

```bash
# Image read / resize / save (Q1)
python Q1.py --img data/input.jpg --scale 0.5

# Live webcam capture (Q2)
python Q2.py --device 0 --out out_video.mp4

# Edge-and-Contour explorer (Q6 extra)
python Q6.py --img data/input.jpg
```

Every script has `--help` for optional flags.

### 2. Interactive notebooks

```bash
jupyter notebook        # then open Task?.ipynb
```

Each notebook:

* Walks through the code line-by-line  
* Shows intermediate images inline  
* Ends with a short “Take-aways” markdown cell

---

## 🌟 What You Learn

* Core OpenCV data types (`cv::Mat` ↔ NumPy)
* Coordinate conventions & color channels  
* Real-time GUI elements: windows, trackbars, mouse callbacks  
* Reading / writing common formats (PNG, JPG, MP4, AVI)  
* Simple UX tricks: FPS overlay, live parameter sliders  
* Edge detection ➜ contour extraction ➜ bounding box & area stats (Task 6)

---

## ✍️ Observations & Notes

* **Color order** – Remember OpenCV uses BGR, not RGB.  
* **Webcam FPS** – `cv2.CAP_DSHOW` on Windows or `cv2.CAP_V4L2` on Linux
  can boost frame-rate if your default backend is slow.  
* **Trackbars** – They return integers only; scale values to floats
  when you need fine-grained alpha or threshold control.  
* **Contour hierarchy** – Useful for differentiating nested shapes in Task 6.  

---
