# AgriYOLO
YOLO-based plant health detection system for identifying healthy and diseased crops from real-world agricultural images.

# 🌱 AgriYOLO

**AgriYOLO** is a YOLO-based plant health detection system designed to classify crops as healthy or diseased using real-world agricultural images.  
The project aims to support smart and sustainable farming through early disease detection.

---

## 🚜 Problem Statement
Crop diseases often go unnoticed until they cause significant yield loss.  
Manual inspection is time-consuming and impractical for large farms.

AgriYOLO automates plant health monitoring using computer vision, enabling faster and more accurate detection.

---

## 🧠 Solution Overview
- Uses **YOLO (You Only Look Once)** for fast and efficient image-based classification
- Trained on real agricultural field images
- Classifies crops into **healthy** and **diseased** categories
- Suitable for real-time applications

---

## 🛠️ Tech Stack
- Python  
- YOLO (Ultralytics)  
- PyTorch  
- OpenCV  
- NumPy  

---

## 📂 Dataset
- Real-world crop images
- Organized into training and testing sets
- Includes healthy and diseased plant samples  

> Dataset is included for academic and demonstration purposes.

---

## ▶️ How to Run

1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train the model
python src/train.py

3️⃣ Predict / Test
python src/predict.py
