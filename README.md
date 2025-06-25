# 🚗 Velocit - Vehicle Detection, Speed Estimation & License Plate Recognition

**Velocit** is an AI-powered application that performs real-time vehicle detection, speed estimation, and license plate recognition using YOLOv8 and EasyOCR. Built with Streamlit for an interactive interface, it allows users to upload traffic videos and view annotated outputs with bounding boxes, speeds, and license numbers.

---

## 🔍 Features

- 🚘 **Vehicle Detection** using YOLOv8
- ⏱️ **Speed Estimation** between video frames
- 🔢 **License Plate Detection & OCR** with EasyOCR
- 🧠 Streamlit-based UI for seamless video upload and processing
- 📁 Clean output directory for annotated results

---

## 🛠 Tech Stack

- **Python**
- **YOLOv8** (Ultralytics)
- **OpenCV**
- **EasyOCR**
- **Streamlit**

---

## 📂 Folder Structure
vehicle_detection_app/

│

├── app.py # Streamlit app

├── yolov8_core.py # Detection and speed logic

├── yolov8n.pt # YOLOv8 model (locally stored)

├── test_videos/ # Input video files (excluded from Git)

├── outputs/ # Processed and annotated video outputs

├── requirements.txt

├── .gitignore

└── README.md

## 🚀 Getting Started

1. **Clone the repository**  
```bash
git clone https://github.com/sreethiii/Velocit.git
cd Velocit
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the app**
```bash
streamlit run app.py
```
