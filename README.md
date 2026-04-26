# 🚗 AI-Powered License Plate Recognition (ALPR)

A deep learning-based system designed to detect and recognize vehicle license plates from images and video streams. This project utilizes state-of-the-art YOLO (You Only Look Once) models for robust object detection and optical character recognition (OCR).

## 🌟 Key Features
- **Vehicle Detection:** Uses YOLOv8/v10 to identify cars and motorcycles in the frame.
- **Plate Localization:** A fine-tuned YOLO model specifically trained to crop license plates.
- **Character Recognition:** Extracts text from localized plates using advanced computer vision techniques.
- **Interactive UI:** Built with Streamlit for easy image uploads and real-time visualization.

## 🛠️ Technical Stack
- **Languages:** Python
- **Deep Learning:** Ultralytics (YOLOv8, YOLOv10)
- **Frameworks:** Streamlit (Frontend), OpenCV (Image Processing)
- **Data Handling:** NumPy, Pandas
- **Models Included:**
  - `license_plate_detector.pt`: Specialized for finding the plate.
  - `PlateReaderyolo.pt`: Specialized for character recognition.

## 📁 Project Structure
```text
├── app/
│   ├── app.py             # Main Streamlit application script
│   └── logo_ensam.png     # Institutional branding
├── models/
│   ├── license_plate_detector.pt
│   ├── PlateReaderyolo.pt
│   └── yolov8n.pt / yolov10n.pt
├── images/                # Sample input images
├── test/                  # Test dataset for validation
├── util.py                # Helper functions for processing and OCR
└── README.md              # Project documentation
