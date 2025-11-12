# 🦺 PPE Compliance Detector

A simple Streamlit web app that detects **Personal Protective Equipment (PPE)** using a custom YOLOv5 model.  
It identifies safety gear like **hardhats, masks, and safety vests** and classifies the image as **Fully, Partially, or Non-Compliant**.

## ⚙️ Features
- Upload an image to detect workers and PPE items  
- Displays bounding boxes and labels on detected objects  
- Shows overall compliance status  
- Download annotated images for reports

#Link to live streamlit: https://ppedetectionfa2-tcwtafqwbi8gvk36czbycp.streamlit.app/

## 🧠 Model
The model (`best_ppe.pt`) was trained on a PPE dataset and loaded via **YOLOv5 (Ultralytics)**.

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py


