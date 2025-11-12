import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import tempfile

# Load model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best_ppe.pt')

model = load_model()

# Compliance map
compliance_map = {
    'Hardhat': '✅ Compliant',
    'Safety Vest': '✅ Compliant',
    'Mask': '✅ Compliant',
    'NO-Hardhat': '❌ Missing Hardhat',
    'NO-Safety Vest': '❌ Missing Vest',
    'NO-Mask': '❌ Missing Mask',
    'Person': '👤 Worker',
    'machinery': '⚙️ Machinery',
    'vehicle': '🚗 Vehicle',
    'Safety Cone': '🟠 Cone'
}

# UI
st.set_page_config(page_title="PPE Compliance Detector", layout="centered")
st.title("🛠️ PPE Compliance Detection")
st.markdown("Upload a construction site image to detect workers and check PPE compliance.")

uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model(image)
        df = results.pandas().xyxy[0]

        # Annotate image
        annotated_img = np.array(image)
        for _, row in df.iterrows():
            label = row['name']
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            color = (0, 255, 0) if 'NO-' not in label else (255, 0, 0)
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_img, compliance_map.get(label, label), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        st.image(annotated_img, caption="🧠 Detection Result", use_column_width=True)

        # Summary
        st.subheader("📋 Compliance Summary")
        for label in df['name'].unique():
            count = (df['name'] == label).sum()
            st.write(f"{compliance_map.get(label, label)}: {count}")
