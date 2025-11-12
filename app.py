# app.py (safe version) — drop-in replacement
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import io
import os

st.set_page_config(page_title="PPE Compliance Detector", layout="centered")
st.title("🛠️ PPE Compliance Detection")

# small map kept from your original
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

MODEL_PATH = "best_ppe.pt"  # ensure this exists in repo root or change to URL

# lazy cv2 importer (won't crash app if cv2 / native libs missing)
def try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

cv2 = try_import_cv2()

# safe model loader (gives readable errors)
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'. Put your best.pt in repo root or provide a URL and change MODEL_PATH.")
    try:
        # load via torch.hub (ultralytics yolov5)
        return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
    except Exception as e:
        raise RuntimeError("Model load failed: " + str(e))

# try to load model, show friendly message if fails
model = None
try:
    model = load_model()
except Exception as e:
    st.error("Model load: " + str(e))
    st.info("You can still upload an image, but detection won't run until model is loaded successfully.")
    # we do not stop here so you can still view UI

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

def draw_boxes_pil(img_pil, df):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        color = (0,255,0) if not str(label).startswith('NO-') else (255,0,0)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        draw.text((x1, max(0,y1-12)), compliance_map.get(label, label), fill=color, font=font)
    return img_pil

def draw_boxes_cv2(img_np, df, cv2):
    out = img_np.copy()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        color = (0,255,0) if not str(label).startswith('NO-') else (0,0,255)  # BGR
        cv2.rectangle(out, (x1,y1),(x2,y2), color, 2)
        cv2.putText(out, compliance_map.get(label, label), (x1, max(10,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

# --- Compliance helper (small drop-in function) ---
def get_compliance(detections):
    """Return (status_label, missing_list). detections = list of class names."""
    names = set(detections)
    required = ['Hardhat', 'Mask', 'Safety Vest']

    # presence / explicit NO- flags
    present = {p: (p in names) for p in required}
    explicit_no = {p: (f"NO-{p}" in names) for p in required}

    # Status logic
    if all(present.values()) and not any(explicit_no.values()):
        status = "🟢 Fully Compliant"
    elif all(explicit_no.values()) and not any(present.values()):
        status = "🔴 Non-Compliant"
    else:
        status = "🟡 Partially Compliant"

    # Determine what's missing (either explicit NO- or simply not detected)
    missing = [p for p in required if explicit_no[p] or not present[p]]

    return status, missing
# --- end compliance helper ---

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.warning("Model not loaded — detection skipped. Fix model (best.pt) or check logs.")
    else:
        with st.spinner("Running detection..."):
            try:
                results = model(img)
                df = results.pandas().xyxy[0]
            except Exception as e:
                st.error("Inference failed: " + str(e))
                st.stop()

        # ---------- PLACE WHERE COMPLIANCE CHECK IS RUN ----------
        # use model outputs to decide Fully / Partially / Non-compliant
        detections = df['name'].tolist() if not df.empty else []
        status, missing = get_compliance(detections)
        st.markdown(f"### Compliance Status: **{status}**")
        if missing:
            st.write("**Missing / Not detected:**", ", ".join(missing))
        else:
            st.write("All required PPE detected ✅")
        # ---------------------------------------------------------

        # annotate and display
        if cv2 is not None:
            # convert PIL to numpy BGR for cv2 drawing
            arr = np.array(img)  # RGB
            bgr = arr[..., ::-1]
            annotated_bgr = draw_boxes_cv2(bgr, df, cv2)
            annotated_rgb = annotated_bgr[..., ::-1]
            st.image(annotated_rgb, caption="🧠 Detection Result (cv2)", use_column_width=True)
            # optional download
            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")
        else:
            # PIL drawing fallback
            annotated = draw_boxes_pil(img.copy(), df)
            st.image(annotated, caption="🧠 Detection Result (PIL fallback)", use_column_width=True)
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")

        # summary
        st.subheader("Compliance summary")
        counts = df['name'].value_counts()
        for label, cnt in counts.items():
            st.write(f"{compliance_map.get(label, label)}: {cnt}")
