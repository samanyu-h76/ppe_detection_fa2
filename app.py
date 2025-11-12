# app.py — updated (keeps your original logic but safer & improved UI)
import streamlit as st
from PIL import Image
import io
import numpy as np
import pandas as pd
from typing import Optional

st.set_page_config(page_title="PPE Compliance Detector", layout="wide")
st.title("🛠️ PPE Compliance Detection (safe + live modes)")

# ---------- User-config / mapping (kept from your original) ----------
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

MODEL_PATH = "best_ppe.pt"  # ensure this file is present in repo root for live mode

# ---------- Helpers ----------
def safe_try_import_cv2():
    """Try import cv2. Return module or None if import fails."""
    try:
        import cv2 as _cv2
        return _cv2
    except Exception:
        return None

def bgr_to_rgb(img_np):
    """If img is BGR (cv2), convert to RGB for display. Expect uint8."""
    if img_np is None:
        return None
    if img_np.ndim == 3 and img_np.shape[2] == 3:
        # assume BGR -> convert to RGB
        return img_np[..., ::-1]
    return img_np

def draw_boxes_cv2(cv2, img_np, detections_df):
    """Draw boxes using cv2 on numpy image; returns numpy image (BGR)."""
    img_out = img_np.copy()
    for _, row in detections_df.iterrows():
        label = row.get('name', '')
        x1, y1, x2, y2 = map(int, [row.get('xmin', 0), row.get('ymin', 0), row.get('xmax', 0), row.get('ymax', 0)])
        color = (0, 255, 0) if not str(label).startswith('NO-') else (0, 0, 255)  # BGR: green/red
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
        display_label = compliance_map.get(label, label)
        cv2.putText(img_out, display_label, (x1, max(10, y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img_out

def draw_boxes_pil(img_pil, detections_df):
    """Fallback drawing with PIL (safe, no cv2)."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for _, row in detections_df.iterrows():
        label = row.get('name', '')
        x1, y1, x2, y2 = map(int, [row.get('xmin', 0), row.get('ymin', 0), row.get('xmax', 0), row.get('ymax', 0)])
        color = (0, 255, 0) if not str(label).startswith('NO-') else (255, 0, 0)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        display_label = compliance_map.get(label, label)
        draw.text((x1, max(0, y1-15)), display_label, fill=color, font=font)
    return img_pil

def decide_overall(df, conf_threshold=0.25):
    """Simple heuristic to decide overall compliance from detections DataFrame."""
    # if no detections -> unknown
    if df is None or df.empty:
        return "Unknown (no detections)", {}
    names = df[df['confidence'] >= conf_threshold]['name'].tolist()
    # check presence of good PPE
    goods = ['Hardhat', 'Mask', 'Safety Vest']
    details = {}
    present = 0
    for g in goods:
        if g in names:
            details[g] = 'present'
            present += 1
        elif f"NO-{g}" in names:
            details[g] = 'explicitly_not_worn'
        else:
            details[g] = 'missing'
    if present == len(goods):
        overall = "🟢 Fully Compliant"
    elif present == 0:
        overall = "🔴 Non-Compliant"
    else:
        overall = "🟡 Partially Compliant"
    return overall, details

# ---------- UI controls ----------
col1, col2 = st.columns([3,1])
with col2:
    mode = st.radio("Mode", ["Live (local/dev only)", "Upload image + CSV (safe)", "Upload image only (no CSV)"], index=1)
    conf_thr = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.01)
    show_raw = st.checkbox("Show raw detections table", value=False)
    download_outputs = st.checkbox("Offer downloads", value=True)

with col1:
    st.write("Upload an image (jpg / png). For safe mode, also upload the CSV produced by your training/Colab run.")
    uploaded_image = st.file_uploader("Image", type=['jpg','jpeg','png'])
    if mode == "Upload image + CSV (safe)":
        uploaded_csv = st.file_uploader("Detections CSV (columns: name, confidence, xmin, ymin, xmax, ymax)", type=['csv'])
    else:
        uploaded_csv = None

process = st.button("Run detection / evaluate compliance")

# ---------- guarded model loader ----------
@st.cache_resource(show_spinner=False)
def load_model_guarded(path: str = MODEL_PATH):
    """
    Try to load model with torch.hub. If torch/cv2 not available or model missing,
    raise an informative exception. Keep import inside function.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Upload it or put it in repo root as {path}.")
    try:
        import torch
    except Exception as e:
        raise RuntimeError("Torch not available in environment: " + str(e))
    # load YOLOv5 custom using hub (may attempt network if not cached)
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
        return model
    except Exception as e:
        raise RuntimeError("Failed to load model via torch.hub: " + str(e))

# ---------- Processing ----------
if process:
    if uploaded_image is None:
        st.error("Upload an image first.")
        st.stop()

    # read image as PIL
    try:
        img_pil = Image.open(uploaded_image).convert("RGB")
    except Exception as e:
        st.error("Failed to read image: " + str(e))
        st.stop()

    detections_df = pd.DataFrame(columns=['name','confidence','xmin','ymin','xmax','ymax'])  # default empty

    if mode == "Upload image + CSV (safe)":
        if uploaded_csv is None:
            st.error("Please upload the CSV for this image.")
            st.stop()
        try:
            df = pd.read_csv(uploaded_csv)
            # normalize: ensure cols exist
            expected = {'name','confidence','xmin','ymin','xmax','ymax'}
            if not expected.issubset(set(df.columns)):
                st.warning("CSV missing some expected columns — trying to proceed anyway.")
            detections_df = df
            st.success("CSV loaded.")
        except Exception as e:
            st.error("Could not load CSV: " + str(e))
            st.stop()

    elif mode == "Upload image only (no CSV)":
        st.info("No CSV provided — fallback to showing the image. For detection you must use Live or generate CSV from Colab.")
        detections_df = pd.DataFrame()  # empty

    else:  # Live mode
        # try to import cv2 (only now)
        cv2 = safe_try_import_cv2()
        if cv2 is None:
            st.error("cv2 not available in this environment. Live mode requires OpenCV (or use 'Upload image + CSV' mode).")
            st.stop()
        # load model
        try:
            model = load_model_guarded()
        except Exception as e:
            st.error("Model load failed: " + str(e))
            st.stop()

        # run inference
        with st.spinner("Running model inference..."):
            try:
                results = model(img_pil)  # yolov5 hub returns results
                # results.pandas().xyxy[0] should give df
                detections_df = results.pandas().xyxy[0]
                # ensure expected numeric columns
                for c in ['xmin','ymin','xmax','ymax','confidence']:
                    if c not in detections_df.columns:
                        detections_df[c] = 0
            except Exception as e:
                st.error("Inference failed: " + str(e))
                st.stop()

    # decide compliance
    overall, details = decide_overall(detections_df, conf_thr)
    st.markdown(f"### Compliance result: **{overall}**")

    # draw annotated image
    if not detections_df.empty:
        # prefer cv2 if available to draw (fast), else PIL fallback
        cv2 = safe_try_import_cv2()
        if cv2 is not None:
            # we must convert PIL->np BGR for cv2 drawing
            img_np = np.array(img_pil)  # RGB
            img_np_bgr = img_np[..., ::-1]  # BGR
            annotated_bgr = draw_boxes_cv2(cv2, img_np_bgr, detections_df)
            annotated_rgb = bgr_to_rgb(annotated_bgr)
            # show with st.image (expects RGB)
            st.image(annotated_rgb, caption="🧠 Detection result", use_column_width=True)
            # allow download: convert numpy->bytes
            if download_outputs:
                pil_out = Image.fromarray(annotated_rgb.astype('uint8'))
                buf = io.BytesIO()
                pil_out.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download annotated image (PNG)", data=buf, file_name="annotated.png", mime="image/png")
        else:
            # PIL fallback drawing
            annotated = draw_boxes_pil(img_pil.copy(), detections_df)
            st.image(annotated, caption="🧠 Detection result (PIL)", use_column_width=True)
            if download_outputs:
                buf = io.BytesIO()
                annotated.save(buf, format="PNG")
                buf.seek(0)
                st.download_button("Download annotated image (PNG)", data=buf, file_name="annotated.png", mime="image/png")
    else:
        st.image(img_pil, caption="Uploaded image (no boxes)", use_column_width=True)

    # show table summarising counts
    if not detections_df.empty and len(detections_df) > 0:
        summary = detections_df['name'].value_counts().rename_axis('label').reset_index(name='count')
        # map to friendly text
        summary['friendly'] = summary['label'].map(compliance_map).fillna(summary['label'])
        st.subheader("📋 Compliance Summary (counts)")
        st.table(summary[['friendly','count']].rename(columns={'friendly':'label'}).set_index('label'))

    if show_raw:
        st.subheader("Raw detections (first rows)")
        st.dataframe(detections_df.head(50))

    # show details dict
    st.write("Per-PPE details:", details)
