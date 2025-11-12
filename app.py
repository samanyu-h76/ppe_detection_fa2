# app.py
import os
import streamlit as st
import torch
from PIL import Image
import numpy as np
import pandas as pd

st.set_page_config(page_title="PPE Compliance Detector", layout="wide")

MODEL_PATH = "best_ppe.pt"  # put best_ppe.pt in the same folder as app.py

# --- helper: compliance rules (edit if your classes use other names) ---
# This dictionary defines pairs: good label -> corresponding "NO-" label
PPE_PAIRS = {
    "Hardhat": "NO-Hardhat",
    "Mask": "NO-Mask",
    "Safety Vest": "NO-Safety Vest",
}

# fallback synonyms (lowercase mapping)
SYNONYMS = {
    "hardhat": "Hardhat",
    "helmet": "Hardhat",
    "mask": "Mask",
    "safety vest": "Safety Vest",
    "vest": "Safety Vest",
    "no-hardhat": "NO-Hardhat",
    "no-mask": "NO-Mask",
    "no-safety vest": "NO-Safety Vest",
    "no-vest": "NO-Safety Vest",
}

@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model file '{path}' not found. Upload it to the repo or same folder.")
        st.stop()
    # using torch.hub load of ultralytics - will load custom model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
    return model

def normalize_name(name):
    """Map detected name to canonical PPE names if possible."""
    if not isinstance(name, str):
        return name
    n = name.strip()
    if n in PPE_PAIRS or n in PPE_PAIRS.values():
        return n
    nl = n.lower()
    return SYNONYMS.get(nl, n)

def aggregate_detections(df, conf_threshold=0.25):
    """
    df: results.pandas().xyxy[0] DataFrame
    returns: dict -> {class_name: best_conf}
    """
    agg = {}
    if df is None or df.empty:
        return agg
    # model might name confidence column 'confidence' or 'conf' or 'confidence' depending on version
    conf_col = None
    for c in ["confidence", "conf", "score"]:
        if c in df.columns:
            conf_col = c
            break
    if conf_col is None:
        # fallback: hope 'confidence' exists; if not, set 1.0
        conf_col = df.columns[4] if len(df.columns) > 4 else None

    for _, row in df.iterrows():
        raw_name = row.get("name", "")
        conf = float(row[conf_col]) if conf_col is not None else 1.0
        name = normalize_name(raw_name)
        # ignore extremely low confidences
        if conf < conf_threshold:
            continue
        if name not in agg or conf > agg[name]:
            agg[name] = conf
    return agg

def decide_compliance(agg):
    """
    agg: dict of class->conf
    return (status, details)
    status: "Fully Compliant" / "Partially Compliant" / "Non-Compliant"
    details: dict with per-item decisions
    """
    details = {}
    # for each required PPE (keys of PPE_PAIRS), check presence/conflict
    required = list(PPE_PAIRS.keys())
    present_count = 0
    missing_count = 0
    not_worn_count = 0

    for good in required:
        bad = PPE_PAIRS[good]
        good_conf = agg.get(good, 0.0)
        bad_conf = agg.get(bad, 0.0)

        # decide based on which one has higher confidence
        # threshold logic: if good_conf >= 0.4 => considered present, unless bad_conf significantly higher
        # if both present but bad_conf > good_conf*1.1 -> prefer bad
        chosen = None
        chosen_conf = 0.0

        if good_conf == 0 and bad_conf == 0:
            chosen = "missing"
            details[good] = {"status": "missing", "good_conf": good_conf, "bad_conf": bad_conf}
            missing_count += 1
            continue

        # choose the label with higher confidence
        if good_conf >= bad_conf:
            chosen = "present"
            chosen_conf = good_conf
        else:
            chosen = "not_worn"
            chosen_conf = bad_conf

        # small sanity bump: if both are close (<0.05 diff), favor 'present' (avoid false negative)
        if abs(good_conf - bad_conf) < 0.05 and good_conf > 0:
            chosen = "present"
            chosen_conf = good_conf

        if chosen == "present":
            present_count += 1
            details[good] = {"status": "present", "good_conf": good_conf, "bad_conf": bad_conf}
        else:
            not_worn_count += 1
            details[good] = {"status": "not_worn", "good_conf": good_conf, "bad_conf": bad_conf}

    # decide overall
    if present_count == len(required):
        overall = "🟢 Fully Compliant"
    elif present_count == 0:
        overall = "🔴 Non-Compliant"
    else:
        overall = "🟡 Partially Compliant"
    return overall, details

# --- UI ---
st.title("🦺 PPE Compliance Detector (Image-level)")

model = load_model()

col1, col2 = st.columns([2, 1])

with col2:
    st.write("### Settings")
    conf_thresh = st.slider("Detection confidence threshold", 0.05, 0.7, 0.25, 0.01)
    show_raw = st.checkbox("Show raw detections table", value=False)

with col1:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Running detection..."):
            results = model(img)  # model handles PIL Image
            # annotated image
            rendered = results.render()[0]  # numpy array BGR? typically RGB-like; convert if needed
            # convert numpy to PIL for display (results.render() returns ndarray uint8)
            try:
                im_out = Image.fromarray(rendered)
            except Exception:
                im_out = img  # fallback

            st.image(im_out, caption="Detections (with boxes)", use_column_width=True)

            df = results.pandas().xyxy[0]  # dataframe with columns including 'name' and confidence
            agg = aggregate_detections(df, conf_threshold=conf_thresh)

            overall, details = decide_compliance(agg)

            st.markdown(f"### Compliance Status: {overall}")

            # show a neat table of PPE items and confidences
            rows = []
            for ppe in PPE_PAIRS.keys():
                info = details.get(ppe, {})
                rows.append({
                    "PPE Item": ppe,
                    "Status": info.get("status", "missing"),
                    "Detected (good) Conf": round(info.get("good_conf", 0.0), 3),
                    "Detected (NO-) Conf": round(info.get("bad_conf", 0.0), 3),
                })
            st.table(pd.DataFrame(rows).set_index("PPE Item"))

            # optionally show raw detections
            if show_raw:
                st.write("Raw detections (filtered):")
                display_df = df.copy()
                # make sure confidence column is named consistently
                for c in ["confidence", "conf", "score"]:
                    if c in display_df.columns:
                        display_df = display_df.rename(columns={c: "confidence"})
                        break
                st.dataframe(display_df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]])

# small footer note
st.markdown(
    """
    **Notes:**  
    - The app decides based on the highest-confidence label for each PPE item (e.g. `Hardhat` vs `NO-Hardhat`).  
    - If you want *person-level* compliance (each person separately), let me know — I can modify the code to group detections by person bounding box.  
    - Tweak the confidence slider if your model is noisy.
    """
)
