import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_SIZE = (150, 150)

st.set_page_config(
    page_title="LEGO Part Identifier",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lego_parts_identifier.keras")

model = load_model()

# =========================
# LOAD CLASS INDICES
# =========================
@st.cache_data
def load_class_indices():
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    # reverse mapping: index -> class name
    return {v: k for k, v in class_indices.items()}

idx_to_class = load_class_indices()

# =========================
# LOAD LEGO DATAFRAME
# =========================
@st.cache_data
def load_parts_df():
    df = pd.read_csv("parts.csv")
    df = df.drop_duplicates()
    df = df[df["part_num"].notna() & df["name"].notna()]
    df["part_num"] = df["part_num"].astype(str)
    return df

parts_df = load_parts_df()

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# PREDICTION (Top 3)
# =========================
def predict_part(img, top_k=3):
    img_array = preprocess_image(img)
    preds = model.predict(img_array)[0]   # shape: (num_classes,)
    
    # get indices of top_k predictions
    top_indices = preds.argsort()[-top_k:][::-1]
    top_classes = [idx_to_class[i] for i in top_indices]
    top_confidences = [float(preds[i]) for i in top_indices]
    
    return list(zip(top_classes, top_confidences))

# =========================
# UI
# =========================
st.title("🧱 LEGO Part Identifier")
st.write("Take a photo or upload an image of a LEGO part.")

camera_image = st.camera_input("Take a photo")
uploaded_image = st.file_uploader(
    "Or upload an image",
    type=["jpg", "jpeg", "png"]
)

image = None
if camera_image:
    image = Image.open(camera_image)
elif uploaded_image:
    image = Image.open(uploaded_image)

# =========================
# RUN PREDICTION
# =========================
if image:
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Identifying LEGO part..."):
        top_predictions = predict_part(image, top_k=3)

    st.subheader("Prediction Results (Top 3)")

    for i, (class_name, confidence) in enumerate(top_predictions, start=1):
        st.write(f"{i}. **{class_name}** — {confidence:.2%}")

    # =========================
    # LOOK UP PART INFO
    # =========================
    part_info = parts_df[parts_df["name"] == top_predictions[0][0]]

    if not part_info.empty:
        st.subheader("LEGO Database Info")
        st.dataframe(
            part_info[["part_num", "name"]],
            use_container_width=True
        )
    else:
        st.warning("Part found by model, but not matched in parts.csv")

else:
    st.info("Please take a photo or upload an image to begin.")
