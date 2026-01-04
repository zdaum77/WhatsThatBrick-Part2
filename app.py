import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
st.set_page_config(page_title="LEGO Part Identifier")
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lego_parts_identifier.keras")
@st.cache_data
def load_classes():
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}
model = load_model()
classes = load_classes()
def predict(img):
    img = img.convert("RGB").resize((96, 96))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)[0]
    top_5 = preds.argsort()[-5:][::-1]
    return [(classes[i], float(preds[i])) for i in top_5]
st.title("LEGO Part Identifier")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
camera = st.camera_input("Or take photo")
image = Image.open(uploaded) if uploaded else (Image.open(camera) if camera else None)
if image:
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, use_container_width=True)
    with col2:
        results = predict(image)
        st.subheader("Results")
        for i, (name, conf) in enumerate(results, 1):
            st.write(f"**{i}. {name}**")
            st.progress(conf, text=f"{conf:.1%}")