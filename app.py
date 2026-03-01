import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Food Freshness Detector", layout="centered")

st.title("🍎 Food Freshness Detection")

@st.cache_resource
def load_my_model():
    return load_model("final_model.keras")

model = load_my_model()

IMG_SIZE = 224

uploaded_file = st.file_uploader("Upload Food Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        prob = prediction[0][0]

        if prob > 0.5:
            label = "Spoiled"
            confidence = prob * 100
            color = "#ff4b4b"
        else:
            label = "Fresh"
            confidence = (1 - prob) * 100
            color = "#00c853"

        # Clean Result Section
        st.markdown("---")

        st.markdown(
            f"<h2 style='text-align:center; color:{color};'>Prediction: {label}</h2>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h4 style='text-align:center;'>Confidence: {confidence:.2f}%</h4>",
            unsafe_allow_html=True
        )