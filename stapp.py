import streamlit as st
from PIL import Image
import os

from src.pipelines.prediction_pipeline import PredictPipeline

# -------------------------------
# Streamlit App Configuration
# -------------------------------
st.set_page_config(
    page_title="Hindi Digit Classifier",
    page_icon="🧮",
    layout="centered"
)

st.title("🧮 Hindi Digit Classification")
st.write("Upload an image of a **Hindi digit** to classify it.")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"]
)

# -------------------------------
# Display & Predict
# -------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=150)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            temp_path = "temp_img.jpg"
            image.save(temp_path)

            try:
                pipeline = PredictPipeline()
                label, confidence = pipeline.predict(temp_path)

                st.success(f"Prediction: **{label}**")
                st.info(f"Confidence: **{confidence:.4f}**")

            except Exception as e:
                st.error(f"Error during prediction: {e}")

            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
