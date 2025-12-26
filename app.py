import streamlit as st
import requests
from PIL import Image
import os
import numpy as np
import cv2

# For direct usage if API is not desired, but let's try to use the pipeline components directly to allow independence from running API 
# OR we can call the API. The prompt said "deploy it with streamlit and fast api". Usually this means Streamlit consumes FastAPI OR Streamlit acts as standalone frontend.
# I will make Streamlit use the pipeline code directly for simplicity in a single container/environment, 
# or I can make it call the API.
# Let's make Streamlit standalone but using the same pipeline code so it doesn't depend on `uvicorn` running in background.

from src.pipelines.prediction_pipeline import PredictPipeline

st.title("Hindi Digit Classification")

st.write("Upload an image of a Hindi digit to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=100)
    
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            # Save to temp
            temp_path = "temp_img.jpg"
            image.save(temp_path)
            
            try:
                pipeline = PredictPipeline()
                label, confidence = pipeline.predict(temp_path)
                
                st.success(f"Prediction: **{label}**")
                st.info(f"Confidence: {confidence:.4f}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
