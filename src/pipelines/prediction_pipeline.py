import sys
import os
import pandas as pd
import numpy as np
import cv2
import json
import tensorflow as tf
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.h5")
        self.label_mapping_path = os.path.join("artifacts", "label_mapping.json")

    def predict(self, image_path):
        try:
            logging.info(f"Loading model from {self.model_path}")
            model = tf.keras.models.load_model(self.model_path)
            
            logging.info(f"Loading label mapping from {self.label_mapping_path}")
            with open(self.label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            
            # Invert mapping: value -> key
            idx_to_label = {v: k for k, v in label_mapping.items()}
            
            # Preprocess
            logging.info("Preprocessing image")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise Exception("Could not read image")
            
            img = cv2.resize(img, (32, 32))
            img_normalized = img.astype('float32') / 255.0
            img_reshaped = img_normalized.reshape(1, 32, 32, 1)
            
            logging.info("Predicting")
            predictions = model.predict(img_reshaped)
            predicted_idx = np.argmax(predictions[0])
            
            predicted_label = idx_to_label.get(predicted_idx, "Unknown")
            confidence = float(predictions[0][predicted_idx])
            
            return predicted_label, confidence
        
        except Exception as e:
            raise CustomException(e, sys)
