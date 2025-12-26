import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import cv2
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # No preprocessor obj to save for just scaling/resizing, but good to have the config
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def transform_data(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining training and testing arrays")

            X_train, y_train = self._process_images(train_df)
            X_test, y_test = self._process_images(test_df)

            logging.info(f"Transformed shapes: X_train={X_train.shape}, X_test={X_test.shape}")

            return (
                X_train,
                y_train,
                X_test,
                y_test,
                # self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)

    def _process_images(self, df):
        image_data = []
        labels = []

        for _, row in df.iterrows():
            try:
                img_path = row['image_path']
                label = row['label']
                
                # Handle absolute/relative paths if necessary. 
                # Assuming paths in CSV are valid relative to CWD or absolute.
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    logging.warning(f"Could not read image: {img_path}")
                    continue
                    
                img = cv2.resize(img, (32, 32))
                image_data.append(img)
                labels.append(label)
            except Exception as e:
                logging.error(f"Error processing image {row['image_path']}: {e}")

        X = np.array(image_data).astype('float32') / 255.0
        X = X.reshape(-1, 32, 32, 1) # Reshape for CNN
        y = np.array(labels)
        
        return X, y
