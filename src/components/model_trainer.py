import os
import sys
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.exception import CustomException
from src.logger import logging
# from src.utils import save_object # Keras models save differently

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Initializing Neural Network Model")
            
            num_classes = len(set(y_train)) # data is 0-indexed labels
            
            model = keras.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            logging.info("Starting training")
            
            history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
            
            logging.info("Training completed. Evaluating model...")
            
            test_loss, test_accuracy = model.evaluate(X_test, y_test)
            logging.info(f"Test Accuracy: {test_accuracy}")
            
            logging.info(f"Saving model to {self.model_trainer_config.trained_model_file_path}")
            model.save(self.model_trainer_config.trained_model_file_path)
            
            return test_accuracy
            
        except Exception as e:
            raise CustomException(e, sys)
