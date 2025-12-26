import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")
            
            # Data Ingestion
            obj = DataIngestion()
            train_data_path, test_data_path = obj.initiate_data_ingestion()
            logging.info("Data Ingestion Completed")
            
            # Data Transformation
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test = data_transformation.transform_data(train_data_path, test_data_path)
            logging.info("Data Transformation Completed")
            
            # Model Training
            model_trainer = ModelTrainer()
            accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
            logging.info(f"Model Training Completed with Accuracy: {accuracy}")
            
            print(f"Training completed successfully. Accuracy: {accuracy}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()
