import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # We assume the user places 'numerals' in data/numerals or data/ directly
            # The original code had: data_path = '/content/numerals/numerals'
            # We'll assume a standard location 'data' which contains the label folders directly or nested
            
            base_data_path = 'data'
            # Check if folders exist directly or inside a valid subdir
            if not os.path.exists(base_data_path):
                 raise FileNotFoundError(f"Data directory '{base_data_path}' not found. Please place your dataset there.")

            # Logic to find where the category folders are
            # We look for the first directory that contains multiple subdirectories (labels)
            target_data_path = None
            for root, dirs, files in os.walk(base_data_path):
                if len(dirs) > 1: # Assuming more than 1 class
                    target_data_path = root
                    break
            
            if target_data_path is None:
                raise Exception("Could not find a directory structure with class folders in 'data/'")

            logging.info(f"Found data at: {target_data_path}")
            
            character_folders = sorted(os.listdir(target_data_path))
            label_mapping = {folder: idx for idx, folder in enumerate(character_folders)}
            
            data_list = []
            
            for label_name, label_idx in label_mapping.items():
                folder_path = os.path.join(target_data_path, label_name)
                if not os.path.isdir(folder_path): continue
                
                image_files = sorted(os.listdir(folder_path))
                # Limiting to 200 as per original script if needed, or take all
                # image_files = image_files[:200] 
                
                for image_file in image_files:
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_path = os.path.join(folder_path, image_file)
                        data_list.append({'image_path': image_path, 'label': label_idx, 'label_name': label_name})

            df = pd.DataFrame(data_list)
            logging.info(f"Read dataset with {len(df)} records")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            # Save label mapping for prediction later
            import json
            with open(os.path.join('artifacts', 'label_mapping.json'), 'w') as f:
                json.dump(label_mapping, f)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
