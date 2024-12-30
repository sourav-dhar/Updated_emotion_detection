import os
import pandas as pd
from utils.utils_data_preprocessing import normalize_text
from utils.custom_logging import setup_logger

logger = setup_logger("data_preprocessing_pipeline")

def main():
    """Main function to preprocess data."""
    try:
        # Paths
        raw_data_path = "./data/raw"
        processed_data_path = "./data/processed"

        # Read data
        logger.info("Reading raw data")
        train_data = pd.read_csv(os.path.join(raw_data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(raw_data_path, "test.csv"))

        # Preprocess data
        logger.info("Preprocessing train data")
        train_processed_data = normalize_text(train_data)
        logger.info("Preprocessing test data")
        test_processed_data = normalize_text(test_data)

        # Save processed data
        os.makedirs(processed_data_path, exist_ok=True)
        train_processed_data.to_csv(os.path.join(processed_data_path, "train_processed_data.csv"), index=False)
        test_processed_data.to_csv(os.path.join(processed_data_path, "test_processed_data.csv"), index=False)
        logger.info("Processed data saved successfully")

    except Exception as e:
        logger.critical("An error occurred in the preprocessing pipeline: %s", e)

if __name__ == "__main__":
    main()
