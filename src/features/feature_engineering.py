import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from typing import Tuple
from utils.custom_logging import setup_logger
from utils.utils_feature_engineering import load_params, read_data, fill_missing_values, apply_tfidf, save_data

# Initialize logger
logger = setup_logger("feature_engineering_pipeline")

# Main Function
def main():
    try:
        # Load parameters
        params_path = 'params.yaml'
        max_features = load_params(params_path)

        # Paths
        train_data_path = "./data/processed/train_processed_data.csv"
        test_data_path = "./data/processed/test_processed_data.csv"

        # Read data
        train_data = read_data(train_data_path)
        test_data = read_data(test_data_path)

        # Fill missing values
        train_data = fill_missing_values(train_data)
        test_data = fill_missing_values(test_data)

        # Extract features and labels
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values

        # Apply Bag of Words
        X_train_tfidf, X_test_tfidf, _ = apply_tfidf(X_train, X_test, max_features)

        # Add labels to the DataFrames
        X_train_tfidf['label'] = y_train
        X_test_tfidf['label'] = y_test

        # Save data
        data_path = os.path.join("data", "interim")
        os.makedirs(data_path, exist_ok=True)
        save_data(X_train_tfidf, os.path.join(data_path, 'train_tfidf.csv'))
        save_data(X_test_tfidf, os.path.join(data_path, 'test_tfidf.csv'))

        logger.info("Feature engineering pipeline executed successfully.")

    except Exception as e:
        logger.critical("An error occurred in the feature engineering pipeline: %s", e)
        raise


if __name__ == "__main__":
    main()
