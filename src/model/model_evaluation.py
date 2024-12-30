import os
import numpy as np
import pandas as pd
from utils.utils_model_evaluation import load_model, read_test_data, evaluate_model, save_metrics
from utils.custom_logging import setup_logger

# Initialize logger
logger = setup_logger("model_evaluation_pipeline")

def main():
    """Main function for model evaluation."""
    try:
        # Paths
        model_path = './models/model.pkl'
        test_data_path = './data/interim/test_tfidf.csv'
        reports_dir = './reports'  # Folder to store reports
        metrics_path = os.path.join(reports_dir, 'metrics.json')

        # Ensure the reports directory exists
        os.makedirs(reports_dir, exist_ok=True)

        # Load model
        logger.info("Loading the trained model")
        clf = load_model(model_path)

        # Read test data
        logger.info("Loading test data")
        test_data = read_test_data(test_data_path)

        # Extract features and labels
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate model
        logger.info("Evaluating the model")
        metrics = evaluate_model(clf, X_test, y_test)

        # Save metrics
        logger.info("Saving evaluation metrics")
        save_metrics(metrics, metrics_path)

        logger.info("Model evaluation completed successfully. Metrics stored at %s", metrics_path)

    except Exception as e:
        logger.critical("An error occurred during the model evaluation pipeline: %s", e)
        raise


if __name__ == "__main__":
    main()
