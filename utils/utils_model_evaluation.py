import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json
from utils.custom_logging import setup_logger

# Initialize logger
logger = setup_logger("utils_model_evaluation")


def load_model(model_path: str):
    """Loads the trained model from a file."""
    try:
        model = pickle.load(open(model_path, 'rb'))
        logger.info("Model loaded successfully from %s", model_path)
        return model
    except FileNotFoundError as e:
        logger.error("Model file not found at %s. Error: %s", model_path, e)
        raise
    except Exception as e:
        logger.error("Error loading the model from %s: %s", model_path, e)
        raise


def read_test_data(test_data_path: str) -> pd.DataFrame:
    """Reads test data from a CSV file."""
    try:
        data = pd.read_csv(test_data_path)
        logger.info("Test data loaded successfully from %s with shape %s", test_data_path, data.shape)
        return data
    except FileNotFoundError as e:
        logger.error("Test data file not found at %s. Error: %s", test_data_path, e)
        raise
    except Exception as e:
        logger.error("Error loading test data from %s: %s", test_data_path, e)
        raise


def evaluate_model(clf, X_test, y_test) -> dict:
    """Evaluates the model and calculates evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info("Model evaluation completed successfully with metrics: %s", metrics)
        return metrics
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str):
    """Saves the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info("Evaluation metrics saved successfully to %s", file_path)
    except Exception as e:
        logger.error("Error saving evaluation metrics to %s: %s", file_path, e)
        raise
