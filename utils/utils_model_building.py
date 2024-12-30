import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from utils.custom_logging import setup_logger

# Initialize logger
logger = setup_logger("utils_model_building")


def load_params(params_path: str) -> dict:
    """Loads model training parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)['model_building']
            logger.info("Successfully loaded model training parameters from %s", params_path)
            return params
    except FileNotFoundError as e:
        logger.error("Parameters file not found at %s. Error: %s", params_path, e)
        raise
    except KeyError as e:
        logger.error("Missing key in parameters file: %s", e)
        raise
    except Exception as e:
        logger.error("Error loading parameters from %s: %s", params_path, e)
        raise


def read_data(file_path: str) -> pd.DataFrame:
    """Reads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info("Successfully read data from %s with shape %s", file_path, data.shape)
        return data
    except FileNotFoundError as e:
        logger.error("Data file not found at %s. Error: %s", file_path, e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("Data file at %s is empty. Error: %s", file_path, e)
        raise
    except Exception as e:
        logger.error("Error reading data from %s: %s", file_path, e)
        raise


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> GradientBoostingClassifier:
    """Trains a Gradient Boosting Classifier."""
    try:
        clf = GradientBoostingClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate']
        )
        clf.fit(X_train, y_train)
        logger.info("Successfully trained the Gradient Boosting Classifier")
        return clf
    except Exception as e:
        logger.error("Error training the Gradient Boosting Classifier: %s", e)
        raise


def save_model(model, file_path: str) -> None:
    """Saves the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info("Successfully saved the model to %s", file_path)
    except Exception as e:
        logger.error("Error saving the model to %s: %s", file_path, e)
        raise
