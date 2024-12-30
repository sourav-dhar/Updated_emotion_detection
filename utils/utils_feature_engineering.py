import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
from typing import Tuple
from utils.custom_logging import setup_logger

# Initialize logger
logger = setup_logger("utils_feature_engineering")


def load_params(params_path: str) -> int:
    """Loads the max_features parameter from the YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            max_features = params['feature_engineering']['max_features']
            logger.info("Successfully loaded max_features parameter: %s", max_features)
            return max_features
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
    """Reads data from the specified CSV file."""
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


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in the DataFrame."""
    try:
        df.fillna('', inplace=True)
        logger.info("Successfully filled missing values in DataFrame.")
        return df
    except Exception as e:
        logger.error("Error filling missing values in DataFrame: %s", e)
        raise


def apply_tfidf(
    X_train: pd.Series, X_test: pd.Series, max_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    """Applies TfidfVectorizer to the training and test data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        logger.info("Successfully applied Bag of Words with max_features=%s", max_features)
        return (
            pd.DataFrame(X_train_tfidf.toarray(), columns=vectorizer.get_feature_names_out()),
            pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out()),
            vectorizer,
        )
    except ValueError as e:
        logger.error("Error in Bag of Words transformation: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during Bag of Words transformation: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Saves the DataFrame to the specified CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logger.info("Successfully saved data to %s", file_path)
    except Exception as e:
        logger.error("Error saving data to %s: %s", file_path, e)
        raise
