import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from typing import Tuple
from utils.custom_logging import setup_logger

logger = setup_logger("utils_data_ingestion")

def load_params(params_path: str) -> float:
    """Loads test size parameter from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info("Parameters loaded successfully from %s", params_path)
            return params['data_ingestion']['test_size']
    except FileNotFoundError as e:
        logger.error("File not found at %s. Error: %s", params_path, e)
        raise
    except KeyError as e:
        logger.error("Missing key in the YAML file: %s", e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def read_data(url: str) -> pd.DataFrame:
    """Reads CSV data from a URL."""
    try:
        df = pd.read_csv(url)
        logger.info("Data read successfully from %s with shape %s", url, df.shape)
        return df
    except Exception as e:
        logger.error("Error reading data from %s: %s", url, e)
        raise

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processes the DataFrame by filtering and encoding sentiments."""
    try:
        if 'tweet_id' in df.columns:
            df.drop(columns=['tweet_id'], inplace=True)
            logger.info("'tweet_id' column dropped")
        df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logger.info("Processed data shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error processing data: %s", e)
        raise

def split_data(df: pd.DataFrame, test_size: float, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test sets."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.info("Train-test split completed: Train=%s, Test=%s", train_data.shape, test_data.shape)
        return train_data, test_data
    except Exception as e:
        logger.error("Error splitting data: %s", e)
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Saves train and test data."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info("Data saved at %s", data_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise
