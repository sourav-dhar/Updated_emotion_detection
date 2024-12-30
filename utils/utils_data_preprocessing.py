import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils.custom_logging import setup_logger

logger = setup_logger("utils_data_preprocessing")

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    """Applies lemmatization to a string."""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logger.error("Error during lemmatization: %s", e)
        raise

def remove_stop_words(text: str) -> str:
    """Removes stop words from a string."""
    try:
        stop_words = set(stopwords.words("english"))
        filtered = [word for word in text.split() if word not in stop_words]
        return " ".join(filtered)
    except Exception as e:
        logger.error("Error during stop word removal: %s", e)
        raise

def removing_numbers(text: str) -> str:
    """Removes numbers from a string."""
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logger.error("Error during number removal: %s", e)
        raise

def lower_case(text: str) -> str:
    """Converts text to lowercase."""
    try:
        return text.lower()
    except Exception as e:
        logger.error("Error during lowercase conversion: %s", e)
        raise

def removing_punctuations(text: str) -> str:
    """Removes punctuations from a string."""
    try:
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error("Error during punctuation removal: %s", e)
        raise

def removing_urls(text: str) -> str:
    """Removes URLs from a string."""
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error("Error during URL removal: %s", e)
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all preprocessing steps to the DataFrame."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        logger.info("Data normalized successfully")
        return df
    except Exception as e:
        logger.error("Error during text normalization: %s", e)
        raise
