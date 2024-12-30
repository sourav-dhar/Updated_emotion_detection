import logging
import os

def setup_logger(logger_name: str, log_file: str = "errors.log") -> logging.Logger:
    """Sets up a logger with console and file handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

if __name__=="__main__":
    
    logger = setup_logger("checking")
    logger.debug("this is for testing purpose")