import os
import pandas as pd
from utils.utils_model_building import load_params, read_data, train_model, save_model
from utils.custom_logging import setup_logger

# Initialize logger
logger = setup_logger("model_building_pipeline")

def main():
    """Main function to handle model training."""
    try:
        # Load parameters
        params_path = 'params.yaml'
        model_params = load_params(params_path)

        # Paths
        train_data_path = "./data/interim/train_bow.csv"
        model_dir = "./models"  # Folder to save the model
        model_save_path = os.path.join(model_dir, "model.pkl")

        # Ensure the model directory exists
        os.makedirs(model_dir, exist_ok=True)

        # Read training data
        logger.info("Reading training data")
        train_data = read_data(train_data_path)

        # Split features and labels
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # Train model
        logger.info("Training the model")
        model = train_model(X_train, y_train, model_params)

        # Save model
        logger.info("Saving the trained model")
        save_model(model, model_save_path)

        logger.info("Model training and saving completed successfully.")

    except Exception as e:
        logger.critical("An error occurred in the model building pipeline: %s", e)
        raise


if __name__ == "__main__":
    main()
