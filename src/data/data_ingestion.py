import os
from utils.utils_data_ingestion import load_params, read_data, process_data, split_data, save_data
from utils.custom_logging import setup_logger

logger = setup_logger("data_ingestion_pipeline")

def main():
    """Main function to execute the data ingestion pipeline."""
    try:
        # Load parameters
        params_path = "params.yaml"
        test_size = load_params(params_path)

        # Read data
        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        raw_df = read_data(url)

        # Process data
        processed_df = process_data(raw_df)

        # Split data
        train_data, test_data = split_data(processed_df, test_size=test_size, random_state=7)

        # Save data
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)

        logger.info("Data ingestion pipeline executed successfully.")
    except Exception as e:
        logger.critical("An error occurred in the pipeline: %s", e)

if __name__ == "__main__":
    main()
