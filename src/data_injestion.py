import os
import logging
import pandas as pd
import requests


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_ingestion.log"),  # log file
        logging.StreamHandler()                     # console
    ]
)
logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, url: str, data_dir: str = "../data", filename: str = "raw.csv"):
        self.url = url
        self.data_dir = data_dir
        self.filepath = os.path.join(data_dir, filename)

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch data from a given URL (CSV).
        """
        try:
            logger.info(f"Fetching data from {self.url}")
            response = requests.get(self.url)
            response.raise_for_status()  # raise error if request fails
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            logger.info(f"Data fetched successfully with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    def save_data(self, df: pd.DataFrame):
        """
        Save dataframe as CSV in the /data directory.
        """
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            df.to_csv(self.filepath, index=False)
            logger.info(f"Data saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise


if __name__ == "__main__":
    DATA_URL = "https://gist.githubusercontent.com/meperezcuello/82a9f1c1c473d6585e750ad2e3c05a41/raw/d42d226d0dd64e7f5395a0eec1b9190a10edbc03/Medical_Cost.csv"

    ingestion = DataIngestion(url=DATA_URL, data_dir="../data", filename="raw.csv")
    df = ingestion.fetch_data()
    ingestion.save_data(df)
