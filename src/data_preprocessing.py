import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreprocessing:
    def __init__(self, raw_data_path: str = "../data/raw.csv",
                 processed_dir: str = "../data/processed",
                 params_path: str = "../params.yaml"):
        self.raw_data_path = raw_data_path
        self.processed_dir = processed_dir
        self.params_path = params_path
        os.makedirs(self.processed_dir, exist_ok=True)

        # Load params
        with open(self.params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.test_size = self.params["split"]["test_size"]
        self.random_state = self.params["split"]["random_state"]

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV"""
        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"Data loaded with shape {df.shape}")
        return df

    def preprocess(self, df: pd.DataFrame):
        """Preprocess dataset: one-hot encode categoricals & scale numeric features"""
        try:
            X = df.drop("charges", axis=1)
            y = df["charges"].values

            categorical_features = ["sex", "smoker", "region"]
            numeric_features = ["age", "bmi", "children"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_features),
                    ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
                ]
            )

            logger.info("Fitting and transforming data...")
            X_processed = preprocessor.fit_transform(X)

            # Save preprocessor
            joblib.dump(preprocessor, os.path.join(self.processed_dir, "preprocessor.pkl"))
            logger.info("Preprocessor object saved.")

            return X_processed, y

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def split_and_save(self, X: np.ndarray, y: np.ndarray):
        """Split train/test sets and save them"""
        try:
            logger.info(f"Splitting data with test_size={self.test_size}, random_state={self.random_state}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Save to files
            np.save(os.path.join(self.processed_dir, "X_train.npy"), X_train)
            np.save(os.path.join(self.processed_dir, "X_test.npy"), X_test)
            np.save(os.path.join(self.processed_dir, "y_train.npy"), y_train)
            np.save(os.path.join(self.processed_dir, "y_test.npy"), y_test)

            logger.info(f"Train and test datasets saved in {self.processed_dir}")

        except Exception as e:
            logger.error(f"Error during train-test split saving: {e}")
            raise


if __name__ == "__main__":
    processor = DataPreprocessing(
        raw_data_path="../data/raw.csv",
        processed_dir="../data/processed",
        params_path="../params.yaml"
    )
    df = processor.load_data()
    X, y = processor.preprocess(df)
    processor.split_and_save(X, y)
