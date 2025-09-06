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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataPreprocessing:
    def __init__(self,
                 raw_data_path: str = os.path.join(ROOT_DIR, "data", "raw.csv"),
                 processed_dir: str = os.path.join(ROOT_DIR, "data", "processed"),
                 params_path: str = os.path.join(ROOT_DIR, "params.yaml")):
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

    def handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        if df.isnull().sum().sum() > 0:
            logger.info("Handling missing values by imputing...")
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna(df[col].median())
        else:
            logger.info("No missing values found.")
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method for numeric features"""
        numeric_cols = ["age", "bmi", "children", "charges"]
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            before = df.shape[0]
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            after = df.shape[0]
            logger.info(f"Outlier removal for {col}: {before - after} rows removed")
        return df

    def preprocess(self, df: pd.DataFrame):
        """Preprocess dataset: missing values, outliers, encoding, scaling"""
        try:
            # Handle missing + outliers
            df = self.handle_missing(df)
            df = self.remove_outliers(df)

            # Features and target
            X = df.drop("charges", axis=1)
            y = df["charges"].values

            # Log transform target (helps regression performance)
            y = np.log1p(y)

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

            # Convert back to DataFrame
            X_cols = numeric_features + list(
                preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
            )
            X_df = pd.DataFrame(X_processed, columns=X_cols)
            y_df = pd.DataFrame(y, columns=["charges"])

            # Save preprocessor
            joblib.dump(preprocessor, os.path.join(self.processed_dir, "preprocessor.pkl"))
            logger.info("Preprocessor object saved.")

            return X_df, y_df

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def split_and_save(self, X: pd.DataFrame, y: pd.DataFrame):
        """Split train/test sets and save them as CSV"""
        try:
            logger.info(f"Splitting data with test_size={self.test_size}, random_state={self.random_state}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Concatenate features + target for saving
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            # Save as CSV
            train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
            test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)

            logger.info(f"Train and test datasets saved as CSV in {self.processed_dir}")

        except Exception as e:
            logger.error(f"Error during train-test split saving: {e}")
            raise


if __name__ == "__main__":
    processor = DataPreprocessing()
    df = processor.load_data()
    X, y = processor.preprocess(df)
    processor.split_and_save(X, y)
