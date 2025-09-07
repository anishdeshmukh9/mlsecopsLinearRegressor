import os
import logging
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FeatureEngineering:
    def __init__(self,
                 processed_dir: str = os.path.join(ROOT_DIR, "data", "processed"),
                 params_path: str = os.path.join(ROOT_DIR, "params.yaml")):
        self.processed_dir = processed_dir
        self.params_path = params_path

        with open(self.params_path, "r") as f:
            self.params = yaml.safe_load(f)

        self.n_components = self.params["feature_eng"]["pca_components"]

    def load_data(self):
        """Load preprocessed train.csv"""
        train_path = os.path.join(self.processed_dir, "train.csv")
        logger.info(f"Loading data from {train_path}")
        df = pd.read_csv(train_path)

        # ensure charges is numeric
        df["charges"] = pd.to_numeric(df["charges"], errors="coerce")

        logger.info(f"Loaded data with shape {df.shape}")
        return df

    def correlation_analysis(self, df: pd.DataFrame):
        """Perform correlation analysis with target"""
        logger.info("Performing correlation analysis...")
        corr = df.corr(numeric_only=True)

        # Save heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation with Target")
        heatmap_path = os.path.join(self.processed_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        logger.info(f"Correlation heatmap saved to {heatmap_path}")

        # Get top correlated features with 'charges'
        target_corr = corr["charges"].abs().sort_values(ascending=False)
        top_features = target_corr.index[1:6].tolist()  # skip 'charges' itself
        logger.info(f"Top correlated features with charges: {top_features}")
        return top_features

    def pca_analysis(self, df: pd.DataFrame):
        """Perform PCA on features"""
        logger.info("Performing PCA analysis...")
        X = df.drop("charges", axis=1)

        pca = PCA(n_components=self.n_components, random_state=42)
        X_pca = pca.fit_transform(X)

        explained_var = pca.explained_variance_ratio_
        logger.info(f"PCA explained variance ratios: {explained_var}")

        # Save PCA model
        joblib.dump(pca, os.path.join(self.processed_dir, "pca_model.pkl"))
        logger.info("PCA model saved.")

        return explained_var

    @staticmethod
    def add_interactions(df: pd.DataFrame):
        """Add custom interaction features"""
        df["smoker_bmi"] = df["smoker_yes"] * df["bmi"]
        df["smoker_age"] = df["smoker_yes"] * df["age"]
        df["age_bmi"] = df["age"] * df["bmi"]
        df["age_children"] = df["age"] * df["children"]
        return df

    def run(self):
        df = self.load_data()

        # Correlation-based feature importance
        top_features = self.correlation_analysis(df)

        # PCA-based dimensionality reduction
        explained_var = self.pca_analysis(df)

        # Save results in YAML
        results = {
            "top_features": top_features,
            "pca_explained_variance": explained_var.tolist(),
            "pca_cumulative_variance": np.cumsum(explained_var).tolist()
        }
        results_path = os.path.join(self.processed_dir, "feature_eng_results.yaml")
        with open(results_path, "w") as f:
            yaml.dump(results, f)

        logger.info(f"Feature engineering results saved to {results_path}")


if __name__ == "__main__":
    fe = FeatureEngineering()
    fe.run()

    # Load processed train/test
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    # Add interaction features
    train = fe.add_interactions(train)
    test = fe.add_interactions(test)

    # Save back
    os.makedirs("data/processed", exist_ok=True)
    train.to_csv("data/processed/train_features.csv", index=False)
    test.to_csv("data/processed/test_features.csv", index=False)

    print("✅ Interaction features added and saved as train_features.csv, test_features.csv")
