import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model_evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelEvaluation:
    def __init__(self, 
                 models_dir=os.path.join(ROOT_DIR, "models"),
                 processed_dir=os.path.join(ROOT_DIR, "data", "processed"),
                 reports_dir=os.path.join(ROOT_DIR, "reports")):
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def load_data(self):
        """Load test dataset without interaction features"""
        test_df = pd.read_csv(os.path.join(self.processed_dir, "test.csv"))
        X_test = test_df.drop("charges", axis=1)
        y_test = test_df["charges"].values
        return X_test, y_test

    def load_model(self):
        """Load only Random Forest model"""
        path = os.path.join(self.models_dir, "random_forest.pkl")
        if os.path.exists(path):
            logger.info("Loaded model: random_forest.pkl")
            return joblib.load(path)
        else:
            logger.error("Random Forest model not found!")
            return None

    def evaluate(self, y_true, y_pred):
        """Compute evaluation metrics"""
        r2 = float(r2_score(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        accuracy = float(1 - (rmse / np.mean(y_true)))  # relative accuracy

        return {
            "R2": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "MAPE (%)": round(mape, 2),
            "Accuracy": round(accuracy, 4),
        }

    def run(self):
        X_test, y_test = self.load_data()
        model = self.load_model()
        if model is None:
            logger.error("Evaluation aborted: no model found.")
            return

        logger.info("Evaluating Random Forest model...")
        y_pred = model.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)

        # Save results to YAML
        report_path = os.path.join(self.reports_dir, "final_evaluation.yaml")
        with open(report_path, "w") as f:
            yaml.dump({"random_forest": metrics}, f)

        logger.info(f"Evaluation report saved to {report_path}")
        print("✅ Evaluation complete. Metrics saved to final_evaluation.yaml")
        print(metrics)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.run()
