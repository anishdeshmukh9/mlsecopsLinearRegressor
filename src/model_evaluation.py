import os
import logging
import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
        """Load test dataset"""
        test_df = pd.read_csv(os.path.join(self.processed_dir, "test.csv"))
        X_test = test_df.drop("charges", axis=1)
        y_test = test_df["charges"].values
        return X_test, y_test

    def load_models(self):
        """Load all trained models"""
        models = {}
        for name in ["linear_regression.pkl", "elasticnet.pkl", "random_forest.pkl"]:
            path = os.path.join(self.models_dir, name)
            if os.path.exists(path):
                models[name.split(".")[0]] = joblib.load(path)
                logger.info(f"Loaded model: {name}")
            else:
                logger.warning(f"Model not found: {name}")
        return models

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
        models = self.load_models()

        final_results = {}
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            y_pred = model.predict(X_test)
            final_results[name] = self.evaluate(y_test, y_pred)

        # Save results to YAML
        report_path = os.path.join(self.reports_dir, "final_evaluation.yaml")
        with open(report_path, "w") as f:
            yaml.dump(final_results, f)

        logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.run()
