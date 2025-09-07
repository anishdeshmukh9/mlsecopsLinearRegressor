import os
import pandas as pd
import mlflow.pyfunc

# MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load the latest Production model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/MedicalCostModel/Production")

# Load new data to predict (use the same feature columns as training)
X_new = pd.read_csv("data/processed/test.csv")  # replace with your new CSV if needed

# Add interaction features exactly as used during training
X_new["smoker_bmi"] = X_new["smoker_yes"] * X_new["bmi"]
X_new["smoker_age"] = X_new["smoker_yes"] * X_new["age"]
X_new["age_bmi"] = X_new["age"] * X_new["bmi"]
X_new["age_children"] = X_new["age"] * X_new["children"]

# Drop target column if present
if "charges" in X_new.columns:
    X_new_features = X_new.drop("charges", axis=1)
else:
    X_new_features = X_new.copy()

# Predict using the loaded model
predictions = model.predict(X_new_features)

# Log predictions
X_new["predicted_charges"] = predictions

# Save predictions to a CSV
os.makedirs("predictions", exist_ok=True)
pred_file = os.path.join("predictions", "predicted_charges.csv")
X_new.to_csv(pred_file, index=False)

print(f"✅ Predictions saved to {pred_file}")
print(X_new[["predicted_charges"]].head())
