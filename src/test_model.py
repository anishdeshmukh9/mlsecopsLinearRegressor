import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

# Load test dataset
data = pd.DataFrame({
    "age": [19,18,28,33,32,31,46,37,37,60,25,62,23,56,27,19,52,23,56,30,60,30,18,34,37,59,63,55,23,31,22,18,19,63,28,19],
    "sex": ["female","male","male","male","male","female","female","female","male","female","male","female","male","female",
            "male","male","female","male","male","male","female","female","male","female","male","female","female","female",
            "male","male","male","female","female","male","male","male"],
    "bmi": [27.9,33.77,33,22.705,28.88,25.74,33.44,27.74,29.83,25.84,26.22,26.29,34.4,39.82,
            42.13,24.6,30.78,23.845,40.3,35.3,36.005,32.4,34.1,31.92,28.025,27.72,23.085,32.775,
            17.385,36.3,35.6,26.315,28.6,28.31,36.4,20.425],
    "children": [0,1,3,0,0,0,1,3,2,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,2,3,0,2,1,2,0,0,5,0,1,0],
    "smoker": ["yes","no","no","no","no","no","no","no","no","no","no","yes","no","no",
               "yes","no","no","no","no","yes","no","no","no","yes","no","no","no","no",
               "no","yes","yes","no","no","no","yes","no"],
    "region": ["southwest","southeast","southeast","northwest","northwest","southeast","southeast",
               "northwest","northeast","northwest","northeast","southeast","southwest","southeast",
               "southeast","southwest","northeast","northeast","southwest","southwest","northeast",
               "southwest","southeast","northeast","northwest","southeast","northeast","northwest",
               "northwest","southwest","southwest","northeast","southwest","northwest","southwest","northwest"],
    "charges": [16884.924,1725.5523,4449.462,21984.47061,3866.8552,3756.6216,8240.5896,7281.5056,
                6406.4107,28923.13692,2721.3208,27808.7251,1826.843,11090.7178,39611.7577,
                1837.237,10797.3362,2395.17155,10602.385,36837.467,13228.84695,4149.736,
                1137.011,37701.8768,6203.90175,14001.1338,14451.83515,12268.63225,2775.19215,
                38711,35585.576,2198.18985,4687.797,13770.0979,51194.55914,1625.43375]
})

# Separate features and target
X = data.drop("charges", axis=1)
y_true = data["charges"].values

# Load preprocessor and model
preprocessor = joblib.load("data/processed/preprocessor.pkl")
model = joblib.load("models/random_forest.pkl")  # change if using another

# Transform features
X_processed = preprocessor.transform(X)

# Predict (log scale)
y_pred_log = model.predict(X_processed)

# Convert back from log to actual charges
y_pred = np.expm1(y_pred_log)  # if trained with log1p
# or use: y_pred = np.exp(y_pred_log) if trained with plain log

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("Predictions (first 5):", y_pred[:5])
print("Actuals (first 5):", y_true[:5])
print("RMSE:", rmse)
