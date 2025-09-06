import numpy as np
import os

processed_dir = "../data/processed"

# Load arrays
X_train = np.load(os.path.join(processed_dir, "X_train.npy"))
X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
y_train = np.load(os.path.join(processed_dir, "y_train.npy"))
y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

# Show shapes
print("Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Show a few rows
print("\nSample of preprocessed X_train:")
print(X_train[:5])  # first 5 rows

print("\nSample of y_train:")
print(y_train[:5])
