import pandas as pd
from linear_regression_scratch import LinearRegressionScratch
import numpy as np
import pyautogui

# -----------------------------
# Load calibration type + data
# -----------------------------
calibration = input("Enter the type of calibration (IRIS_ONLY, FACE_CENTER, HEAD_POSE): ")
file_name = input("Enter File Name (without .csv) : ")
name = input("What do you wish this dataset be called?")
df = pd.read_csv(f"{file_name}.csv")

# -----------------------------
# Choose feature columns
# -----------------------------
if calibration == "IRIS_ONLY":
    feature_columns = [
        "leftdx",
        "leftdy",
        "rightdx",
        "rightdy"
    ]

elif calibration == "FACE_CENTER":
    feature_columns = [
        "leftdx",
        "leftdy",
        "rightdx",
        "rightdy",
        "face_center_x",
        "face_center_y"
    ]

elif calibration == "HEAD_POSE":
    feature_columns = [
        "leftdx",
        "leftdy",
        "rightdx",
        "rightdy",
        "face_center_x",
        "face_center_y",
        "face_width",
        "face_height",
        "face_yaw",
        "face_pitch",
        "face_roll"
    ]

else:
    raise ValueError("Invalid calibration type. Use IRIS_ONLY, FACE_CENTER, or HEAD_POSE.")

target_columns = ["dot_x", "dot_y"]

# -----------------------------
# Check required columns exist
# -----------------------------
required_columns = feature_columns + target_columns

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# -----------------------------
# Dataset exploration
# -----------------------------
print("\n==============================")
print("DATASET INFORMATION")
print("==============================")
print("Calibration type:", calibration)
print("Total rows/samples:", len(df))
print("Feature columns:", feature_columns)
print("Target columns:", target_columns)

print("\n==============================")
print("FEATURE SUMMARY STATISTICS")
print("==============================")
feature_stats = df[feature_columns].describe().T
feature_stats["range"] = feature_stats["max"] - feature_stats["min"]
print(feature_stats)

print("\n==============================")
print("TARGET SUMMARY STATISTICS")
print("==============================")
target_stats = df[target_columns].describe().T
target_stats["range"] = target_stats["max"] - target_stats["min"]
print(target_stats)

# Save exploration stats
feature_stats.to_csv(f"feature_stats_{file_name}_{name}.csv")
target_stats.to_csv(f"target_stats_{file_name}_{name}.csv")

# -----------------------------
# Train/test split
# -----------------------------

n = len(df)
train_size = int(n * 0.8)

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()
print("\n==============================")
print("SPLIT METHOD")
print("==============================")
print("Split by row")
print("\nTraining samples:", len(train_df))
print("Testing samples:", len(test_df))

# -----------------------------
# Create X/y arrays
# -----------------------------
X_train = train_df[feature_columns].values
y_train = train_df[target_columns].values

X_test = test_df[feature_columns].values
y_test = test_df[target_columns].values

# -----------------------------
# Normalize using TRAINING stats only
# -----------------------------
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# -----------------------------
# Train model
# -----------------------------
model = LinearRegressionScratch(
    learning_rate=0.15,
    epochs=10000,
    calibration_type=calibration
)

model.fit(X_train_norm, y_train)

# -----------------------------
# Predict on test set
# -----------------------------
preds_test = model.predict(X_test_norm)

# -----------------------------
# Convert normalized coordinate error to pixel error
# -----------------------------
frame_width, frame_height = 640, 480

x_errors_pixels = np.abs(preds_test[:, 0] - y_test[:, 0]) * frame_width
y_errors_pixels = np.abs(preds_test[:, 1] - y_test[:, 1]) * frame_height

errors_pixels = np.sqrt(
    x_errors_pixels ** 2 +
    y_errors_pixels ** 2
)

# -----------------------------
# Model performance statistics
# -----------------------------
performance_stats = {
    "calibration": calibration,
    "train_samples": len(train_df),
    "test_samples": len(test_df),

    "mean_pixel_error": np.mean(errors_pixels),
    "median_pixel_error": np.median(errors_pixels),
    "std_pixel_error": np.std(errors_pixels),
    "min_pixel_error": np.min(errors_pixels),
    "max_pixel_error": np.max(errors_pixels),
    "q25_pixel_error": np.percentile(errors_pixels, 25),
    "q75_pixel_error": np.percentile(errors_pixels, 75),

    "mean_x_error_pixels": np.mean(x_errors_pixels),
    "median_x_error_pixels": np.median(x_errors_pixels),
    "mean_y_error_pixels": np.mean(y_errors_pixels),
    "median_y_error_pixels": np.median(y_errors_pixels),
}

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

for key, value in performance_stats.items():
    print(f"{key}: {value}")

performance_df = pd.DataFrame([performance_stats])
performance_df.to_csv(f"performance_stats_{file_name}_{name}.csv", index=False)

# -----------------------------
# Save per-sample predictions
# -----------------------------
results_df = test_df.copy()

results_df["pred_dot_x"] = preds_test[:, 0]
results_df["pred_dot_y"] = preds_test[:, 1]

results_df["x_error_pixels"] = x_errors_pixels
results_df["y_error_pixels"] = y_errors_pixels
results_df["total_error_pixels"] = errors_pixels

results_df.to_csv(f"prediction_results_{file_name}_{name}.csv", index=False)

# -----------------------------
# Save model parameters
# -----------------------------
np.savez(
    f"linear_regression_parameters_{file_name}_{name}.npz",
    weights=model.weights,
    bias=model.bias,
    X_mean=X_mean,
    X_std=X_std,
    feature_columns=np.array(feature_columns)
)

print("\nSaved files:")
print(f"- feature_stats_{file_name}_{name}.csv")
print(f"- target_stats_{file_name}_{name}.csv")
print(f"- performance_stats_{file_name}_{name}.csv")
print(f"- prediction_results_{file_name}_{name}.csv")
print(f"- linear_regression_parameters_{file_name}_{name}.npz")