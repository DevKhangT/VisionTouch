import pandas as pd
from linear_regression_scratch import LinearRegressionScratch
import numpy as np
import pyautogui

# --------------------------------------------------
# Add synthesized features
# --------------------------------------------------
def synthesize_features(df):
    eps = 1e-8

    # Average iris position
    df["avg_iris_x"] = (df["leftdx"] + df["rightdx"]) / 2
    df["avg_iris_y"] = (df["leftdy"] + df["rightdy"]) / 2

    # Iris relative to face center
    df["left_iris_rel_face_x"] = df["leftdx"] - df["face_center_x"]
    df["left_iris_rel_face_y"] = df["leftdy"] - df["face_center_y"]

    df["right_iris_rel_face_x"] = df["rightdx"] - df["face_center_x"]
    df["right_iris_rel_face_y"] = df["rightdy"] - df["face_center_y"]

    df["avg_iris_rel_face_x"] = df["avg_iris_x"] - df["face_center_x"]
    df["avg_iris_rel_face_y"] = df["avg_iris_y"] - df["face_center_y"]

    # Normalize iris-relative features by face size
    df["left_iris_rel_face_x_norm"] = df["left_iris_rel_face_x"] / (df["face_width"] + eps)
    df["left_iris_rel_face_y_norm"] = df["left_iris_rel_face_y"] / (df["face_height"] + eps)

    df["right_iris_rel_face_x_norm"] = df["right_iris_rel_face_x"] / (df["face_width"] + eps)
    df["right_iris_rel_face_y_norm"] = df["right_iris_rel_face_y"] / (df["face_height"] + eps)

    df["avg_iris_rel_face_x_norm"] = df["avg_iris_rel_face_x"] / (df["face_width"] + eps)
    df["avg_iris_rel_face_y_norm"] = df["avg_iris_rel_face_y"] / (df["face_height"] + eps)

    # Difference between eyes
    df["iris_dx"] = df["rightdx"] - df["leftdx"]
    df["iris_dy"] = df["rightdy"] - df["leftdy"]

    # Head pose interaction features
    df["yaw_avg_iris_x_interaction"] = df["face_yaw"] * df["avg_iris_rel_face_x_norm"]
    df["pitch_avg_iris_y_interaction"] = df["face_pitch"] * df["avg_iris_rel_face_y_norm"]
    df["roll_avg_iris_x_interaction"] = df["face_roll"] * df["avg_iris_rel_face_x_norm"]
    df["roll_avg_iris_y_interaction"] = df["face_roll"] * df["avg_iris_rel_face_y_norm"]

    # Simple nonlinear features
    df["yaw_squared"] = df["face_yaw"] ** 2
    df["pitch_squared"] = df["face_pitch"] ** 2
    df["roll_squared"] = df["face_roll"] ** 2
    df["avg_iris_rel_face_x_norm_squared"] = df["avg_iris_rel_face_x_norm"] ** 2
    df["avg_iris_rel_face_y_norm_squared"] = df["avg_iris_rel_face_y_norm"] ** 2

    return df


# --------------------------------------------------
# Define ablation feature sets
# --------------------------------------------------
ablations = {
    "RAW_IRIS": [
        "leftdx", "leftdy",
        "rightdx", "rightdy"
    ],

    "RAW_FACE_CENTER": [
        "leftdx", "leftdy",
        "rightdx", "rightdy",
        "face_center_x", "face_center_y"
    ],

    "RAW_HEAD_POSE": [
        "leftdx", "leftdy",
        "rightdx", "rightdy",
        "face_yaw", "face_pitch", "face_roll"
    ],

    "RAW_ALL": [
        "leftdx", "leftdy",
        "rightdx", "rightdy",
        "face_center_x", "face_center_y",
        "face_width", "face_height",
        "face_yaw", "face_pitch", "face_roll"
    ],

    "AVG_IRIS_ONLY": [
        "avg_iris_x", "avg_iris_y"
    ],

    "AVG_IRIS_WITH_EYE_DIFF": [
        "avg_iris_x", "avg_iris_y",
        "iris_dx", "iris_dy"
    ],

    "RELATIVE_IRIS_TO_FACE": [
        "left_iris_rel_face_x",
        "left_iris_rel_face_y",
        "right_iris_rel_face_x",
        "right_iris_rel_face_y"
    ],

    "NORMALIZED_RELATIVE_IRIS_TO_FACE": [
        "left_iris_rel_face_x_norm",
        "left_iris_rel_face_y_norm",
        "right_iris_rel_face_x_norm",
        "right_iris_rel_face_y_norm"
    ],

    "AVG_NORMALIZED_RELATIVE_IRIS": [
        "avg_iris_rel_face_x_norm",
        "avg_iris_rel_face_y_norm"
    ],

    "RELATIVE_PLUS_HEAD_POSE": [
        "avg_iris_rel_face_x_norm",
        "avg_iris_rel_face_y_norm",
        "face_yaw",
        "face_pitch",
        "face_roll"
    ],

    "INTERACTION_FEATURES": [
        "avg_iris_rel_face_x_norm",
        "avg_iris_rel_face_y_norm",
        "face_yaw",
        "face_pitch",
        "face_roll",
        "yaw_avg_iris_x_interaction",
        "pitch_avg_iris_y_interaction",
        "roll_avg_iris_x_interaction",
        "roll_avg_iris_y_interaction"
    ],

    "SQUARED_AND_INTERACTIONS": [
        "avg_iris_rel_face_x_norm",
        "avg_iris_rel_face_y_norm",
        "face_yaw",
        "face_pitch",
        "face_roll",
        "yaw_avg_iris_x_interaction",
        "pitch_avg_iris_y_interaction",
        "roll_avg_iris_x_interaction",
        "roll_avg_iris_y_interaction",
        "yaw_squared",
        "pitch_squared",
        "roll_squared",
        "avg_iris_rel_face_x_norm_squared",
        "avg_iris_rel_face_y_norm_squared"
    ]
}


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
dataset = input("Enter the dataset name without .csv: ")

df = pd.read_csv(dataset + ".csv")
df = synthesize_features(df)

target_columns = ["dot_x", "dot_y"]

screen_width, screen_height = pyautogui.size()

# --------------------------------------------------
# Fixed row-based train/test split
# --------------------------------------------------
# fixed row split was used instead of dot_ids since it wasn't recorded
# Keep this split the same for every ablation.
n = len(df)
train_size = int(n * 0.8)

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

print("\n==============================")
print("GLOBAL DATASET SPLIT")
print("==============================")
print("Dataset:", dataset)
print("Total samples:", len(df))
print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))
print("Split method: fixed row-based 80/20 split")
print("Note: dot_id was not recorded, so split-by-dot is unavailable.")


# --------------------------------------------------
# Run all ablations
# --------------------------------------------------
all_performance_results = []
all_head_condition_results = []

for ablation_name, feature_columns in ablations.items():

    print("\n\n==============================")
    print(f"RUNNING ABLATION: {ablation_name}")
    print("==============================")

    required_columns = feature_columns + target_columns

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column for {ablation_name}: {col}")

    # -----------------------------
    # Dataset exploration
    # -----------------------------
    feature_stats = df[feature_columns].describe().T
    feature_stats["range"] = feature_stats["max"] - feature_stats["min"]

    target_stats = df[target_columns].describe().T
    target_stats["range"] = target_stats["max"] - target_stats["min"]

    feature_stats.to_csv(f"{dataset}_feature_stats_{ablation_name}.csv")
    target_stats.to_csv(f"{dataset}_target_stats_{ablation_name}.csv")

    print("Feature columns:", feature_columns)
    print("Number of features:", len(feature_columns))

    # -----------------------------
    # Create arrays
    # -----------------------------
    X_train = train_df[feature_columns].values
    y_train = train_df[target_columns].values

    X_test = test_df[feature_columns].values
    y_test = test_df[target_columns].values

    # -----------------------------
    # Normalize using training stats only
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
        calibration_type=ablation_name
    )

    model.fit(X_train_norm, y_train)

    # -----------------------------
    # Predict on test set
    # -----------------------------
    preds_test = model.predict(X_test_norm)

    # -----------------------------
    # Error metrics in pixels
    # -----------------------------
    x_errors_pixels = np.abs(preds_test[:, 0] - y_test[:, 0]) * screen_width
    y_errors_pixels = np.abs(preds_test[:, 1] - y_test[:, 1]) * screen_height

    total_errors_pixels = np.sqrt(
        x_errors_pixels ** 2 +
        y_errors_pixels ** 2
    )

    performance_stats = {
        "dataset": dataset,
        "ablation": ablation_name,
        "num_features": len(feature_columns),
        "features": ", ".join(feature_columns),

        "train_samples": len(train_df),
        "test_samples": len(test_df),

        "mean_pixel_error": np.mean(total_errors_pixels),
        "median_pixel_error": np.median(total_errors_pixels),
        "std_pixel_error": np.std(total_errors_pixels),
        "min_pixel_error": np.min(total_errors_pixels),
        "max_pixel_error": np.max(total_errors_pixels),
        "q25_pixel_error": np.percentile(total_errors_pixels, 25),
        "q75_pixel_error": np.percentile(total_errors_pixels, 75),

        "mean_x_error_pixels": np.mean(x_errors_pixels),
        "median_x_error_pixels": np.median(x_errors_pixels),
        "mean_y_error_pixels": np.mean(y_errors_pixels),
        "median_y_error_pixels": np.median(y_errors_pixels),
    }

    all_performance_results.append(performance_stats)

    print("\nMODEL PERFORMANCE")
    for key, value in performance_stats.items():
        if key != "features":
            print(f"{key}: {value}")

    # -----------------------------
    # Save per-sample predictions
    # -----------------------------
    results_df = test_df.copy()
    results_df["ablation"] = ablation_name

    results_df["pred_dot_x"] = preds_test[:, 0]
    results_df["pred_dot_y"] = preds_test[:, 1]

    results_df["x_error_pixels"] = x_errors_pixels
    results_df["y_error_pixels"] = y_errors_pixels
    results_df["total_error_pixels"] = total_errors_pixels

    results_df.to_csv(f"{dataset}_prediction_results_{ablation_name}.csv", index=False)

    # -----------------------------
    # Save model parameters
    # -----------------------------
    np.savez(
        f"{dataset}_linear_regression_parameters_{ablation_name}.npz",
        weights=model.weights,
        bias=model.bias,
        X_mean=X_mean,
        X_std=X_std,
        feature_columns=np.array(feature_columns)
    )


# --------------------------------------------------
# Save combined ablation results
# --------------------------------------------------
ablation_results_df = pd.DataFrame(all_performance_results)
ablation_results_df = ablation_results_df.sort_values("mean_pixel_error")

ablation_results_df.to_csv(f"{dataset}_stage2_ablation_results.csv", index=False)

print("\n\n==============================")
print("FINAL ABLATION RESULTS")
print("==============================")
print(ablation_results_df[
    [
        "ablation",
        "num_features",
        "mean_pixel_error",
        "median_pixel_error",
        "std_pixel_error",
        "max_pixel_error",
        "mean_x_error_pixels",
        "mean_y_error_pixels"
    ]
])

print(f"\nSaved combined ablation results to: {dataset}_stage2_ablation_results.csv")