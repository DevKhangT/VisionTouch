import pandas as pd
import numpy as np
import pyautogui
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

ALPHAS = [0.01, 0.1, 1.0]

L1_RATIOS = [0.5]

POLY_DEGS = [2]

N_NEIGHBORS = [5, 10]

KERNEL = ["rbf"]

C = [0.5, 1.0, 2.0, 5.0, 10.0]
EPSILONS = [0.05, 0.1, 0.2]

MAX_DEPTH = [5, 10]

MIN_SAMPLE_LEAF = [3, 5]

N_ESTIMATORS = [100, 300]

MAX_ITER = [2000]

LEARNING_RATE = [0.001]

HIDDEN_LAYERS_SIZES = [(32,), (64, 32)]

ACTIVATION = ["relu"]

SOLVER = ["adam"]

WEIGHTS = ["distance"]
# number of frames lagged
LAGS = [0, 3, 5, 6, 8, 9, 10, 11, 12, 14, 16]


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

frame_width, frame_height = 640, 480

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
dataset = input("Enter the dataset name without .csv: ")
RESULTS_DIR = Path("..") / "results" / dataset
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(dataset + ".csv")
df = synthesize_features(df)
duration = (df["current_timestamp"].iloc[-1] - df["current_timestamp"].iloc[0]) / 1000
fps = (len(df) - 1) / duration
all_performance_results = []
all_model_results = {}
def evaluate_model(preds_test, y_test, ablation_name, model_name, model_parameters, lag):

    # -----------------------------
    # Error metrics in pixels
    # -----------------------------
    x_errors_pixels = np.abs(preds_test[:, 0] - y_test[:, 0]) * frame_width
    y_errors_pixels = np.abs(preds_test[:, 1] - y_test[:, 1]) * frame_height

    total_errors_pixels = np.sqrt(
        x_errors_pixels ** 2 +
        y_errors_pixels ** 2
    )
    
    performance_stats = {
        "dataset": dataset,
        "ablation": ablation_name,
        "model": model_name,
        "model_parameters": model_parameters,
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
        "p90_pixel_error": np.percentile(total_errors_pixels, 90),
        "lag": lag,
        "fps": fps,
        "lag_seconds": lag / fps
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
    results_df["model"] = model_name
    results_df["model_parameters"] = model_parameters
    results_df["pred_dot_x"] = preds_test[:, 0]
    results_df["pred_dot_y"] = preds_test[:, 1]

    results_df["x_error_pixels"] = x_errors_pixels
    results_df["y_error_pixels"] = y_errors_pixels
    results_df["total_error_pixels"] = total_errors_pixels

    if (ablation_name, model_name) not in all_model_results:
        all_model_results[(ablation_name, model_name)] = []
    all_model_results[(ablation_name, model_name)].append(results_df)


# --------------------------------------------------
# Define ablation feature sets
# --------------------------------------------------
ablations = {
    "RELATIVE_IRIS_TO_FACE": [
        "left_iris_rel_face_x",
        "left_iris_rel_face_y",
        "right_iris_rel_face_x",
        "right_iris_rel_face_y"
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
    # ALPHAS = [0.001, 0.01, 0.1, 1.0]
    # L1_RATIOS = [0.2, 0.5, 0.8]
    # POLY_DEGS = [2, 3]
    # N_NEIGHBORS = [3, 5, 10, 12]
    # KERNEL = ["rbf", "sigmoid", "poly"]
    # C = ALPHAS.copy()
    # EPSILONS = [0.01, 0.1, 0.5]
    # MAX_DEPTH = [5, 10, 12]
    # MIN_SAMPLE_LEAF = [3, 4, 5]
    # N_ESTIMATORS = [50, 100, 500, 1000]
    # MAX_ITER = [2000, 5000, 10000]
    # LEARNING_RATE = [0.001, 0.005, 0.01]
    # HIDDEN_LAYERS_SIZES = [(64,), (32, ), (64, 32)]
    # ACTIVATION = ["identity", "logistic", "tanh", "relu"]
    # SOLVER = ["adam", 'sgd']
    # WEIGHTS = ["distance", "uniform"]

for lag in LAGS:
    all_performance_results = []
    all_model_results = {}
    df_copy = df.copy()
    df_copy["target_dot_x"] = df_copy["dot_x"].shift(lag)
    df_copy["target_dot_y"] = df_copy["dot_y"].shift(lag)
    target_columns = ["target_dot_x", "target_dot_y"]
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    LAG_DIR = RESULTS_DIR / f"lag_{lag}"
    TARGET_STATS_DIR = LAG_DIR / "target_stats"
    PREDICTIONS_DIR = LAG_DIR / "predictions"
    FEATURE_STATS_DIR = LAG_DIR / "feature_stats"
    MODEL_RESULTS_DIR = LAG_DIR / "model_results"
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok = True)
    FEATURE_STATS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_STATS_DIR.mkdir(parents=True, exist_ok=True)
    LAG_DIR.mkdir(parents=True, exist_ok=True)

    
    for ablation_name, feature_columns in ablations.items():


        print("\n\n==============================")
        print(f"RUNNING ABLATION: {ablation_name}")
        print("==============================")

        required_columns = feature_columns + target_columns

        for col in required_columns:
            if col not in df_copy.columns:
                raise ValueError(f"Missing required column for {ablation_name}: {col}")
        # --------------------------------------------------
        # Fixed row-based train/test split
        # --------------------------------------------------
        # fixed row split was used instead of dot_ids since it wasn't recorded
        # Keep this split the same for every ablation.
        df_clean = df_copy.dropna(subset=required_columns)

        n = len(df_clean)
        train_size = int(n * 0.8)

        train_df = df_clean.iloc[:train_size].copy()
        test_df = df_clean.iloc[train_size:].copy()
        print("\n==============================")
        print("GLOBAL DATASET SPLIT")
        print("==============================")
        print("Dataset:", dataset)
        print("Total samples:", len(df_clean))
        print("Training samples:", len(train_df))
        print("Testing samples:", len(test_df))
        print("Split method: fixed row-based 80/20 split")
        print("Note: dot_id was not recorded, so split-by-dot is unavailable.")
        # -----------------------------
        # Dataset exploration
        # -----------------------------
        feature_stats = df_clean[feature_columns].describe().T
        feature_stats["range"] = feature_stats["max"] - feature_stats["min"]

        target_stats = df_clean[target_columns].describe().T
        target_stats["range"] = target_stats["max"] - target_stats["min"]

        feature_stats.to_csv(FEATURE_STATS_DIR / f"{dataset}_feature_stats_{ablation_name}_lag_{lag}.csv")
        target_stats.to_csv(TARGET_STATS_DIR / f"{dataset}_target_stats_{ablation_name}_lag_{lag}.csv")

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
        
        # Use Pipeline to chain cross validation steps and make it easier to add more models in the future
        
        # -----------------------------
        # Linear Regression Updated
        # -----------------------------
        linear_model = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])
        linear_model.fit(X_train, y_train)
        # X_trained_scaled = scaler.fit_transform(X_train)
        # regressor.fit(X_trained_scaled, y_train)
        linear_preds = linear_model.predict(X_test)
        # X_test_scaled scaler.transform(X_test)
        # regressor.predict(X_test_scaled)
        evaluate_model(linear_preds, y_test, ablation_name, "linear", None, lag)

        # -----------------------------
        # Ridge Regression
        # -----------------------------
        for alpha in ALPHAS:
            ridge_model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=alpha))
            ])
            ridge_model.fit(X_train, y_train)
            ridge_preds = ridge_model.predict(X_test)
            evaluate_model(ridge_preds, y_test, ablation_name, "ridge", {
                "alpha": alpha
            }, lag)

        # -----------------------------
        # Lasso Regression
        # -----------------------------
        for alpha in ALPHAS:
            lasso_model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Lasso(alpha=alpha))
            ])
            lasso_model.fit(X_train, y_train)
            lasso_preds = lasso_model.predict(X_test)
            evaluate_model(lasso_preds, y_test, ablation_name, "lasso", {
                "alpha": alpha
            }, lag)
        # -----------------------------
        # ElasticNet Regression
        # -----------------------------
        for alpha in ALPHAS:
            for ratio in L1_RATIOS:
                for max_iter in MAX_ITER:
                    elasticnet_model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("regressor", ElasticNet(alpha = alpha,l1_ratio = ratio, max_iter = max_iter))
                    ])
                    elasticnet_model.fit(X_train, y_train)
                    elasticnet_preds = elasticnet_model.predict(X_test)
                    evaluate_model(elasticnet_preds, y_test, ablation_name, "elasticnet", {
                    "alpha": alpha,
                    "l1_ratio": ratio,
                    "max_iter": max_iter
                    }, lag)
        # -----------------------------
        # Polynomial Regression
        # -----------------------------
        for degree in POLY_DEGS:
            for alpha in ALPHAS:
                poly_ridge_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly_features", PolynomialFeatures(degree = degree, include_bias=False)),
                    ("regressor", Ridge(alpha=alpha))
                ])
                poly_ridge_model.fit(X_train, y_train)
                poly_ridge_preds = poly_ridge_model.predict(X_test)
                evaluate_model(poly_ridge_preds, y_test, ablation_name, "polynomial_ridge", {
                    "degree": degree,
                    "alpha": alpha
                }, lag)
        # -----------------------------
        # KNN Regressor
        # -----------------------------
        for weight in WEIGHTS:
            for neighbor in N_NEIGHBORS:
                KNN_model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", KNeighborsRegressor(n_neighbors=neighbor, weights = weight))
                ])
                KNN_model.fit(X_train, y_train)
                KNN_preds = KNN_model.predict(X_test)
                evaluate_model(KNN_preds, y_test, ablation_name, "knn", {
                    "n_neighbors": neighbor,
                    "weights": weight
                    }, lag)
        # -----------------------------
        # SVR
        # -----------------------------
        # polynomial and sigmoid
        # C is for the penalty magnitude of errors, higher C means less tolerance for errors and can lead to overfitting
        # epsilon defines the margin of toleranced
        for kernel in KERNEL:
            for c in C:
                for epsilon in EPSILONS:
                    SVR_model = Pipeline([
                        ("scaler", StandardScaler()),
                        ("regressor", MultiOutputRegressor(
                            SVR(kernel=kernel, C=c, epsilon=epsilon, gamma = "scale")
                            ))

                    ])
                    SVR_model.fit(X_train, y_train)
                    SVR_preds = SVR_model.predict(X_test)
                    evaluate_model(SVR_preds, y_test, ablation_name, "svr", {
                        "kernel": kernel,
                        "C": c,
                        "epsilon": epsilon
                    }, lag)
        # -----------------------------
        # Decision Tree Regressor
        # -----------------------------
        for max_depth in MAX_DEPTH:
            for min_sample_leaf in MIN_SAMPLE_LEAF:
                DTree_model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_sample_leaf, random_state = 42)
                DTree_model.fit(X_train, y_train)
                DTree_preds = DTree_model.predict(X_test)
                evaluate_model(DTree_preds, y_test, ablation_name, "decision_tree", {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_sample_leaf
                }, lag)

        # -----------------------------
        # Random Forest Regressor
        # -----------------------------
        #oob_score tests on data out of the sample bag
        for n_estimator in N_ESTIMATORS:
            for max_depth in MAX_DEPTH:
                for min_sample_leaf in MIN_SAMPLE_LEAF:
                    RFTree_model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth, min_samples_leaf=min_sample_leaf, random_state=42, n_jobs=-1, oob_score=True)
                    RFTree_model.fit(X_train, y_train)
                    RFTree_preds = RFTree_model.predict(X_test)
                    evaluate_model(RFTree_preds, y_test, ablation_name, "random_forest", {
                        "n_estimators": n_estimator,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_sample_leaf
                    }, lag)
        # -----------------------------
        # Gradient Boosting Regressor 
        # -----------------------------
        for n_estimator in N_ESTIMATORS:
            for learning_rate in LEARNING_RATE:
                for max_depth in MAX_DEPTH:
                    for min_sample_leaf in MIN_SAMPLE_LEAF:
                        GDB_model = MultiOutputRegressor(
                            GradientBoostingRegressor(n_estimators=n_estimator, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_sample_leaf, random_state=42)
                        )
                        GDB_model.fit(X_train, y_train)
                        GDB_preds = GDB_model.predict(X_test)
                        evaluate_model(GDB_preds, y_test, ablation_name, "gradient_boosting", {
                            "n_estimators": n_estimator,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_sample_leaf
                        }, lag)
        # -----------------------------
        # HistGradientBoostingRegressor
        # -----------------------------
        for max_iter in MAX_ITER:
            for learning_rate in LEARNING_RATE:
                for max_depth in MAX_DEPTH:
                    for min_sample_leaf in MIN_SAMPLE_LEAF:
                        HGB_model = MultiOutputRegressor(
                            HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth, min_samples_leaf=min_sample_leaf, random_state=42)
                        )
                        HGB_model.fit(X_train, y_train)
                        HGB_preds = HGB_model.predict(X_test)
                        evaluate_model(HGB_preds, y_test, ablation_name, "hist_gradient_boosting", {
                            "max_iter": max_iter,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth,
                            "min_samples_leaf": min_sample_leaf
                        }, lag)
        # -----------------------------
        # MLP Regressor
        # -----------------------------
        for hidden_layer_sizes in HIDDEN_LAYERS_SIZES:
            for activation in ACTIVATION:
                for solver in SOLVER:
                    for alpha in ALPHAS:
                        for learning_rate in LEARNING_RATE:
                            for max_iter in MAX_ITER:
                                MLP_model = Pipeline([
                                    ("scaler", StandardScaler()),
                                    ("regressor", MLPRegressor(
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        activation=activation,
                                        solver = solver,
                                        alpha = alpha,
                                        learning_rate_init = learning_rate,
                                        max_iter = max_iter,
                                        random_state = 42
                                    ))
                                ])
                                try:
                                    MLP_model.fit(X_train, y_train)
                                    MLP_preds = MLP_model.predict(X_test)

                                    evaluate_model(MLP_preds, y_test, ablation_name, "mlp", {
                                        "hidden_layer_sizes": hidden_layer_sizes,
                                        "activation": activation,
                                        "solver": solver,
                                        "alpha": alpha,
                                        "learning_rate_init": learning_rate,
                                        "max_iter": max_iter
                                    }, lag)

                                except Exception as e:
                                    print("MLP failed with params:", {
                                        "hidden_layer_sizes": hidden_layer_sizes,
                                        "activation": activation,
                                        "solver": solver,
                                        "alpha": alpha,
                                        "learning_rate_init": learning_rate,
                                        "max_iter": max_iter
                                    })
                                    print("Error:", e)

    # --------------------------------------------------
    # Save per sample prediction results
    # --------------------------------------------------
    for (ablation_name, model_name), dfs in all_model_results.items():
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(PREDICTIONS_DIR / f"{dataset}_prediction_results_{ablation_name}_{model_name}_lag_{lag}.csv", index=False)
        print(f"Saved prediction results for {ablation_name} - {model_name} to {dataset}_lag_{lag}_prediction_results_{ablation_name}_{model_name}.csv")
    # --------------------------------------------------
    # Save combined ablation results
    # --------------------------------------------------
    model_results_df = pd.DataFrame(all_performance_results)
    model_results_df = model_results_df.sort_values("mean_pixel_error")

    model_results_df.to_csv(MODEL_RESULTS_DIR / f"{dataset}_lag_{lag}_stage3_models_results.csv", index=False)

    print("\n\n==============================")
    print("FINAL MODEL RESULTS")
    print("==============================")
    print(model_results_df[
        [
            "ablation",
            "model",
            "model_parameters",
            "num_features",
            "mean_pixel_error",
            "median_pixel_error",
            "std_pixel_error",
            "max_pixel_error",
            "mean_x_error_pixels",
            "mean_y_error_pixels"
        ]
    ])

    print(f"\nSaved combined model results to: {dataset}_lag_{lag}_stage3_models_results.csv")