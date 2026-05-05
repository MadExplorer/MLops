import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.models import infer_signature
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler


DATA_PATH = Path("df_clear.csv")
MODEL_PATH = Path("insurance_model.pkl")
METRICS_PATH = Path("metrics.json")

TARGET_COLUMN = "charges"

FEATURE_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Сначала запусти download.py"
        )

    df = pd.read_csv(path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"В датасете нет нужных колонок: {missing_columns}")

    df = df[required_columns].copy()
    df = df.dropna()
    df = df.drop_duplicates().reset_index(drop=True)

    return df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    df = load_dataset(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[[TARGET_COLUMN]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        shuffle=True,
    )

    feature_scaler = StandardScaler()
    target_transformer = PowerTransformer()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_transformed = target_transformer.fit_transform(y_train)
    y_test_transformed = target_transformer.transform(y_test)

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.001, 0.05, 0.1, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "fit_intercept": [False, True],
    }

    base_model = SGDRegressor(
        random_state=42,
        max_iter=5000,
        tol=1e-3,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=params,
        cv=3,
        n_jobs=-1,
        scoring="r2",
    )

    mlflow.set_experiment("insurance_charges_model")

    with mlflow.start_run():
        grid_search.fit(X_train_scaled, y_train_transformed.ravel())

        best_model = grid_search.best_estimator_

        y_pred_transformed = best_model.predict(X_test_scaled).reshape(-1, 1)
        y_pred = target_transformer.inverse_transform(y_pred_transformed)
        y_test_original = y_test.to_numpy()

        metrics = evaluate_model(y_test_original, y_pred)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("r2", metrics["r2"])

        signature = infer_signature(X_train_scaled, best_model.predict(X_train_scaled))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        model_bundle = {
            "model": best_model,
            "feature_scaler": feature_scaler,
            "target_transformer": target_transformer,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "best_params": grid_search.best_params_,
            "metrics": metrics,
        }

        joblib.dump(model_bundle, MODEL_PATH)

        METRICS_PATH.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Model saved to:", MODEL_PATH)
        print("Metrics saved to:", METRICS_PATH)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        print("Best params:")
        print(json.dumps(grid_search.best_params_, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()