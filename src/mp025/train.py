"""
MLflow training pipeline for MP025 Fishing Boat Catch Prediction.
Trains 3 candidate models, logs metrics and SHA256 parameter to MLflow (sqlite:///mlflow.db),
selects champion model, registers it, and exports to models/model.joblib.
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import yaml

from src.mp025.data_clean import clean_dataset, load_config
from src.mp025.features import build_pipeline, get_feature_target_split


def get_data_sha256(raw_data_path: str = "data/train.csv") -> str:
    """Compute exact SHA256 of original raw training dataset as required by rubric."""
    path = Path(raw_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at {path.resolve()}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_findings_tracking(
    findings_path: str,
    experiment_name: str,
    best_run_id: str,
    registered_model_name: str,
    champion_version: int,
) -> None:
    """Update tracking block in findings.yaml while preserving all previous Stage 1 findings."""
    fpath = Path(findings_path)
    if not fpath.exists():
        return

    with open(fpath, "r", encoding="utf-8") as f:
        findings = yaml.safe_load(f)

    if "tracking" not in findings:
        findings["tracking"] = {}

    findings["tracking"]["experiment_name"] = experiment_name
    findings["tracking"]["best_run_id"] = best_run_id
    findings["tracking"]["registered_model_name"] = registered_model_name
    findings["tracking"]["champion_version"] = champion_version

    with open(fpath, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False, default_flow_style=False)


def run_training_experiment(config_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Execute full Phase 3 training pipeline:
    1. Loads config and calculates data_sha256.
    2. Loads cleaned dataset and prepares train/val splits.
    3. Initializes MLflow tracking with SQLite backend.
    4. Trains 3 regression candidates (Ridge, Random Forest, Gradient Boosting).
    5. Logs data_sha256, hyperparameters, MAE, RMSE, R2, and cloudpickle models.
    6. Identifies best model by validation MAE.
    7. Saves champion model pipeline to models/model.joblib.
    8. Registers champion model in MLflow model registry and sets alias 'champion'.
    9. Updates findings.yaml tracking section.
    """
    # Silence hint noise
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

    config = load_config(config_path)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    raw_path = data_cfg.get("raw_train_path", "data/train.csv")
    cleaned_path = data_cfg.get("cleaned_train_path", "data/processed/train_cleaned.csv")
    tracking_uri = train_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = train_cfg.get("experiment_name", "mp025_fishing_boat_catch")
    reg_model_name = train_cfg.get("registered_model_name", "fishing_boat_catch_champion")
    model_artifact_path = train_cfg.get("model_artifact_path", "models/model.joblib")
    test_size = train_cfg.get("test_size", 0.2)
    random_state = train_cfg.get("random_state", 42)

    # 1. Compute required data_sha256
    data_sha256 = get_data_sha256(raw_path)
    print(f"Dataset SHA256: {data_sha256}")

    # 2. Load or prepare cleaned data
    cleaned_file = Path(cleaned_path)
    if not cleaned_file.exists():
        print(f"Generating cleaned dataset at {cleaned_path}...")
        clean_df = clean_dataset(raw_path, config_path=config_path, save_path=cleaned_path)
    else:
        clean_df = pd.read_csv(cleaned_file)

    # 3. Feature/target split and train/validation split
    X, y = get_feature_target_split(clean_df, target_col="catch_kg")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train split: {X_train.shape}, Val split: {X_val.shape}")

    # 4. Setup MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")

    # 5. Define candidate models
    candidates = [
        (
            "ridge_regression",
            Ridge(alpha=1.0, random_state=random_state),
            {"alpha": 1.0},
        ),
        (
            "random_forest_regressor",
            RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1
            ),
            {"n_estimators": 100, "max_depth": 8},
        ),
        (
            "gradient_boosting_regressor",
            GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=4, random_state=random_state
            ),
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4},
        ),
    ]

    results = []

    # 6. Train and track candidates
    for name, estimator, params in candidates:
        pipeline = build_pipeline(model=estimator)
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id

            # Log parameters
            mlflow.log_param("candidate_name", name)
            mlflow.log_param("estimator_class", estimator.__class__.__name__)
            mlflow.log_param("data_sha256", data_sha256)
            for k, v in params.items():
                mlflow.log_param(k, v)

            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Predict
            y_pred_train = pipeline.predict(X_train)
            y_pred_val = pipeline.predict(X_val)

            # Evaluate metrics
            mae_val = mean_absolute_error(y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            r2_val = r2_score(y_val, y_pred_val)

            mae_train = mean_absolute_error(y_train, y_pred_train)
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            r2_train = r2_score(y_train, y_pred_train)

            # Log metrics (must contain 'mae' per handbook rule)
            mlflow.log_metric("mae", mae_val)
            mlflow.log_metric("mae_val", mae_val)
            mlflow.log_metric("rmse_val", rmse_val)
            mlflow.log_metric("r2_val", r2_val)
            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("rmse_train", rmse_train)
            mlflow.log_metric("r2_train", r2_train)

            # Log model artifact with cloudpickle format (safe for MLflow 3.15+)
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                serialization_format="cloudpickle",
            )

            print(
                f"Candidate: {name:28s} | Val MAE: {mae_val:6.2f} | Val RMSE: {rmse_val:6.2f} | Val R2: {r2_val:5.3f} | Run ID: {run_id}"
            )

            results.append({
                "name": name,
                "run_id": run_id,
                "pipeline": pipeline,
                "mae_val": mae_val,
                "rmse_val": rmse_val,
                "r2_val": r2_val,
            })

    # 7. Select champion (lowest validation MAE)
    results_sorted = sorted(results, key=lambda x: x["mae_val"])
    champion = results_sorted[0]
    best_run_id = champion["run_id"]
    best_pipeline = champion["pipeline"]
    print(f"\nChampion Selected: {champion['name']} (Val MAE: {champion['mae_val']:.2f})")

    # 8. Save champion pipeline to models/model.joblib
    out_model_path = Path(model_artifact_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, out_model_path)
    print(f"Saved champion model to: {out_model_path}")

    # 9. Register champion in MLflow model registry and assign alias 'champion'
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    reg_version = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
    champion_version = int(reg_version.version)

    # Set alias champion
    client.set_registered_model_alias(
        name=reg_model_name,
        alias="champion",
        version=str(champion_version),
    )
    print(f"Registered model '{reg_model_name}' v{champion_version} with alias '@champion'")

    # 10. Update findings.yaml tracking section
    update_findings_tracking(
        findings_path="findings.yaml",
        experiment_name=experiment_name,
        best_run_id=best_run_id,
        registered_model_name=reg_model_name,
        champion_version=champion_version,
    )
    print("Updated findings.yaml tracking section.")

    return {
        "results": results_sorted,
        "champion": champion,
        "champion_version": champion_version,
        "model_artifact_path": str(out_model_path),
        "data_sha256": data_sha256,
    }


if __name__ == "__main__":
    run_training_experiment()
