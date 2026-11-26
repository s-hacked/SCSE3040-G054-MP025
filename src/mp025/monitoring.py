"""
Monitoring, retraining, and promotion module for MP025 Fishing Boat Catch Prediction.

Responsibilities (Abhibhav Manhas – S24CSEU2780):
  - PSI-based data drift monitoring (train vs. later)
  - Harbour-wise MAE fairness monitoring using v1 champion model
  - V2 model retraining on train + first 70% of later (by date)
  - V1 vs V2 evaluation on last 30% of later
  - 12% promotion rule: promote v2 only if MAE ≤ v1_MAE × 0.88
  - MLflow registration and champion alias management
  - findings.yaml update
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from src.mp025.data_clean import clean_dataset, load_config
from src.mp025.features import build_pipeline, get_feature_target_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mp025.monitoring")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m",
]
PSI_N_BINS = 10
PSI_EPSILON = 1e-6
PROMOTION_THRESHOLD = 0.88  # v2 must achieve MAE ≤ v1 × 0.88 (12% lower)


# ===========================================================================
# PSI
# ===========================================================================

def compute_psi_single(
    reference: pd.Series,
    comparison: pd.Series,
    n_bins: int = PSI_N_BINS,
    epsilon: float = PSI_EPSILON,
) -> float:
    """
    Compute Population Stability Index (PSI) for a single numeric feature.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    Bin boundaries are fixed on the *reference* (training) distribution so
    that comparisons remain consistent across time.

    Parameters
    ----------
    reference : pd.Series
        Reference distribution (cleaned training data).
    comparison : pd.Series
        Comparison distribution (later.csv).
    n_bins : int
        Number of equal-width bins, default 10.
    epsilon : float
        Small value added to empty-bin counts to prevent log(0).

    Returns
    -------
    float
        PSI value. 0 = no shift; < 0.1 = stable; 0.1-0.25 = moderate drift;
        > 0.25 = significant drift.
    """
    ref_clean = reference.dropna()
    cmp_clean = comparison.dropna()

    # Build bin edges from reference distribution
    min_val = float(ref_clean.min())
    max_val = float(ref_clean.max())

    # Guard against constant feature
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    # Extend edges slightly so max value falls inside the last bin
    bin_edges[-1] += 1e-9

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cmp_counts, _ = np.histogram(cmp_clean, bins=bin_edges)

    # Convert to proportions with epsilon smoothing
    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * n_bins)
    cmp_pct = (cmp_counts + epsilon) / (cmp_counts.sum() + epsilon * n_bins)

    psi = float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))
    return round(psi, 6)


def compute_psi_all_features(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    features: list[str] = NUMERIC_FEATURES,
    n_bins: int = PSI_N_BINS,
) -> Dict[str, float]:
    """
    Compute PSI for every specified numeric feature.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training data (reference distribution).
    later_df : pd.DataFrame
        Later dataset (comparison distribution).
    features : list[str]
        Feature names to evaluate.
    n_bins : int
        Number of bins for each feature.

    Returns
    -------
    dict mapping feature name -> PSI value
    """
    results: Dict[str, float] = {}
    for col in features:
        if col not in train_df.columns:
            logger.warning("Feature %s not found in train_df, skipping PSI.", col)
            results[col] = None
            continue
        if col not in later_df.columns:
            logger.warning("Feature %s not found in later_df, skipping PSI.", col)
            results[col] = None
            continue
        psi = compute_psi_single(train_df[col], later_df[col], n_bins=n_bins)
        logger.info("PSI  %-30s = %.6f", col, psi)
        results[col] = psi
    return results


# ===========================================================================
# Harbour-wise MAE (Group fairness monitoring)
# ===========================================================================

def compute_mae_by_harbour(
    model,
    later_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate MAE separately for each harbour group using the given model.

    Uses the same feature extraction pipeline as training (get_feature_target_split),
    which drops leakage columns (kg_sold_at_auction), record_id, date, and catch_kg.

    Parameters
    ----------
    model : fitted sklearn Pipeline
        The loaded v1 champion model.
    later_df : pd.DataFrame
        later.csv data with both features and catch_kg target.

    Returns
    -------
    dict mapping harbour name -> MAE (float)
    """
    if "catch_kg" not in later_df.columns:
        raise ValueError("later_df must contain catch_kg column for MAE computation.")
    if "harbour" not in later_df.columns:
        raise ValueError("later_df must contain harbour column.")

    X, y = get_feature_target_split(later_df, target_col="catch_kg")
    preds = model.predict(X)

    results: Dict[str, float] = {}
    for harbour in later_df["harbour"].unique():
        mask = later_df["harbour"].values == harbour
        mae = float(mean_absolute_error(y[mask], preds[mask]))
        results[harbour] = round(mae, 4)
        logger.info("MAE  harbour=%-15s = %.4f", harbour, mae)

    return results


# ===========================================================================
# V2 Retraining
# ===========================================================================

def chronological_split_later(
    later_df: pd.DataFrame,
    train_fraction: float = 0.70,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort later.csv by date and split into train / check portions.

    Parameters
    ----------
    later_df : pd.DataFrame
        later.csv (expected to be clean).
    train_fraction : float
        Fraction of rows (sorted by date) used for retraining, default 0.70.

    Returns
    -------
    (later_train, later_check) DataFrames
    """
    df_sorted = later_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_fraction)
    later_train = df_sorted.iloc[:split_idx].copy()
    later_check = df_sorted.iloc[split_idx:].copy()
    logger.info(
        "Chronological split: later_train=%d rows, later_check=%d rows",
        len(later_train),
        len(later_check),
    )
    return later_train, later_check


def train_v2_model(
    train_df: pd.DataFrame,
    later_train_df: pd.DataFrame,
    data_sha256: str,
    config: dict,
) -> Tuple[object, str]:
    """
    Train v2 model on cleaned train.csv + first 70% of later.csv.

    Uses the same GradientBoostingRegressor architecture and preprocessing pipeline
    as the v1 champion training for a fair comparison.

    MLflow run is logged to the existing experiment.

    Parameters
    ----------
    train_df : pd.DataFrame
        Cleaned training data.
    later_train_df : pd.DataFrame
        First 70% of later.csv (sorted by date).
    data_sha256 : str
        SHA-256 hash of original train.csv.
    config : dict
        Project params.yaml config dict.

    Returns
    -------
    (fitted_pipeline, run_id)
    """
    train_cfg = config.get("training", {})
    tracking_uri = train_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = train_cfg.get("experiment_name", "mp025_fishing_boat_catch")
    random_state = train_cfg.get("random_state", 42)

    # Combine train + later_train (drop target-leaking and metadata cols via feature split)
    combined = pd.concat([train_df, later_train_df], ignore_index=True)
    X_train, y_train = get_feature_target_split(combined, target_col="catch_kg")

    logger.info("V2 training dataset: %d rows", len(X_train))

    # Same architecture as v1 champion search: GradientBoostingRegressor
    estimator = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=random_state,
    )
    pipeline_v2 = build_pipeline(model=estimator)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="v2_retrained_gbr") as run:
        run_id = run.info.run_id

        mlflow.log_param("candidate_name", "v2_retrained_gbr")
        mlflow.log_param("estimator_class", "GradientBoostingRegressor")
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("v2_train_rows", len(X_train))
        mlflow.log_param("later_train_rows", len(later_train_df))

        pipeline_v2.fit(X_train, y_train)

        # Log train MAE for traceability
        y_pred_train = pipeline_v2.predict(X_train)
        mae_train = float(mean_absolute_error(y_train, y_pred_train))
        mlflow.log_metric("mae_train", mae_train)
        logger.info("V2 training MAE (on combined train set): %.4f", mae_train)

        mlflow.sklearn.log_model(
            sk_model=pipeline_v2,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

    logger.info("V2 MLflow run_id: %s", run_id)
    return pipeline_v2, run_id


# ===========================================================================
# Evaluation and Promotion
# ===========================================================================

def evaluate_v1_v2(
    v1_model,
    v2_model,
    later_check_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Evaluate v1 and v2 on the same held-out last 30% of later.csv.

    Parameters
    ----------
    v1_model : fitted sklearn Pipeline
        Loaded v1 champion model.
    v2_model : fitted sklearn Pipeline
        Freshly trained v2 model.
    later_check_df : pd.DataFrame
        Last 30% of later.csv (sorted by date, unseen during v2 training).

    Returns
    -------
    (mae_v1, mae_v2) floats
    """
    X_check, y_check = get_feature_target_split(later_check_df, target_col="catch_kg")

    preds_v1 = v1_model.predict(X_check)
    preds_v2 = v2_model.predict(X_check)

    mae_v1 = round(float(mean_absolute_error(y_check, preds_v1)), 4)
    mae_v2 = round(float(mean_absolute_error(y_check, preds_v2)), 4)

    logger.info("V1 MAE on check set: %.4f", mae_v1)
    logger.info("V2 MAE on check set: %.4f", mae_v2)
    return mae_v1, mae_v2


def apply_promotion_rule(
    mae_v1: float,
    mae_v2: float,
    v2_run_id: str,
    v2_pipeline,
    config: dict,
) -> Tuple[str, int]:
    """
    Apply the 12% promotion rule.

    Promote v2 ONLY if: mae_v2 <= mae_v1 * 0.88

    If promoted:
      - Save v2 joblib to models/model.joblib (overwrite champion artifact)
      - Register v2 in MLflow model registry
      - Set 'champion' alias to v2 version
      - Log mae_v2 metric on the v2 run

    Parameters
    ----------
    mae_v1, mae_v2 : float
        Evaluation MAEs on the check set.
    v2_run_id : str
        MLflow run id of v2 training run.
    v2_pipeline : fitted sklearn Pipeline
    config : dict

    Returns
    -------
    (decision: str, champion_version: int)
      decision is 'promote' or 'retain_v1'
    """
    train_cfg = config.get("training", {})
    tracking_uri = train_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    reg_model_name = train_cfg.get(
        "registered_model_name", "fishing_boat_catch_champion"
    )

    threshold = mae_v1 * PROMOTION_THRESHOLD
    logger.info(
        "Promotion check: v2_MAE=%.4f vs threshold=%.4f (v1_MAE=%.4f × 0.88)",
        mae_v2,
        threshold,
        mae_v1,
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if mae_v2 <= threshold:
        decision = "promote"
        logger.info("DECISION: PROMOTE v2 (improvement ≥ 12%)")

        # Log final eval metric on v2 run
        with mlflow.start_run(run_id=v2_run_id):
            mlflow.log_metric("mae", mae_v2)
            mlflow.log_metric("mae_check", mae_v2)
            mlflow.log_metric("mae_v1_check", mae_v1)

        # Register v2
        model_uri = f"runs:/{v2_run_id}/model"
        reg_version = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
        v2_version = int(reg_version.version)

        # Set champion alias to v2
        client.set_registered_model_alias(
            name=reg_model_name, alias="champion", version=str(v2_version)
        )

        # Overwrite joblib champion artifact
        joblib.dump(v2_pipeline, "models/model.joblib")
        logger.info(
            "Registered '%s' v%d as @champion, saved models/model.joblib",
            reg_model_name,
            v2_version,
        )
        return decision, v2_version

    else:
        decision = "retain_v1"
        logger.info("DECISION: RETAIN v1 (improvement < 12%)")

        # Log metrics on v2 run for audit trail even though we don't promote
        with mlflow.start_run(run_id=v2_run_id):
            mlflow.log_metric("mae", mae_v2)
            mlflow.log_metric("mae_check", mae_v2)
            mlflow.log_metric("mae_v1_check", mae_v1)

        # Champion alias stays on v1 (version 1)
        v1_version = 1
        return decision, v1_version


# ===========================================================================
# findings.yaml helpers
# ===========================================================================

def load_findings(path: str = "findings.yaml") -> dict:
    """Load findings.yaml without modifying unrelated keys."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_findings(findings: dict, path: str = "findings.yaml") -> None:
    """Write findings.yaml preserving the original key ordering as much as possible."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(findings, f, sort_keys=False, default_flow_style=False)
    logger.info("findings.yaml updated.")


def update_findings_monitoring(
    findings: dict,
    psi: Dict[str, float],
    most_drifted: str,
    mae_by_harbour: Dict[str, float],
    worst_group: str,
    mae_v1_check: float,
    mae_v2_check: float,
    decision: str,
) -> dict:
    """Update only the monitoring block in findings, preserving all other keys."""
    if "monitoring" not in findings:
        findings["monitoring"] = {}
    m = findings["monitoring"]

    m["psi"] = {col: psi.get(col) for col in NUMERIC_FEATURES}
    m["most_drifted_feature"] = most_drifted
    m["mae_by_group"] = {
        "North Jetty": mae_by_harbour.get("North Jetty"),
        "South Jetty": mae_by_harbour.get("South Jetty"),
        "Lagoon": mae_by_harbour.get("Lagoon"),
    }
    m["worst_served_group"] = worst_group
    m["mae_v1_on_check"] = mae_v1_check
    m["mae_v2_on_check"] = mae_v2_check
    m["decision"] = decision
    return findings


# ===========================================================================
# Holdout predictions
# ===========================================================================

def predict_holdout(
    model,
    holdout_path: str,
    output_path: str = "predictions/holdout_predictions.csv",
) -> pd.DataFrame:
    """
    Generate predictions on holdout_features.csv using the champion model.

    Parameters
    ----------
    model : fitted sklearn Pipeline
    holdout_path : str
        Path to holdout_features.csv.
    output_path : str
        Output CSV path.

    Returns
    -------
    DataFrame with columns [record_id, prediction]
    """
    holdout_df = pd.read_csv(holdout_path)
    record_ids = holdout_df["record_id"].values

    X, _ = get_feature_target_split(holdout_df, target_col="catch_kg")
    preds = model.predict(X)

    out = pd.DataFrame({"record_id": record_ids, "prediction": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Holdout predictions saved to %s (%d rows)", output_path, len(out))
    return out


# ===========================================================================
# Save monitoring report
# ===========================================================================

def save_monitoring_report(
    psi: Dict[str, float],
    most_drifted: str,
    mae_by_harbour: Dict[str, float],
    worst_group: str,
    mae_v1_check: float,
    mae_v2_check: float,
    decision: str,
    champion_version: int,
    output_path: str = "reports/monitoring_report.json",
) -> None:
    """Save full monitoring results to a JSON report file."""
    report = {
        "psi": psi,
        "most_drifted_feature": most_drifted,
        "mae_by_harbour": mae_by_harbour,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "promotion_decision": decision,
        "champion_version_after": champion_version,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Monitoring report saved to %s", output_path)


# ===========================================================================
# Main pipeline entry point
# ===========================================================================

def run_monitoring_pipeline(config_path: str = "params.yaml") -> dict:
    """
    Full monitoring and retraining pipeline.

    Steps:
      1. Load config and paths from params.yaml
      2. Load and clean train.csv
      3. Load later.csv
      4. Compute PSI (train vs later) for all 6 numeric features
      5. Load v1 champion from models/model.joblib
      6. Compute harbour-wise MAE on later.csv
      7. Chronological 70/30 split of later.csv
      8. Train v2 on train + first 70% of later
      9. Evaluate v1 and v2 on last 30% of later
      10. Apply 12% promotion rule
      11. Save monitoring report to reports/
      12. Update findings.yaml
      13. Generate holdout predictions

    Returns
    -------
    dict with all computed results
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    raw_train_path = data_cfg.get("raw_train_path", "data/train.csv")
    cleaned_path = data_cfg.get("cleaned_train_path", "data/processed/train_cleaned.csv")
    later_path = "data/later.csv"
    holdout_path = "data/holdout_features.csv"

    # 1. SHA-256 of original train.csv
    data_sha256 = hashlib.sha256(Path(raw_train_path).read_bytes()).hexdigest()
    logger.info("data_sha256: %s", data_sha256)

    # 2. Clean train.csv
    logger.info("Cleaning train.csv ...")
    cleaned_path_obj = Path(cleaned_path)
    if cleaned_path_obj.exists():
        train_clean = pd.read_csv(cleaned_path_obj)
        logger.info("Loaded cached cleaned train: %d rows", len(train_clean))
    else:
        train_clean = clean_dataset(
            raw_train_path, config_path=config_path, save_path=cleaned_path
        )
        logger.info("Cleaned and saved train: %d rows", len(train_clean))

    # 3. Load later.csv (clean per spec, no cleaning needed)
    later_df = pd.read_csv(later_path)
    logger.info("later.csv loaded: %d rows", len(later_df))

    # 4. PSI
    logger.info("--- PSI ---")
    psi_results = compute_psi_all_features(train_clean, later_df)
    most_drifted = max(
        (k for k, v in psi_results.items() if v is not None),
        key=lambda k: psi_results[k] or 0.0,
    )
    logger.info("Most drifted feature: %s (PSI=%.6f)", most_drifted, psi_results[most_drifted])

    # 5. Load v1 champion
    logger.info("--- Loading v1 champion ---")
    v1_model = joblib.load("models/model.joblib")

    # 6. Harbour-wise MAE on full later.csv
    logger.info("--- Harbour-wise MAE ---")
    mae_by_harbour = compute_mae_by_harbour(v1_model, later_df)
    worst_group = max(mae_by_harbour, key=lambda k: mae_by_harbour[k])
    logger.info("Worst-served harbour: %s (MAE=%.4f)", worst_group, mae_by_harbour[worst_group])

    # 7. Chronological 70/30 split
    logger.info("--- 70/30 split of later.csv ---")
    later_train, later_check = chronological_split_later(later_df, train_fraction=0.70)

    # 8. Train v2
    logger.info("--- Training v2 ---")
    v2_pipeline, v2_run_id = train_v2_model(train_clean, later_train, data_sha256, config)

    # 9. Evaluate
    logger.info("--- Evaluating v1 vs v2 on check set ---")
    mae_v1_check, mae_v2_check = evaluate_v1_v2(v1_model, v2_pipeline, later_check)

    # 10. Promotion rule
    logger.info("--- Promotion rule ---")
    decision, champion_version = apply_promotion_rule(
        mae_v1_check, mae_v2_check, v2_run_id, v2_pipeline, config
    )

    # 11. Save report
    save_monitoring_report(
        psi=psi_results,
        most_drifted=most_drifted,
        mae_by_harbour=mae_by_harbour,
        worst_group=worst_group,
        mae_v1_check=mae_v1_check,
        mae_v2_check=mae_v2_check,
        decision=decision,
        champion_version=champion_version,
    )

    # 12. Update findings.yaml
    findings = load_findings("findings.yaml")
    findings = update_findings_monitoring(
        findings=findings,
        psi=psi_results,
        most_drifted=most_drifted,
        mae_by_harbour=mae_by_harbour,
        worst_group=worst_group,
        mae_v1_check=mae_v1_check,
        mae_v2_check=mae_v2_check,
        decision=decision,
    )
    save_findings(findings, "findings.yaml")

    # 13. Holdout predictions using final champion
    if decision == "promote":
        champion_model = v2_pipeline
    else:
        champion_model = v1_model
    predict_holdout(champion_model, holdout_path)

    result = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_harbour": mae_by_harbour,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1_check,
        "mae_v2_on_check": mae_v2_check,
        "decision": decision,
        "champion_version": champion_version,
        "v2_run_id": v2_run_id,
    }
    logger.info("Pipeline complete: %s", json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "params.yaml"
    run_monitoring_pipeline(cfg)
