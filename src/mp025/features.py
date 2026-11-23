"""
Feature engineering and preprocessing pipeline module for MP025 Fishing Boat Catch dataset.
"""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_NUMERICAL_COLS = [
    "trip_hours",
    "distance_from_shore_km",
    "crew",
    "wave_height_m",
    "water_temp_c",
    "net_length_m",
]

DEFAULT_CATEGORICAL_COLS = [
    "boat",
    "harbour",
]

DEFAULT_EXCLUDED_COLS = [
    "record_id",
    "date",
    "kg_sold_at_auction",
]


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "catch_kg",
    excluded_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Split DataFrame into feature matrix X and target Series y.
    Strictly excludes metadata and leakage columns (record_id, date, kg_sold_at_auction).
    If target_col is absent (e.g. holdout dataset), returns (X, None).
    """
    if excluded_cols is None:
        excluded_cols = DEFAULT_EXCLUDED_COLS

    cols_to_drop = list(set(excluded_cols))
    if target_col in df.columns:
        cols_to_drop.append(target_col)
        y = df[target_col].copy()
    else:
        y = None

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).copy()
    return X, y


def build_preprocessor(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Construct a Scikit-Learn ColumnTransformer:
    - StandardScaler for numerical features
    - OneHotEncoder (handle_unknown='ignore', sparse_output=False) for categorical features
    """
    num_cols = numerical_cols if numerical_cols is not None else DEFAULT_NUMERICAL_COLS
    cat_cols = categorical_cols if categorical_cols is not None else DEFAULT_CATEGORICAL_COLS

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    model=None,
) -> Pipeline:
    """
    Construct an end-to-end Scikit-Learn Pipeline combining the preprocessor
    with an optional model estimator.
    """
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    steps = [("preprocessor", preprocessor)]

    if model is not None:
        steps.append(("model", model))

    return Pipeline(steps=steps)
