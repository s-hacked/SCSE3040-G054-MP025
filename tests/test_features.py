"""
Synthetic-data unit tests for feature preprocessing module (src/mp025/features.py).
Tests run entirely on synthetic mock data without requiring raw CSV files.
"""

import numpy as np
import pandas as pd
import pytest
from src.mp025.features import (
    get_feature_target_split,
    build_preprocessor,
    build_pipeline,
    DEFAULT_NUMERICAL_COLS,
    DEFAULT_CATEGORICAL_COLS,
)


@pytest.fixture
def synthetic_cleaned_df() -> pd.DataFrame:
    """Create synthetic cleaned DataFrame."""
    return pd.DataFrame({
        "record_id": [f"R{i:03d}" for i in range(1, 6)],
        "date": ["2026-01-01"] * 5,
        "trip_hours": [10.0, 12.5, 8.0, 15.0, 6.5],
        "distance_from_shore_km": [20.0, 35.0, 15.0, 45.0, 10.0],
        "crew": [4, 6, 3, 8, 2],
        "wave_height_m": [1.2, 2.0, 0.8, 2.5, 0.5],
        "water_temp_c": [25.0, 24.5, 26.0, 23.5, 27.0],
        "net_length_m": [800, 1100, 600, 1300, 500],
        "boat": ["country_boat", "motorised", "trawler", "country_boat", "motorised"],
        "harbour": ["North Jetty", "South Jetty", "Lagoon", "North Jetty", "Lagoon"],
        "catch_kg": [420.0, 510.0, 310.0, 640.0, 230.0],
    })


def test_feature_target_split_with_target(synthetic_cleaned_df):
    """Verify feature/target split extracts y, preserves features, and excludes metadata/leakage."""
    X, y = get_feature_target_split(synthetic_cleaned_df, target_col="catch_kg")

    assert y is not None
    assert len(y) == 5
    assert y.name == "catch_kg"
    assert "catch_kg" not in X.columns
    assert "record_id" not in X.columns
    assert "date" not in X.columns
    assert "kg_sold_at_auction" not in X.columns

    for col in DEFAULT_NUMERICAL_COLS + DEFAULT_CATEGORICAL_COLS:
        assert col in X.columns


def test_feature_target_split_without_target(synthetic_cleaned_df):
    """Verify holdout features (lacking target catch_kg) return y=None without error."""
    holdout_df = synthetic_cleaned_df.drop(columns=["catch_kg"])
    X, y = get_feature_target_split(holdout_df, target_col="catch_kg")

    assert y is None
    assert len(X) == 5
    assert "record_id" not in X.columns


def test_preprocessing_pipeline_shape(synthetic_cleaned_df):
    """Verify preprocessor transforms numerical + categorical features into expected 2D array."""
    X, _ = get_feature_target_split(synthetic_cleaned_df)
    preprocessor = build_preprocessor()

    X_trans = preprocessor.fit_transform(X)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == 5

    # 6 numerical features + OneHot categories (boat has 3 categories, harbour has 3 categories) = 6 + 3 + 3 = 12
    assert X_trans.shape[1] == 12


def test_unknown_category_handling():
    """Verify handle_unknown='ignore' allows unseen categories during inference without error."""
    train_data = pd.DataFrame({
        "trip_hours": [10.0, 15.0],
        "distance_from_shore_km": [20.0, 30.0],
        "crew": [4, 6],
        "wave_height_m": [1.0, 2.0],
        "water_temp_c": [25.0, 24.0],
        "net_length_m": [800, 1000],
        "boat": ["country_boat", "motorised"],
        "harbour": ["North Jetty", "South Jetty"],
    })

    test_unseen = pd.DataFrame({
        "trip_hours": [12.0],
        "distance_from_shore_km": [25.0],
        "crew": [5],
        "wave_height_m": [1.5],
        "water_temp_c": [24.5],
        "net_length_m": [900],
        "boat": ["submarine"],  # Unseen boat category
        "harbour": ["Unknown Harbour"],  # Unseen harbour category
    })

    preprocessor = build_preprocessor()
    preprocessor.fit(train_data)

    # Should not raise any ValueError
    X_unseen_trans = preprocessor.transform(test_unseen)
    assert X_unseen_trans.shape[0] == 1
    assert not np.isnan(X_unseen_trans).any()


def test_build_pipeline_wrapper():
    """Verify build_pipeline returns a valid Scikit-Learn Pipeline."""
    pipeline = build_pipeline()
    assert "preprocessor" in pipeline.named_steps
