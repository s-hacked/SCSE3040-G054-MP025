"""
Tests for trained champion model artifact (models/model.joblib).
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def champion_model():
    """Load the saved champion model artifact."""
    model_path = Path("models/model.joblib")
    if not model_path.exists():
        pytest.skip("models/model.joblib artifact not found.")
    return joblib.load(model_path)


def test_champion_model_loads(champion_model):
    """Verify saved champion model exists and is a valid Scikit-Learn Pipeline."""
    assert hasattr(champion_model, "predict")
    assert hasattr(champion_model, "named_steps")


def test_champion_model_inference(champion_model):
    """Verify model performs inference on valid input features."""
    sample_input = pd.DataFrame([{
        "trip_hours": 12.0,
        "distance_from_shore_km": 25.0,
        "crew": 5,
        "wave_height_m": 1.5,
        "water_temp_c": 26.0,
        "net_length_m": 850,
        "boat": "country_boat",
        "harbour": "South Jetty",
    }])

    pred = champion_model.predict(sample_input)
    assert len(pred) == 1
    assert isinstance(pred[0], (float, np.floating))
    assert pred[0] > 0.0


def test_champion_model_handles_unknown_category(champion_model):
    """Verify model handles unseen categorical levels gracefully without crashing."""
    sample_unknown = pd.DataFrame([{
        "trip_hours": 10.0,
        "distance_from_shore_km": 20.0,
        "crew": 4,
        "wave_height_m": 1.2,
        "water_temp_c": 25.0,
        "net_length_m": 800,
        "boat": "unseen_cat",
        "harbour": "unseen_harbour",
    }])

    pred = champion_model.predict(sample_unknown)
    assert len(pred) == 1
    assert not np.isnan(pred[0])
