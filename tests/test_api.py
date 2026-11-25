from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)
SAMPLE = {
    "trip_hours": 12.0,
    "distance_from_shore_km": 25.0,
    "crew": 5,
    "wave_height_m": 1.5,
    "water_temp_c": 26.0,
    "net_length_m": 850,
    "boat": "country_boat",
    "harbour": "South Jetty",
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_with_mock_model(monkeypatch):
    model = Mock()
    model.predict.return_value = [125.4]
    monkeypatch.setattr(main, "load_model", lambda: model)
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert response.json() == {"prediction": 125.4}
    assert model.predict.call_args.args[0].iloc[0].to_dict() == SAMPLE


@pytest.mark.parametrize("field,value", [
    ("distance_from_shore_km", 151), ("wave_height_m", -0.1),
    ("unexpected", 1),
])
def test_invalid_input(monkeypatch, field, value):
    loader = Mock(side_effect=AssertionError("Invalid input reached model"))
    monkeypatch.setattr(main, "load_model", loader)
    response = client.post("/predict", json={**SAMPLE, field: value})
    assert response.status_code == 422
    loader.assert_not_called()


def test_missing_field(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(main, "load_model", loader)
    payload = SAMPLE.copy()
    del payload["boat"]
    assert client.post("/predict", json=payload).status_code == 422
    loader.assert_not_called()


def test_boundary_values(monkeypatch):
    model = Mock()
    model.predict.return_value = [100.0]
    monkeypatch.setattr(main, "load_model", lambda: model)
    response = client.post("/predict", json={**SAMPLE,
        "distance_from_shore_km": 150, "wave_height_m": 0})
    assert response.status_code == 200


def test_real_model_prediction():
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)
