import json
import logging
import sys
from unittest.mock import Mock

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


def test_predict_emits_single_json_line_to_stdout(monkeypatch, capsys):
    """Verify each /predict call writes one valid JSON log line to stdout."""
    model = Mock()
    model.predict.return_value = [125.4]
    monkeypatch.setattr(main, "load_model", lambda: model)
    capsys.readouterr()

    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1

    log_entry = json.loads(lines[0])
    assert log_entry["level"] == "INFO"
    assert log_entry["logger"] == "mp025.api"
    assert log_entry["message"] == "prediction_served"
    assert log_entry["event"] == "prediction"
    assert log_entry["endpoint"] == "/predict"
    assert log_entry["boat"] == "country_boat"
    assert log_entry["harbour"] == "South Jetty"
    assert log_entry["prediction"] == 125.4
    assert isinstance(log_entry["latency_ms"], float)
    assert log_entry["latency_ms"] >= 0.0
    assert "timestamp" in log_entry


def test_predict_multiple_calls_emit_one_json_line_each(monkeypatch, capsys):
    """Verify multiple predictions produce one JSON line per prediction."""
    model = Mock()
    model.predict.side_effect = [[110.0], [220.5]]
    monkeypatch.setattr(main, "load_model", lambda: model)
    capsys.readouterr()

    assert client.post("/predict", json=SAMPLE).status_code == 200
    assert client.post("/predict", json={**SAMPLE, "harbour": "Lagoon"}).status_code == 200

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [entry["prediction"] for entry in parsed] == [110.0, 220.5]
    assert [entry["harbour"] for entry in parsed] == ["South Jetty", "Lagoon"]


def test_logging_configured_for_stdout_only_no_file_handler():
    """Verify logger uses stdout StreamHandler and never writes to files."""
    handlers = main.logger.handlers
    assert len(handlers) >= 1
    assert any(
        isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        for h in handlers
    )
    assert not any(isinstance(h, logging.FileHandler) for h in handlers)
    assert any(isinstance(h.formatter, main.JsonFormatter) for h in handlers)


def test_predict_logging_uses_python_logging_and_excludes_pii(monkeypatch, caplog):
    """Verify prediction log comes from stdlib logging and contains no personal details."""
    model = Mock()
    model.predict.return_value = [150.25]
    monkeypatch.setattr(main, "load_model", lambda: model)

    with caplog.at_level(logging.INFO, logger="mp025.api"):
        response = client.post("/predict", json=SAMPLE)

    assert response.status_code == 200
    records = [r for r in caplog.records if r.name == "mp025.api"]
    assert len(records) == 1

    formatted = main.JsonFormatter().format(records[0])
    assert "\n" not in formatted
    payload = json.loads(formatted)
    for forbidden_key in ("name", "email", "phone", "address", "customer_id"):
        assert forbidden_key not in payload


def test_invalid_request_does_not_emit_prediction_log(capsys):
    """Verify rejected 422 requests do not log a prediction event."""
    capsys.readouterr()
    response = client.post("/predict", json={**SAMPLE, "distance_from_shore_km": 200})
    assert response.status_code == 422

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines == []
