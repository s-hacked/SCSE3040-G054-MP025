from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

_RESERVED_LOG_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Format log records as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


class _StdoutStreamHandler(logging.StreamHandler):
    """StreamHandler bound dynamically to sys.stdout for runtime and test capture."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def configure_logging() -> logging.Logger:
    """Configure JSON structured logger writing exclusively to stdout."""
    log = logging.getLogger("mp025.api")
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = _StdoutStreamHandler()
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    return log


logger = configure_logging()
app = FastAPI(title="MP025 Fishing Boat Catch API")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    trip_hours: float
    distance_from_shore_km: float = Field(..., le=150)
    crew: int
    wave_height_m: float = Field(..., ge=0)
    water_temp_c: float
    net_length_m: float
    boat: str
    harbour: str


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    start = perf_counter()
    features = pd.DataFrame([request.model_dump()])
    prediction = float(load_model().predict(features)[0])
    latency_ms = round((perf_counter() - start) * 1000, 3)

    logger.info(
        "prediction_served",
        extra={
            "event": "prediction",
            "endpoint": "/predict",
            "boat": request.boat,
            "harbour": request.harbour,
            "prediction": prediction,
            "latency_ms": latency_ms,
        },
    )
    return {"prediction": prediction}
