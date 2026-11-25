from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

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
    features = pd.DataFrame([request.model_dump()])
    prediction = load_model().predict(features)[0]
    return {"prediction": float(prediction)}
