"""Pydantic response models for the API -- kept separate from main.py so the shape of
every response is declared in one place and FastAPI can generate accurate OpenAPI docs
(visible at /docs) straight from these classes, with no manual schema duplication."""
from datetime import datetime
from typing import Dict, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class PointForecastResponse(BaseModel):
    target: Literal["load_mw", "wind_mw", "solar_mw"]
    model: Literal["lstm", "transformer"]
    as_of: datetime = Field(description="Timestamp of the most recent known reading used as input.")
    forecast_for: datetime = Field(description="Timestamp being predicted -- horizon steps past as_of.")
    predicted_mw: float


class QuantileForecastResponse(BaseModel):
    target: Literal["load_mw", "wind_mw", "solar_mw"]
    model: Literal["probabilistic"] = "probabilistic"
    as_of: datetime
    forecast_for: datetime
    q10_mw: float = Field(description="10th percentile -- 90% chance the actual value exceeds this.")
    q50_mw: float = Field(description="Median forecast.")
    q90_mw: float = Field(description="90th percentile -- 90% chance the actual value is below this.")


class PolicyActionResponse(BaseModel):
    policy: Literal["PPO", "SAC", "MARL"]
    battery_power_mw: float = Field(description="Positive = discharge, negative = charge.")
    gen_dispatch_mw: Dict[int, float] = Field(description="Generator dispatch in MW, keyed by bus number.")
    total_gen_mw: float
    price_signal_usd_per_mwh: float
    description: str
    reward: float
    scope_note: str = (
        "This action comes from a real inference call against the trained policy's own "
        "synthetic IEEE 14-bus training environment (see policy_inference.py) -- it is NOT "
        "a projection onto the real Belgian grid state reported by the /forecast endpoints."
    )


class ErrorResponse(BaseModel):
    detail: str