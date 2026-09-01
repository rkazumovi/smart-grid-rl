"""
FastAPI service exposing the trained forecasters (LSTM, Transformer, probabilistic-LSTM)
and RL policies (PPO, SAC, MARL) from this project as HTTP endpoints.

Run locally (from the project root, venv active). Use --app-dir, not a dotted
"src.api.main" path -- src/ has its own __init__.py, so importing through it would create
a second, separate copy of every module under src/api/ alongside the one this file's own
sys.path.insert already loads (the exact "two copies of one module" trap that broke
grid_intelligence/agents.py earlier in this project -- --app-dir sidesteps it the same way
every other script here does, by never importing anything through the "src." prefix):
    uvicorn main:app --app-dir src\api --reload --port 8000
Then open http://localhost:8000/docs for interactive OpenAPI docs, or:
    curl http://localhost:8000/health
    curl http://localhost:8000/forecast/load_mw
    curl http://localhost:8000/forecast/solar_mw?model=transformer
    curl http://localhost:8000/forecast/wind_mw/quantiles
    curl http://localhost:8000/policy/MARL

Every endpoint here calls a REAL trained checkpoint under outputs/ -- there is no mocked
or hardcoded response. If a checkpoint hasn't been trained yet, the endpoint returns a
404 naming the exact training command to run, rather than a generic error.

Prometheus metrics (request count/latency/in-progress by path, method, status code -- see
prometheus.yml and the "Deployment infra" section of the README for how Prometheus/Grafana
scrape and visualize these) are exposed at /metrics via prometheus-fastapi-instrumentator,
instrumented immediately below rather than folded into every endpoint's own code.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api import model_loader
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PointForecastResponse,
    PolicyActionResponse,
    QuantileForecastResponse,
)

API_VERSION = "1.0.0"

app = FastAPI(
    title="Smart Grid RL -- Forecasting & Policy API",
    description=(
        "Serves the trained day-ahead load/wind/solar forecasters and RL dispatch "
        "policies from the Smart Grid Energy Optimization project."
    ),
    version=API_VERSION,
)

# expose_app=True (default) registers /metrics on this same app -- one process, one port,
# no separate metrics server to run or containerize. should_instrument_requests_inprogress
# adds a gauge (requests currently being handled) alongside the default counters/
# histograms, since a service that only ever reports "requests so far" can't show whether
# it's currently backed up -- the gauge is what a real "is this thing overloaded right now"
# Grafana panel would key off.
Instrumentator(should_instrument_requests_inprogress=True).instrument(app).expose(app)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(status="ok", version=API_VERSION)


@app.get(
    "/forecast/{target}",
    response_model=PointForecastResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["forecasting"],
)
def forecast(
    target: str,
    model: str = Query("lstm", description="Which point forecaster to use."),
):
    if target not in ("load_mw", "wind_mw", "solar_mw"):
        raise HTTPException(status_code=422, detail="target must be one of: load_mw, wind_mw, solar_mw")
    if model not in ("lstm", "transformer"):
        raise HTTPException(status_code=422, detail="model must be 'lstm' or 'transformer' (use /quantiles for probabilistic)")

    try:
        as_of, forecast_for, predicted_mw = model_loader.predict_point(model, target)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PointForecastResponse(
        target=target, model=model, as_of=as_of, forecast_for=forecast_for, predicted_mw=predicted_mw,
    )


@app.get(
    "/forecast/{target}/quantiles",
    response_model=QuantileForecastResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["forecasting"],
)
def forecast_quantiles(target: str):
    if target not in ("load_mw", "wind_mw", "solar_mw"):
        raise HTTPException(status_code=422, detail="target must be one of: load_mw, wind_mw, solar_mw")

    try:
        as_of, forecast_for, q10, q50, q90 = model_loader.predict_quantiles(target)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return QuantileForecastResponse(
        target=target, as_of=as_of, forecast_for=forecast_for, q10_mw=q10, q50_mw=q50, q90_mw=q90,
    )


@app.get(
    "/policy/{policy_name}",
    response_model=PolicyActionResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["optimization"],
)
def policy_action(policy_name: str, seed: int = Query(0, description="GridEnv reset seed.")):
    if policy_name not in ("PPO", "SAC", "MARL"):
        raise HTTPException(status_code=422, detail="policy_name must be one of: PPO, SAC, MARL")

    try:
        decoded, description, reward, _info = model_loader.get_policy_action(policy_name, seed=seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Policy inference unavailable -- missing dependency ({e}). "
                   f"Install stable_baselines3 to enable this endpoint.",
        )

    return PolicyActionResponse(
        policy=policy_name,
        battery_power_mw=decoded["battery_power_mw"],
        gen_dispatch_mw=decoded["gen_dispatch_mw"],
        total_gen_mw=decoded["total_gen_mw"],
        price_signal_usd_per_mwh=decoded["price_signal_usd_per_mwh"],
        description=description,
        reward=float(reward),
    )