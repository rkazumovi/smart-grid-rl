"""
Loads trained checkpoints (LSTM/Transformer/probabilistic-LSTM forecasters, RL policies)
and runs a single genuine forward inference call for the API layer in main.py.

Design choice worth stating explicitly: a forecast here is built from the LAST seq_len
(96 = 24h) rows of whatever history this service has on disk (data/processed/*_test.csv,
falling back to *_val.csv), predicting horizon (96 = 24h) steps past the most recent
known reading. This is deliberately NOT the same thing evaluate_vs_elia.py and the
notebooks do (which score against an already-known future actual, for backtesting
accuracy) -- an API endpoint that claims to forecast "the future" has to actually predict
past the edge of its own data, with no peeking, or the "forecast" it returns is fiction.
Every response also reports `as_of` (the last known timestamp used) and `forecast_for`
(the timestamp being predicted) so a caller can see exactly how that gap was bridged.

Checkpoints are cached in-process after first load (they're small; re-loading them from
disk and rebuilding the model object on every request would be wasted work under real
traffic) -- see the _CHECKPOINT_CACHE dict below. This is deliberately a plain dict, not
functools.lru_cache, so a failed load (missing checkpoint) is never cached as a "result"
and is retried on the next request rather than being remembered as a permanent failure.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from forecasting.features import add_cyclical_time_features
from forecasting.lstm_model import LSTMForecaster, Standardizer, TIME_FEATURE_COLS, NON_NEGATIVE_TARGETS
from forecasting.transformer_model import TransformerForecaster
from forecasting.probabilistic import QuantileLSTM

DATA_DIR = os.path.join("data", "processed")
OUTPUTS_DIR = "outputs"

_CHECKPOINT_CACHE: dict = {}


def _short(target: str) -> str:
    return target.replace("_mw", "")


def _checkpoint_path(model_type: str, target: str) -> str:
    short = _short(target)
    if model_type == "lstm":
        return os.path.join(OUTPUTS_DIR, f"lstm_{short}_forecaster.pt")
    elif model_type == "transformer":
        return os.path.join(OUTPUTS_DIR, f"transformer_{short}_forecaster.pt")
    elif model_type == "probabilistic":
        return os.path.join(OUTPUTS_DIR, f"probabilistic_lstm_{short}_forecaster.pt")
    raise ValueError(f"Unknown model_type '{model_type}'")


def _load_checkpoint(model_type: str, target: str) -> dict:
    """Loads (and caches) the checkpoint dict + reconstructed model for one (model_type,
    target) pair. Raises FileNotFoundError with a clear, actionable message if the
    checkpoint hasn't been trained yet -- callers in main.py turn that into a 404."""
    key = (model_type, target)
    if key in _CHECKPOINT_CACHE:
        return _CHECKPOINT_CACHE[key]

    path = _checkpoint_path(model_type, target)
    if not os.path.exists(path):
        train_script = {
            "lstm": "lstm_model.py",
            "transformer": "transformer_model.py",
            "probabilistic": "probabilistic.py",
        }[model_type]
        raise FileNotFoundError(
            f"No trained '{model_type}' checkpoint for target '{target}' at {path}. "
            f"Train it first, e.g.: python src/forecasting/{train_script} --target {target}"
        )

    checkpoint = torch.load(path, map_location="cpu")
    scaler = Standardizer()
    scaler.mean, scaler.std = checkpoint["scaler_mean"], checkpoint["scaler_std"]
    seq_len, horizon = checkpoint["seq_len"], checkpoint["horizon"]
    input_size = 1 + len(TIME_FEATURE_COLS)

    if model_type == "lstm":
        model = LSTMForecaster(input_size=input_size)
    elif model_type == "transformer":
        model = TransformerForecaster(input_size=input_size)
    elif model_type == "probabilistic":
        model = QuantileLSTM(input_size=input_size)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    entry = {"model": model, "scaler": scaler, "seq_len": seq_len, "horizon": horizon}
    _CHECKPOINT_CACHE[key] = entry
    return entry


def _load_latest_history(target: str, min_rows: int) -> pd.DataFrame:
    """The freshest available history for `target`: test split if it has enough rows,
    otherwise val (train/val/test are chronological -- see build_dataset.py -- so test is
    always the most recent slice when it exists and is long enough)."""
    for split in ("test", "val"):
        path = os.path.join(DATA_DIR, f"{target}_{split}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["timestamp"])[["timestamp", target]]
            if len(df) >= min_rows:
                return df
    raise FileNotFoundError(
        f"Not enough rows of {target} history in {DATA_DIR} (need >= {min_rows}, "
        f"i.e. seq_len + horizon) -- run build_dataset.py first."
    )


def _build_latest_window(target: str, seq_len: int, horizon: int):
    """The last seq_len rows of real history, feature-engineered exactly like training
    (add_cyclical_time_features), as one (1, seq_len, input_size) tensor ready for the
    model -- plus the timestamp of the most recent known reading and the timestamp
    horizon steps past it (what's actually being forecast)."""
    df = _load_latest_history(target, min_rows=seq_len)
    df = add_cyclical_time_features(df, timestamp_col="timestamp")
    window_df = df.tail(seq_len).reset_index(drop=True)

    as_of = window_df["timestamp"].iloc[-1]
    forecast_for = as_of + pd.Timedelta(minutes=15 * horizon)

    target_values = window_df[target].values.astype(np.float32)
    time_feats = window_df[TIME_FEATURE_COLS].values.astype(np.float32)
    return target_values, time_feats, as_of, forecast_for


def predict_point(model_type: str, target: str):
    """Returns (as_of, forecast_for, predicted_mw) for the 'lstm' or 'transformer' model
    types (both output a single point estimate)."""
    entry = _load_checkpoint(model_type, target)
    model, scaler, seq_len, horizon = entry["model"], entry["scaler"], entry["seq_len"], entry["horizon"]

    target_values, time_feats, as_of, forecast_for = _build_latest_window(target, seq_len, horizon)
    target_scaled = scaler.transform(target_values)
    X = np.concatenate([target_scaled.reshape(-1, 1), time_feats], axis=1)
    X_tensor = torch.from_numpy(X).unsqueeze(0)  # (1, seq_len, input_size)

    with torch.no_grad():
        pred_scaled = model(X_tensor).item()
    pred_mw = float(scaler.inverse_transform(np.array([pred_scaled]))[0])
    if target in NON_NEGATIVE_TARGETS:
        pred_mw = max(pred_mw, 0.0)

    return as_of, forecast_for, pred_mw


def predict_quantiles(target: str):
    """Returns (as_of, forecast_for, q10_mw, q50_mw, q90_mw) from the probabilistic-LSTM
    checkpoint -- same quantile-crossing sort-fix and non-negative clip as
    probabilistic.py's own __main__ block, applied to this one live prediction."""
    entry = _load_checkpoint("probabilistic", target)
    model, scaler, seq_len, horizon = entry["model"], entry["scaler"], entry["seq_len"], entry["horizon"]

    target_values, time_feats, as_of, forecast_for = _build_latest_window(target, seq_len, horizon)
    target_scaled = scaler.transform(target_values)
    X = np.concatenate([target_scaled.reshape(-1, 1), time_feats], axis=1)
    X_tensor = torch.from_numpy(X).unsqueeze(0)

    with torch.no_grad():
        preds_scaled = model(X_tensor).numpy()  # (1, 3)
    preds_mw = scaler.inverse_transform(preds_scaled)[0]  # (3,) -- [q10, q50, q90] order, per QUANTILES
    preds_mw = np.sort(preds_mw)  # enforce q10 <= q50 <= q90, same as probabilistic.py

    if target in NON_NEGATIVE_TARGETS:
        preds_mw = np.clip(preds_mw, 0.0, None)

    q10, q50, q90 = (float(v) for v in preds_mw)
    return as_of, forecast_for, q10, q50, q90


def get_policy_action(policy_name: str, seed: int = 0):
    """Thin re-export of grid_intelligence.policy_inference.get_policy_action so main.py
    only has to import from this one module. Import is deferred to call time (not module
    import time) because it pulls in stable_baselines3 and the GridEnv simulator, which
    the forecast endpoints above don't need -- a service that only wants to serve
    forecasts shouldn't fail to start just because that optional dependency is missing."""
    from grid_intelligence.policy_inference import get_policy_action as _get_policy_action
    return _get_policy_action(policy_name, seed=seed)