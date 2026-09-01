"""
Fair, apples-to-apples comparison of our LSTM day-ahead load forecaster against Elia's
own real "Day-ahead 6PM forecast" (published the day before delivery and never updated,
same as our model: both make a genuine 24h-ahead prediction, neither peeks at anything
closer to real time).

Both models are scored on EXACTLY the same set of timestamps (any timestamp missing from
either series is dropped from both), on the val split only -- not train, not test.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from forecasting.lstm_model import LSTMForecaster, SequenceDataset, Standardizer, TIME_FEATURE_COLS


def get_lstm_predictions(checkpoint_path: str, val_raw: pd.DataFrame):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    scaler = Standardizer()
    scaler.mean, scaler.std = checkpoint["scaler_mean"], checkpoint["scaler_std"]
    seq_len, horizon = checkpoint["seq_len"], checkpoint["horizon"]

    model = LSTMForecaster(input_size=1 + len(TIME_FEATURE_COLS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ds = SequenceDataset(val_raw, target_col="load_mw", standardizer=scaler, seq_len=seq_len, horizon=horizon)
    preds_mw, timestamps = [], []
    with torch.no_grad():
        for idx in range(len(ds)):
            window, _ = ds[idx]
            pred_scaled = model(window.unsqueeze(0)).item()
            preds_mw.append(scaler.inverse_transform(np.array([pred_scaled]))[0])
            timestamps.append(ds.target_timestamp(idx))
    return pd.DataFrame({"timestamp": timestamps, "lstm_pred_mw": preds_mw})


if __name__ == "__main__":
    checkpoint_path = os.path.join("outputs", "lstm_load_forecaster.pt")
    val_path = os.path.join("data", "processed", "load_mw_val.csv")
    merged_path = os.path.join("data", "processed", "grid_merged.csv")

    missing = [p for p in (checkpoint_path, val_path, merged_path) if not os.path.exists(p)]
    if missing:
        print("Missing file(s):")
        for m in missing:
            print(f"  {m}")
        print("Run data_loader.py, build_dataset.py, and lstm_model.py first (in that order).")
        sys.exit(1)

    val_raw = pd.read_csv(val_path, parse_dates=["timestamp"])[["timestamp", "load_mw"]]
    merged = pd.read_csv(merged_path, parse_dates=["timestamp"])

    if "elia_dayahead_forecast_load_mw" not in merged.columns:
        print("grid_merged.csv doesn't have elia_dayahead_forecast_load_mw yet.")
        print("Update data_loader.py's load_elia_total_load() and rerun it to regenerate grid_merged.csv.")
        sys.exit(1)

    print("Running the trained LSTM over the validation split...")
    lstm_df = get_lstm_predictions(checkpoint_path, val_raw)

    # Align on timestamp: actual measured load + Elia's real day-ahead forecast + our LSTM's
    # prediction, all for the exact same instants.
    actuals = merged[["timestamp", "load_mw", "elia_dayahead_forecast_load_mw"]]
    combined = lstm_df.merge(actuals, on="timestamp", how="inner")

    n_before = len(combined)
    combined = combined.dropna(subset=["load_mw", "elia_dayahead_forecast_load_mw", "lstm_pred_mw"])
    n_after = len(combined)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} timestamps with a missing value in either series "
              f"(kept {n_after}, evaluated identically for both models).")

    def rmse(pred, actual):
        return float(np.sqrt(np.mean((pred - actual) ** 2)))

    def mae(pred, actual):
        return float(np.mean(np.abs(pred - actual)))

    lstm_rmse = rmse(combined["lstm_pred_mw"].values, combined["load_mw"].values)
    lstm_mae = mae(combined["lstm_pred_mw"].values, combined["load_mw"].values)
    elia_rmse = rmse(combined["elia_dayahead_forecast_load_mw"].values, combined["load_mw"].values)
    elia_mae = mae(combined["elia_dayahead_forecast_load_mw"].values, combined["load_mw"].values)

    print(f"\nEvaluated on {n_after} aligned day-ahead predictions "
          f"({combined['timestamp'].min()} -> {combined['timestamp'].max()})\n")
    print(f"{'Model':<30} {'RMSE (MW)':>12} {'MAE (MW)':>12}")
    print("-" * 56)
    print(f"{'Our LSTM (24h ahead)':<30} {lstm_rmse:>12.1f} {lstm_mae:>12.1f}")
    print(f"{'Elia Day-ahead 6PM forecast':<30} {elia_rmse:>12.1f} {elia_mae:>12.1f}")

    if lstm_rmse < elia_rmse:
        pct = (elia_rmse - lstm_rmse) / elia_rmse * 100
        print(f"\nOur LSTM beats Elia's own day-ahead forecast by {pct:.1f}% RMSE.")
    else:
        pct = (lstm_rmse - elia_rmse) / elia_rmse * 100
        print(f"\nElia's own day-ahead forecast beats our LSTM by {pct:.1f}% RMSE -- "
              f"a real grid operator's operational forecast likely uses inputs we don't have "
              f"(weather forecasts, planned outages, calendar/holiday data), so this is a "
              f"meaningful, honest gap to report, not a bug to chase.")