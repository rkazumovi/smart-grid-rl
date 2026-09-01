"""
LSTM point forecaster: given a 24h window (96 steps at 15-min resolution) of a target's
past values + cyclical time features, predict the target 24h (96 steps) ahead.

Now generalized (via --target) to forecast any of the three grid quantities produced by
build_dataset.py: load_mw (default, matches every earlier run), wind_mw, or solar_mw. The
architecture, windowing, and training loop are identical across all three -- only the input
column and, for the two generation targets, a physical post-processing clip differ.

LSTM cell equations (see notebooks/forecasting_theory.ipynb for the full derivation once
it's built):
    f_t = sigmoid(W_f [h_{t-1}, x_t] + b_f)    forget gate
    i_t = sigmoid(W_i [h_{t-1}, x_t] + b_i)    input gate
    c~_t = tanh(W_c [h_{t-1}, x_t] + b_c)      candidate cell content
    c_t = f_t * c_{t-1} + i_t * c~_t           new cell state
    o_t = sigmoid(W_o [h_{t-1}, x_t] + b_o)    output gate
    h_t = o_t * tanh(c_t)                      new hidden state

The forget gate is what lets a signal from 96 steps back survive: a plain RNN multiplies
its hidden state by a weight matrix every step, so a 96-step-old signal has been through
that matrix 96 times and has either vanished or exploded. The forget gate lets the network
hold c_t nearly constant (f_t near 1) across long gaps and only update it when something
actually changes.

Normalization is fit on the TRAINING split only and reused for val/test -- using
validation/test statistics to normalize would leak future information into training.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from forecasting.features import add_cyclical_time_features

SEQ_LEN = 96      # 24h of history at 15-min resolution
HORIZON = 96      # predict 24h ahead -- a real day-ahead forecast
TIME_FEATURE_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]

# wind_mw and solar_mw are physical generation quantities that can never be negative (see
# data_loader.py, which already clips small negative wind readings caused by Elia's DSO
# upscaling noise). A trained regressor has no knowledge of this constraint and can predict
# a small negative number, especially near zero-output periods (nighttime solar, calm
# wind) -- so predictions for these two targets are clipped to >= 0 as a physically honest
# post-processing step, same spirit as the clip already applied to the raw data itself.
NON_NEGATIVE_TARGETS = ("wind_mw", "solar_mw")


class Standardizer:
    """Fit mean/std on one series, apply to any series. Fit ONLY on the training split."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, series: np.ndarray):
        self.mean = float(np.mean(series))
        self.std = float(np.std(series)) + 1e-8
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        return (series - self.mean) / self.std

    def inverse_transform(self, series: np.ndarray) -> np.ndarray:
        return series * self.std + self.mean


class SequenceDataset(Dataset):
    """Sliding windows of [target_scaled, *time_features] -> target_scaled at t+HORIZON."""

    def __init__(self, df: pd.DataFrame, target_col: str, standardizer: Standardizer,
                 seq_len: int = SEQ_LEN, horizon: int = HORIZON):
        df = add_cyclical_time_features(df, timestamp_col="timestamp")
        target_scaled = standardizer.transform(df[target_col].values.astype(np.float32))
        time_feats = df[TIME_FEATURE_COLS].values.astype(np.float32)
        self.X = np.concatenate([target_scaled.reshape(-1, 1), time_feats], axis=1)
        self.target_scaled = target_scaled
        # NOTE: keep this as a pandas Series, not .values -- calling .values on a
        # timezone-aware datetime column silently drops the tz info (converts to naive
        # datetime64[ns]), which then fails to merge against any tz-aware timestamp column.
        self.timestamps = df["timestamp"].reset_index(drop=True)
        self.seq_len = seq_len
        self.horizon = horizon
        self.n_samples = len(df) - seq_len - horizon + 1
        if self.n_samples <= 0:
            raise ValueError(f"Not enough rows ({len(df)}) for seq_len={seq_len} + horizon={horizon}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        window = self.X[idx: idx + self.seq_len]
        y = self.target_scaled[idx + self.seq_len + self.horizon - 1]
        return torch.from_numpy(window), torch.tensor(y, dtype=torch.float32)

    def target_timestamp(self, idx):
        """The timestamp the prediction at index idx is actually FOR (t+horizon), used to
        align our predictions with Elia's own forecast at that same instant."""
        return self.timestamps.iloc[idx + self.seq_len + self.horizon - 1]


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size) -- final layer's hidden state at the last step
        return self.head(last_hidden).squeeze(-1)


def naive_persistence_mse(dataset: SequenceDataset) -> float:
    """Baseline: predict the current value (at the end of the window) as the forecast for
    t+HORIZON. Since HORIZON=96=1 day, this is exactly the seasonal-naive "tomorrow same
    hour = today same hour" baseline -- any model worth using has to beat this. Note this
    is a much stronger baseline for load and solar (strong daily cycle) than for wind
    (weather-driven, weaker day-to-day repetition) -- we still report it for all three so
    every target is judged against the same honest bar."""
    errors = []
    for idx in range(len(dataset)):
        window, y = dataset[idx]
        pred = window[-1, 0].item()  # last observed (scaled) target value in the window
        errors.append((pred - y.item()) ** 2)
    return float(np.mean(errors))


def train_lstm(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_losses.append(loss_fn(pred, y).item())

        print(f"  epoch {epoch}/{epochs}: train MSE (scaled) = {np.mean(train_losses):.4f}, "
              f"val MSE (scaled) = {np.mean(val_losses):.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LSTM day-ahead forecaster for one grid quantity.")
    parser.add_argument("--target", default="load_mw", choices=["load_mw", "wind_mw", "solar_mw"],
                         help="Which column to forecast (default: load_mw, matches every earlier run).")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    target = args.target

    train_path = os.path.join("data", "processed", f"{target}_train.csv")
    val_path = os.path.join("data", "processed", f"{target}_val.csv")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print(f"Missing {train_path} or {val_path} -- run build_dataset.py first.")
        sys.exit(1)

    # These files already have lag/rolling feature columns from build_dataset.py, but the
    # LSTM builds its own sliding windows from the raw [timestamp, target] pair -- it
    # doesn't need pre-computed lags, that's the whole point of feeding it a sequence.
    train_raw = pd.read_csv(train_path, parse_dates=["timestamp"])[["timestamp", target]]
    val_raw = pd.read_csv(val_path, parse_dates=["timestamp"])[["timestamp", target]]

    scaler = Standardizer().fit(train_raw[target].values)
    print(f"Target: {target}")
    print(f"Fit standardizer on TRAIN only: mean={scaler.mean:.2f} MW, std={scaler.std:.2f} MW")

    train_ds = SequenceDataset(train_raw, target_col=target, standardizer=scaler)
    val_ds = SequenceDataset(val_raw, target_col=target, standardizer=scaler)
    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    naive_val_mse = naive_persistence_mse(val_ds)
    print(f"\nNaive persistence baseline (predict = value 24h ago) val MSE (scaled) = {naive_val_mse:.4f}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = LSTMForecaster(input_size=1 + len(TIME_FEATURE_COLS))
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("\nTraining...")
    model = train_lstm(model, train_loader, val_loader, epochs=args.epochs, device=device)

    # Report final val error in real MW, not scaled units, so it's actually interpretable.
    model.to("cpu")
    model.eval()
    errors_mw, n_clipped = [], 0
    with torch.no_grad():
        for X, y in val_loader:
            pred_scaled = model(X).numpy()
            y_scaled = y.numpy()
            pred_mw = scaler.inverse_transform(pred_scaled)
            y_mw = scaler.inverse_transform(y_scaled)
            if target in NON_NEGATIVE_TARGETS:
                n_clipped += int((pred_mw < 0).sum())
                pred_mw = np.clip(pred_mw, 0.0, None)
            errors_mw.extend((pred_mw - y_mw).tolist())
    errors_mw = np.array(errors_mw)

    if target in NON_NEGATIVE_TARGETS and n_clipped > 0:
        print(f"\n(clipped {n_clipped} negative {target} prediction(s) to 0.0 -- physically "
              f"impossible for a generation quantity, same reasoning as data_loader.py's "
              f"wind clip)")

    print(f"\nFinal val MAE = {np.mean(np.abs(errors_mw)):.1f} MW, "
          f"RMSE = {np.sqrt(np.mean(errors_mw ** 2)):.1f} MW")
    naive_rmse_mw = np.sqrt(naive_val_mse) * scaler.std
    print(f"Naive persistence RMSE (for comparison) = {naive_rmse_mw:.1f} MW")

    os.makedirs("outputs", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "target": target,
    }, os.path.join("outputs", f"lstm_{target.replace('_mw', '')}_forecaster.pt"))
    print(f"\nSaved outputs/lstm_{target.replace('_mw', '')}_forecaster.pt")