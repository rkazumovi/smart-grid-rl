"""
Probabilistic forecasting: extends the LSTM backbone from lstm_model.py to output three
quantiles (10th, 50th, 90th percentile) instead of one point estimate, trained with the
pinball (quantile) loss:

    L_tau(y, y_hat) = tau * (y - y_hat)        if y >= y_hat
                    = (1 - tau) * (y_hat - y)   if y < y_hat

For tau=0.5 this reduces to MAE (symmetric under/over-estimation cost) -- minimizing it
recovers the median. For tau=0.9, underestimating (y > y_hat) costs 9x as much as
overestimating, which pushes y_hat up until only ~10% of actual outcomes exceed it: that's
exactly the definition of the 90th percentile. Training all three quantiles jointly (one
network, three output heads, losses summed) gives a genuine [q10, q90] uncertainty band
instead of a single number that hides how confident the model actually is.

Known issue: nothing forces q10 <= q50 <= q90 at every single prediction (each quantile is
a separate output computed independently) -- this is the well-documented "quantile
crossing" problem. We report how often it happens, then fix it at inference time by
sorting the three predicted values, which is a standard, honest practical remedy (not a
retraining fix, but a correct one: sorting three numbers that were MEANT to be ordered
cannot make the forecast worse, only enforces the constraint the loss doesn't guarantee).

Now generalized (via --target) the same way lstm_model.py was: it can forecast any of the
three grid quantities produced by build_dataset.py (load_mw default, wind_mw, solar_mw).
For wind_mw/solar_mw, all three quantiles (q10, q50, q90) get the same non-negative clip
lstm_model.py already applies to its single point prediction -- a generation quantity
can't go below 0 MW at any confidence level, so a negative lower or median quantile is
just as physically wrong as a negative point forecast.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forecasting.lstm_model import SequenceDataset, Standardizer, TIME_FEATURE_COLS, NON_NEGATIVE_TARGETS

QUANTILES = (0.1, 0.5, 0.9)


class QuantileLSTM(nn.Module):
    """Same LSTM backbone as LSTMForecaster, but the head outputs len(QUANTILES) values."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.1, quantiles=QUANTILES):
        super().__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # (batch, len(quantiles))


def pinball_loss(preds: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """preds: (batch, n_quantiles), target: (batch,). Returns the mean pinball loss
    summed across quantiles."""
    target = target.unsqueeze(-1)  # (batch, 1) to broadcast against (batch, n_quantiles)
    errors = target - preds
    losses = []
    for i, tau in enumerate(quantiles):
        e = errors[:, i]
        losses.append(torch.max(tau * e, (tau - 1) * e))
    return torch.stack(losses, dim=1).mean()


def train_quantile_lstm(model, train_loader, val_loader, quantiles=QUANTILES,
                         epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            preds = model(X)
            loss = pinball_loss(preds, y, quantiles)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_losses.append(pinball_loss(model(X), y, quantiles).item())

        print(f"  epoch {epoch}/{epochs}: train pinball loss (scaled) = {np.mean(train_losses):.4f}, "
              f"val pinball loss (scaled) = {np.mean(val_losses):.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a probabilistic (quantile) LSTM day-ahead forecaster for one grid quantity.")
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

    train_raw = pd.read_csv(train_path, parse_dates=["timestamp"])[["timestamp", target]]
    val_raw = pd.read_csv(val_path, parse_dates=["timestamp"])[["timestamp", target]]

    scaler = Standardizer().fit(train_raw[target].values)
    print(f"Target: {target}")
    print(f"Fit standardizer on TRAIN only: mean={scaler.mean:.2f} MW, std={scaler.std:.2f} MW")

    train_ds = SequenceDataset(train_raw, target_col=target, standardizer=scaler)
    val_ds = SequenceDataset(val_raw, target_col=target, standardizer=scaler)
    print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = QuantileLSTM(input_size=1 + len(TIME_FEATURE_COLS))
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters, "
          f"quantiles = {QUANTILES}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("\nTraining...")
    model = train_quantile_lstm(model, train_loader, val_loader, epochs=args.epochs, device=device)

    model.to("cpu")
    model.eval()
    all_preds_mw, all_actuals_mw = [], []
    with torch.no_grad():
        for X, y in val_loader:
            preds_scaled = model(X).numpy()  # (batch, 3)
            preds_mw = scaler.inverse_transform(preds_scaled)
            actuals_mw = scaler.inverse_transform(y.numpy())
            all_preds_mw.append(preds_mw)
            all_actuals_mw.append(actuals_mw)
    all_preds_mw = np.concatenate(all_preds_mw, axis=0)   # (n, 3): [q10, q50, q90]
    all_actuals_mw = np.concatenate(all_actuals_mw, axis=0)

    q10, q50, q90 = all_preds_mw[:, 0], all_preds_mw[:, 1], all_preds_mw[:, 2]

    n_crossed = np.sum((q10 > q50) | (q50 > q90))
    print(f"\nQuantile crossing (q10>q50 or q50>q90, before fixing): "
          f"{n_crossed}/{len(q10)} ({100 * n_crossed / len(q10):.1f}%)")

    # Enforce monotonicity by sorting -- a valid post-hoc fix since these three numbers
    # were always meant to be ordered; sorting can only correct violations, not introduce them.
    sorted_preds = np.sort(all_preds_mw, axis=1)

    n_clipped = 0
    if target in NON_NEGATIVE_TARGETS:
        # Clipping is applied AFTER sorting, not before: clipping is monotonic (it can
        # only move a value up towards 0, never past a value that was already above it),
        # so it cannot undo the ordering the sort just enforced.
        n_clipped = int((sorted_preds < 0).sum())
        sorted_preds = np.clip(sorted_preds, 0.0, None)

    q10, q50, q90 = sorted_preds[:, 0], sorted_preds[:, 1], sorted_preds[:, 2]

    if target in NON_NEGATIVE_TARGETS and n_clipped > 0:
        print(f"\n(clipped {n_clipped} negative {target} quantile prediction(s) across "
              f"q10/q50/q90 to 0.0 -- physically impossible for a generation quantity, "
              f"same reasoning as data_loader.py's wind clip)")

    median_mae = np.mean(np.abs(q50 - all_actuals_mw))
    median_rmse = np.sqrt(np.mean((q50 - all_actuals_mw) ** 2))
    print(f"\nMedian (q50) forecast: MAE = {median_mae:.1f} MW, RMSE = {median_rmse:.1f} MW")

    coverage = np.mean((all_actuals_mw >= q10) & (all_actuals_mw <= q90))
    print(f"Empirical coverage of the [q10, q90] interval: {100 * coverage:.1f}% "
          f"(target: 80%, since q90 - q10 should bracket the middle 80% of outcomes)")
    mean_interval_width = np.mean(q90 - q10)
    print(f"Mean [q10, q90] interval width: {mean_interval_width:.1f} MW")

    os.makedirs("outputs", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "seq_len": train_ds.seq_len,
        "horizon": train_ds.horizon,
        "quantiles": QUANTILES,
        "target": target,
    }, os.path.join("outputs", f"probabilistic_lstm_{target.replace('_mw', '')}_forecaster.pt"))
    print(f"\nSaved outputs/probabilistic_lstm_{target.replace('_mw', '')}_forecaster.pt")