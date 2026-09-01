"""
Transformer point forecaster: same task and same input format as lstm_model.py -- a 24h
window (96 steps at 15-min resolution) of [target, cyclical time features] -> the target
24h (96 steps) ahead -- so its result is directly comparable to the LSTM's.

Now generalized (via --target) the same way lstm_model.py was: it can forecast any of the
three grid quantities produced by build_dataset.py (load_mw default, wind_mw, solar_mw).
The architecture, windowing, and training loop are identical across all three -- only the
input column and, for the two generation targets, a physical post-processing clip differ.

Self-attention (see notebooks/forecasting_theory.ipynb for the full derivation once it's
built): for the whole window packed into a matrix X (seq_len x d_model),
    Q = X W_Q,  K = X W_K,  V = X W_V
    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
Q K^T is a (seq_len x seq_len) matrix of pairwise relevance scores between every pair of
positions; dividing by sqrt(d_k) keeps those dot products from growing large with
dimension (which would push softmax into a near-one-hot regime with vanishing gradients).
The output at each position is a weighted average of every position's value vector,
weighted by relevance -- so information from 672 steps back reaches the output in ONE
step, unlike the LSTM's forget gate which has to relay it through 672 sequential updates.

Multi-head attention just runs several of these in parallel with different learned
projections and concatenates the results, so different heads can specialize (e.g. one
head attending mostly to yesterday-same-hour, another to last-week-same-hour).

Since attention has no built-in sense of order (softmax over Q K^T is permutation-
invariant), a sinusoidal positional encoding is added to the embedded input:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
so nearby positions get nearby encodings and the model can distinguish "15 minutes ago"
from "a week ago" even though attention itself doesn't track position.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forecasting.lstm_model import SequenceDataset, Standardizer, naive_persistence_mse, TIME_FEATURE_COLS, NON_NEGATIVE_TARGETS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model), not a learned parameter

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1), :]


class TransformerForecaster(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size). No causal mask: this is a fixed, fully-observed
        # history window, not autoregressive generation, so every position may attend to
        # every other position -- there's no future leakage since the prediction target
        # (t+HORIZON) lies entirely outside this window.
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        last_position = h[:, -1, :]  # (batch, d_model) -- same "read out the last step" convention as the LSTM
        return self.head(last_position).squeeze(-1)


def train_transformer(model, train_loader, val_loader, epochs=5, lr=1e-3, device="cpu"):
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
                val_losses.append(loss_fn(model(X), y).item())

        print(f"  epoch {epoch}/{epochs}: train MSE (scaled) = {np.mean(train_losses):.4f}, "
              f"val MSE (scaled) = {np.mean(val_losses):.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer day-ahead forecaster for one grid quantity.")
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

    naive_val_mse = naive_persistence_mse(val_ds)
    print(f"\nNaive persistence baseline val MSE (scaled) = {naive_val_mse:.4f}")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = TransformerForecaster(input_size=1 + len(TIME_FEATURE_COLS))
    print(f"\nModel: {sum(p.numel() for p in model.parameters())} parameters")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    print("\nTraining...")
    model = train_transformer(model, train_loader, val_loader, epochs=args.epochs, device=device)

    model.to("cpu")
    model.eval()
    errors_mw, n_clipped = [], 0
    with torch.no_grad():
        for X, y in val_loader:
            pred_mw = scaler.inverse_transform(model(X).numpy())
            y_mw = scaler.inverse_transform(y.numpy())
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
        "seq_len": train_ds.seq_len,
        "horizon": train_ds.horizon,
        "target": target,
    }, os.path.join("outputs", f"transformer_{target.replace('_mw', '')}_forecaster.pt"))
    print(f"\nSaved outputs/transformer_{target.replace('_mw', '')}_forecaster.pt")