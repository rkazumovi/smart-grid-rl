"""
Time-series feature engineering for the load/renewable forecasting module.

Math notes (full writeup goes in notebooks/forecasting_theory.ipynb once we build the
models that consume these features):

Cyclical time encoding
-----------------------
A raw hour-of-day value h in {0, ..., 23} has a false discontinuity: hour 23 and hour 0
are one step apart in real time but 23 apart numerically. Feeding that gap to a neural
net teaches it a wrong distance metric. The standard fix is to embed the periodic
variable on the unit circle:

    hour_sin = sin(2*pi*h / 24)
    hour_cos = cos(2*pi*h / 24)

so that hour 23 and hour 0 map to nearby points on the circle. The same construction
is applied to day-of-week (period 7) and month-of-year (period 12).

Lag and rolling features
-------------------------
For a target series x_t (e.g. grid load in MW), the model is given:
    x_{t-1}   (previous hour       -- short-term persistence)
    x_{t-24}  (same hour, yesterday -- daily seasonality)
    x_{t-168} (same hour, last week -- weekly seasonality)
    rolling_mean_24 = (1/24) * sum_{k=1}^{24} x_{t-k}   (recent local level)
    rolling_std_24  = std of the same window            (recent local volatility)

These are the classical, cheap-to-compute features that let even a small model capture
most of the seasonal structure in grid load before the recurrent/attention layers have
to learn anything about long-range dependence from scratch.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def add_cyclical_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Add sin/cos encodings for hour-of-day, day-of-week, and month-of-year."""
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    hour = ts.dt.hour + ts.dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = ts.dt.dayofweek  # Monday=0
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    month = ts.dt.month - 1  # 0-indexed
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    df["is_weekend"] = (dow >= 5).astype(float)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str,
                                  lags=(1, 24, 168), rolling_windows=(24,)) -> pd.DataFrame:
    """
    Add lag_{k} and rolling_mean_{w}/rolling_std_{w} features for the target column.
    Assumes df is already sorted in ascending time order with a fixed (e.g. hourly) step.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    for w in rolling_windows:
        shifted = df[target_col].shift(1)  # never leak the current-step target into its own features
        df[f"rolling_mean_{w}"] = shifted.rolling(window=w).mean()
        df[f"rolling_std_{w}"] = shifted.rolling(window=w).std()
    return df


def build_features(df: pd.DataFrame, target_col: str = "load_mw", timestamp_col: str = "timestamp",
                    lags=(1, 24, 168), rolling_windows=(24,), dropna: bool = True) -> pd.DataFrame:
    """Full feature pipeline: cyclical time features + lag/rolling features on target_col."""
    out = add_cyclical_time_features(df, timestamp_col=timestamp_col)
    out = add_lag_and_rolling_features(out, target_col=target_col, lags=lags, rolling_windows=rolling_windows)
    if dropna:
        out = out.dropna().reset_index(drop=True)
    return out


def train_val_test_split_by_time(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Chronological split (never shuffle time series!): the model must only ever be
    evaluated on data that comes strictly after everything it was trained on, or the
    "forecast" is silently peeking at the future.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].reset_index(drop=True), \
        df.iloc[train_end:val_end].reset_index(drop=True), \
        df.iloc[val_end:].reset_index(drop=True)


if __name__ == "__main__":
    # Synthetic hourly load series for 90 days: daily + weekly seasonality + trend + noise,
    # standing in for real data until we wire up the Elia/ENTSO-E/NREL feeds in the next step.
    rng = np.random.default_rng(0)
    n_hours = 24 * 90
    timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")
    hour_of_day = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values

    daily_shape = 500 * np.sin(2 * np.pi * (hour_of_day - 6) / 24.0) + 500  # peak ~ midday-evening
    weekend_discount = np.where(day_of_week >= 5, -150.0, 0.0)
    trend = np.linspace(0, 100, n_hours)  # slow seasonal drift
    noise = rng.normal(0, 30, n_hours)
    load_mw = 1000 + daily_shape + weekend_discount + trend + noise

    raw = pd.DataFrame({"timestamp": timestamps, "load_mw": load_mw})
    print(f"Synthetic raw series: {len(raw)} hourly rows, "
          f"load range [{raw['load_mw'].min():.1f}, {raw['load_mw'].max():.1f}] MW")

    featured = build_features(raw, target_col="load_mw")
    print(f"\nAfter feature engineering: {len(featured)} rows "
          f"(lost {len(raw) - len(featured)} rows to the max lag, 168h = 1 week, as expected)")
    print(f"Columns: {list(featured.columns)}")

    print("\nSanity checks:")
    radius = featured["hour_sin"] ** 2 + featured["hour_cos"] ** 2
    print(f"  hour_sin^2 + hour_cos^2 (should be ~1.0 always): "
          f"min={radius.min():.6f}, max={radius.max():.6f}")

    raw_indexed = raw.set_index("timestamp")["load_mw"]
    check_row = featured.iloc[100]
    expected_lag1 = raw_indexed.loc[check_row["timestamp"] - pd.Timedelta(hours=1)]
    print(f"  lag_1 check at row 100: feature={check_row['lag_1']:.3f}, "
          f"expected from raw series={expected_lag1:.3f}, "
          f"match={np.isclose(check_row['lag_1'], expected_lag1)}")

    dow_check = pd.to_datetime(featured["timestamp"]).dt.dayofweek
    weekend_matches = ((dow_check >= 5).astype(float) == featured["is_weekend"]).all()
    print(f"  is_weekend matches pandas dayofweek>=5 for all rows: {weekend_matches}")

    train_df, val_df, test_df = train_val_test_split_by_time(featured)
    print(f"\nChronological split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"  train ends at {train_df['timestamp'].iloc[-1]}, "
          f"val starts at {val_df['timestamp'].iloc[0]} (should be immediately after)")
    print(f"  val ends at {val_df['timestamp'].iloc[-1]}, "
          f"test starts at {test_df['timestamp'].iloc[0]} (should be immediately after)")