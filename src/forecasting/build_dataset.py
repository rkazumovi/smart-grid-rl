"""
Turn the merged real Elia dataset (data/processed/grid_merged.csv, 15-min resolution)
into three independent, model-ready forecasting datasets -- one each for load, wind, and
solar -- with chronological train/val/test splits.

Why three separate targets instead of one multi-output model: load, wind, and solar have
different physical drivers (grid demand patterns vs. weather), and using a target's own
lagged history as a feature (as features.py does) means each target needs its own lag
columns computed from its own series. Keeping them as separate univariate problems for
now avoids a subtler issue too: solar_mw and wind_mw are simultaneous MEASUREMENTS, not
forecasts, so using today's actual wind_mw to help "forecast" today's actual load_mw
would be leaking a same-timestamp measurement into what should be a forward-looking
prediction. (A model that fuses power sources for a real multi-step-ahead forecast is a
reasonable extension later, but it needs each source's own forecast, not its measurement,
as the cross-feature -- exactly the elia_forecast_load_mw column already gives us for load.)

Lag/rolling windows are expressed in 15-min steps to match this dataset's native
resolution (unlike features.py's own __main__ demo, which used hourly synthetic data):
    lag_1   = 15 minutes ago       (short-term persistence)
    lag_96  = 24 hours ago         (24h * 4 steps/hour = 96)
    lag_672 = 1 week ago           (7d * 96 steps/day = 672)
    rolling_mean_96 / rolling_std_96 = trailing 24h window
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from forecasting.features import build_features, train_val_test_split_by_time

LAGS_15MIN = (1, 96, 672)
ROLLING_15MIN = (96,)


def build_target_dataset(merged: pd.DataFrame, target_col: str) -> pd.DataFrame:
    subset = merged[["timestamp", target_col]].copy()
    return build_features(subset, target_col=target_col, lags=LAGS_15MIN, rolling_windows=ROLLING_15MIN)


if __name__ == "__main__":
    merged_path = os.path.join("data", "processed", "grid_merged.csv")
    if not os.path.exists(merged_path):
        print(f"Missing {merged_path} -- run data_loader.py first.")
        sys.exit(1)

    merged = pd.read_csv(merged_path, parse_dates=["timestamp"])
    print(f"Loaded {len(merged)} rows from {merged_path}")

    out_dir = os.path.join("data", "processed")
    for target_col in ("load_mw", "wind_mw", "solar_mw"):
        featured = build_target_dataset(merged, target_col)
        train_df, val_df, test_df = train_val_test_split_by_time(featured)

        print(f"\n{target_col}: {len(merged)} raw rows -> {len(featured)} feature rows "
              f"(lost {len(merged) - len(featured)} to the {max(LAGS_15MIN)}-step max lag = "
              f"{max(LAGS_15MIN) / 96:.1f} days)")
        print(f"  train={len(train_df)} ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
        print(f"  val=  {len(val_df)} ({val_df['timestamp'].min()} -> {val_df['timestamp'].max()})")
        print(f"  test= {len(test_df)} ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")

        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            split_df.to_csv(os.path.join(out_dir, f"{target_col}_{split_name}.csv"), index=False)

    print(f"\nWrote 9 files (3 targets x train/val/test) to {out_dir}")