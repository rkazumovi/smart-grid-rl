"""
Loaders for the three real Elia Open Data Portal CSVs (Belgian grid):
  ods001 -- Total load (measured + Elia's own forecasts), one row per 15-min timestamp.
  ods031 -- Wind generation, but broken down into one row PER TIMESTAMP PER
            (offshore/onshore x region x grid-connection-type) combination -- there is
            NO single national total column, it has to be summed across those rows.
  ods032 -- Solar generation, same idea but broken down per region only.

Two things about this data that are easy to get wrong silently:

1. Timezone / DST. Elia's Datetime column already carries an explicit UTC offset
   (+01:00 in winter / +02:00 in summer, i.e. CET/CEST with the DST switch baked in).
   Parsing these as *naive* local timestamps is dangerous: on the one night per year
   clocks fall back, the same naive wall-clock time (e.g. 02:30) occurs twice, and on
   the night they spring forward, one wall-clock hour never occurs at all -- both cases
   silently corrupt a naive time index. Because the offset is already in the string,
   `pd.to_datetime(..., utc=True)` resolves every timestamp to one unambiguous instant
   and sidesteps both problems entirely, so we always parse with utc=True.

2. Aggregation. ods031/ods032 must be grouped by Datetime and summed across every
   region/type combination to get one national-total generation figure per timestamp --
   using the raw per-region rows directly would undercount by roughly the number of
   regions -- UNLESS one of those "regions" is itself an aggregate of the others, in
   which case naively summing everything double- (or triple-) counts. describe_categories()
   and check_for_aggregate_row() below exist specifically to catch that before it happens;
   see SOLAR_AGGREGATE_REGIONS for the real case this caught.

3. Row order. Elia's export is newest-first; every loader below re-sorts ascending,
   since every downstream consumer (feature lags, chronological train/val/test splits)
   assumes ascending time order.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _read_elia_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["Datetime"], utc=True)
    return df


def load_elia_total_load(path: str) -> pd.DataFrame:
    """Returns [timestamp, load_mw, elia_forecast_load_mw, elia_dayahead_forecast_load_mw],
    ascending by timestamp.

    Two different Elia forecast columns are kept, and they are NOT interchangeable:
      - elia_forecast_load_mw ("Most recent forecast") continuously updates as the
        delivery time approaches -- it's a nowcast, an easy target, not comparable to a
        genuine day-ahead model.
      - elia_dayahead_forecast_load_mw ("Day-ahead 6PM forecast") is published the
        previous day at 6PM and never updated after that -- this is the real,
        apples-to-apples baseline for our own 24h-ahead LSTM/Transformer forecasts.
    """
    df = _read_elia_csv(path)
    out = df[["timestamp", "Total Load", "Most recent forecast", "Day-ahead 6PM forecast"]].rename(
        columns={"Total Load": "load_mw", "Most recent forecast": "elia_forecast_load_mw",
                 "Day-ahead 6PM forecast": "elia_dayahead_forecast_load_mw"}
    )
    out = out.dropna(subset=["load_mw"]).drop_duplicates(subset=["timestamp"], keep="last")
    return out.sort_values("timestamp").reset_index(drop=True)


def load_elia_wind(path: str) -> pd.DataFrame:
    """Returns [timestamp, wind_mw] -- summed across ALL region/onshore-offshore/
    grid-connection-type rows for each timestamp, i.e. Belgium-wide total wind generation.
    Unlike solar, these categories genuinely are a disjoint partition (Wallonia + Flanders
    for onshore, "Federal" for offshore since it sits in federal waters, outside any
    region) -- confirmed by check_for_aggregate_row() finding no aggregate signature and
    by the row counts being internally consistent across all three grouping columns.

    Negative values are clipped to 0: wind generation cannot physically be negative, but
    Elia's DSO "upscaled" estimate (statistically extrapolated from a sample of monitored
    installations to the full regional fleet) can dip slightly below zero from estimation
    noise during near-zero-output periods. This is a known property of the upscaling
    model, not a real negative reading.
    """
    df = _read_elia_csv(path)
    totals = df.groupby("timestamp", as_index=False)["Measured & Upscaled"].sum()
    totals = totals.rename(columns={"Measured & Upscaled": "wind_mw"})
    n_negative = (totals["wind_mw"] < 0).sum()
    if n_negative > 0:
        print(f"  (clipping {n_negative} negative wind_mw reading(s) to 0.0 -- "
              f"DSO upscaling estimation noise, not physically real)")
        totals["wind_mw"] = totals["wind_mw"].clip(lower=0.0)
    return totals.sort_values("timestamp").reset_index(drop=True)


# Elia's solar "Region" column is a flat listing of a real 3-level Belgian administrative
# hierarchy, not a disjoint partition: Belgium (national total) contains Flanders and
# Wallonia (each the sum of their provinces) plus Brussels (no provinces of its own), while
# the 10 provinces are the actual leaves. Summing all 14 categories triple-counts most of
# the country. These are the three aggregate rows to exclude before summing.
SOLAR_AGGREGATE_REGIONS = ("Belgium", "Flanders", "Wallonia")


def load_elia_solar(path: str, aggregate_regions=SOLAR_AGGREGATE_REGIONS) -> pd.DataFrame:
    """Returns [timestamp, solar_mw] -- summed across only the LEAF region rows per
    timestamp (the 10 provinces + Brussels), excluding the Belgium/Flanders/Wallonia
    aggregate rows that would otherwise triple-count most of the country.
    """
    df = _read_elia_csv(path)
    leaves = df[~df["Region"].isin(aggregate_regions)]
    totals = leaves.groupby("timestamp", as_index=False)["Measured & Upscaled"].sum()
    return totals.rename(columns={"Measured & Upscaled": "solar_mw"}).sort_values("timestamp").reset_index(drop=True)


def describe_categories(path: str, group_cols) -> None:
    """Print every distinct value (and row count) for each grouping column, plus flag
    any category whose value looks like it's an aggregate of the others rather than a
    disjoint part -- see check_for_aggregate_row() below for the detection logic."""
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    print(f"  {os.path.basename(path)}: {len(df)} total rows")
    for col in group_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        print(f"    {col!r}: {df[col].nunique()} distinct value(s) -> {dict(counts)}")


def check_for_aggregate_row(path: str, group_cols, value_col: str = "Measured & Upscaled",
                             tol: float = 0.02) -> None:
    """
    Heuristic aggregate-row detector: if a "total"/"Belgium"-style row X sits alongside
    disjoint parts that sum to X, then at any given timestamp:
        value(X) == sum(all other rows) == total_sum - value(X)
        =>  value(X) == total_sum / 2   (exactly, if there's one aggregate + its parts)
    So we pick the timestamp with the most rows (most likely to include every category),
    and flag any category whose value is suspiciously close to exactly half the total --
    that's the signature of double-counting a whole alongside its parts.
    This won't catch every possible multi-level hierarchy, but it catches the common
    "regions + a national total" case, which is what produced solar_mw ~4x too high here.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["Datetime"], utc=True)
    counts_per_ts = df.groupby("timestamp").size()
    busiest_ts = counts_per_ts.idxmax()
    sample = df[df["timestamp"] == busiest_ts]

    total = sample[value_col].sum()
    if total == 0:
        return
    flagged = []
    for _, row in sample.iterrows():
        half_total = total / 2.0
        if half_total != 0 and abs(row[value_col] - half_total) / abs(half_total) < tol:
            key = tuple(row[c] for c in group_cols)
            flagged.append((key, row[value_col]))
    if flagged:
        print(f"    WARNING: at {busiest_ts}, these categories are suspiciously close to "
              f"HALF the total ({total:.1f}) -- likely an aggregate row double-counted "
              f"alongside its own parts, exclude it before summing:")
        for key, val in flagged:
            print(f"      {dict(zip(group_cols, key))} = {val:.2f}")
    else:
        print(f"    No aggregate-row signature detected at {busiest_ts} "
              f"(total={total:.1f} across {len(sample)} rows) -- looks like a clean partition.")


def merge_grid_datasets(load_df: pd.DataFrame, wind_df: pd.DataFrame, solar_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join the three series on timestamp -- all three are PT15M resolution,
    so an inner join keeps only timestamps where all three actually have data (real
    open-data feeds occasionally have small gaps in one series but not the others)."""
    merged = load_df.merge(wind_df, on="timestamp", how="inner").merge(solar_df, on="timestamp", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    load_path = os.path.join("data", "raw", "elia_load.csv")
    wind_path = os.path.join("data", "raw", "elia_wind.csv")
    solar_path = os.path.join("data", "raw", "elia_solar.csv")

    missing = [p for p in (load_path, wind_path, solar_path) if not os.path.exists(p)]
    if missing:
        print("Missing file(s), run this from the smart-grid-rl directory:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    print("=== Category breakdown (checking for double-counted aggregate rows) ===")
    describe_categories(wind_path, ["Offshore/onshore", "Region", "Grid connection type"])
    check_for_aggregate_row(wind_path, ["Offshore/onshore", "Region", "Grid connection type"])
    describe_categories(solar_path, ["Region"])
    check_for_aggregate_row(solar_path, ["Region"])
    print()

    load_df = load_elia_total_load(load_path)
    wind_df = load_elia_wind(wind_path)
    solar_df = load_elia_solar(solar_path)

    print(f"Load:  {len(load_df)} rows, {load_df['timestamp'].min()} -> {load_df['timestamp'].max()}")
    print(f"Wind:  {len(wind_df)} rows, {wind_df['timestamp'].min()} -> {wind_df['timestamp'].max()}")
    print(f"Solar: {len(solar_df)} rows, {solar_df['timestamp'].min()} -> {solar_df['timestamp'].max()}")

    print("\nSanity checks:")
    for name, d in [("load", load_df), ("wind", wind_df), ("solar", solar_df)]:
        is_sorted = d["timestamp"].is_monotonic_increasing
        n_dupes = d["timestamp"].duplicated().sum()
        print(f"  {name}: sorted ascending = {is_sorted}, duplicate timestamps = {n_dupes}")

    print(f"\n  load_mw range:  [{load_df['load_mw'].min():.1f}, {load_df['load_mw'].max():.1f}] MW "
          f"(sanity: Belgium's total grid load is roughly 6,000-14,000 MW)")
    print(f"  wind_mw range:  [{wind_df['wind_mw'].min():.1f}, {wind_df['wind_mw'].max():.1f}] MW")
    print(f"  solar_mw range: [{solar_df['solar_mw'].min():.1f}, {solar_df['solar_mw'].max():.1f}] MW "
          f"(sanity: should include 0.0 at night, peak should be in the low thousands of MW "
          f"-- Belgium's total installed solar capacity, not tens of thousands)")

    merged = merge_grid_datasets(load_df, wind_df, solar_df)
    print(f"\nMerged dataset: {len(merged)} rows (from load={len(load_df)}, "
          f"wind={len(wind_df)}, solar={len(solar_df)} -- inner join keeps only timestamps common to all three)")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nFirst 3 merged rows:\n{merged.head(3).to_string(index=False)}")

    out_path = os.path.join("data", "processed")
    os.makedirs(out_path, exist_ok=True)
    merged.to_csv(os.path.join(out_path, "grid_merged.csv"), index=False)
    print(f"\nSaved merged dataset to {os.path.join(out_path, 'grid_merged.csv')}")