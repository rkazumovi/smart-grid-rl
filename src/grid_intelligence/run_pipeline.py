"""
Runs the full four-agent Grid Intelligence pipeline once, using the most recent row of
real merged grid data (data/processed/grid_merged.csv) as the "actual" values, Elia's
own real day-ahead forecast as the load forecast, and a real inference call against
whichever trained RL policy POLICY_NAME names (default "MARL") for the proposed action --
see policy_inference.py's module docstring for the important scope caveat: that policy
runs on its own synthetic IEEE 14-bus environment, not the real Belgian grid the
forecasts above describe. If the checkpoint isn't found or inference fails for any
reason, this falls back to a clearly-labeled placeholder action rather than crashing the
whole pipeline over what is, after all, just one input field.

Requires:
  1. The RAG index already built: python src\\grid_intelligence\\build_rag_index.py
  2. An LLM backend, chosen via the LLM_BACKEND environment variable:
       LLM_BACKEND=ollama (default) -- local, free, needs Ollama installed and a model
         pulled (see ollama_client.py). Override the model with OLLAMA_MODEL, the server
         URL with OLLAMA_URL.
       LLM_BACKEND=anthropic -- real API calls, needs ANTHROPIC_API_KEY set (see
         llm_client.py) and incurs a small real cost per run.
  3. (Optional, for a real optimization recommendation instead of the placeholder) a
     trained checkpoint under outputs/ for whichever POLICY_NAME you set ("MARL", "PPO",
     or "SAC" -- default "MARL"), plus stable_baselines3 installed.
Run from the project root:
  python src\\grid_intelligence\\run_pipeline.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from grid_intelligence.rag import RagRetriever
from grid_intelligence.graph import build_grid_intelligence_graph


def build_llm_client():
    backend = os.environ.get("LLM_BACKEND", "ollama").lower()
    if backend == "anthropic":
        from grid_intelligence.llm_client import AnthropicClient
        return AnthropicClient()
    elif backend == "ollama":
        from grid_intelligence.ollama_client import OllamaClient
        return OllamaClient()
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}' -- use 'ollama' or 'anthropic'.")


def get_proposed_action(policy_name: str, seed: int = 0) -> str:
    """Tries a real inference call against the trained `policy_name` checkpoint (see
    policy_inference.py); falls back to a clearly-labeled placeholder, with a printed
    reason, if the checkpoint is missing or inference fails for any reason (e.g.
    stable_baselines3 not installed, a shape mismatch from a stale checkpoint)."""
    try:
        from grid_intelligence.policy_inference import get_policy_action
        _, description, _, _ = get_policy_action(policy_name, seed=seed)
        return description
    except Exception as e:
        print(f"  (live {policy_name} policy inference unavailable ({e}) -- using a placeholder action)")
        return (
            "[PLACEHOLDER -- no live policy inference available] "
            "Discharge battery at 200 kW during the 18:00-20:00 peak window; "
            "import the remaining shortfall from the grid."
        )


def build_initial_state_from_latest_row(merged_path: str) -> dict:
    df = pd.read_csv(merged_path, parse_dates=["timestamp"])
    row = df.iloc[-1]
    load_forecast = row.get("elia_dayahead_forecast_load_mw")
    policy_name = os.environ.get("POLICY_NAME", "MARL")
    return {
        "question": "What should the grid operator know going into the next day-ahead window?",
        "load_actual_mw": float(row["load_mw"]),
        "load_forecast_mw": float(load_forecast) if pd.notna(load_forecast) else None,
        "wind_actual_mw": float(row["wind_mw"]),
        # No standalone next-day wind/solar forecast is wired into grid_merged.csv itself
        # (those come from outputs/lstm_wind_forecaster.pt / lstm_solar_forecaster.pt at
        # inference time) -- left as None here rather than guessing a number.
        "wind_forecast_mw": None,
        "solar_actual_mw": float(row["solar_mw"]),
        "solar_forecast_mw": None,
        "policy_name": policy_name,
        "proposed_action": get_proposed_action(policy_name),
    }


if __name__ == "__main__":
    merged_path = os.path.join("data", "processed", "grid_merged.csv")
    if not os.path.exists(merged_path):
        print(f"Missing {merged_path} -- run src/forecasting/data_loader.py first.")
        sys.exit(1)

    print("Building initial state from the most recent row of real merged grid data...")
    initial_state = build_initial_state_from_latest_row(merged_path)
    for k, v in initial_state.items():
        print(f"  {k}: {v}")

    print("\nLoading RAG index...")
    retriever = RagRetriever()

    backend = os.environ.get("LLM_BACKEND", "ollama").lower()
    print(f"Connecting to LLM backend: {backend}...")
    llm = build_llm_client()

    print("\nRunning the 4-agent pipeline...\n")
    graph = build_grid_intelligence_graph(retriever, llm)
    final_state = graph.invoke(initial_state)

    for title, key in [
        ("FORECAST INTERPRETATION", "forecast_summary"),
        ("OPTIMIZATION ADVICE", "optimization_summary"),
        ("RISK ASSESSMENT", "risk_summary"),
        ("FINAL REPORT", "final_report"),
    ]:
        print("=" * 60)
        print(title)
        print("=" * 60)
        print(final_state[key])
        print()