"""
Shared state passed between the four LangGraph nodes in the Grid Intelligence pipeline
(graph.py). Every field is written by exactly one node, so no merge/reducer logic is
needed -- each agent only ever writes to its own field(s), and LangGraph's default
"last write wins" behavior for a plain (non-Annotated) TypedDict field is exactly right
here since nothing is ever written twice.
"""
from typing import TypedDict


class GridIntelligenceState(TypedDict, total=False):
    # ---- input, provided by the caller (see run_pipeline.py) ----
    question: str                    # what the operator/user actually asked
    load_actual_mw: float
    load_forecast_mw: float
    wind_actual_mw: float
    wind_forecast_mw: float
    solar_actual_mw: float
    solar_forecast_mw: float
    policy_name: str                 # e.g. "MARL", "PPO", "SAC", "Heuristic"
    proposed_action: str             # plain-language description of what the policy recommends right now

    # ---- written by each agent, in pipeline order ----
    forecast_summary: str            # Forecaster Interpreter's output
    optimization_summary: str        # Optimization Advisor's output
    risk_summary: str                # Risk/Anomaly Detector's output
    final_report: str                # Report Synthesizer's output -- the end result