"""
The four Grid Intelligence agents. Each is a LangGraph node: a function of the shared
GridIntelligenceState (state.py) that returns a partial-state dict with just the
field(s) it's responsible for. Every node is built by a factory function (make_*_node)
that takes the RAG retriever and LLM client as arguments -- plain dependency injection,
so a test can pass a stub LLM client (anything with a .complete(system, user) method)
without touching a real API key, and graph.py decides which retriever and client every
node actually gets at run time.

Pipeline order: Forecaster Interpreter -> Optimization Advisor -> Risk/Anomaly Detector
-> Report Synthesizer. Each of the first three retrieves project-specific context via
RAG before calling the LLM, grounding its explanation in this project's real, verified
results rather than the model's own possibly-stale recollection of them.
"""
from .state import GridIntelligenceState


def _format_context(chunks) -> str:
    if not chunks:
        return "(no relevant context retrieved)"
    return "\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)


def _fmt_mw(value) -> str:
    """A missing forecast/actual (None) is formatted as the words "not available" rather
    than the literal string "None" -- an LLM shown "forecast: None" in plain text can and
    did (observed in real testing) still narrate a comparison against it, e.g. claiming
    actual production "is consistent with" a forecast that was never actually given. This
    alone doesn't guarantee the model won't do that anyway, so the system prompts below
    also explicitly instruct against it -- see the "not available" instruction there."""
    return f"{value:.1f} MW" if value is not None else "not available"


def make_forecaster_node(retriever, llm):
    def forecaster_interpreter(state: GridIntelligenceState) -> dict:
        query = "day-ahead load wind solar forecasting LSTM Transformer probabilistic results"
        context = _format_context(retriever.retrieve(query, k=4))
        user = (
            f"Operator question: {state.get('question', '(none given)')}\n\n"
            f"Current numbers:\n"
            f"  Load  -- actual: {_fmt_mw(state.get('load_actual_mw'))}, day-ahead forecast: {_fmt_mw(state.get('load_forecast_mw'))}\n"
            f"  Wind  -- actual: {_fmt_mw(state.get('wind_actual_mw'))}, day-ahead forecast: {_fmt_mw(state.get('wind_forecast_mw'))}\n"
            f"  Solar -- actual: {_fmt_mw(state.get('solar_actual_mw'))}, day-ahead forecast: {_fmt_mw(state.get('solar_forecast_mw'))}\n\n"
            f"Relevant project documentation (retrieved via RAG):\n{context}"
        )
        system = (
            "You are the Forecaster Interpreter agent in a smart-grid decision-support pipeline. "
            "Explain, in plain operator-facing language, what the load/wind/solar forecasts say and "
            "how the current actuals compare to what was forecast. Ground any claim about model "
            "accuracy or limitations (e.g. 'wind is less reliable because...') in the retrieved "
            "documentation, not assumption. If a forecast is listed as 'not available', say plainly "
            "that no forecast was available for that quantity -- never claim actual production "
            "'matches' or 'is consistent with' a forecast that wasn't given to you. Be concise: "
            "3-5 sentences."
        )
        return {"forecast_summary": llm.complete(system, user)}
    return forecaster_interpreter


def make_optimizer_node(retriever, llm):
    def optimization_advisor(state: GridIntelligenceState) -> dict:
        query = "PPO SAC MARL MADDPG benchmark cost improvement heuristic optimization results"
        context = _format_context(retriever.retrieve(query, k=4))
        user = (
            f"Policy in use: {state.get('policy_name', '(unspecified)')}\n"
            f"Proposed action right now: {state.get('proposed_action', '(none given)')}\n\n"
            f"Forecast interpretation from the previous agent:\n{state.get('forecast_summary', '')}\n\n"
            f"Relevant project documentation (retrieved via RAG):\n{context}"
        )
        system = (
            "You are the Optimization Advisor agent in a smart-grid decision-support pipeline. "
            "Explain what the proposed action means in practice and why it's reasonable given the "
            "forecast interpretation you were given, citing this project's actual measured benchmark "
            "performance (cost improvement vs. a rule-based heuristic) from the retrieved "
            "documentation -- including honestly noting if the policy in use fell short of its "
            "original target. Be concise: 3-5 sentences."
        )
        return {"optimization_summary": llm.complete(system, user)}
    return optimization_advisor


def make_risk_node(retriever, llm):
    def risk_detector(state: GridIntelligenceState) -> dict:
        query = "forecast error limitations quantile coverage wind uncertainty risk"
        context = _format_context(retriever.retrieve(query, k=4))

        # A few cheap, deterministic checks up front -- an LLM shouldn't be the only
        # thing standing between "actual load far exceeds forecast" and a human noticing.
        flags = []
        for name in ("load", "wind", "solar"):
            actual = state.get(f"{name}_actual_mw")
            forecast = state.get(f"{name}_forecast_mw")
            if actual is not None and forecast not in (None, 0):
                pct_diff = abs(actual - forecast) / abs(forecast) * 100
                if pct_diff > 15:
                    flags.append(
                        f"{name}: actual differs from forecast by {pct_diff:.1f}% "
                        f"(actual={actual} MW, forecast={forecast} MW)"
                    )

        user = (
            "Deterministic threshold checks (>15% actual-vs-forecast deviation flagged):\n"
            + ("\n".join(flags) if flags else "  none triggered")
            + f"\n\nForecast interpretation:\n{state.get('forecast_summary', '')}\n\n"
            f"Optimization summary:\n{state.get('optimization_summary', '')}\n\n"
            f"Relevant project documentation (retrieved via RAG):\n{context}"
        )
        system = (
            "You are the Risk/Anomaly Detector agent in a smart-grid decision-support pipeline. "
            "Given the deterministic deviation flags, the forecast interpretation, and the "
            "optimization summary, state plainly whether there is anything an operator should be "
            "cautious about right now -- and if a threshold was NOT triggered, say so rather than "
            "manufacturing a concern. Ground any statement about how reliable a forecast is (e.g. "
            "known weaker wind forecasts, under-calibrated uncertainty bands) in the retrieved "
            "documentation. Be concise: 3-5 sentences."
        )
        return {"risk_summary": llm.complete(system, user)}
    return risk_detector


def make_synthesizer_node(llm):
    def report_synthesizer(state: GridIntelligenceState) -> dict:
        user = (
            f"Operator question: {state.get('question', '(none given)')}\n\n"
            f"Forecast interpretation:\n{state.get('forecast_summary', '')}\n\n"
            f"Optimization summary:\n{state.get('optimization_summary', '')}\n\n"
            f"Risk summary:\n{state.get('risk_summary', '')}"
        )
        system = (
            "You are the Report Synthesizer agent, the last step in a smart-grid decision-support "
            "pipeline. Combine the three prior agents' outputs into one coherent, well-organized "
            "report for a grid operator: what's forecast, what's recommended, and what to watch out "
            "for. Do not introduce any new claim that isn't already supported by the three summaries "
            "you were given. Keep it under 200 words."
        )
        return {"final_report": llm.complete(system, user)}
    return report_synthesizer