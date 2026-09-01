"""
Wires the four Grid Intelligence agents (pipeline_agents.py) into a single LangGraph
pipeline: Forecaster Interpreter -> Optimization Advisor -> Risk/Anomaly Detector ->
Report Synthesizer, each stage reading the previous stages' output plus its own
RAG-retrieved context. A straight-line pipeline (not a branching/looping graph) was
chosen because each stage's output is exactly what the next stage needs as input --
there's no decision point here where the graph needs to branch.
"""
from langgraph.graph import StateGraph, START, END

from .state import GridIntelligenceState
from .pipeline_agents import make_forecaster_node, make_optimizer_node, make_risk_node, make_synthesizer_node


def build_grid_intelligence_graph(retriever, llm):
    """retriever: a grid_intelligence.rag.RagRetriever (or anything with a
    .retrieve(query, k) method). llm: a grid_intelligence.llm_client.AnthropicClient (or
    any object with a .complete(system, user) method) -- injected here, not constructed
    inside the nodes, so a test can pass a stub for either without touching a real API
    key or a real FAISS index."""
    graph = StateGraph(GridIntelligenceState)
    graph.add_node("forecaster_interpreter", make_forecaster_node(retriever, llm))
    graph.add_node("optimization_advisor", make_optimizer_node(retriever, llm))
    graph.add_node("risk_detector", make_risk_node(retriever, llm))
    graph.add_node("report_synthesizer", make_synthesizer_node(llm))
    graph.add_edge(START, "forecaster_interpreter")
    graph.add_edge("forecaster_interpreter", "optimization_advisor")
    graph.add_edge("optimization_advisor", "risk_detector")
    graph.add_edge("risk_detector", "report_synthesizer")
    graph.add_edge("report_synthesizer", END)
    return graph.compile()