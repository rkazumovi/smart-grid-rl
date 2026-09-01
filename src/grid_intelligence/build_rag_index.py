"""
Build (or rebuild) the FAISS RAG index over this project's own documentation --
README.md, the two theory notebooks, and every docstring under src/ -- so the four Grid
Intelligence agents (agents.py) can retrieve real, verified project content instead of
relying on an LLM's own recollection.

Run this once, from the project root, after any of those source documents change:
    python src\\grid_intelligence\\build_rag_index.py
The four agents load the saved index at run time rather than rebuilding it on every query.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid_intelligence.rag import build_index

if __name__ == "__main__":
    build_index(project_root=".")