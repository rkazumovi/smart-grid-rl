"""
Retrieval-augmented generation (RAG) over this project's own documentation: the README,
the theory notebooks (notebooks/forecasting_theory.ipynb, rl_theory.ipynb), and the
module/function docstrings already written throughout src/. This lets the four Grid
Intelligence agents (agents.py) answer questions grounded in real, verified project
content (e.g. "why did the LSTM beat the Transformer", "what does the MARL benchmark
show") instead of an LLM's own possibly-outdated or invented recollection of the numbers.

Pipeline: gather text -> split into overlapping character chunks -> embed each chunk
with a local sentence-transformers model (no API key needed, runs on CPU or GPU) ->
store the vectors in a FAISS flat index (exact nearest-neighbor search -- the corpus
here is small, a few hundred chunks, so there's no need for an approximate index) -> at
query time, embed the query the same way and retrieve the top-k closest chunks by
cosine similarity.
"""
import os
import json
import glob
import ast
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small (~80MB), fast, runs fine on CPU
CHUNK_SIZE = 800   # characters
CHUNK_OVERLAP = 100

INDEX_DIR = os.path.join("outputs", "rag_index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")


def _chunk_text(text: str, source: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Fixed-size character chunking with overlap -- simple and robust, no dependency on
    any particular tokenizer, and the overlap avoids cutting a sentence in half right at
    a chunk boundary and losing context on both sides of the cut."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        piece = text[start: start + chunk_size].strip()
        if piece:
            chunks.append({"text": piece, "source": source})
        start += step
    return chunks


def _extract_notebook_text(path: str) -> str:
    """Pull markdown + code cell source out of a .ipynb, skipping outputs (which can
    contain large base64 image blobs we don't want in the text corpus)."""
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        text = "".join(src) if isinstance(src, list) else str(src)
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docstrings(path: str) -> str:
    """Pull the module docstring plus every function/class docstring out of a .py file
    via the ast module -- this is where most of this project's real design rationale
    lives (why a fix was made, what a known limitation is), not just inline comments."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=path)
    except (SyntaxError, UnicodeDecodeError):
        return ""
    parts = []
    module_doc = ast.get_docstring(tree)
    if module_doc:
        parts.append(module_doc)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            if doc:
                parts.append(f"{node.name}: {doc}")
    return "\n\n".join(parts)


def gather_project_documents(project_root: str):
    """Collect (text, source_label) pairs from README.md, the two theory notebooks, and
    every .py file's docstrings under src/. Missing files are skipped with a printed
    note rather than raising -- a fresh checkout without notebooks built yet should
    still be able to index whatever does exist."""
    documents = []

    readme_path = os.path.join(project_root, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            documents.append((f.read(), "README.md"))
    else:
        print(f"  (skipping README.md -- not found at {readme_path})")

    for nb_name in ("forecasting_theory.ipynb", "rl_theory.ipynb"):
        nb_path = os.path.join(project_root, "notebooks", nb_name)
        if os.path.exists(nb_path):
            documents.append((_extract_notebook_text(nb_path), f"notebooks/{nb_name}"))
        else:
            print(f"  (skipping notebooks/{nb_name} -- not found)")

    py_files = sorted(glob.glob(os.path.join(project_root, "src", "**", "*.py"), recursive=True))
    for py_path in py_files:
        doc_text = _extract_docstrings(py_path)
        if doc_text.strip():
            rel = os.path.relpath(py_path, project_root)
            documents.append((doc_text, rel))

    return documents


def build_index(project_root: str = "."):
    print("Gathering project documents...")
    documents = gather_project_documents(project_root)
    print(f"  {len(documents)} document(s) found")

    all_chunks = []
    for text, source in documents:
        all_chunks.extend(_chunk_text(text, source))
    print(f"  {len(all_chunks)} chunk(s) after splitting (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    if not all_chunks:
        raise RuntimeError("No text found to index -- check that README.md/notebooks/src exist under project_root.")

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' (downloads once, then cached locally)...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Normalized embeddings + inner product = cosine similarity, without a separate
    # normalization step at query time beyond normalizing the query itself.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    print(f"\nSaved FAISS index ({index.ntotal} vectors, dim={embeddings.shape[1]}) to {INDEX_PATH}")
    print(f"Saved chunk texts to {CHUNKS_PATH}")
    return index, all_chunks


class RagRetriever:
    """Loads a previously built index once; call .retrieve(query, k) as many times as
    needed without reloading the embedding model or the index each time."""

    def __init__(self, index_path: str = INDEX_PATH, chunks_path: str = CHUNKS_PATH):
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            raise FileNotFoundError(
                f"No RAG index found at {index_path} -- run build_rag_index.py first."
            )
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, query: str, k: int = 4):
        """Returns the top-k chunks as a list of {"text", "source", "score"} dicts,
        highest similarity first."""
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            results.append({"text": chunk["text"], "source": chunk["source"], "score": float(score)})
        return results