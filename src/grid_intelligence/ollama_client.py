"""
Local LLM backend via Ollama's REST API (http://localhost:11434 by default) -- zero
per-call cost, fully offline once a model is pulled. Same .complete(system, user)
interface as llm_client.AnthropicClient, so agents.py and graph.py don't know or care
which backend they're talking to -- run_pipeline.py picks one via the LLM_BACKEND
environment variable.

One-time setup:
  1. Install Ollama: https://ollama.com/download (Windows installer).
  2. Pull a model, e.g.:  ollama pull llama3.1:8b
     An 8B model is a reasonable fit for an RTX 4060 Laptop GPU's ~8GB VRAM. If that's
     too slow or runs out of memory, a smaller model works too: ollama pull llama3.2:3b
     (then set OLLAMA_MODEL=llama3.2:3b).
  3. Ollama runs its own local server automatically after install on Windows -- no
     separate "start the server" step is normally needed.
"""
import os

import requests

DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


class OllamaClient:
    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL, base_url: str = DEFAULT_OLLAMA_URL, timeout: int = 120):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # Fail fast with a clear, actionable message here rather than a cryptic
        # connection error deep inside the pipeline if Ollama isn't running.
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} ({e}). Make sure Ollama is "
                f"installed and running (https://ollama.com/download) before running "
                f"this script."
            ) from e

        available = [m["name"] for m in resp.json().get("models", [])]
        if available and not any(self.model == m or m.startswith(self.model) for m in available):
            print(
                f"WARNING: model '{self.model}' not found among Ollama's pulled models "
                f"{available}. Pull it first with: ollama pull {self.model}"
            )

    def complete(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]