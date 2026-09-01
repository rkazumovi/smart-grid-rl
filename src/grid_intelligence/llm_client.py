"""
Thin wrapper around the Anthropic Messages API. Kept as its own small class -- rather
than calling anthropic.Anthropic() directly inside every agent -- for two reasons: one
place to read the API key/model name from the environment, and one place to substitute a
stub in tests without needing a real API key or network access. Every agent in agents.py
takes an "llm" argument and only ever calls llm.complete(system, user); anything with
that method works, including a test stub.
"""
import os
import anthropic

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


class AnthropicClient:
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 700):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Get a key from https://console.anthropic.com/ "
                "and set it before running this script, e.g. in cmd.exe:\n"
                "    set ANTHROPIC_API_KEY=sk-ant-...\n"
                "(this only lasts for the current terminal session -- set it via "
                "System Properties > Environment Variables for a permanent value). "
                "If 'claude-sonnet-4-5' isn't available on your account, also set "
                "ANTHROPIC_MODEL to a model name you do have access to."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        """One-shot call: a system prompt (the agent's role/instructions) and a single
        user message (the task + retrieved context). Returns the plain text of the
        reply -- Anthropic can return multiple content blocks (e.g. thinking blocks),
        so this joins only the actual text blocks."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")