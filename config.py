"""Configuration for seAI. Reads from environment / a local .env file."""
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env support is optional; env vars can also be set directly.

# --- Model -------------------------------------------------------------------
# The chat model that powers the agent. Defaults to a local Ollama model.
# First run:  ollama pull qwen2.5-coder:7b
MODEL_ID = os.getenv("MODEL_ID", "qwen2.5-coder:7b")

# --- Provider endpoint (OpenAI-compatible) -----------------------------------
# The agent talks to any OpenAI-compatible endpoint. Default is local Ollama,
# so it runs offline and free. Override in .env to use a hosted provider.
BASE_URL = os.getenv("BASE_URL", "http://localhost:11434/v1")

# API key for the endpoint. Ollama ignores it, so a placeholder is fine.
API_KEY = os.getenv("API_KEY", "ollama")

# --- Agent behaviour ---------------------------------------------------------
# File tools are sandboxed to this directory; the agent can't escape it.
WORKSPACE = os.path.abspath(os.getenv("WORKSPACE", "./workspace"))

# Safety cap: max tool-use iterations per user turn (prevents infinite loops).
MAX_STEPS = int(os.getenv("MAX_STEPS", "12"))

# Lower temperature = more deterministic output, which is better for code.
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
