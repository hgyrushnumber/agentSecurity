"""agentSecurity domain package.

Heavy training/evaluation code lives here so the FastAPI control plane never
imports torch/transformers at startup. Only pure-Python helpers (agents.common)
are safe to import from the API process.
"""

__version__ = "0.1.0"
