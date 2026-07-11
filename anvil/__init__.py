"""ANVIL — Adaptive Notes, Verification, Inference & Liaison harness.

A lean, local-first agent harness that climbs an Ollama model ladder
(local -> Ollama Cloud) only as far as a task needs, takes notes constantly,
recalls them under a token budget, schedules its own jobs, and reaches you on
Discord.

The core runs on the Python standard library alone. Optional packages
(discord.py, tiktoken, apscheduler) light up richer features.
"""

__version__ = "0.1.0"
__all__ = [
    "config",
    "providers",
    "router",
    "context",
    "memory",
    "scheduler",
    "comms",
    "pipeline",
]
