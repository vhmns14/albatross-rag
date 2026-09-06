"""
telemetry.py - Latency waterfall & cost tracker for Albatross RAG.
Ponytail: Stdlib time/dataclass first. Zero heavy telemetry daemons.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List

# Pricing per 1M tokens (USD) as of 2024-2026 reference
MODEL_PRICING = {
    # Claude Opus
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-opus-4.6": {"input": 15.0, "output": 75.0},
    "opus": {"input": 15.0, "output": 75.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    # Mistral
    "mistral-large-latest": {"input": 2.0, "output": 6.0},
    "mistral": {"input": 2.0, "output": 6.0},
    "mistral-small-latest": {"input": 0.2, "output": 0.6},
    # Groq (Free / Near Zero)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "groq": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
}

@dataclass
class StepTiming:
    step_name: str
    duration_ms: float

@dataclass
class QueryTelemetry:
    model: str
    mode: str  # "naive" | "hybrid"
    timings: List[StepTiming] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    def add_step(self, step_name: str, duration_ms: float):
        self.timings.append(StepTiming(step_name=step_name, duration_ms=round(duration_ms, 2)))

    def calculate_cost(self):
        pricing = MODEL_PRICING.get(self.model, {"input": 1.0, "output": 3.0})
        cost_in = (self.input_tokens / 1_000_000) * pricing["input"]
        cost_out = (self.output_tokens / 1_000_000) * pricing["output"]
        self.estimated_cost_usd = round(cost_in + cost_out, 6)
        return self.estimated_cost_usd

class TelemetryTimer:
    """Context manager to measure sub-millisecond execution times."""
    def __init__(self, step_name: str, telemetry: QueryTelemetry):
        self.step_name = step_name
        self.telemetry = telemetry
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.telemetry.add_step(self.step_name, duration_ms)

def estimate_tokens(text: str) -> int:
    """Approximate token count safely."""
    if not text:
        return 0
    return max(1, len(text) // 4)
