"""
llm_monitor.py
──────────────
LLM Monitoring & Analytics

Records every LLM call (prompt, response, metadata) and provides
analytics: token usage, latency, model distribution, etc.
"""

from __future__ import annotations
import json
import datetime
from pathlib import Path
from typing import Any


class LLMMonitor:
    """
    Lightweight monitor that logs every LLM interaction.

    Public interface (called by main.py):
        record(prompt, response, metadata)
        export_json(path)
        summary() -> dict
    """

    def __init__(self):
        self._records: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────────

    def record(self, prompt: str, response: str, metadata: dict) -> None:
        """
        Store one LLM call record with timestamp and all metadata.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            **metadata,
        }
        self._records.append(entry)

    def export_json(self, path: Any) -> None:
        """Save all records to a JSON file."""
        out = {
            "exported_at": datetime.datetime.now().isoformat(),
            "total_calls": len(self._records),
            "summary": self.summary(),
            "records": self._records,
        }
        Path(path).write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def summary(self) -> dict:
        """
        Return aggregate statistics over all recorded calls.
        """
        if not self._records:
            return {"total_calls": 0}

        total_calls = len(self._records)
        total_tokens = sum(r.get("total_tokens", 0) for r in self._records)
        avg_latency = (
            sum(r.get("elapsed_sec", 0) for r in self._records) / total_calls
        )

        # Model distribution
        model_dist: dict[str, int] = {}
        for r in self._records:
            m = r.get("model", "unknown")
            model_dist[m] = model_dist.get(m, 0) + 1

        # Token histogram
        buckets = {"<100": 0, "100-500": 0, "500-1000": 0, ">1000": 0}
        for r in self._records:
            t = r.get("total_tokens", 0)
            if t < 100:
                buckets["<100"] += 1
            elif t < 500:
                buckets["100-500"] += 1
            elif t < 1000:
                buckets["500-1000"] += 1
            else:
                buckets[">1000"] += 1

        # Latency stats
        latencies = [r.get("elapsed_sec", 0) for r in self._records]
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "average_latency_sec": round(avg_latency, 3),
            "min_latency_sec": round(min_latency, 3),
            "max_latency_sec": round(max_latency, 3),
            "model_distribution": model_dist,
            "token_histogram": buckets,
        }
