"""llm_monitor.py
Monitoring and lightweight analytics for the MKT3434 baseline.
Compatibility with the provided GUI is preserved.
"""

from __future__ import annotations

import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class LLMMonitor:
    def __init__(self):
        self._records: list[dict[str, Any]] = []

    def record(self, prompt: str, response: str, metadata: dict) -> None:
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
        if not self._records:
            return {"total_calls": 0}

        total_calls = len(self._records)
        total_tokens = sum(int(r.get("total_tokens", 0) or 0) for r in self._records)
        latencies = [float(r.get("elapsed_sec", r.get("elapsed_orchestration_sec", 0)) or 0) for r in self._records]
        avg_latency = sum(latencies) / total_calls if total_calls else 0.0

        model_dist: dict[str, int] = {}
        for record in self._records:
            model = record.get("model", "unknown")
            model_dist[model] = model_dist.get(model, 0) + 1

        buckets = {"<100": 0, "100-500": 0, "500-1000": 0, ">1000": 0}
        for record in self._records:
            tokens = int(record.get("total_tokens", 0) or 0)
            if tokens < 100:
                buckets["<100"] += 1
            elif tokens < 500:
                buckets["100-500"] += 1
            elif tokens < 1000:
                buckets["500-1000"] += 1
            else:
                buckets[">1000"] += 1

        agent_distribution: dict[str, int] = defaultdict(int)
        agent_avg_score: dict[str, list[float]] = defaultdict(list)
        for record in self._records:
            for trace in record.get("agent_trace", []) or []:
                agent = trace.get("agent", "unknown")
                agent_distribution[agent] += 1
                if isinstance(trace.get("overall_score"), (int, float)):
                    agent_avg_score[agent].append(float(trace["overall_score"]))

        agent_quality = {
            agent: round(sum(scores) / len(scores), 3)
            for agent, scores in agent_avg_score.items()
            if scores
        }

        return {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "average_latency_sec": round(avg_latency, 3),
            "min_latency_sec": round(min(latencies) if latencies else 0.0, 3),
            "max_latency_sec": round(max(latencies) if latencies else 0.0, 3),
            "model_distribution": model_dist,
            "token_histogram": buckets,
            "agent_distribution": dict(agent_distribution),
            "agent_average_quality": agent_quality,
        }

    def records(self) -> list[dict[str, Any]]:
        return list(self._records)
