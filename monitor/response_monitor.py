"""LLM response monitoring and logging."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResponseMonitor:
    """Records and persists LLM query/response events for later analysis."""

    def __init__(self) -> None:
        self._log: list[dict[str, Any]] = []

    def log_response(
        self,
        prompt: str,
        response: str,
        model: str,
        provider: str,
        elapsed: float,
        tokens: dict[str, int] | None = None,
        sources: list[str] | None = None,
    ) -> str:
        """Record one LLM response event and return a formatted summary line."""
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "provider": provider,
            "prompt": prompt[:200],
            "response_length": len(response),
            "elapsed_seconds": round(elapsed, 2),
            "tokens": tokens or {},
            "sources": sources or [],
        }
        self._log.append(entry)

        token_str = (
            f"tokens={tokens.get('total_tokens', '?')}" if tokens else "tokens=?"
        )
        sources_str = (
            f"sources=[{', '.join(Path(s).name for s in (sources or [])[:3])}]"
        )
        return (
            f"Query | model={model} | {token_str} | "
            f"time={elapsed:.1f}s | {sources_str}"
        )

    def get_log(self) -> list[dict[str, Any]]:
        """Return all logged entries."""
        return self._log.copy()

    def save_log(self, path: str) -> None:
        """Save the log to a JSON or plain-text file."""
        p = Path(path)
        if p.suffix == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._log, f, indent=2, ensure_ascii=False)
        else:
            with open(p, "w", encoding="utf-8") as f:
                for entry in self._log:
                    f.write(
                        f"[{entry['timestamp']}] "
                        f"model={entry['model']} | "
                        f"tokens={entry['tokens'].get('total_tokens', '?')} | "
                        f"time={entry['elapsed_seconds']}s | "
                        f"prompt={entry['prompt'][:100]}\n"
                    )

    def clear(self) -> None:
        """Clear all logged entries."""
        self._log.clear()
