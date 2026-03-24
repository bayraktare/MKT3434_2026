"""mcp_handler.py
Graceful MCP handler with offline built-in tools for the MKT3434 baseline.
"""

from __future__ import annotations

import ast
import csv
import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MCPHandler:
    _latest_instance: "MCPHandler | None" = None

    def __init__(self, server_url: str = "http://localhost:8765"):
        self.server_url = server_url.rstrip("/")
        self._connected = False
        self._tools: list[dict] = []
        self._remote_available = False
        MCPHandler._latest_instance = self

    def connect(self) -> None:
        try:
            data = self._post_json("/list_tools", {})
            self._tools = data.get("tools", [])
            self._connected = True
            self._remote_available = True
            logger.info("Connected to MCP server at %s with %d tools.", self.server_url, len(self._tools))
        except Exception as exc:
            logger.warning("MCP server unavailable, switching to offline tools: %s", exc)
            self._connected = True
            self._remote_available = False
            self._tools = self._builtin_tools()

    def disconnect(self) -> None:
        self._connected = False
        self._remote_available = False
        self._tools = []
        logger.info("MCP disconnected.")

    def list_tools(self) -> list[dict]:
        if not self._connected:
            raise RuntimeError("Not connected to an MCP server.")
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict) -> Any:
        if not self._connected:
            raise RuntimeError("Not connected to an MCP server.")

        if self._remote_available:
            try:
                return self._post_json("/call_tool", {"name": name, "arguments": arguments})
            except Exception as exc:
                logger.error("Remote MCP call failed, falling back if possible: %s", exc)

        return self._call_builtin(name, arguments)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @classmethod
    def get_active_handler(cls) -> "MCPHandler | None":
        return cls._latest_instance

    # ------------------------------------------------------------------
    # Remote helpers
    # ------------------------------------------------------------------
    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.server_url}{endpoint}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Offline tools
    # ------------------------------------------------------------------
    def _builtin_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "calculator",
                "description": "Safely evaluate arithmetic expressions.",
                "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
            },
            {
                "name": "file_reader",
                "description": "Read a local text-like file and return its content preview.",
                "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {
                "name": "structured_lookup",
                "description": "Search a JSON or CSV file for rows/items containing a query string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                    },
                },
            },
        ]

    def _call_builtin(self, name: str, arguments: dict[str, Any]) -> Any:
        if name == "calculator":
            expr = str(arguments.get("expression", "")).replace("^", "**")
            return {"tool": name, "result": self._safe_eval(expr), "expression": expr}
        if name == "file_reader":
            path = Path(arguments.get("path", ""))
            if not path.exists() or not path.is_file():
                return {"tool": name, "error": f"File not found: {path}"}
            text = path.read_text(encoding="utf-8", errors="ignore")
            return {"tool": name, "path": str(path), "preview": text[:4000]}
        if name == "structured_lookup":
            return self._structured_lookup(arguments)
        return {"error": f"Unknown tool: {name}"}

    def _structured_lookup(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = Path(arguments.get("path", ""))
        query = str(arguments.get("query", "")).lower().strip()
        if not path.exists() or not path.is_file():
            return {"tool": "structured_lookup", "error": f"File not found: {path}"}

        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            serial = json.dumps(data, ensure_ascii=False)
            return {
                "tool": "structured_lookup",
                "matches": [serial[:2000]] if query in serial.lower() else [],
            }

        if path.suffix.lower() == ".csv":
            matches = []
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    serial = json.dumps(row, ensure_ascii=False)
                    if query in serial.lower():
                        matches.append(row)
                    if len(matches) >= 10:
                        break
            return {"tool": "structured_lookup", "matches": matches}

        return {"tool": "structured_lookup", "error": "Only JSON and CSV are supported in offline mode."}

    def _safe_eval(self, expression: str) -> Any:
        allowed_nodes = {
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.Mod,
            ast.USub,
            ast.UAdd,
            ast.Constant,
            ast.Load,
            ast.FloorDiv,
        }
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                raise ValueError(f"Unsafe expression component: {type(node).__name__}")
        return eval(compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {})
