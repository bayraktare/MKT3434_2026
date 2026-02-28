"""
mcp_handler.py
──────────────
MCP (Model Context Protocol) Handler

Manages the connection to an MCP server and exposes tool-calling
capabilities to the RAG pipeline.

Reference: https://modelcontextprotocol.io/docs
"""

from __future__ import annotations
import json
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MCPHandler:
    """
    Thin wrapper around an MCP server connection via HTTP.

    Public interface (called by main.py):
        connect()
        disconnect()
        list_tools()  -> list[dict]
        call_tool(name, arguments) -> Any
    """

    def __init__(self, server_url: str = "http://localhost:8765"):
        self.server_url = server_url
        self._connected = False
        self._tools: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Establish connection to the MCP server and fetch available tools.
        Attempts an HTTP connection to the configured server URL.
        If the server is not reachable, logs a warning and operates
        in disconnected mode (no tools available but no crash).
        """
        try:
            # Try to connect and list tools
            response = requests.post(
                f"{self.server_url}/list_tools",
                json={},
                timeout=5,
            )
            response.raise_for_status()
            data = response.json()
            self._tools = data.get("tools", [])
            self._connected = True
            logger.info(
                f"MCP connected to {self.server_url}. "
                f"Available tools: {len(self._tools)}"
            )
        except requests.exceptions.ConnectionError:
            # Server not running — connect gracefully with no tools
            logger.warning(
                f"MCP server at {self.server_url} is not reachable. "
                "Connecting in offline mode (no tools available)."
            )
            self._connected = True
            self._tools = []
        except Exception as exc:
            logger.warning(f"MCP connection issue: {exc}. Connecting in offline mode.")
            self._connected = True
            self._tools = []

    def disconnect(self) -> None:
        """Close the MCP server connection gracefully."""
        self._connected = False
        self._tools = []
        logger.info("MCP disconnected.")

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Access
    # ─────────────────────────────────────────────────────────────────────────

    def list_tools(self) -> list[dict]:
        """Return the list of tools exposed by the connected MCP server."""
        if not self._connected:
            raise RuntimeError("Not connected to an MCP server.")
        return self._tools

    def call_tool(self, name: str, arguments: dict) -> Any:
        """
        Invoke a named tool on the MCP server.
        """
        if not self._connected:
            raise RuntimeError("Not connected to an MCP server.")

        try:
            response = requests.post(
                f"{self.server_url}/call_tool",
                json={"name": name, "arguments": arguments},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error(f"MCP server not reachable when calling tool '{name}'.")
            return {"error": "MCP server not reachable"}
        except Exception as exc:
            logger.error(f"MCP tool call failed: {exc}")
            return {"error": str(exc)}

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected
