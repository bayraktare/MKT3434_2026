"""
MCP (Model Context Protocol) Client for MKT3434 Term Project

Provides a lightweight integration layer that lets an LLM call registered
local tools via the Model Context Protocol pattern.  Students can extend
this module to connect to full MCP servers (e.g. filesystem, web-search,
database) using the official `mcp` Python SDK.

References
----------
* https://modelcontextprotocol.io
* https://github.com/modelcontextprotocol/python-sdk
"""

import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MCPTool:
    """Descriptor for a single MCP-compatible tool."""

    def __init__(self, name: str, description: str, handler: Callable) -> None:
        self.name = name
        self.description = description
        self.handler = handler

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description}


class MCPClient:
    """
    Lightweight MCP client that manages local tools and can be extended
    to forward calls to remote MCP servers.

    Usage
    -----
    >>> client = MCPClient()
    >>> client.register_tool("get_weather", "Return weather for a city",
    ...                      lambda city: f"Sunny in {city}")
    >>> client.call_tool("get_weather", {"city": "Istanbul"})
    'Sunny in Istanbul'
    >>> print(client.get_tool_context())
    Available MCP tools:
      - get_weather: Return weather for a city
    """

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}
        self._call_log: List[Dict[str, Any]] = []
        self._register_builtin_tools()

    # ------------------------------------------------------------------
    # Built-in demonstration tools
    # ------------------------------------------------------------------

    def _register_builtin_tools(self) -> None:
        """Register a small set of built-in demonstration tools."""
        self.register_tool(
            name="list_sources",
            description="List the document sources currently loaded into the RAG engine.",
            handler=lambda sources="": sources or "No sources loaded yet.",
        )
        self.register_tool(
            name="get_timestamp",
            description="Return the current UTC date and time.",
            handler=self._tool_get_timestamp,
        )

    @staticmethod
    def _tool_get_timestamp(**_kwargs) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_tool(self, name: str, description: str, handler: Callable) -> None:
        """Register a callable as an MCP tool."""
        self._tools[name] = MCPTool(name=name, description=description, handler=handler)
        logger.info("MCP tool registered: '%s'", name)

    def list_tools(self) -> List[Dict[str, str]]:
        """Return a list of tool descriptors (name + description)."""
        return [tool.to_dict() for tool in self._tools.values()]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Any:
        """
        Invoke the named tool with the supplied keyword arguments.

        Parameters
        ----------
        name : str
            Registered tool name.
        arguments : dict, optional
            Keyword arguments forwarded to the tool handler.

        Returns
        -------
        Any
            The return value of the tool handler.
        """
        if name not in self._tools:
            raise ValueError(f"Unknown MCP tool: '{name}'. Available: {list(self._tools)}")

        args = arguments or {}
        result = self._tools[name].handler(**args)

        entry = {"tool": name, "arguments": args, "result": result}
        self._call_log.append(entry)
        logger.info("MCP call: %s", json.dumps(entry, default=str))
        return result

    def get_tool_context(self) -> str:
        """
        Build a plain-text description of all registered tools suitable
        for injection into an LLM system prompt.
        """
        if not self._tools:
            return ""
        lines = ["Available MCP tools:"]
        for tool in self._tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)

    def get_call_log(self) -> List[Dict[str, Any]]:
        """Return the full history of tool calls made in this session."""
        return list(self._call_log)
