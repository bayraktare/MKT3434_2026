"""Basic MCP (Model Context Protocol) client for connecting to MCP servers."""

from __future__ import annotations

import asyncio
from typing import Any


class MCPClient:
    """Connects to stdio-based MCP servers and exposes their tools."""

    def __init__(self) -> None:
        self._servers: dict[str, dict[str, Any]] = {}
        self._cached_tools: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Server registration
    # ------------------------------------------------------------------

    def add_server(
        self, name: str, command: str, args: list[str] | None = None
    ) -> None:
        """Register a stdio-based MCP server by name."""
        self._servers[name] = {"command": command, "args": args or []}
        self._cached_tools.clear()

    def remove_server(self, name: str) -> None:
        """Remove a previously registered server."""
        self._servers.pop(name, None)
        self._cached_tools.clear()

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, str]]:
        """Return the cached list of tools from all registered servers."""
        if not self._cached_tools and self._servers:
            self._cached_tools = asyncio.run(self._gather_all_tools())
        return self._cached_tools.copy()

    async def _gather_all_tools(self) -> list[dict[str, str]]:
        tools: list[dict[str, str]] = []
        for server_config in self._servers.values():
            tools.extend(await self._list_server_tools(server_config))
        return tools

    async def _list_server_tools(
        self, server_config: dict[str, Any]
    ) -> list[dict[str, str]]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"],
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_response = await session.list_tools()
                    return [
                        {
                            "name": t.name,
                            "description": t.description or "",
                        }
                        for t in tools_response.tools
                    ]
        except Exception as exc:
            print(f"Warning: could not list tools from MCP server: {exc}")
            return []

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a named tool synchronously (blocks until complete)."""
        return asyncio.run(self._call_tool_async(tool_name, arguments))

    async def _call_tool_async(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> Any:
        for server_config in self._servers.values():
            tools = await self._list_server_tools(server_config)
            if any(t["name"] == tool_name for t in tools):
                return await self._invoke_tool(server_config, tool_name, arguments)
        raise ValueError(
            f"Tool {tool_name!r} not found in any registered MCP server."
        )

    async def _invoke_tool(
        self,
        server_config: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"],
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)
