"""LangChain tool adapters for MCP (Model Context Protocol) tools.

This module provides adapters that wrap MCP tools as LangChain tools,
enabling seamless integration between MCP servers and LangChain agents.

Example:
    from src.tools.mcp_tools import create_mcp_tool_adapters
    from src.core.mcp_manager import get_mcp_manager
    
    manager = get_mcp_manager()
    mcp_tools = await manager.discover_tools("filesystem")
    langchain_tools = create_mcp_tool_adapters(mcp_tools, "filesystem")
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from ..logger import setup_logger
from .json_schema_gemini import sanitize_json_schema_for_gemini


logger = setup_logger()


def _json_prop_to_python_type(prop_schema: Dict[str, Any]) -> Any:
    """Map a single JSON Schema property to a Python type (Gemini-safe array ``items``)."""
    prop_schema = sanitize_json_schema_for_gemini(prop_schema)
    prop_type = prop_schema.get("type", "string")

    if prop_type == "array":
        items = prop_schema.get("items") or {"type": "string"}
        if not isinstance(items, dict):
            return List[Any]
        items = sanitize_json_schema_for_gemini(items)
        inner_t = items.get("type", "string")
        if inner_t == "string":
            return list[str]
        if inner_t == "integer":
            return list[int]
        if inner_t == "number":
            return list[float]
        if inner_t == "boolean":
            return list[bool]
        if inner_t == "object":
            return list[dict]
        if inner_t == "array":
            return List[Any]
        return List[Any]

    if prop_type == "object":
        return dict

    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return type_mapping.get(prop_type, str)


def _create_args_schema(
    tool_name: str,
    input_schema: Dict[str, Any],
) -> Type[BaseModel]:
    """Create a Pydantic model from JSON schema for tool arguments.

    Args:
        tool_name: Name of the tool (used for model naming).
        input_schema: JSON schema for the tool's input.

    Returns:
        A Pydantic BaseModel class representing the schema.
    """
    input_schema = sanitize_json_schema_for_gemini(dict(input_schema or {}))
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    field_definitions = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            prop_schema = {}
        description = prop_schema.get("description", "")
        default = ... if prop_name in required else None

        python_type = _json_prop_to_python_type(prop_schema)

        if prop_name not in required:
            python_type = Optional[python_type]

        field_definitions[prop_name] = (
            python_type,
            Field(default=default, description=description),
        )

    model_name = f"{tool_name.replace('-', '_').title()}Args"
    if not field_definitions:
        return create_model(model_name)

    return create_model(model_name, **field_definitions)


class MCPToolAdapter(BaseTool):
    """Adapter that wraps an MCP tool as a LangChain tool.

    This adapter handles the translation between LangChain's tool interface
    and MCP's tool calling protocol.

    Attributes:
        name: Tool name.
        description: Tool description.
        mcp_server: Name of the MCP server providing this tool.
        mcp_tool_name: Original tool name on the MCP server.
        args_schema: Pydantic model for argument validation.
    """

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    mcp_server: str = Field(..., description="MCP server name")
    mcp_tool_name: str = Field(..., description="Original MCP tool name")
    args_schema: Type[BaseModel] = Field(..., description="Arguments schema")

    def _run(self, **kwargs: Any) -> str:
        """Synchronous execution - wraps async call.

        Args:
            **kwargs: Tool arguments.

        Returns:
            Tool execution result as string.
        """
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._arun(**kwargs))

            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._arun(**kwargs)).result(
                    timeout=120
                )
        except Exception as e:
            error_msg = f"Error executing MCP tool {self.name}: {e}"
            logger.error(error_msg)
            return error_msg

    async def _arun(self, **kwargs: Any) -> str:
        """Asynchronous execution via MCP manager.

        Tool invocations often use ``asyncio.run()`` in a thread with a **new** event loop. MCP
        stdio sessions belong to one loop; reusing the singleton across loops breaks calls and
        stdio teardown (anyio cancel-scope errors). Clear connections before each call and close
        after. ``MCP_CROSS_THREAD_LOCK`` avoids overlapping connect/disconnect races.

        Args:
            **kwargs: Tool arguments.

        Returns:
            Tool execution result as string.
        """
        from ..core.mcp_manager import MCP_CROSS_THREAD_LOCK, get_mcp_manager

        manager = get_mcp_manager()
        with MCP_CROSS_THREAD_LOCK:
            await manager.close_all()
            try:
                return await manager.call_tool(
                    self.mcp_server,
                    self.mcp_tool_name,
                    kwargs,
                )
            finally:
                await manager.close_all()


def create_mcp_tool_adapter(
    tool_name: str,
    tool_description: str,
    input_schema: Dict[str, Any],
    server_name: str,
) -> MCPToolAdapter:
    """Create a LangChain tool adapter from MCP tool info.

    Args:
        tool_name: Name of the MCP tool.
        tool_description: Description of the tool.
        input_schema: JSON schema for tool input.
        server_name: Name of the MCP server.

    Returns:
        MCPToolAdapter instance.
    """
    args_schema = _create_args_schema(tool_name, input_schema)

    # Create a prefixed name to avoid conflicts
    prefixed_name = f"mcp_{server_name}_{tool_name}"

    return MCPToolAdapter(
        name=prefixed_name,
        description=f"[MCP:{server_name}] {tool_description}",
        mcp_server=server_name,
        mcp_tool_name=tool_name,
        args_schema=args_schema,
    )


def create_mcp_tool_adapters(
    mcp_tools: List[Any],
    server_name: str,
) -> List[MCPToolAdapter]:
    """Create LangChain tool adapters from a list of MCP tools.

    Args:
        mcp_tools: List of MCPTool objects.
        server_name: Name of the MCP server.

    Returns:
        List of MCPToolAdapter instances.
    """
    adapters = []
    for tool in mcp_tools:
        try:
            adapter = create_mcp_tool_adapter(
                tool_name=tool.name,
                tool_description=tool.description,
                input_schema=tool.input_schema,
                server_name=server_name,
            )
            adapters.append(adapter)
            logger.debug(f"Created adapter for MCP tool: {tool.name}")
        except Exception as e:
            logger.warning(f"Failed to create adapter for {tool.name}: {e}")

    return adapters


async def get_mcp_tools_async(server_names: List[str]) -> List[MCPToolAdapter]:
    """Asynchronously get LangChain tools from MCP servers.

    Args:
        server_names: List of MCP server names to get tools from.

    Returns:
        List of MCPToolAdapter instances.
    """
    from ..core.mcp_manager import (
        MCP_CROSS_THREAD_LOCK,
        _mcp_server_dependencies_met,
        get_mcp_manager,
    )

    manager = get_mcp_manager()
    with MCP_CROSS_THREAD_LOCK:
        await manager.close_all()
        all_tools = []

        try:
            for server_name in server_names:
                if not _mcp_server_dependencies_met(server_name):
                    continue
                try:
                    mcp_tools = await manager.discover_tools(server_name)
                    adapters = create_mcp_tool_adapters(mcp_tools, server_name)
                    all_tools.extend(adapters)
                    logger.info(
                        f"Loaded {len(adapters)} tools from MCP server: {server_name}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load tools from {server_name}: {e}"
                    )
        finally:
            await manager.close_all()

        return all_tools


def get_mcp_tools_sync(server_names: List[str]) -> List[MCPToolAdapter]:
    """Synchronously get LangChain tools from MCP servers.

    This is a convenience wrapper for sync contexts.

    Args:
        server_names: List of MCP server names to get tools from.

    Returns:
        List of MCPToolAdapter instances.
    """
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(get_mcp_tools_async(server_names))

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run, get_mcp_tools_async(server_names)
            ).result(timeout=180)
    except Exception as e:
        logger.error(f"Failed to get MCP tools: {e}")
        return []
