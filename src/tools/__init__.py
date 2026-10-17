"""Deterministic, JSON-returning analytics tools — the only interface agents use.

Importing this package registers every tool in ``REGISTRY``. Use
``call_tool(name, arguments)`` to execute one with validation and provenance.
"""
from src.tools import (allowance_tools, credit_tools, macro_tools, model_tools,  # noqa: F401  (registration)
                       portfolio_tools, reconciliation_tools)
from src.tools.base import (REGISTRY, ToolCallLog, ToolCallRecord, ToolContext, ToolError, call_tool,  # noqa: F401
                            get_context, list_tools, set_context, tool_schemas)
