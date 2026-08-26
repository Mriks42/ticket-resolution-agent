"""Failures a tool can raise.

These are observations for the agent loop, not crashes. Per CLAUDE.md a failed
tool narrows the agent's options; it does not take the run down. The MCP server
turns a `ToolError` into an `isError` tool result, which the agent sees in
OBSERVE and can plan around.
"""

from __future__ import annotations


class ToolError(Exception):
    """A tool ran and could not produce an answer."""


class ToolInputError(ToolError):
    """The arguments were not usable — unknown id, out-of-range amount."""
