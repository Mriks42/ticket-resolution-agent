"""MCP server: JSON-RPC 2.0 over stdio, stdlib only.

See `docs/adr/0002-stdlib-mcp-server-over-official-sdk.md` for why this is
hand-rolled rather than built on the `mcp` package.

Methods implemented: `initialize`, `tools/list`, `tools/call`, `ping`.
Notifications (messages with no `id`) are accepted and not answered.

Two things about permissions are worth stating plainly, because they are the
reason this file exists at all rather than a decorator on each handler:

**Scopes come from the process that starts the server, never from the client.**
`MCPServer(granted_scopes=...)` is set by whoever spawns the server — the agent
runner, `evals/run.py`, a test. `initialize` reports the granted set back so the
client knows what it has, but nothing a client sends can widen it. A permission
model the client can talk its way past is not a permission model.

**The scope check runs before argument validation.** A refusal must not depend
on, or leak, whether the arguments would have been accepted.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, IO

from src.mcp.audit import AuditLog
from src.tools.catalog import CATALOG, Scope
from src.tools.context import ToolContext
from src.tools.errors import ToolError
from src.paths import resolve_var_dir

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "ticket-resolution-agent"
SERVER_VERSION = "0.1.0"

# JSON-RPC 2.0 reserved codes.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Application code, in the -32000..-32099 implementation-defined band.
# A scope refusal is a protocol-level rejection, not a tool result: the tool
# never ran, so there is no result to report as `isError`.
SCOPE_DENIED = -32001

UNKNOWN_RUN_ID = "unknown"


class MCPServer:
    def __init__(
        self,
        *,
        granted_scopes: frozenset[Scope] | set[Scope],
        var_dir: str | Path | None = None,
        dataset_dir: str | Path | None = None,
        audit: AuditLog | None = None,
        context: ToolContext | None = None,
    ) -> None:
        self.granted_scopes = frozenset(granted_scopes)
        resolved_var = resolve_var_dir(var_dir)
        self.context = context or ToolContext.build(
            var_dir=resolved_var, dataset_dir=dataset_dir
        )
        self.audit = audit or AuditLog.in_var_dir(resolved_var)

    # --- dispatch ----------------------------------------------------------

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle one JSON-RPC message. Returns None for notifications."""
        req_id = request.get("id")
        is_notification = "id" not in request

        method = request.get("method")
        params = request.get("params") or {}

        if not isinstance(method, str):
            response = _error(req_id, INVALID_REQUEST, "missing method")
        elif method == "initialize":
            response = _ok(req_id, self._initialize())
        elif method == "ping":
            response = _ok(req_id, {})
        elif method == "tools/list":
            response = _ok(req_id, {"tools": self._list_tools()})
        elif method == "tools/call":
            response = self._call_tool(req_id, params)
        elif method.startswith("notifications/"):
            response = _ok(req_id, {})
        else:
            response = _error(
                req_id, METHOD_NOT_FOUND, f"unknown method: {method}"
            )

        return None if is_notification else response

    def _initialize(self) -> dict:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            # Reported, not accepted. See the module docstring.
            "grantedScopes": self._scope_values(),
        }

    def _list_tools(self) -> list[dict]:
        """All six tools, annotated with scope, gating, and whether permitted.

        Tools the caller cannot currently call are listed rather than hidden:
        the agent loop needs the full surface to plan against, and a tool that
        vanishes is indistinguishable from one that does not exist.
        """
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
                "requiredScope": spec.scope.value,
                "gated": spec.gated,
                "permitted": spec.scope in self.granted_scopes,
            }
            for spec in CATALOG.values()
        ]

    # --- tools/call --------------------------------------------------------

    def _call_tool(self, req_id: Any, params: dict) -> dict:
        name = params.get("name")
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        meta = params.get("_meta") or {}
        run_id = meta.get("run_id") or UNKNOWN_RUN_ID

        started = time.perf_counter()

        def audit(status, required_scope=None, detail=None) -> None:
            self.audit.record(
                run_id=run_id,
                tool=name if isinstance(name, str) else repr(name),
                args=arguments,
                caller_scopes=self._scope_values(),
                required_scope=required_scope,
                status=status,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                detail=detail,
            )

        spec = CATALOG.get(name) if isinstance(name, str) else None
        if spec is None:
            audit("refused", detail=f"unknown tool: {name!r}")
            return _error(
                req_id, METHOD_NOT_FOUND, f"unknown tool: {name!r}", {"tool": name}
            )

        # Before argument validation, deliberately.
        if spec.scope not in self.granted_scopes:
            audit("refused", spec.scope.value, "caller was not granted this scope")
            return _error(
                req_id,
                SCOPE_DENIED,
                f"{spec.name} requires scope {spec.scope.value!r}",
                {
                    "tool": spec.name,
                    "requiredScope": spec.scope.value,
                    "grantedScopes": self._scope_values(),
                },
            )

        try:
            payload = spec.handler(self.context, **arguments)
        except ToolError as exc:
            # The tool ran and could not answer. That is an observation for the
            # agent, so it comes back as a result with isError, not as a
            # transport error.
            audit("error", spec.scope.value, str(exc))
            return _ok(req_id, _content(str(exc), is_error=True))
        except TypeError as exc:
            audit("error", spec.scope.value, f"bad arguments: {exc}")
            return _error(
                req_id, INVALID_PARAMS, f"bad arguments for {spec.name}: {exc}"
            )
        except Exception as exc:  # noqa: BLE001 - logged, then re-reported
            audit("error", spec.scope.value, f"{type(exc).__name__}: {exc}")
            return _error(
                req_id, INTERNAL_ERROR, f"{spec.name} failed: {type(exc).__name__}"
            )

        audit("ok", spec.scope.value)
        return _ok(req_id, _content(json.dumps(payload, ensure_ascii=False, default=str)))

    def _scope_values(self) -> list[str]:
        return sorted(s.value for s in self.granted_scopes)


# --- JSON-RPC envelope helpers ---------------------------------------------


def _ok(req_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str, data: dict | None = None) -> dict:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": error}


def _content(text: str, is_error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


# --- stdio transport -------------------------------------------------------


def serve_stdio(server: MCPServer, stdin: IO[str], stdout: IO[str]) -> None:
    """Read newline-delimited JSON-RPC from stdin, write responses to stdout.

    A malformed line is answered with a parse error and the loop continues; one
    bad message from a client must not take the server down mid-run.
    """
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, _error(None, PARSE_ERROR, f"invalid JSON: {exc}"))
            continue

        if not isinstance(request, dict):
            _write(stdout, _error(None, INVALID_REQUEST, "expected a JSON object"))
            continue

        response = server.handle(request)
        if response is not None:
            _write(stdout, response)


def _write(stdout: IO[str], message: dict) -> None:
    stdout.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    stdout.flush()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="MCP server over stdio for the support triage agent tools."
    )
    parser.add_argument(
        "--scopes",
        default="",
        help=(
            "Comma-separated scopes granted to the caller, e.g. "
            "'customer:read,policy:read'. Use 'all' to grant every scope. "
            "Empty grants nothing, and every call is refused."
        ),
    )
    parser.add_argument(
        "--var-dir",
        type=Path,
        default=None,
        help="Override where the store and audit log are written.",
    )
    args = parser.parse_args(argv)

    if args.scopes.strip() == "all":
        granted = frozenset(Scope)
    else:
        names = [s.strip() for s in args.scopes.split(",") if s.strip()]
        try:
            granted = frozenset(Scope(name) for name in names)
        except ValueError as exc:
            parser.error(
                f"{exc}; valid scopes: {', '.join(s.value for s in Scope)}"
            )

    server = MCPServer(granted_scopes=granted, var_dir=args.var_dir)
    serve_stdio(server, sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
