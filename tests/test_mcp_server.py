"""MCP server: JSON-RPC framing, scope enforcement, and the audit trail.

The load-bearing claim being tested is that scope enforcement lives in the
server. A client cannot widen its own permissions by asking nicely, and a
refused call leaves the same audit evidence a successful one does.
"""

from __future__ import annotations

import io
import json

import pytest

from src.mcp.audit import AuditLog
from src.mcp.server import (
    METHOD_NOT_FOUND,
    SCOPE_DENIED,
    MCPServer,
    serve_stdio,
)
from src.tools.catalog import CATALOG, Scope

ALL_SCOPES = frozenset(Scope)
READ_ONLY = frozenset(
    {Scope.CUSTOMER_READ, Scope.TICKETS_READ, Scope.POLICY_READ}
)


def build(tmp_path, scopes=ALL_SCOPES) -> MCPServer:
    return MCPServer(granted_scopes=scopes, var_dir=tmp_path / "var")


def rpc(server: MCPServer, method: str, params=None, req_id=1):
    request = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        request["params"] = params
    return server.handle(request)


def call_tool(server: MCPServer, name: str, arguments: dict, run_id="run-test"):
    return rpc(
        server,
        "tools/call",
        {"name": name, "arguments": arguments, "_meta": {"run_id": run_id}},
    )


# --- protocol --------------------------------------------------------------


def test_initialize_reports_protocol_and_granted_scopes(tmp_path) -> None:
    response = rpc(build(tmp_path, READ_ONLY), "initialize", {})
    result = response["result"]
    assert result["protocolVersion"]
    assert set(result["capabilities"]) >= {"tools"}
    assert sorted(result["grantedScopes"]) == sorted(s.value for s in READ_ONLY)


def test_tools_list_advertises_scope_and_gated_flags(tmp_path) -> None:
    tools = rpc(build(tmp_path, READ_ONLY), "tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == set(CATALOG)
    by_name = {t["name"]: t for t in tools}
    assert by_name["issue_refund"]["gated"] is True
    assert by_name["issue_refund"]["requiredScope"] == Scope.REFUND_WRITE.value
    assert by_name["issue_refund"]["permitted"] is False
    assert by_name["retrieve_policy"]["permitted"] is True
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"


def test_unknown_method_is_a_jsonrpc_error(tmp_path) -> None:
    response = rpc(build(tmp_path), "tools/enumerate")
    assert response["error"]["code"] == METHOD_NOT_FOUND


def test_notification_gets_no_response(tmp_path) -> None:
    """A JSON-RPC message with no id must not be answered."""
    server = build(tmp_path)
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_malformed_json_line_returns_a_parse_error_and_keeps_serving(
    tmp_path,
) -> None:
    server = build(tmp_path)
    stdin = io.StringIO(
        "{not json\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        + "\n"
    )
    stdout = io.StringIO()
    serve_stdio(server, stdin, stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["error"]["code"] == -32700
    assert responses[1]["id"] == 2
    assert "tools" in responses[1]["result"]


# --- scope enforcement -----------------------------------------------------


def test_permitted_tool_call_succeeds(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY)
    response = call_tool(server, "lookup_customer", {"customer_id": "CUST-1042"})
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["owner_email"] == "r.mehta@brightfold.example"


def test_caller_without_the_scope_is_refused(tmp_path) -> None:
    """The headline case: read scopes must not buy a refund."""
    server = build(tmp_path, READ_ONLY)
    response = call_tool(
        server, "issue_refund", {"ticket_id": "T-001", "amount": 29.0, "reason": "x"}
    )
    assert "result" not in response
    error = response["error"]
    assert error["code"] == SCOPE_DENIED
    assert error["data"]["requiredScope"] == Scope.REFUND_WRITE.value
    assert error["data"]["tool"] == "issue_refund"


def test_a_refused_write_leaves_the_store_untouched(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY)
    call_tool(
        server, "issue_refund", {"ticket_id": "T-001", "amount": 29.0, "reason": "x"}
    )
    assert server.context.store.records("refunds") == []


def test_a_caller_with_no_scopes_at_all_is_refused_everything(tmp_path) -> None:
    server = build(tmp_path, frozenset())
    for name, spec in CATALOG.items():
        response = call_tool(server, name, {})
        assert response["error"]["code"] == SCOPE_DENIED, name
        assert response["error"]["data"]["requiredScope"] == spec.scope.value


def test_client_cannot_grant_itself_a_scope_via_params(tmp_path) -> None:
    """Scopes come from the process that started the server, never the client."""
    server = build(tmp_path, READ_ONLY)
    rpc(server, "initialize", {"grantedScopes": [s.value for s in Scope]})
    response = call_tool(
        server,
        "issue_refund",
        {"ticket_id": "T-001", "amount": 29.0, "reason": "x", "scopes": ["refund:write"]},
    )
    assert response["error"]["code"] == SCOPE_DENIED


def test_scope_is_checked_before_arguments_are_validated(tmp_path) -> None:
    """A refusal must not leak whether the arguments would have been accepted."""
    server = build(tmp_path, READ_ONLY)
    response = call_tool(server, "issue_refund", {"amount": -5})
    assert response["error"]["code"] == SCOPE_DENIED


def test_granting_the_write_scope_lets_the_call_through(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY | {Scope.REFUND_WRITE})
    response = call_tool(
        server, "issue_refund", {"ticket_id": "T-001", "amount": 29.0, "reason": "x"}
    )
    assert response["result"]["isError"] is False
    assert len(server.context.store.records("refunds")) == 1


# --- audit -----------------------------------------------------------------

AUDIT_FIELDS = {
    "run_id",
    "tool",
    "args",
    "caller_scopes",
    "status",
    "latency_ms",
    "timestamp",
}


def test_successful_call_is_audited_with_every_required_field(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY)
    call_tool(server, "lookup_customer", {"customer_id": "CUST-1042"}, run_id="run-a")

    records = server.audit.records()
    assert len(records) == 1
    record = records[0]
    assert AUDIT_FIELDS <= set(record)
    assert record["run_id"] == "run-a"
    assert record["tool"] == "lookup_customer"
    assert record["args"] == {"customer_id": "CUST-1042"}
    assert record["status"] == "ok"
    assert record["caller_scopes"] == sorted(s.value for s in READ_ONLY)
    assert record["latency_ms"] >= 0


def test_refusal_is_audited(tmp_path) -> None:
    """A refused call must leave the same evidence a successful one does."""
    server = build(tmp_path, READ_ONLY)
    call_tool(
        server,
        "issue_refund",
        {"ticket_id": "T-001", "amount": 29.0, "reason": "x"},
        run_id="run-b",
    )

    records = server.audit.records()
    assert len(records) == 1
    record = records[0]
    assert AUDIT_FIELDS <= set(record)
    assert record["status"] == "refused"
    assert record["tool"] == "issue_refund"
    assert record["run_id"] == "run-b"
    assert record["required_scope"] == Scope.REFUND_WRITE.value
    assert record["caller_scopes"] == sorted(s.value for s in READ_ONLY)
    assert record["args"] == {"ticket_id": "T-001", "amount": 29.0, "reason": "x"}


def test_a_failing_tool_is_audited_as_an_error(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY)
    response = call_tool(server, "lookup_customer", {"customer_id": "CUST-9999"})
    assert response["result"]["isError"] is True

    record = server.audit.records()[0]
    assert record["status"] == "error"
    assert "CUST-9999" in record["detail"]


def test_audit_log_is_append_only_across_calls(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY)
    call_tool(server, "lookup_customer", {"customer_id": "CUST-1042"})
    first = server.audit.path.read_text(encoding="utf-8")

    call_tool(server, "issue_refund", {"ticket_id": "T-001", "amount": 1, "reason": "x"})
    second = server.audit.path.read_text(encoding="utf-8")

    assert second.startswith(first)
    statuses = [r["status"] for r in server.audit.records()]
    assert statuses == ["ok", "refused"]


def test_audit_log_is_json_lines(tmp_path) -> None:
    server = build(tmp_path, READ_ONLY)
    call_tool(server, "retrieve_policy", {"query": "refund window"})
    call_tool(server, "send_reply", {"ticket_id": "T-001", "body": "hi"})

    lines = server.audit.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        assert set(json.loads(line)) >= AUDIT_FIELDS


def test_audit_survives_a_new_server_over_the_same_var_dir(tmp_path) -> None:
    """Append-only means across processes, not just within one."""
    first = build(tmp_path, READ_ONLY)
    call_tool(first, "lookup_customer", {"customer_id": "CUST-1042"})

    second = build(tmp_path, READ_ONLY)
    call_tool(second, "lookup_customer", {"customer_id": "CUST-1108"})

    assert len(second.audit.records()) == 2


def test_audit_dirs_are_isolated_per_var_dir(tmp_path) -> None:
    a = build(tmp_path / "a", READ_ONLY)
    b = build(tmp_path / "b", READ_ONLY)
    call_tool(a, "lookup_customer", {"customer_id": "CUST-1042"})
    assert len(a.audit.records()) == 1
    assert b.audit.records() == []


def test_run_id_is_recorded_as_unknown_when_the_client_omits_it(tmp_path) -> None:
    """Missing correlation must be visible in the trace, not silently blank."""
    server = build(tmp_path, READ_ONLY)
    rpc(server, "tools/call", {"name": "retrieve_policy", "arguments": {"query": "x"}})
    assert server.audit.records()[0]["run_id"] == "unknown"


def test_unknown_tool_is_rejected_and_audited(tmp_path) -> None:
    server = build(tmp_path)
    response = call_tool(server, "drop_database", {})
    assert response["error"]["code"] == METHOD_NOT_FOUND
    record = server.audit.records()[0]
    assert record["tool"] == "drop_database"
    assert record["status"] == "refused"


def test_audit_log_can_be_pointed_at_an_explicit_path(tmp_path) -> None:
    """Eval runs pass their own path so runs do not interleave."""
    path = tmp_path / "nested" / "run-17.jsonl"
    server = MCPServer(
        granted_scopes=READ_ONLY,
        var_dir=tmp_path / "var",
        audit=AuditLog(path),
    )
    call_tool(server, "lookup_customer", {"customer_id": "CUST-1042"})
    assert path.exists()
    assert len(server.audit.records()) == 1
