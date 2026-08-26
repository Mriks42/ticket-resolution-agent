# 2. A stdlib JSON-RPC MCP server over the official `mcp` SDK

## Status

Accepted

## Context

CLAUDE.md specifies "MCP (Model Context Protocol) server exposing agent tools"
with per-tool permission scopes and an append-only audit record per call. It
also says: prefer stdlib and the existing stack, and ask before adding a
dependency.

What the server actually has to do in this project is narrow. One client — our
own agent loop — connects over stdio and calls six tools. There is no second
consumer, no third-party client, and no HTTP transport requirement: the FastAPI
service is this project's network surface, and the MCP server sits inside the
agent process behind it.

The protocol surface that need covers is four methods: `initialize`,
`tools/list`, `tools/call`, and `ping`, over newline-delimited JSON-RPC 2.0.

## Options considered

**The official `mcp` Python SDK**

- Maintained implementation of the spec, tracking revisions as they land.
- Gives the full protocol surface: resources, prompts, sampling, roots,
  progress and cancellation notifications, structured output schemas, and
  the Streamable HTTP transport in addition to stdio.
- Interoperates with third-party MCP clients that have been tested against it.
- Dependency cost, measured rather than estimated:

  ```
  uv pip compile - --python-version 3.11   # input: "mcp"
  ```

  resolves `mcp==2.1.1` to **30 packages**. Twelve are already in this
  project's environment (pydantic, anyio, starlette, uvicorn, click, h11,
  idna, colorama, typing-extensions and friends). The other **18 are new**:

  `attrs`, `cffi`, `cryptography`, `httpcore2`, `httpx2`, `jsonschema`,
  `jsonschema-specifications`, `mcp`, `mcp-types`, `opentelemetry-api`,
  `pycparser`, `pyjwt`, `python-multipart`, `pywin32`, `referencing`,
  `rpds-py`, `sse-starlette`, `truststore`.

  That takes the declared runtime dependency list from 3 to 4, and the
  installed set from 20 to 38. Most of the additions serve transports and
  features this project does not use: `cryptography` + `cffi` + `pycparser`
  pull in a C-extension build chain, `pyjwt` and `truststore` exist for OAuth
  and TLS trust in the HTTP transport, and `opentelemetry-api` is a second
  tracing story alongside the JSON Lines logs ADR 0001 already settled on.

**Hand-rolled JSON-RPC 2.0 over stdio, stdlib only**

- `json` for framing, a dispatch table for four methods, file I/O for the
  audit log. Roughly 250 lines in `src/mcp/`, no new packages.
- Scope enforcement is hand-built in either case. This is the decisive point.
  The SDK has no permission model matching what CLAUDE.md requires — a scope
  per tool, plus a gate keyed on action type and retrieval coverage. Choosing
  the SDK would not have saved that work; it would have added a layer to
  thread it through.
- The audit record has to wrap every call including refused ones. Owning the
  dispatch path makes that a single choke point rather than a middleware hook
  that has to be kept in front of every entry point.
- Terraform/Lambda packaging stays small, and there is no C-extension build
  step in CI.

## Decision

Hand-roll the server: stdlib JSON-RPC 2.0 over stdio, in `src/mcp/server.py`.
Implement `initialize`, `tools/list`, `tools/call` and `ping` against protocol
revision `2025-06-18`, and accept notifications without answering them.

Scopes are set by the process that spawns the server (`MCPServer(granted_scopes=...)`,
or `--scopes` on the CLI) and merely *reported* by `initialize`. Nothing a
client sends can widen them.

## Consequences

- **Protocol conformance is ours to maintain.** We implement one revision and
  a four-method subset of it. When the spec moves, nothing tells us; a
  version bump is a code change here, not a dependency bump. The subset is
  small enough that this is a contained cost, and `tests/test_mcp_server.py`
  pins the framing behaviour we rely on — but it is a real, recurring cost
  and it is the main thing this decision buys down dependencies with.
- **We give up the richer MCP feature set.** Resources, prompts, sampling,
  roots, progress and cancellation notifications, and structured output
  schemas are not implemented. If any of those becomes load-bearing —
  serving the policy corpus as MCP *resources* rather than through
  `retrieve_policy` is the most plausible candidate — the honest move is to
  swap to the SDK rather than grow a second implementation of the spec. The
  catalog in `src/tools/catalog.py` is deliberately declarative (name,
  description, scope, JSON schema, handler) so that swap is a rewrite of
  `src/mcp/`, not of the tools.
- **stdio only.** No Streamable HTTP or SSE transport, so no remote MCP
  clients. This is fine while the only client is our own agent loop in the
  same process tree, and it is the assumption to re-check first if that
  changes.
- **No third-party client interop.** The server has not been exercised
  against Claude Desktop or an IDE client, and should not be advertised as
  working with them without testing it.
- We keep a 3-dependency runtime and a CI install with no compiled
  extensions.

### What would reverse this

Any one of: a second consumer of the tools outside this process; a need for
resources/prompts/sampling; or a requirement to serve MCP over HTTP. None are
on the roadmap through the sessions currently planned.
