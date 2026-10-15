"""Shared fixtures.

`api` is a scripted HTTP server on a loopback port, for testing provider
clients against the bytes they really send and receive.

`ctx`, `server`, `writer`, `trace` and `make` are the agent-loop rig: real
tools on the lexical backend in an isolated var dir, a read-only server for the
model's calls and a write-scoped one only the gate reaches, sharing one store
and one audit log.

`pg` gives a test its own Postgres schema, dropped afterwards, so tests never
read or write the tables a dev run or an eval run uses. Tests that need it are
skipped, with the reason shown, when Postgres is not reachable — they need
`docker compose up -d`, not a code change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.agent.config import AgentConfig
from src.agent.executor import MCPToolExecutor
from src.agent.loop import READ_SCOPES, WRITE_SCOPES, AgentLoop
from src.agent.trace import TraceLog
from src.config import read_dotenv
from src.mcp.server import MCPServer
from src.tools.context import ToolContext
from src.tools.retrieval import RetrievalConfig


@pytest.fixture
def pg():
    psycopg = pytest.importorskip("psycopg")
    # Read, never load: copying .env into os.environ here would leak a
    # developer's RETRIEVAL_BACKEND into every test that runs after this one.
    url = os.environ.get("DATABASE_URL") or read_dotenv().get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set; run `docker compose up -d`")
    try:
        conn = psycopg.connect(url, autocommit=True, connect_timeout=2)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable ({exc.__class__.__name__}); run `docker compose up -d`")

    schema = f"test_{uuid.uuid4().hex[:12]}"
    # The extension must exist before search_path moves, or the first test on
    # a fresh database would install it into a schema that is about to be
    # dropped.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    try:
        yield conn
    finally:
        if not conn.closed:
            conn.execute(f"DROP SCHEMA {schema} CASCADE")
            conn.close()
        else:
            with psycopg.connect(url, autocommit=True) as cleanup:
                cleanup.execute(f"DROP SCHEMA {schema} CASCADE")


class HashingEmbedder:
    """Deterministic bag-of-words vectors, for tests that need the dense
    index's plumbing without a network call.

    It shares vocabulary-matching behaviour with the lexical index, which is
    the point: a test can predict what it ranks first. It is never used by
    `evals/` — eval numbers come from the real model only.
    """

    def __init__(self, model: str = "test-hashing", dims: int = 256) -> None:
        self.model = model
        self.dims = dims
        self.calls: list[list[str]] = []

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        # The last dimension is a constant so no text maps to the zero vector,
        # whose cosine distance is undefined.
        vec[-1] = 1.0
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % (self.dims - 1)
            vec[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]


@pytest.fixture
def hashing_embedder() -> HashingEmbedder:
    return HashingEmbedder()


class ScriptedAPI:
    """Replays `(status, body, delay)` per request and records what it got."""

    def __init__(self) -> None:
        self.script: list[tuple[int, object, float]] = []
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def _handler(self):
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                length = int(self.headers.get("Content-Length", 0))
                api.requests.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                status, body, delay = api.script.pop(0)
                if delay:
                    time.sleep(delay)
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client timed out and hung up; expected

            def log_message(self, *args) -> None:
                pass

        return Handler

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def api():
    server = ScriptedAPI()
    server.start()
    yield server
    server.stop()


# --- agent loop rig ----------------------------------------------------------


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext.build(var_dir=tmp_path / "var", retrieval=RetrievalConfig())


@pytest.fixture
def server(ctx, tmp_path) -> MCPServer:
    return MCPServer(granted_scopes=READ_SCOPES, context=ctx, var_dir=tmp_path / "var")


@pytest.fixture
def writer(ctx, tmp_path) -> MCPServer:
    return MCPServer(granted_scopes=WRITE_SCOPES, context=ctx, var_dir=tmp_path / "var")


@pytest.fixture
def trace(tmp_path) -> TraceLog:
    return TraceLog(tmp_path / "var" / "trace" / "trace.jsonl")


@pytest.fixture
def make(server, writer, trace):
    def build(llm, *, executor=None, write_executor=None, clock=time.monotonic, spend=None, **config):
        return AgentLoop(
            llm=llm,
            executor=executor or MCPToolExecutor(server, timeout=5),
            writer=write_executor or MCPToolExecutor.for_writes(writer, timeout=5),
            trace=trace,
            config=AgentConfig(**config),
            clock=clock,
            spend=spend,
        )

    return build
