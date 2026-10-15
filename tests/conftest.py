"""Shared fixtures.

`pg` gives a test its own Postgres schema, dropped afterwards, so tests never
read or write the tables a dev run or an eval run uses. Tests that need it are
skipped, with the reason shown, when Postgres is not reachable — they need
`docker compose up -d`, not a code change.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import uuid

import pytest

from src.config import read_dotenv


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
