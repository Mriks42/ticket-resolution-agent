"""The HTTP API (ADR 0012).

    TRA_API_KEY_SUBMIT=... TRA_API_KEY_APPROVE=... \
        uv run uvicorn src.api.app:create_app_from_env --factory

    GET  /health                    no key
    POST /tickets                   submit key: queue a run, 202 with its run_id
    GET  /runs/{run_id}             submit key: status, result, trace events
    GET  /approvals                 approve key: actions the gate held, pending
    POST /approvals/{approval_id}   approve key: approve (execute) or reject

**Two keys, one per side of the gate.** The submit key cannot approve and the
approve key cannot submit, mirroring the read and write scope split
(ADR 0002): whoever can put a ticket in front of the agent cannot also release
the refund it asked for. Keys come from the environment and are compared in
constant time. The app refuses to build without both, or with the same key
twice, rather than serving an API that is open by accident.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException

from src.agent.loop import LoopAgent
from src.api.models import ApprovalDecision, IncomingTicket
from src.api.service import AgentService, ApprovalConflict
from src.config import load_dotenv
from src.tools.errors import ToolInputError


def create_app(service: AgentService, *, submit_key: str, approve_key: str) -> FastAPI:
    if not submit_key or not approve_key:
        raise ValueError("both TRA_API_KEY_SUBMIT and TRA_API_KEY_APPROVE must be set")
    if hmac.compare_digest(submit_key.encode(), approve_key.encode()):
        raise ValueError("the submit and approve keys must differ, or either can approve")

    def holding(key: str, side: str):
        def check(x_api_key: str | None = Header(default=None)) -> None:
            if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), key.encode()):
                raise HTTPException(status_code=401, detail=f"a valid {side} key is required")
        return [Depends(check)]

    app = FastAPI(title="Support triage agent", version="0.1.0")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/tickets", status_code=202, dependencies=holding(submit_key, "submit"))
    async def submit(ticket: IncomingTicket) -> dict:
        try:
            run_id = service.submit(ticket)
        except ToolInputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return {"run_id": run_id, "status": "queued"}

    @app.get("/runs/{run_id}", dependencies=holding(submit_key, "submit"))
    async def run(run_id: str) -> dict:
        found = service.run(run_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no run {run_id!r}")
        return found

    @app.get("/approvals", dependencies=holding(approve_key, "approve"))
    async def approvals() -> dict:
        return {"pending": service.pending_approvals()}

    @app.post("/approvals/{approval_id}", dependencies=holding(approve_key, "approve"))
    async def decide(approval_id: str, decision: ApprovalDecision) -> dict:
        try:
            return await service.decide(approval_id, decision)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no approval {approval_id!r}") from None
        except ApprovalConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    return app


def create_app_from_env() -> FastAPI:
    """For uvicorn's --factory: the real agent, configured from `.env` and
    the environment, as evals/run.py configures it."""
    load_dotenv()
    agent = LoopAgent()
    return create_app(
        AgentService(agent=agent, context=agent.tool_context),
        submit_key=os.environ.get("TRA_API_KEY_SUBMIT", ""),
        approve_key=os.environ.get("TRA_API_KEY_APPROVE", ""),
    )
