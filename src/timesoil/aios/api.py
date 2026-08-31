from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from .agents import AgentState, AgentWorkflow
from .economics import CHDDEconomicsAdapter, EconomicResult
from .llm import APPROVED_MODEL, LLMConfig, TatneftLLMClient
from .tools import GROUNDED_ROLE_TOOLS, build_grounded_tool_registry
from .ui import OPERATOR_PAGE, UI_HEADERS


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(APIModel):
    status: Literal["ok"]


class QwenCapability(APIModel):
    model: Literal["qwen3.6-35b-a3b"]
    configured: bool
    connectivity_verified: bool


class TrackCapability(APIModel):
    component_available: bool
    certified: bool


class Track2Capability(TrackCapability):
    model_z_trained: bool


class CHDDCapability(APIModel):
    component_available: bool
    ready: bool


class CapabilitiesResponse(APIModel):
    qwen: QwenCapability
    track1: TrackCapability
    track2: Track2Capability
    chdd: CHDDCapability


class AgentExperimentRequest(APIModel):
    context: dict[str, Any]


class AgentToolEvidenceResponse(APIModel):
    call_id: str
    tool: str
    output: dict[str, Any]


class AgentDecisionResponse(APIModel):
    role: str
    summary: str
    recommendation: str
    evidence: list[str]
    approved: bool
    tools: list[AgentToolEvidenceResponse]


class AgentExperimentResponse(APIModel):
    run_id: str
    complete: bool
    critic_approved: bool
    decisions: list[AgentDecisionResponse]


type CHDDNumber = StrictInt | StrictFloat


class CHDDRecord(APIModel):
    DATA: date
    well: str = Field(min_length=1, max_length=128)
    WLPT: CHDDNumber
    WLPR: CHDDNumber
    WOMT: CHDDNumber
    WOMR: CHDDNumber
    WWIR: CHDDNumber
    WWIT: CHDDNumber
    THP: CHDDNumber
    BHP: CHDDNumber
    WEFF: CHDDNumber
    WLPT_Diff: CHDDNumber
    WOMT_Diff: CHDDNumber
    WWIT_Diff: CHDDNumber


class CHDDRequest(APIModel):
    records: list[CHDDRecord] = Field(min_length=1)
    start_year: int = Field(ge=1900, le=9999)


class CHDDResponse(APIModel):
    run_id: str
    total_chdd_m: float
    profitability_index: float
    start_date: str
    max_date: str
    diagnostics: dict[str, Any]


async def get_agent_workflow() -> AsyncIterator[AgentWorkflow]:
    try:
        config = LLMConfig.from_env()
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qwen3.6 is not configured",
        ) from None
    async with TatneftLLMClient(config) as client:
        yield AgentWorkflow(
            client,
            build_grounded_tool_registry(),
            role_tools=GROUNDED_ROLE_TOOLS,
        )


def get_chdd_adapter() -> CHDDEconomicsAdapter:
    try:
        return CHDDEconomicsAdapter.from_env()
    except (OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CHDD is not configured",
        ) from None


def get_runs_dir() -> Path:
    configured = os.environ.get("AIOS_RUNS_DIR", "").strip()
    return Path(configured or "results/aios-runs").resolve()


AgentWorkflowDep = Annotated[AgentWorkflow, Depends(get_agent_workflow)]
CHDDAdapterDep = Annotated[CHDDEconomicsAdapter, Depends(get_chdd_adapter)]
RunsDirDep = Annotated[Path, Depends(get_runs_dir)]


def _qwen_configured() -> bool:
    try:
        LLMConfig.from_env()
    except ValueError:
        return False
    return True


def _chdd_ready() -> bool:
    try:
        CHDDEconomicsAdapter.from_env()
    except (OSError, ValueError):
        return False
    return True


app = FastAPI(title="TimesOil AIOS", version="1")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def operator_page() -> HTMLResponse:
    return HTMLResponse(OPERATOR_PAGE, headers=UI_HEADERS)


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/v1/capabilities")
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        qwen=QwenCapability(
            model=APPROVED_MODEL,
            configured=_qwen_configured(),
            connectivity_verified=False,
        ),
        track1=TrackCapability(component_available=True, certified=False),
        track2=Track2Capability(
            component_available=True,
            certified=False,
            model_z_trained=False,
        ),
        chdd=CHDDCapability(component_available=True, ready=_chdd_ready()),
    )


@app.post("/v1/experiments/agents")
async def run_agent_experiment(
    request: AgentExperimentRequest,
    workflow: AgentWorkflowDep,
) -> AgentExperimentResponse:
    state: AgentState = await workflow.run(request.context)
    return AgentExperimentResponse(
        run_id=state.run_id,
        complete=state.complete,
        critic_approved=state.critic_approved,
        decisions=[
            AgentDecisionResponse(
                role=decision.role.value,
                summary=decision.summary,
                recommendation=decision.recommendation,
                evidence=list(decision.evidence),
                approved=decision.approved,
                tools=[
                    AgentToolEvidenceResponse(
                        call_id=item.call_id,
                        tool=item.tool,
                        output=dict(item.output),
                    )
                    for item in decision.tool_evidence
                ],
            )
            for decision in state.decisions
        ],
    )


@app.post("/v1/economics/chdd")
def run_chdd(
    request: CHDDRequest,
    adapter: CHDDAdapterDep,
    runs_dir: RunsDirDep,
) -> CHDDResponse:
    run_id = uuid4().hex
    result: EconomicResult = adapter.calculate(
        [record.model_dump(mode="json") for record in request.records],
        start_year=request.start_year,
        output_dir=runs_dir / run_id,
    )
    return CHDDResponse(
        run_id=run_id,
        total_chdd_m=result.total_chdd_m,
        profitability_index=result.profitability_index,
        start_date=result.start_date,
        max_date=result.max_date,
        diagnostics=dict(result.diagnostics),
    )
