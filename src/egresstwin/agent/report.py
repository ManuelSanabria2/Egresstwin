"""Structured agent report submitted through submit_report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from egresstwin.scenarios.spec import Modification

Objective = Literal["minimize_t95", "minimize_peak_density"]

OBJECTIVE_METRIC: dict[str, str] = {
    "minimize_t95": "t95_s",
    "minimize_peak_density": "peak_density_per_m2",
}


class Recommendation(BaseModel):
    """One recommended change set, backed by a simulated scenario."""

    modifications: list[Modification] = Field(min_length=1)
    scenario_id: str
    claimed_metrics: dict[str, float] = Field(min_length=1)
    rationale: str = ""


class RobustnessClaim(BaseModel):
    """Claimed N-1 outcome for one recommended scenario."""

    scenario_id: str
    blocked_exit_id: str
    claimed_t95_s: float = Field(gt=0)


class AgentReport(BaseModel):
    """Final report: recommendations, robustness, limitations."""

    baseline_id: str
    objective: Objective
    recommendations: list[Recommendation] = Field(default_factory=list)
    robustness: list[RobustnessClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    summary: str = ""
