"""Scenario specifications: named modification sets over a Layout."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from egresstwin.sim.events import SimEvent

Point = tuple[float, float]


class RemoveObstacleModification(BaseModel):
    """Remove one removable obstacle, freeing its cells."""

    type: Literal["remove_obstacle"] = "remove_obstacle"
    obstacle_id: str


class AddExitModification(BaseModel):
    """Open a new exit along the given meter-space segment."""

    type: Literal["add_exit"] = "add_exit"
    segment: tuple[Point, Point]
    exit_id: str | None = None


class WidenExitModification(BaseModel):
    """Widen an existing exit to width_m, keeping its midpoint."""

    type: Literal["widen_exit"] = "widen_exit"
    exit_id: str
    width_m: float = Field(gt=0)


Modification = Annotated[
    RemoveObstacleModification | AddExitModification | WidenExitModification,
    Field(discriminator="type"),
]


class ScenarioSpec(BaseModel):
    """One scenario: modifications plus exits blocked for the whole run."""

    name: str = Field(min_length=1)
    modifications: list[Modification] = Field(default_factory=list)
    blocked_exit_ids: list[str] = Field(default_factory=list)
    max_time_s: float = Field(default=300.0, gt=0)
    events: list[SimEvent] = Field(default_factory=list)  # timed live events (P07)
    # Pre-movement (reaction) times, uniform in [min, max]; 0 = instant.
    pre_movement_min_s: float = Field(default=0.0, ge=0)
    pre_movement_max_s: float = Field(default=0.0, ge=0)
    # Fixed desired speed for every agent (verification cases); None
    # keeps the engine's population distribution.
    desired_speed_mps: float | None = Field(default=None, gt=0)
