"""Scenario specifications: named modification sets over a Layout."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

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
