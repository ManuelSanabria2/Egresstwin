"""Perception: image -> Layout (Spark + SAM box prompts + OpenCV)."""

from egresstwin.perception.pipeline import PipelineResult, image_to_layout
from egresstwin.perception.plan_reader import (
    BoxPx,
    DimensionAnnotation,
    ExitDetection,
    PerceptionError,
    PlanRead,
    read_plan,
)

__all__ = [
    "BoxPx",
    "DimensionAnnotation",
    "ExitDetection",
    "PerceptionError",
    "PipelineResult",
    "PlanRead",
    "image_to_layout",
    "read_plan",
]
