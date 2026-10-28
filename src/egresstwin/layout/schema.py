"""Layout model: walkable grid, obstacles, exits and spawn zones.

Coordinates are meters in a grid frame: x grows with columns, y grows
with rows (image-like). Cell (row, col) has centre
((col + 0.5) * cell_size_m, (row + 0.5) * cell_size_m). The walkable
mask and obstacle masks are binary grids of shape (rows, cols); they
are serialised to .npz (see Layout.save_npz), not to JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

Point = tuple[float, float]


def rasterize_polygon(
    polygon: list[Point], shape: tuple[int, int], cell_size_m: float
) -> np.ndarray:
    """Rasterise a meter-space polygon to a bool grid (cell centres inside)."""
    rows, cols = shape
    img = Image.new("1", (cols, rows), 0)
    draw = ImageDraw.Draw(img)
    pts = [(x / cell_size_m, y / cell_size_m) for x, y in polygon]
    draw.polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def polygon_area_m2(polygon: list[Point]) -> float:
    """Shoelace area of a meter-space polygon."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(polygon, [*polygon[1:], polygon[0]], strict=True):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


class Obstacle(BaseModel):
    """A blocked region given by polygon (meters) or by grid mask."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    label: str = ""
    removable: bool = False
    polygon: list[Point] | None = None
    mask: np.ndarray | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _check_geometry(self) -> Obstacle:
        if self.polygon is None and self.mask is None:
            raise ValueError(f"obstacle {self.id!r} needs a polygon or a mask")
        if self.polygon is not None and len(self.polygon) < 3:
            raise ValueError(f"obstacle {self.id!r} polygon needs >= 3 points")
        return self

    def rasterized(self, shape: tuple[int, int], cell_size_m: float) -> np.ndarray:
        """Bool grid of this obstacle at the layout resolution."""
        if self.mask is not None:
            if self.mask.shape != shape:
                raise ValueError(f"obstacle {self.id!r} mask shape {self.mask.shape} != {shape}")
            return self.mask.astype(bool)
        assert self.polygon is not None
        return rasterize_polygon(self.polygon, shape, cell_size_m)


class Exit(BaseModel):
    """An exit as a segment in meters; width is the segment length."""

    id: str
    segment: tuple[Point, Point]

    @property
    def width_m(self) -> float:
        (x1, y1), (x2, y2) = self.segment
        return float(np.hypot(x2 - x1, y2 - y1))

    @model_validator(mode="after")
    def _check_width(self) -> Exit:
        if self.width_m <= 0:
            raise ValueError(f"exit {self.id!r} segment has zero length")
        return self

    @property
    def midpoint(self) -> Point:
        (x1, y1), (x2, y2) = self.segment
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SpawnZone(BaseModel):
    """Where people appear: a polygon plus a headcount or a density."""

    id: str = ""
    polygon: list[Point]
    count: int | None = Field(default=None, ge=0)
    density_per_m2: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _check_amount(self) -> SpawnZone:
        if len(self.polygon) < 3:
            raise ValueError("spawn zone polygon needs >= 3 points")
        if (self.count is None) == (self.density_per_m2 is None):
            raise ValueError("spawn zone needs exactly one of count / density_per_m2")
        return self

    def planned_count(self) -> int:
        """People to spawn (density zones use the polygon area)."""
        if self.count is not None:
            return self.count
        assert self.density_per_m2 is not None
        return int(round(self.density_per_m2 * polygon_area_m2(self.polygon)))


class Layout(BaseModel):
    """A space ready for validation and simulation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    meters_per_pixel: float | None = Field(default=None, gt=0)
    cell_size_m: float = Field(default=0.1, gt=0)
    walkable_mask: np.ndarray = Field(exclude=True)
    obstacles: list[Obstacle] = Field(default_factory=list)
    exits: list[Exit] = Field(default_factory=list)
    spawn_zones: list[SpawnZone] = Field(default_factory=list)
    source_image_hash: str = ""

    @model_validator(mode="after")
    def _check_grid(self) -> Layout:
        if self.walkable_mask.ndim != 2 or self.walkable_mask.size == 0:
            raise ValueError("walkable_mask must be a non-empty 2D grid")
        ids = [e.id for e in self.exits]
        if len(ids) != len(set(ids)):
            raise ValueError("exit ids must be unique")
        return self

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape (rows, cols)."""
        return tuple(self.walkable_mask.shape)  # type: ignore[return-value]

    @property
    def width_m(self) -> float:
        return self.shape[1] * self.cell_size_m

    @property
    def height_m(self) -> float:
        return self.shape[0] * self.cell_size_m

    def effective_walkable(self, removed_obstacle_ids: frozenset[str] = frozenset()) -> np.ndarray:
        """Walkable mask minus obstacle masks (removed ones excluded)."""
        out = self.walkable_mask.astype(bool).copy()
        for obs in self.obstacles:
            if obs.id not in removed_obstacle_ids:
                out &= ~obs.rasterized(self.shape, self.cell_size_m)
        return out

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        """Grid cell (row, col) containing a meter-space point."""
        col = min(max(int(x / self.cell_size_m), 0), self.shape[1] - 1)
        row = min(max(int(y / self.cell_size_m), 0), self.shape[0] - 1)
        return row, col

    def save_npz(self, path: str | Path) -> None:
        """Serialise mask(s) plus JSON metadata to a single .npz file."""
        meta = self.model_dump(mode="json", exclude={"walkable_mask"})
        arrays: dict[str, Any] = {
            "walkable": self.walkable_mask.astype(np.uint8),
            "meta": np.array(json.dumps(meta)),
        }
        for i, obs in enumerate(self.obstacles):
            if obs.mask is not None:
                arrays[f"obstacle_mask_{i}"] = obs.mask.astype(np.uint8)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> Layout:
        """Inverse of save_npz."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            walkable = data["walkable"].astype(bool)
            for i, obs in enumerate(meta.get("obstacles", [])):
                key = f"obstacle_mask_{i}"
                if key in data:
                    obs["_mask"] = data[key].astype(bool)
        obstacles = []
        for obs in meta.get("obstacles", []):
            mask = obs.pop("_mask", None)
            obstacles.append(Obstacle(**obs, mask=mask))
        meta["obstacles"] = obstacles
        return cls(**meta, walkable_mask=walkable)
