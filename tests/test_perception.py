"""Perception tests with mock fixtures (no network, no real API cost).

The fake client returns the exact ground-truth doors of synthetic
plan B, so the pipeline must reproduce its walkable zone with
IoU >= 0.9, find all its exits, and produce a simulable Layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from egresstwin.config import Settings
from egresstwin.layout.edit import (
    add_exit,
    mark_obstacle_walkable,
    paint_wall,
    remove_exit,
)
from egresstwin.layout.schema import Exit, Layout, Obstacle, SpawnZone
from egresstwin.layout.validate import validate_layout
from egresstwin.meta.client import MetaClient
from egresstwin.perception.cv_fallback import extract_walls, mask_iou
from egresstwin.perception.pipeline import image_to_layout
from egresstwin.perception.plan_reader import PerceptionError, parse_plan_data, read_plan
from egresstwin.perception.scale import from_two_points, parse_annotation_meters
from egresstwin.perception.segmenter import (
    SegmentedObject,
    masks_to_obstacles,
    parse_sam_wire,
    segment_boxes,
    segment_concepts,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN_B = ROOT / "data" / "samples" / "spike_plan_synth.png"
GT_JSON = ROOT / "data" / "samples" / "spike_plan_synth_gt.json"
GT_MASK = ROOT / "data" / "samples" / "spike_plan_synth_gtmask.png"


@dataclass
class FakeResult:
    data: dict = field(default_factory=dict)
    text: str = ""
    cost_usd: float = 0.0
    cached: bool = False
    model: str = "fake"


class FakeClient:
    """Spark + SAM test double seeded from plan B ground truth."""

    def __init__(self, *, with_doors: bool = True, with_scale: bool = True) -> None:
        gt = json.loads(GT_JSON.read_text(encoding="utf-8"))
        self._doors = gt["doors"] if with_doors else []
        self._with_scale = with_scale
        self.concept_calls: list[str] = []

    def vision_structured(self, image_path, prompt, schema, schema_name="PlanRead"):  # noqa: ANN001, ANN202
        exits = [
            {"label": d["kind"], "confidence": 0.95, "box_px": d["box"]} for d in self._doors
        ]
        data = {
            "image_type": "floor_plan",
            "exits": exits,
            "dimension_annotations": (
                [{"text": "13.6 m", "box_px": [372, 578, 60, 14]}] if self._with_scale else []
            ),
            "scale_annotation": "13.6 m" if self._with_scale else None,
            "notes": "mock read",
            "confidence": 0.95,
        }
        return FakeResult(data=data, cost_usd=0.01, model="muse-spark-1.3")

    def segment_image_boxes(self, image_path, boxes):  # noqa: ANN001, ANN202
        segments = [
            {
                "label": "door",
                "confidence": 0.97,
                "box": {"x": b[0], "y": b[1], "w": b[2], "h": b[3]},
            }
            for b in boxes
        ]
        return FakeResult(data={"segments": segments}, cost_usd=0.0025, model="sam-3.1")

    def segment_image(self, image_path, concept):  # noqa: ANN001, ANN202
        self.concept_calls.append(concept)
        segments = [{"label": concept, "confidence": 0.4, "box": {"x": 1, "y": 1, "w": 5, "h": 5}}]
        return FakeResult(data={"segments": segments}, cost_usd=0.0025, model="sam-3.1")


def run_plan_b(tmp_path: Path, client: FakeClient | None = None, **kwargs):  # noqa: ANN202
    return image_to_layout(
        PLAN_B,
        client=client or FakeClient(),
        output_dir=tmp_path / "out",
        scale_two_points=((60, 572), (740, 572)),
        scale_distance_m=13.6,
        **kwargs,
    )


def test_synth_plan_b_iou_and_all_exits(tmp_path: Path) -> None:
    result = run_plan_b(tmp_path)
    layout = result.layout
    gt_mask = np.array(Image.open(GT_MASK)) > 0
    gt_grid = (
        cv2.resize(
            gt_mask.astype(np.float32), (layout.shape[1], layout.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        > 0.5
    )
    assert mask_iou(layout.effective_walkable(), gt_grid) >= 0.9
    gt = json.loads(GT_JSON.read_text(encoding="utf-8"))
    assert len(layout.exits) == len(gt["exits"]) == 2
    mpp = gt["meters_per_pixel"]
    for gt_exit in gt["exits"]:
        x, y, w, h = gt_exit["box"]
        centre = ((x + w / 2) * mpp, (y + h / 2) * mpp)
        nearest = min(
            float(np.hypot(e.midpoint[0] - centre[0], e.midpoint[1] - centre[1]))
            for e in layout.exits
        )
        assert nearest < 1.0
    assert layout.spawn_zones and layout.spawn_zones[0].planned_count() > 0
    assert validate_layout(layout) is not None


def test_synth_plan_b_layout_is_simulable(tmp_path: Path) -> None:
    from egresstwin.sim.engine import Scenario, run

    result = run_plan_b(tmp_path)
    layout = result.layout.model_copy(deep=True)
    layout.spawn_zones = [
        SpawnZone(
            id="z", polygon=[(8.0, 5.0), (10.0, 5.0), (10.0, 7.0), (8.0, 7.0)], count=3
        )
    ]
    sim = run(layout, Scenario(max_time_s=90.0), seed=1)
    assert np.isfinite(sim.exit_times).all()


def test_hand_style_plan_a_also_produces_layout() -> None:
    plan_a = ROOT / "data" / "samples" / "spike_plan_hand.png"
    result = image_to_layout(
        plan_a,
        client=FakeClient(),
        scale_two_points=((60, 572), (740, 572)),
        scale_distance_m=13.6,
    )
    assert len(result.layout.exits) == 2
    validate_layout(result.layout)


def test_report_overlay_and_costs(tmp_path: Path) -> None:
    result = run_plan_b(tmp_path)
    assert result.overlay_path is not None and result.overlay_path.is_file()
    assert result.report_path is not None and result.report_path.is_file()
    assert result.layout_path is not None and result.layout_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["source_image_hash"] == result.layout.source_image_hash
    assert report["scale"]["source"] == "user_two_points"
    assert report["scale"]["assumed"] is False
    assert report["costs"]["total_usd"] == pytest.approx(0.0125)
    assert len(report["exits"]) == 2
    assert all("confidence" in e for e in report["exits"])
    loaded = Layout.load_npz(result.layout_path)
    assert [e.id for e in loaded.exits] == [e.id for e in result.layout.exits]


def test_assumed_scale_is_flagged(tmp_path: Path) -> None:
    result = image_to_layout(PLAN_B, client=FakeClient(with_scale=False))
    assert result.report["scale"]["assumed"] is True
    assert result.report["scale"]["source"] == "assumed"
    assert any("assumed" in w for w in result.warnings)


def test_annotation_scale_is_not_assumed(tmp_path: Path) -> None:
    result = image_to_layout(PLAN_B, client=FakeClient())
    assert result.report["scale"]["source"] == "annotation"
    assert result.report["scale"]["assumed"] is False
    assert result.layout.meters_per_pixel == pytest.approx(0.02, rel=0.02)


def test_blank_image_fails_clearly(tmp_path: Path) -> None:
    blank = tmp_path / "blank.png"
    Image.new("RGB", (200, 200), "white").save(blank)
    with pytest.raises(PerceptionError, match="blank"):
        image_to_layout(blank, client=FakeClient())


def test_missing_and_corrupt_images_fail_clearly(tmp_path: Path) -> None:
    with pytest.raises(PerceptionError, match="not found"):
        image_to_layout(tmp_path / "nope.png", client=FakeClient())
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"this is not an image")
    with pytest.raises(PerceptionError, match="not readable"):
        image_to_layout(corrupt, client=FakeClient())


def test_no_exits_fails_clearly() -> None:
    with pytest.raises(PerceptionError, match="no exits"):
        image_to_layout(PLAN_B, client=FakeClient(with_doors=False))


def test_plan_reader_converts_normalized_grid() -> None:
    data = {
        "image_type": "floor_plan",
        "exits": [
            {
                "label": "exit", "confidence": 0.9,
                "x_min": 100, "y_min": 200, "x_max": 200, "y_max": 400,
            }
        ],
        "dimension_annotations": [
            {"text": "13.6 m", "x_min": 0, "y_min": 900, "x_max": 100, "y_max": 950}
        ],
        "scale_annotation": "13.6 m",
        "notes": "",
        "confidence": 0.9,
    }
    read = parse_plan_data(data, 800, 600)
    assert read.exits[0].box.as_list() == [80, 120, 80, 120]
    assert read.image_type == "floor_plan"


def test_read_plan_and_vision_cache(settings: Settings, tmp_path: Path) -> None:
    from PIL import ImageDraw

    sample_image = tmp_path / "plan.png"
    img = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(img).rectangle([40, 40, 280, 200], outline="black", width=4)
    img.save(sample_image)
    client = MetaClient(settings)
    first = read_plan(sample_image, client)
    assert first.cost_usd > 0 and first.cached is False
    second = read_plan(sample_image, client)
    assert second.cached is True and second.cost_usd == 0.0
    assert second.exits[0].box.as_list() == [120, 40, 60, 140]


def test_sam_wire_parser_and_concept_requests(tmp_path: Path) -> None:
    wire = "1<|box;x1=10;y1=20;x2=30;y2=50;c=0.9|><|mask;data=4,5,!abc;c=0.9|>"
    records = parse_sam_wire(wire)
    assert records[0]["box"] == [10, 20, 20, 30]
    assert records[0]["confidence"] == pytest.approx(0.9)
    client = FakeClient()
    result = segment_concepts(
        client, PLAN_B, ("tree", "car"), image_shape=(600, 800), threshold=0.5
    )
    assert client.concept_calls == ["tree", "car"]  # one request per concept
    assert result.objects == []  # 0.4 confidence filtered at 0.5

    class RecordClient:
        def segment_image_boxes(self, image_path, boxes):  # noqa: ANN001, ANN202
            return FakeResult(
                data={"records": records}, cost_usd=0.0025, model="sam-3.1"
            )

    boxed = segment_boxes(
        RecordClient(), PLAN_B, [[10, 20, 20, 30]], image_shape=(600, 800)
    )
    assert len(boxed.objects) == 1
    assert boxed.objects[0].confidence == pytest.approx(0.9)
    # Undecodable payload falls back to the box raster.
    assert boxed.objects[0].mask[20:50, 10:30].all()


def test_cv_fallback_finds_plan_b_gaps() -> None:
    cv = extract_walls(PLAN_B)
    assert len(cv.gaps) == 2
    assert cv.walkable.mean() > 0.3


def test_known_door_scale_and_sam_obstacles() -> None:
    result = image_to_layout(PLAN_B, client=FakeClient(), known_door=([180, 452, 80, 88], 1.6))
    assert result.report["scale"]["source"] == "known_door"
    assert result.layout.meters_per_pixel == pytest.approx(0.02)

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    objects = [
        SegmentedObject(label="tree", confidence=0.9, mask=mask, box=[2, 2, 3, 3]),
        SegmentedObject(label="door", confidence=0.9, mask=mask, box=[2, 2, 3, 3]),
    ]
    obstacles = masks_to_obstacles(objects)
    assert len(obstacles) == 1  # doors are openings, not obstacles
    assert obstacles[0]["label"] == "tree"
    assert obstacles[0]["confidence"] == pytest.approx(0.9)
    assert obstacles[0]["mask"].sum() == 9


def test_scale_helpers() -> None:
    est = from_two_points((60, 572), (740, 572), 13.6)
    assert est.meters_per_pixel == pytest.approx(0.02)
    assert est.assumed is False
    assert parse_annotation_meters("13.6 m") == pytest.approx(13.6)
    assert parse_annotation_meters("no scale here") is None


def _room() -> Layout:
    rows, cols, cell = 24, 40, 0.25
    walkable = np.zeros((rows, cols), dtype=bool)
    walkable[1:-1, 1:-1] = True
    return Layout(
        cell_size_m=cell,
        walkable_mask=walkable,
        obstacles=[
            Obstacle(
                id="o1",
                label="block",
                polygon=[(4.0, 2.0), (5.0, 2.0), (5.0, 3.0), (4.0, 3.0)],
            )
        ],
        exits=[Exit(id="E", segment=((10.0, 2.0), (10.0, 4.0)))],
        spawn_zones=[
            SpawnZone(id="z", polygon=[(0.5, 0.5), (3.0, 0.5), (3.0, 3.0), (0.5, 3.0)], count=5)
        ],
    )


def test_manual_edits_are_pure() -> None:
    layout = _room()
    added = add_exit(layout, ((0.0, 2.0), (0.0, 4.0)), exit_id="W")
    assert [e.id for e in layout.exits] == ["E"]
    assert [e.id for e in added.exits] == ["E", "W"]
    removed = remove_exit(added, "W")
    assert [e.id for e in removed.exits] == ["E"]
    with pytest.raises(ValueError, match="no exit"):
        remove_exit(layout, "missing")

    freed = mark_obstacle_walkable(layout, "o1")
    assert freed.obstacles == []
    assert layout.obstacles[0].id == "o1"  # original untouched
    cell = freed.cell_of(4.5, 2.5)
    assert freed.walkable_mask[cell]

    walled = paint_wall(layout, [(2.0, 1.0), (2.0, 5.0)], width_m=0.25)
    assert len(layout.obstacles) == 1 and len(walled.obstacles) == 2
    wall_cell = walled.cell_of(2.0, 3.0)
    assert not walled.effective_walkable()[wall_cell]
    assert layout.effective_walkable()[wall_cell]
