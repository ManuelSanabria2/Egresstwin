"""Image -> Layout pipeline (the P01 decision, productionised).

Spark reads doors and scale, SAM refines Spark boxes (box prompts,
threshold 0.5), OpenCV supplies walls, exits are CV outer-wall gaps
matched to Spark doors, and the result is validated. Manual edits in
layout/edit.py fix whatever the AI got wrong. All API calls go
through MetaClient: disk-cached by image hash and cost-logged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from egresstwin.layout.schema import Exit, Layout, Obstacle, SpawnZone
from egresstwin.layout.validate import LayoutValidationError, validate_layout
from egresstwin.perception import scale as scale_mod
from egresstwin.perception.cv_fallback import extract_walls
from egresstwin.perception.plan_reader import (
    PerceptionError,
    PlanRead,
    check_image,
    image_sha256,
    read_plan,
)
from egresstwin.perception.scale import ScaleEstimate
from egresstwin.perception.segmenter import (
    PLAN_CONCEPTS,
    SegmentedObject,
    masks_to_obstacles,
    segment_boxes,
    segment_concepts,
)

LOW_CONFIDENCE = 0.7


@dataclass
class PipelineResult:
    """Layout plus everything the report/overlay were built from."""

    layout: Layout
    report: dict[str, Any]
    plan_read: PlanRead
    overlay_path: Path | None = None
    report_path: Path | None = None
    layout_path: Path | None = None
    warnings: list[str] = field(default_factory=list)


def _resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA
    )
    return resized > 0.5


def _resolve_scale(
    plan: PlanRead,
    interior: np.ndarray | None,
    *,
    scale_two_points: tuple[tuple[float, float], tuple[float, float]] | None,
    scale_distance_m: float | None,
    known_door: tuple[list[int], float] | None,
) -> ScaleEstimate:
    if scale_two_points is not None:
        if scale_distance_m is None:
            raise PerceptionError("scale_distance_m is required with scale_two_points")
        a, b = scale_two_points
        return scale_mod.from_two_points(a, b, scale_distance_m)
    if known_door is not None:
        box, width_m = known_door
        door_px = float(min(box[2], box[3])) if len(box) == 4 else 0.0
        return scale_mod.from_known_door(door_px, width_m)
    texts = [plan.scale_annotation] + [a.text for a in plan.dimension_annotations]
    for text in texts:
        meters = scale_mod.parse_annotation_meters(text)
        if meters is None:
            continue
        if interior is not None and interior.any():
            xs = np.where(interior.any(axis=0))[0]
            reference_px = float(xs.max() - xs.min() + 1)
            return scale_mod.from_annotation(
                meters,
                reference_px,
                detail=f"annotation {text!r} over outer width {reference_px:.0f} px",
            )
        if plan.dimension_annotations and plan.dimension_annotations[0].box is not None:
            box = plan.dimension_annotations[0].box
            assert box is not None
            return scale_mod.from_annotation(meters, box.w, detail=f"annotation {text!r}")
    if plan.exits:
        widths = sorted(min(d.box.w, d.box.h) for d in plan.exits)
        return scale_mod.assumed_from_door(widths[len(widths) // 2])
    raise PerceptionError(
        "cannot resolve scale: give two reference points, a known door width, "
        "or use an image with a readable dimension annotation"
    )


def _gap_side(gap: list[int], bbox: tuple[int, int, int, int]) -> str:
    x1, x2, y1, y2 = bbox
    cx, cy = gap[0] + gap[2] / 2.0, gap[1] + gap[3] / 2.0
    distances = {
        "left": abs(cx - x1), "right": abs(cx - x2),
        "top": abs(cy - y1), "bottom": abs(cy - y2),
    }
    return min(distances, key=lambda k: distances[k])


def _build_exits(
    plan: PlanRead,
    gaps: list[list[int]],
    interior: np.ndarray,
    walkable_grid: np.ndarray,
    mpp: float,
    cell: float,
    sam_objects: list[SegmentedObject],
) -> tuple[list[Exit], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Exits from outer-wall gaps matched to Spark doors; rest are doors."""
    warnings: list[str] = []
    ys, xs = np.where(interior)
    bbox = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
    # Group gaps by (side, matched Spark detection).
    groups: dict[tuple[str, int], list[list[int]]] = {}
    unmatched_gaps: list[tuple[str, list[int]]] = []
    for gap in gaps:
        side = _gap_side(gap, bbox)
        centre = (gap[0] + gap[2] / 2.0, gap[1] + gap[3] / 2.0)
        best, best_dist = -1, float("inf")
        for i, det in enumerate(plan.exits):
            dist = float(np.hypot(det.box.centre[0] - centre[0], det.box.centre[1] - centre[1]))
            if dist < best_dist:
                best, best_dist = i, dist
        if best >= 0 and best_dist <= 120.0:
            groups.setdefault((side, best), []).append(gap)
        else:
            unmatched_gaps.append((side, gap))

    exits: list[Exit] = []
    exit_info: list[dict[str, Any]] = []
    used_detections: set[int] = set()

    def carve_throat(side: str, segment: tuple) -> None:
        """Open a 0.6 m walkable throat outward through the exit.

        Without it the wall field pushes agents back from the last
        walkable row and the engine's exit capture never triggers.
        """
        (x1, y1), (x2, y2) = segment
        depth = max(1, round(0.6 / cell))
        rows, cols = walkable_grid.shape
        if side in ("top", "bottom"):
            c_a = int(np.clip(min(x1, x2) / cell, 0, cols - 1))
            c_b = int(np.clip(max(x1, x2) / cell, 0, cols - 1))
            row = int(np.clip(y1 / cell, 0, rows - 1))
            lo, hi = (
                (row, min(rows, row + depth + 1))
                if side == "bottom"
                else (max(0, row - depth), row + 1)
            )
            walkable_grid[lo:hi, c_a : c_b + 1] = True
        else:
            r_a = int(np.clip(min(y1, y2) / cell, 0, rows - 1))
            r_b = int(np.clip(max(y1, y2) / cell, 0, rows - 1))
            col = int(np.clip(x1 / cell, 0, cols - 1))
            lo, hi = (
                (col, min(cols, col + depth + 1))
                if side == "right"
                else (max(0, col - depth), col + 1)
            )
            walkable_grid[r_a : r_b + 1, lo:hi] = True

    def snap_segment(side: str, a_px: float, b_px: float, edge_px: float) -> tuple:
        """Meter segment along one wall, snapped to the walkable edge row/col."""
        if side in ("top", "bottom"):
            col_a = int(np.clip(a_px * mpp / cell, 0, walkable_grid.shape[1] - 1))
            col_b = int(np.clip(b_px * mpp / cell, 0, walkable_grid.shape[1] - 1))
            cols = slice(min(col_a, col_b), max(col_a, col_b) + 1)
            rows = np.where(walkable_grid[:, cols].any(axis=1))[0]
            if len(rows):
                row = int(rows.min()) if side == "top" else int(rows.max())
            else:
                row = 0
            y_m = (row + 0.5) * cell
            return ((a_px * mpp, y_m), (b_px * mpp, y_m))
        row_a = int(np.clip(a_px * mpp / cell, 0, walkable_grid.shape[0] - 1))
        row_b = int(np.clip(b_px * mpp / cell, 0, walkable_grid.shape[0] - 1))
        rows = slice(min(row_a, row_b), max(row_a, row_b) + 1)
        cols = np.where(walkable_grid[rows, :].any(axis=0))[0]
        if len(cols):
            col = int(cols.min()) if side == "left" else int(cols.max())
        else:
            col = 0
        x_m = (col + 0.5) * cell
        return ((x_m, a_px * mpp), (x_m, b_px * mpp))

    for (side, det_idx), group in sorted(groups.items()):
        det = plan.exits[det_idx]
        used_detections.add(det_idx)
        if side in ("top", "bottom"):
            a = min(g[0] for g in group)
            b = max(g[0] + g[2] for g in group)
            edge = bbox[3] if side == "bottom" else bbox[2]
        else:
            a = min(g[1] for g in group)
            b = max(g[1] + g[3] for g in group)
            edge = bbox[1] if side == "right" else bbox[0]
        segment = snap_segment(side, float(a), float(b), float(edge))
        conf = det.confidence
        sam_match = _sam_confidence(det.box.as_list(), sam_objects)
        source = "spark+opencv_gap"
        if sam_match is not None:
            conf = (conf + sam_match) / 2.0
            source = "spark+sam+opencv_gap"
        exit_id = f"E{len(exits) + 1}"
        exits.append(Exit(id=exit_id, segment=segment))
        carve_throat(side, segment)
        exit_info.append(
            {"id": exit_id, "label": det.label, "confidence": round(conf, 4),
             "source": source, "box_px": det.box.as_list(), "width_m": exits[-1].width_m}
        )
        if conf < LOW_CONFIDENCE:
            warnings.append(f"exit {exit_id} has low confidence {conf:.2f}; verify it manually")

    # Spark detections explicitly labelled exit but with no matched gap.
    for i, det in enumerate(plan.exits):
        if i in used_detections or det.label.strip().lower() != "exit":
            continue
        cx, cy = det.box.centre
        side = _gap_side([round(cx), round(cy), 1, 1], bbox)
        if side in ("top", "bottom"):
            segment = snap_segment(side, det.box.x, det.box.x + det.box.w, cy)
        else:
            segment = snap_segment(side, det.box.y, det.box.y + det.box.h, cx)
        exit_id = f"E{len(exits) + 1}"
        exits.append(Exit(id=exit_id, segment=segment))
        carve_throat(side, segment)
        exit_info.append(
            {"id": exit_id, "label": det.label, "confidence": round(det.confidence, 4),
             "source": "spark_label", "box_px": det.box.as_list(), "width_m": exits[-1].width_m}
        )
        warnings.append(f"exit {exit_id} comes from a Spark label with no wall gap; verify it")
        used_detections.add(i)

    if unmatched_gaps:
        warnings.append(
            f"{len(unmatched_gaps)} wall gap(s) matched no Spark door and were ignored; "
            "add any real exit manually"
        )
    doors = [
        {"label": d.label, "confidence": round(d.confidence, 4), "box_px": d.box.as_list()}
        for i, d in enumerate(plan.exits)
        if i not in used_detections
    ]
    return exits, exit_info, doors, warnings


def _sam_confidence(box: list[int], objects: list[SegmentedObject]) -> float | None:
    best: float | None = None
    best_iou = 0.0
    for obj in objects:
        ax2, ay2 = box[0] + box[2], box[1] + box[3]
        bx2, by2 = obj.box[0] + obj.box[2], obj.box[1] + obj.box[3]
        iw = max(0, min(ax2, bx2) - max(box[0], obj.box[0]))
        ih = max(0, min(ay2, by2) - max(box[1], obj.box[1]))
        union = box[2] * box[3] + obj.box[2] * obj.box[3] - iw * ih
        iou = iw * ih / union if union else 0.0
        if iou > best_iou:
            best_iou, best = iou, obj.confidence
    return best if best_iou > 0.1 else None


def _save_overlay(
    path: Path, image_path: Path, layout: Layout, mpp: float, exit_info: list[dict[str, Any]]
) -> None:
    with Image.open(image_path) as im:
        base = im.convert("RGB")
    w_px, h_px = base.size
    walk_up = _resize_bool(layout.walkable_mask, (h_px, w_px))
    tint = np.zeros((h_px, w_px, 4), dtype=np.uint8)
    tint[walk_up] = (0, 200, 0, 70)
    for obs in layout.obstacles:
        mask = obs.rasterized(layout.shape, layout.cell_size_m)
        tint[_resize_bool(mask, (h_px, w_px))] = (220, 0, 0, 90)
    overlay = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(tint))
    draw = ImageDraw.Draw(overlay)
    for info in exit_info:
        box = info["box_px"]
        rect = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        draw.rectangle(rect, outline=(0, 120, 255), width=3)
    for ex in layout.exits:
        (x1, y1), (x2, y2) = ex.segment
        draw.line([x1 / mpp, y1 / mpp, x2 / mpp, y2 / mpp], fill=(255, 0, 0), width=5)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(path)


def image_to_layout(
    image_path: str | Path,
    *,
    client: Any = None,
    output_dir: str | Path | None = None,
    cell_size_m: float = 0.1,
    scale_two_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
    scale_distance_m: float | None = None,
    known_door: tuple[list[int], float] | None = None,
    concepts: tuple[str, ...] = PLAN_CONCEPTS,
    sam_threshold: float = 0.5,
    spawn_count: int = 50,
) -> PipelineResult:
    """Turn one plan image into a validated Layout plus report and overlay."""
    path = Path(image_path)
    width, height = check_image(path)
    digest = image_sha256(path)
    if client is None:
        from egresstwin.meta.client import MetaClient

        client = MetaClient()
    warnings: list[str] = []

    plan = read_plan(path, client)
    if plan.image_type == "aerial_photo":
        warnings.append("aerial-photo path is deferred (P01); results need manual review")
    elif plan.image_type == "other":
        warnings.append("Spark did not recognise a floor plan; relying on OpenCV and manual checks")

    cv = extract_walls(path)
    scale = _resolve_scale(
        plan,
        cv.interior,
        scale_two_points=scale_two_points,
        scale_distance_m=scale_distance_m,
        known_door=known_door,
    )
    if scale.assumed:
        warnings.append(f"scale is assumed, not measured: {scale.detail}")
    mpp = scale.meters_per_pixel

    cols = max(1, round(width * mpp / cell_size_m))
    rows = max(1, round(height * mpp / cell_size_m))
    walkable = _resize_bool(cv.walkable, (rows, cols))
    # Door slabs sit in the wall gaps: carve Spark door boxes (clipped to
    # the interior) back into walkable, or interior walls seal every room.
    if plan.exits:
        door_mask = np.zeros((height, width), dtype=bool)
        for det in plan.exits:
            b = det.box
            x0, y0 = max(round(b.x), 0), max(round(b.y), 0)
            x1 = min(round(b.x + b.w), width)
            y1 = min(round(b.y + b.h), height)
            if x1 > x0 and y1 > y0:
                door_mask[y0:y1, x0:x1] = True
        # Dilate a few px so the slab outline and dilated wall ink
        # around the opening are cleared too, not just the slab fill.
        door_mask = cv2.dilate(
            door_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=3,
        ) > 0
        interior_grid = _resize_bool(cv.interior, (rows, cols))
        walkable |= _resize_bool(door_mask, (rows, cols)) & interior_grid

    boxes = [d.box.as_list() for d in plan.exits]
    sam_box = segment_boxes(
        client, path, boxes, labels=[d.label for d in plan.exits],
        image_shape=(height, width), threshold=sam_threshold,
    )
    sam_concepts = segment_concepts(
        client, path, concepts, image_shape=(height, width), threshold=sam_threshold
    )
    if boxes and not sam_box.objects:
        warnings.append("SAM box prompts returned nothing above threshold; using Spark boxes")
    if concepts and not sam_concepts.objects:
        warnings.append("SAM concept prompts returned no matches (valid zero result)")

    obstacles: list[Obstacle] = []
    obstacle_info: list[dict[str, Any]] = []
    for desc in masks_to_obstacles(sam_concepts.objects):
        grid_mask = _resize_bool(np.asarray(desc["mask"]), (rows, cols))
        if not grid_mask.any():
            continue
        obstacles.append(Obstacle(id=str(desc["id"]), label=str(desc["label"]), mask=grid_mask))
        obstacle_info.append(
            {
                "id": desc["id"],
                "label": desc["label"],
                "confidence": round(float(desc["confidence"]), 4),
                "box_px": desc["box"],
                "source": "sam_concept",
            }
        )
        if float(desc["confidence"]) < LOW_CONFIDENCE:
            conf = float(desc["confidence"])
            warnings.append(f"obstacle {desc['id']} has low confidence {conf:.2f}")

    # Prune tiny isolated walkable specks (e.g. gaps between stair
    # treads enclosed by ink): real rooms are far larger than 1 m2 and
    # anything bigger that stays disconnected fails validation below.
    from scipy import ndimage

    labelled, n_comp = ndimage.label(walkable)
    if n_comp > 1:
        sizes = ndimage.sum_labels(walkable, labelled, range(1, n_comp + 1))
        speck_cells = max(1, round(1.0 / (cell_size_m**2)))
        pruned = 0
        for comp, size in enumerate(sizes, start=1):
            if size < speck_cells and size < sizes.max():
                walkable[labelled == comp] = False
                pruned += int(size)
        if pruned:
            warnings.append(
                f"pruned {pruned} isolated walkable cells smaller than 1 m2 (ink specks)"
            )

    exits, exit_info, doors, exit_warnings = _build_exits(
        plan, cv.gaps, cv.interior, walkable, mpp, cell_size_m, sam_box.objects
    )
    warnings.extend(exit_warnings)
    if not exits:
        raise PerceptionError(
            "no exits found in the image: check that it is a floor plan with visible "
            "exit openings, or add an exit manually with layout.edit.add_exit"
        )

    # Default spawn zone: walkable bounding box, inset.
    eff = walkable.copy()
    for obs in obstacles:
        eff &= ~obs.rasterized((rows, cols), cell_size_m)
    ys, xs = np.where(eff)
    x_a, x_b = (xs.min() + 0.5) * cell_size_m, (xs.max() + 0.5) * cell_size_m
    y_a, y_b = (ys.min() + 0.5) * cell_size_m, (ys.max() + 0.5) * cell_size_m
    ix, iy = 0.15 * (x_b - x_a), 0.15 * (y_b - y_a)
    spawn = SpawnZone(
        id="spawn-1",
        polygon=[
            (x_a + ix, y_a + iy),
            (x_b - ix, y_a + iy),
            (x_b - ix, y_b - iy),
            (x_a + ix, y_b - iy),
        ],
        count=spawn_count,
    )

    layout = Layout(
        meters_per_pixel=mpp,
        cell_size_m=cell_size_m,
        walkable_mask=walkable,
        obstacles=obstacles,
        exits=exits,
        spawn_zones=[spawn],
        source_image_hash=digest,
    )
    try:
        validation_warnings = validate_layout(layout)
    except LayoutValidationError as exc:
        raise PerceptionError(f"layout needs manual correction: {exc}") from exc
    warnings.extend(validation_warnings)

    total_cost = plan.cost_usd + sam_box.cost_usd + sam_concepts.cost_usd
    report: dict[str, Any] = {
        "source_image": str(path),
        "source_image_hash": digest,
        "image_size_px": {"w": width, "h": height},
        "image_type": plan.image_type,
        "models": {"spark": plan.model, "sam": sam_box.model or sam_concepts.model},
        "cell_size_m": cell_size_m,
        "grid_shape": {"rows": rows, "cols": cols},
        "meters_per_pixel": mpp,
        "scale": scale.as_dict(),
        "exits": exit_info,
        "doors": doors,
        "obstacles": obstacle_info,
        "spawn_zones": [{"id": spawn.id, "count": spawn.count, "polygon": spawn.polygon}],
        "plan_notes": plan.notes,
        "plan_confidence": plan.confidence,
        "warnings": warnings,
        "validation_warnings": validation_warnings,
        "costs": {
            "spark_usd": round(plan.cost_usd, 8),
            "sam_usd": round(sam_box.cost_usd + sam_concepts.cost_usd, 8),
            "total_usd": round(total_cost, 8),
            "cached": bool(plan.cached and sam_box.cached),
        },
        "walkable_cells": int(eff.sum()),
    }

    overlay_path = report_path = layout_path = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        overlay_path = out / "overlay.png"
        _save_overlay(overlay_path, path, layout, mpp, exit_info)
        layout_path = out / "layout.npz"
        layout.save_npz(layout_path)
        report_path = out / "layout_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return PipelineResult(
        layout=layout, report=report, plan_read=plan,
        overlay_path=overlay_path, report_path=report_path, layout_path=layout_path,
        warnings=warnings,
    )
