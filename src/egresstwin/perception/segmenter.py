"""SAM 3.1 segmentation following the P01 spike decision.

P01 found SAM concept prompts return zero matches on line plans at
every threshold, while box prompts seeded by Spark boxes scored
recall 1.00 with confidence 0.957-0.980. Production therefore uses
SAM only in box-prompt mode on plans, with threshold 0.5. Concept
mode stays available (one request per concept, as the API requires)
for the deferred photo path. Wire output is parsed here and masks
are decoded with the official meta-sam-parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Concepts decided in P01: none on floor plans (all returned zero).
PLAN_CONCEPTS: tuple[str, ...] = ()
# Deferred photo path candidates (P01: re-test on a real photo first).
AERIAL_CONCEPTS: tuple[str, ...] = ("building", "tent", "stage", "tree", "car")
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DOOR_LABELS = {"door", "doorway", "exit"}


@dataclass
class SegmentedObject:
    """One SAM instance: label, confidence and a full-image bool mask."""

    label: str
    confidence: float
    mask: np.ndarray  # bool, image shape (rows, cols)
    box: list[int]  # [x, y, w, h] pixels

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence, "box": list(self.box)}


@dataclass
class SegmentationResult:
    """Objects from SAM plus the API cost that produced them."""

    objects: list[SegmentedObject] = field(default_factory=list)
    cost_usd: float = 0.0
    cached: bool = False
    model: str = ""


_BOX_RE = re.compile(r"<\|box;([^|]*)\|>")
_MASK_RE = re.compile(r"<\|mask;([^|]*)\|>")
_ID_RE = re.compile(r"(\d+)\s*$")


def _fields(raw: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in raw.split(";") if "=" in kv)


def parse_sam_wire(text: str) -> list[dict[str, Any]]:
    """Parse SAM output_text lines into box/mask records (docs grammar)."""
    records: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        pos = 0
        pending_id: int | None = None
        pending_box: dict[str, Any] | None = None
        while pos < len(line):
            mb, mm = _BOX_RE.search(line, pos), _MASK_RE.search(line, pos)
            nxt = min([m for m in (mb, mm) if m], key=lambda m: m.start(), default=None)
            if nxt is None:
                break
            head = line[pos : nxt.start()]
            mid = _ID_RE.search(head)
            if mid:
                pending_id = int(mid.group(1))
            f = _fields(nxt.group(1))
            if nxt.re is _BOX_RE:
                pending_box = {
                    "object_id": pending_id,
                    "box": [
                        int(f["x1"]),
                        int(f["y1"]),
                        int(f["x2"]) - int(f["x1"]),
                        int(f["y2"]) - int(f["y1"]),
                    ],
                    "confidence": float(f["c"]) if "c" in f else None,
                }
            else:
                h_s, w_s, payload = f["data"].split(",", 2)
                rec = dict(pending_box or {"object_id": pending_id, "box": [], "confidence": None})
                rec.update(
                    {
                        "mask_h": int(h_s),
                        "mask_w": int(w_s),
                        "mask_enc": payload[:1],
                        "mask_payload": payload,
                        "confidence": float(f["c"]) if "c" in f else rec.get("confidence"),
                    }
                )
                records.append(rec)
                pending_box = None
            pos = nxt.end()
    return records


def decode_mask(record: dict[str, Any]) -> np.ndarray | None:
    """Decode one mask record with the official parser, or None."""
    if not record.get("mask_payload"):
        return None
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError:
        return None
    enc = "one_bit" if record.get("mask_enc") == "!" else "lossless"
    mask = SegmentationMask(
        encoding=enc,
        payload=str(record["mask_payload"]),
        width=int(record["mask_w"]),
        height=int(record["mask_h"]),
    )
    try:
        raw = decode_mask_to_raster(mask)
    except Exception:  # noqa: BLE001 - one bad mask must not kill the run
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(int(record["mask_h"]), int(record["mask_w"]))


def _mask_at_box(
    record: dict[str, Any], image_shape: tuple[int, int]
) -> np.ndarray:
    """Full-image bool mask for one record (decoded mask or its box)."""
    out = np.zeros(image_shape, dtype=bool)
    box = [int(v) for v in record.get("box") or []]
    if len(box) != 4:
        return out
    x, y, w, h = box
    rows, cols = image_shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, cols), min(y + h, rows)
    if x1 <= x0 or y1 <= y0:
        return out
    raster = decode_mask(record)
    if raster is not None:
        import cv2

        if raster.shape != (h, w):
            raster = cv2.resize(raster, (w, h), interpolation=cv2.INTER_NEAREST)
        out[y0:y1, x0:x1] = raster[y0 - y : y1 - y, x0 - x : x1 - x] > 0
    else:
        out[y0:y1, x0:x1] = True
    return out


def _objects_from_result(
    result: Any, image_shape: tuple[int, int], labels: list[str], threshold: float
) -> list[SegmentedObject]:
    data: dict[str, Any] = dict(getattr(result, "data", {}) or {})
    records: list[dict[str, Any]] = []
    if data.get("raw"):
        records = parse_sam_wire(str(data["raw"]))
    records += [dict(r) for r in data.get("records", []) if isinstance(r, dict)]
    objects: list[SegmentedObject] = []
    if records:
        for i, rec in enumerate(records):
            conf = rec.get("confidence")
            conf_f = float(conf) if conf is not None else 1.0
            if conf_f < threshold or not rec.get("box"):
                continue
            label = labels[i] if i < len(labels) else str(data.get("concept", "object"))
            objects.append(
                SegmentedObject(
                    label=label,
                    confidence=conf_f,
                    mask=_mask_at_box(rec, image_shape),
                    box=[int(v) for v in rec["box"]],
                )
            )
        return objects
    for i, seg in enumerate(data.get("segments", [])):
        if not isinstance(seg, dict):
            continue
        conf_f = float(seg.get("confidence", 1.0))
        if conf_f < threshold:
            continue
        b = seg.get("box") or {}
        if isinstance(b, dict):
            box = [int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("w", 0)), int(b.get("h", 0))]
        else:
            box = [int(v) for v in b]  # type: ignore[arg-type]
        label = str(seg.get("label") or (labels[i] if i < len(labels) else "object"))
        rec = {"box": box, "confidence": conf_f}
        # Keep any inline mask payload a fixture/transport may provide.
        for key in ("mask_payload", "mask_enc", "mask_w", "mask_h"):
            if key in seg:
                rec[key] = seg[key]
        objects.append(
            SegmentedObject(
                label=label, confidence=conf_f, mask=_mask_at_box(rec, image_shape), box=box
            )
        )
    return objects


def segment_boxes(
    client: Any,
    image_path: str | Path,
    boxes: list[list[int]],
    *,
    labels: list[str] | None = None,
    image_shape: tuple[int, int],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SegmentationResult:
    """SAM box prompts seeded by Spark boxes (the P01 main path)."""
    if not boxes:
        return SegmentationResult()
    if hasattr(client, "segment_image_boxes"):
        result = client.segment_image_boxes(image_path, boxes)
    elif hasattr(client, "segment_boxes"):
        result = client.segment_boxes(image_path, boxes)
    else:
        raise ValueError("client does not support SAM box prompts")
    return SegmentationResult(
        objects=_objects_from_result(
            result, image_shape, labels or ["door"] * len(boxes), threshold
        ),
        cost_usd=float(getattr(result, "cost_usd", 0.0) or 0.0),
        cached=bool(getattr(result, "cached", False)),
        model=str(getattr(result, "model", "") or ""),
    )


def segment_concepts(
    client: Any,
    image_path: str | Path,
    concepts: tuple[str, ...] | list[str] = PLAN_CONCEPTS,
    *,
    image_shape: tuple[int, int],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> SegmentationResult:
    """SAM concept prompts: one request per concept. Zero matches is valid."""
    objects: list[SegmentedObject] = []
    cost, cached, model = 0.0, True, ""
    for concept in concepts:
        result = client.segment_image(image_path, concept)
        cost += float(getattr(result, "cost_usd", 0.0) or 0.0)
        cached = cached and bool(getattr(result, "cached", False))
        model = str(getattr(result, "model", "") or model)
        for obj in _objects_from_result(result, image_shape, [concept], threshold):
            obj.label = concept
            objects.append(obj)
    if not concepts:
        cached = False
    return SegmentationResult(objects=objects, cost_usd=cost, cached=cached, model=model)


def masks_to_obstacles(
    objects: list[SegmentedObject],
    *,
    id_prefix: str = "sam",
) -> list[dict[str, Any]]:
    """Obstacle descriptors (label + confidence + mask) from SAM objects.

    Door/exit instances are openings, not obstacles, and are skipped;
    the pipeline turns the rest into Layout obstacles at grid size.
    """
    out: list[dict[str, Any]] = []
    for i, obj in enumerate(objects):
        if obj.label.lower() in DOOR_LABELS:
            continue
        out.append(
            {
                "id": f"{id_prefix}-{i + 1}",
                "label": obj.label,
                "confidence": obj.confidence,
                "box": list(obj.box),
                "mask": obj.mask,
            }
        )
    return out
