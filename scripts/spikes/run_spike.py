"""Run the P01 perception spike end to end and write results.

Usage (from repo root, real mode):
  .venv/Scripts/python.exe scripts/spikes/generate_assets.py
  .venv/Scripts/python.exe scripts/spikes/run_spike.py
Outputs: runs/spikes/results.json, overlays in docs/figures/spike/.
API responses are cached in runs/spikes/cache/ (re-runs cost 0).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs" / "spikes" / "mplconfig")
)

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from meta_api import build_box_prompt, decode_mask, sam_segment, spark_read_image  # noqa: E402
from opencv_walls import extract_walls, mask_iou  # noqa: E402

SAMPLES = ROOT / "data" / "samples"
FIGURES = ROOT / "docs" / "figures" / "spike"
RESULTS = ROOT / "runs" / "spikes" / "results.json"
MATCH_IOU = 0.30

IMAGES: list[dict[str, Any]] = [
    {
        "id": "A_hand_substitute",
        "file": "spike_plan_hand.png",
        "kind": "plan",
        "concepts": ["door", "doorway", "wall", "stairs"],
    },
    {
        "id": "B_synth_plan",
        "file": "spike_plan_synth.png",
        "kind": "plan",
        "concepts": ["door", "doorway", "wall", "stairs"],
    },
    {
        "id": "C_aerial_substitute",
        "file": "spike_aerial_synth.png",
        "kind": "aerial",
        "concepts": ["building", "tent", "stage", "fence", "tree", "car"],
    },
]
CONCEPT_GT = {"door": "doors", "doorway": "doors", "wall": "walls", "stairs": "stairs"}


def box_iou(a: list[int], b: list[int]) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - iw * ih
    return iw * ih / union if union else 0.0


def match_boxes(
    pred: list[list[int]], gt: list[list[int]], thr: float = MATCH_IOU
) -> dict[str, Any]:
    """Greedy one-to-one matching by box IoU."""
    pairs = sorted(
        ((box_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt)), reverse=True
    )
    used_p, used_g, ious = set(), set(), []
    for v, i, j in pairs:
        if v >= thr and i not in used_p and j not in used_g:
            used_p.add(i), used_g.add(j), ious.append(v)
    return {
        "tp": len(ious),
        "fp": len(pred) - len(ious),
        "fn": len(gt) - len(ious),
        "recall": len(ious) / len(gt) if gt else None,
        "precision": len(ious) / len(pred) if pred else None,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def boxes_of(items: list[dict[str, Any]]) -> list[list[int]]:
    return [list(map(int, it["box"])) for it in items]


def raster_from_records(records: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    """Union of decoded SAM masks placed at their boxes (fallback: boxes)."""
    out = np.zeros(shape, dtype=np.uint8)
    for rec in records:
        if not rec.get("box"):
            continue
        x, y, w, h = [int(v) for v in rec["box"]]
        raster = decode_mask(rec) if rec.get("mask_payload") else None
        if raster is not None:
            if raster.shape != (h, w):
                raster = cv2.resize(raster, (w, h), interpolation=cv2.INTER_NEAREST)
            out[y : y + h, x : x + w] = np.maximum(out[y : y + h, x : x + w], raster)
        else:
            out[y : y + h, x : x + w] = 1
    return out


def boxes_mask(boxes: list[list[int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    for x, y, w, h in boxes:
        out[y : y + h, x : x + w] = 1
    return out


def save_overlay(
    img_id: str,
    image: np.ndarray,
    gt: dict[str, Any],
    spark: dict[str, Any],
    sam_by_concept: dict[str, list[dict[str, Any]]],
) -> Path:
    """One overlay figure: GT green, Spark blue, SAM orange + mask tint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.imshow(image)
    for key in ("exits", "doors", "building", "tent", "stage", "tree", "car", "stairs"):
        for it in gt.get(key, []) if isinstance(gt.get(key), list) else []:
            x, y, w, h = it["box"]
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="lime", lw=1.6))
    for it in spark.get("exits_or_doors", []):
        x, y, w, h = it["box_px"]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="dodgerblue", lw=1.6))
        ax.text(
            x, max(y - 4, 2), f"spark {it.get('confidence', 0):.2f}", color="dodgerblue", fontsize=7
        )
    tint = np.zeros((*image.shape[:2], 4), dtype=np.float32)
    for concept, records in sam_by_concept.items():
        for rec in records:
            if not rec.get("box"):
                continue
            x, y, w, h = [int(v) for v in rec["box"]]
            ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor="orange", lw=1.4))
            ax.text(
                x,
                y + h + 12,
                f"sam:{concept} {rec.get('confidence') or 0:.2f}",
                color="darkorange",
                fontsize=7,
            )
            raster = decode_mask(rec) if rec.get("mask_payload") else None
            if raster is not None:
                if raster.shape != (h, w):
                    raster = cv2.resize(raster, (w, h), interpolation=cv2.INTER_NEAREST)
                region = tint[y : y + h, x : x + w]
                region[raster > 0] = (1.0, 0.55, 0.0, 0.35)
    ax.imshow(tint)
    ax.set_title(f"{img_id}: GT lime | Spark blue | SAM orange")
    ax.axis("off")
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{img_id}_overlay.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def save_cv_overlay(img_id: str, image_path: Path, cv_res: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].imshow(img), axes[0].set_title("input"), axes[0].axis("off")
    (
        axes[1].imshow(cv_res["walls"], cmap="gray"),
        axes[1].set_title("OpenCV walls"),
        axes[1].axis("off"),
    )
    (
        axes[2].imshow(cv_res["walkable"], cmap="gray"),
        axes[2].set_title("OpenCV walkable"),
        axes[2].axis("off"),
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{img_id}_opencv.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def run_image(spec: dict[str, Any]) -> dict[str, Any]:
    path = SAMPLES / spec["file"]
    gt = json.loads(
        (SAMPLES / spec["file"].replace(".png", "_gt.json")).read_text(encoding="utf-8")
    )
    gt_walk = np.array(Image.open(SAMPLES / spec["file"].replace(".png", "_gtmask.png")))
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    shape = gt_walk.shape
    total_cost = 0.0
    # Task 1: Spark vision.
    spark = spark_read_image(path)
    total_cost += spark["cost_usd"]
    spark_boxes = [it["box_px"] for it in spark.get("exits_or_doors", [])]
    exits_gt = boxes_of(gt.get("exits", []))
    res: dict[str, Any] = {
        "id": spec["id"],
        "file": spec["file"],
        "kind": spec["kind"],
        "spark": {k: spark.get(k) for k in ("image_type", "scale_annotation", "confidence")},
        "spark_exits_match": match_boxes(spark_boxes, exits_gt) if exits_gt else None,
        "spark_doors_match": match_boxes(spark_boxes, boxes_of(gt.get("doors", [])))
        if gt.get("doors")
        else None,
        "spark_n_boxes": len(spark_boxes),
    }
    # Task 2: SAM concept prompts, one request per concept.
    sam: dict[str, Any] = {}
    sam_records: dict[str, list[dict[str, Any]]] = {}
    for concept in spec["concepts"]:
        out = sam_segment(path, concept)
        total_cost += out["cost_usd"]
        recs = out["records"]
        sam_records[concept] = recs
        gt_key = CONCEPT_GT.get(concept, concept)
        gt_boxes = boxes_of(gt.get(gt_key, [])) if isinstance(gt.get(gt_key), list) else []
        confs = [r["confidence"] for r in recs if r.get("confidence") is not None]
        sam[concept] = {
            "n": len(recs),
            "mean_conf": float(np.mean(confs)) if confs else None,
            "match_vs_gt": match_boxes([r["box"] for r in recs if r.get("box")], gt_boxes)
            if gt_boxes
            else None,
            "kept_at_0.3": sum(1 for c in confs if c >= 0.3),
            "kept_at_0.5": sum(1 for c in confs if c >= 0.5),
            "kept_at_0.7": sum(1 for c in confs if c >= 0.7),
        }
    res["sam_concepts"] = sam
    # Task 3: SAM box prompts seeded by Spark boxes (single request).
    if spark_boxes:
        prompt = build_box_prompt(spark_boxes, shape[1], shape[0])
        out = sam_segment(path, prompt)
        total_cost += out["cost_usd"]
        recs = out["records"]
        sam_records["box_prompt"] = recs
        res["sam_box_prompt"] = {
            "n": len(recs),
            "match_vs_exits": match_boxes([r["box"] for r in recs if r.get("box")], exits_gt)
            if exits_gt
            else None,
            "match_vs_doors": match_boxes(
                [r["box"] for r in recs if r.get("box")], boxes_of(gt.get("doors", []))
            )
            if gt.get("doors")
            else None,
        }
    else:
        res["sam_box_prompt"] = {"n": 0, "note": "skipped: Spark returned no boxes"}
    # Task 5 (exits): SAM door/doorway recall against GT exits.
    for concept in ("door", "doorway"):
        if concept in sam_records and exits_gt:
            res[f"sam_{concept}_exits_match"] = match_boxes(
                [r["box"] for r in sam_records[concept] if r.get("box")], exits_gt
            )
    # Task 4 + walkable IoU variants.
    if spec["kind"] == "plan":
        cv_res = extract_walls(path)
        wall_sam = raster_from_records(sam_records.get("wall", []), shape)
        interior = cv_res["interior"]
        wall_dil = cv2.dilate(
            cv_res["walls"], cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2
        )
        sam_dil = cv2.dilate(
            (wall_sam * 255).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )
        variants = {
            "interior_only_baseline": interior,
            "opencv": cv_res["walkable"],
            "sam_wall": cv2.bitwise_and(interior, cv2.bitwise_not(sam_dil)),
            "opencv_plus_sam": cv2.bitwise_and(
                interior, cv2.bitwise_not(cv2.bitwise_or(wall_dil, sam_dil))
            ),
        }
        res["walkable_iou"] = {k: round(mask_iou(v, gt_walk), 4) for k, v in variants.items()}
        res["opencv_gaps"] = cv_res["gaps"]
        res["opencv_gaps_vs_exits"] = match_boxes(cv_res["gaps"], exits_gt) if exits_gt else None
        save_cv_overlay(spec["id"], path, cv_res)
    else:
        obstacle_recs: list[dict[str, Any]] = []
        for c in ("building", "stage", "tent", "tree", "car"):
            obstacle_recs += sam_records.get(c, [])
        sam_walk = 1 - boxes_mask([r["box"] for r in obstacle_recs if r.get("box")], shape)
        res["walkable_iou"] = {"sam_obstacle_boxes": round(mask_iou(sam_walk * 255, gt_walk), 4)}
    res["overlay"] = str(save_overlay(spec["id"], image, gt, spark, sam_records).relative_to(ROOT))
    res["cost_usd"] = round(total_cost, 6)
    return res


def main() -> int:
    results = [run_image(spec) for spec in IMAGES]
    total = round(sum(r["cost_usd"] for r in results), 6)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {"match_iou_threshold": MATCH_IOU, "total_cost_usd": total, "images": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"total charged this run: USD {total:.4f} (cached calls cost 0)")
    for r in results:
        print(f"\n== {r['id']} ({r['file']}) ==")
        print(" spark:", r["spark"], "| exits match:", r["spark_exits_match"])
        for c, s in r["sam_concepts"].items():
            print(f" sam[{c}]: n={s['n']} mean_conf={s['mean_conf']} match={s['match_vs_gt']}")
        print(" sam_box_prompt:", r["sam_box_prompt"])
        print(" walkable_iou:", r["walkable_iou"])
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
