"""Generate spike assets with exact ground truth (P01).

Outputs in data/samples/:
  spike_plan_synth.png  (asset B: synthetic rectangle floor plan)
  spike_plan_hand.png   (asset A substitute: hand-drawn style, same layout)
  spike_aerial_synth.png (asset C substitute: synthetic top-down open space)
Each image gets a sibling *_gt.json (boxes, scale) and *_gtmask.png
(walkable = 255). A/C are project-generated stand-ins until Manuel
supplies the real A/C; they are labelled as such in data/ASSETS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples"
W, H = 800, 600
M_PER_PX = 0.02  # outer width 680 px = 13.6 m

# Plan geometry (x1, y1, x2, y2), shared by B and the A substitute.
OUTER = (60, 60, 740, 540)
WALL_T = 8
WALL_RECTS = [
    (60, 60, 740, 68),
    (60, 60, 68, 540),  # top, left
    (60, 532, 180, 540),
    (260, 532, 740, 540),  # bottom (gap = exit E1)
    (732, 60, 740, 420),
    (732, 480, 740, 540),  # right (gap = exit E2)
    (396, 60, 404, 200),
    (396, 260, 404, 360),  # interior vertical (gap D1)
    (400, 356, 540, 364),
    (600, 356, 740, 364),  # interior horiz. (gap D2)
]
STAIRS = (68, 384, 220, 532)
COLUMN = (240, 240, 280, 280)
# Doors: slab rectangles drawn next to each opening; GT box = gap + slab.
DOORS = [
    {"id": "D1", "kind": "door", "box": [360, 200, 44, 60]},
    {"id": "D2", "kind": "door", "box": [540, 316, 60, 48]},
    {"id": "E1", "kind": "exit", "box": [180, 452, 80, 88]},
    {"id": "E2", "kind": "exit", "box": [652, 420, 88, 60]},
]


def plan_gt() -> dict:
    return {
        "image_size": {"w": W, "h": H},
        "meters_per_pixel": M_PER_PX,
        "scale_annotation": "13.6 m (outer width)",
        "exits": [d for d in DOORS if d["kind"] == "exit"],
        "doors": DOORS,
        "walls": [{"box": [x1, y1, x2 - x1, y2 - y1]} for x1, y1, x2, y2 in WALL_RECTS],
        "stairs": [{"box": [STAIRS[0], STAIRS[1], STAIRS[2] - STAIRS[0], STAIRS[3] - STAIRS[1]]}],
        "obstacles": [
            {
                "label": "stairs",
                "box": [STAIRS[0], STAIRS[1], STAIRS[2] - STAIRS[0], STAIRS[3] - STAIRS[1]],
            },
            {
                "label": "column",
                "box": [COLUMN[0], COLUMN[1], COLUMN[2] - COLUMN[0], COLUMN[3] - COLUMN[1]],
            },
        ],
    }


def plan_walkable() -> np.ndarray:
    """Walkable = outer interior minus walls, stairs and column."""
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[OUTER[1] + WALL_T : OUTER[3] - WALL_T, OUTER[0] + WALL_T : OUTER[2] - WALL_T] = 255
    for x1, y1, x2, y2 in WALL_RECTS:
        mask[y1:y2, x1:x2] = 0
    mask[STAIRS[1] : STAIRS[3], STAIRS[0] : STAIRS[2]] = 0
    mask[COLUMN[1] : COLUMN[3], COLUMN[0] : COLUMN[2]] = 0
    return mask


def draw_plan(hand: bool) -> Image.Image:
    rng = np.random.default_rng(7 if hand else 0)
    bg = (250, 248, 240) if hand else (255, 255, 255)
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)

    def j(v: int) -> int:
        return v + (int(rng.integers(-3, 4)) if hand else 0)

    for x1, y1, x2, y2 in WALL_RECTS:
        if hand:  # outline strokes instead of filled rects
            d.line([(j(x1), j(y1)), (j(x2), j(y1))], fill=(20, 20, 20), width=4)
            d.line([(j(x1), j(y2)), (j(x2), j(y2))], fill=(20, 20, 20), width=4)
            d.line([(j(x1), j(y1)), (j(x1), j(y2))], fill=(20, 20, 20), width=4)
            d.line([(j(x2), j(y1)), (j(x2), j(y2))], fill=(20, 20, 20), width=4)
        else:
            d.rectangle([x1, y1, x2, y2], fill=(30, 30, 30))
    # Stairs: block outline + tread lines.
    d.rectangle([STAIRS[0], STAIRS[1], STAIRS[2], STAIRS[3]], outline=(30, 30, 30), width=3)
    for y in range(STAIRS[1] + 16, STAIRS[3], 16):
        d.line([(STAIRS[0], y), (STAIRS[2], y)], fill=(30, 30, 30), width=2)
    d.rectangle([COLUMN[0], COLUMN[1], COLUMN[2], COLUMN[3]], fill=(120, 120, 120))
    # Doors: light slabs (+ swing arcs in hand style).
    for door in DOORS:
        x, y, w, h = door["box"]
        d.rectangle([x, y, x + w, y + h], fill=(200, 200, 200), outline=(30, 30, 30), width=2)
        if hand:
            d.arc(
                [x - 10, y - 10, x + w + 10, y + h + 10],
                start=0,
                end=90,
                fill=(30, 30, 30),
                width=2,
            )
    d.text((186, 556), "EXIT", fill=(0, 130, 0))
    d.text((640, 444), "EXIT", fill=(0, 130, 0))
    d.text((90, 448), "STAIRS", fill=(30, 30, 30))
    # Dimension line under the plan: outer width = 13.6 m.
    d.line([(60, 572), (740, 572)], fill=(200, 0, 0), width=2)
    d.line([(60, 566), (60, 578)], fill=(200, 0, 0), width=2)
    d.line([(740, 566), (740, 578)], fill=(200, 0, 0), width=2)
    d.text((372, 578), "13.6 m", fill=(200, 0, 0))
    return img


CW, CH = 900, 650
TENTS = [(520, 120), (660, 100), (600, 220)]  # triangle apexes, size 70x56
TREES = [
    (120, 420, 30),
    (220, 480, 26),
    (420, 480, 34),
    (760, 340, 28),
    (150, 580, 24),
    (520, 560, 30),
]
CARS = [(560, 580), (620, 580), (680, 580), (740, 580)]  # 44x18 each
BUILDING = (80, 80, 380, 280)
STAGE = (600, 420, 840, 560)
FENCE = (500, 60, 860, 300)


def aerial_gt() -> dict:
    def box(x1: int, y1: int, x2: int, y2: int) -> list[int]:
        return [x1, y1, x2 - x1, y2 - y1]

    return {
        "image_size": {"w": CW, "h": CH},
        "meters_per_pixel": None,
        "scale_annotation": None,
        "exits": [],
        "doors": [],
        "building": [{"box": box(*BUILDING)}],
        "stage": [{"box": box(*STAGE)}],
        "tent": [{"box": [ax - 35, ay, 70, 56]} for ax, ay in TENTS],
        "fence": [{"box": box(*FENCE)}],
        "tree": [{"box": [cx - r, cy - r, 2 * r, 2 * r]} for cx, cy, r in TREES],
        "car": [{"box": [x, y, 44, 18]} for x, y in CARS],
        "obstacles": (
            [{"label": "building", "box": box(*BUILDING)}, {"label": "stage", "box": box(*STAGE)}]
            + [{"label": "tent", "box": [ax - 35, ay, 70, 56]} for ax, ay in TENTS]
            + [{"label": "tree", "box": [cx - r, cy - r, 2 * r, 2 * r]} for cx, cy, r in TREES]
            + [{"label": "car", "box": [x, y, 44, 18]} for x, y in CARS]
        ),
    }


def aerial_walkable() -> np.ndarray:
    """Open ground = frame minus obstacle footprints (fence lines excluded)."""
    mask = np.full((CH, CW), 255, dtype=np.uint8)
    for label, b in [(o["label"], o["box"]) for o in aerial_gt()["obstacles"]]:
        x, y, w, h = b
        if label == "tree":
            yy, xx = np.mgrid[y : y + h, x : x + w]
            r = w // 2
            mask[y : y + h, x : x + w][(xx - x - r) ** 2 + (yy - y - r) ** 2 <= r * r] = 0
        elif label == "tent":
            for row in range(h):
                half = int((row / max(h - 1, 1)) * (w // 2))
                mask[y + row, x + w // 2 - half : x + w // 2 + half + 1] = 0
        else:
            mask[y : y + h, x : x + w] = 0
    return mask


def draw_aerial() -> Image.Image:
    rng = np.random.default_rng(3)
    img = Image.new("RGB", (CW, CH), (96, 142, 72))
    d = ImageDraw.Draw(img)
    for _ in range(2600):  # grass speckle
        x, y = int(rng.integers(0, CW)), int(rng.integers(0, CH))
        c = (80 + int(rng.integers(0, 40)), 125 + int(rng.integers(0, 40)), 60)
        d.point((x, y), fill=c)
    d.rectangle([430, 0, 490, CH], fill=(178, 178, 168))  # paths
    d.rectangle([0, 300, CW, 340], fill=(178, 178, 168))
    d.rectangle([*BUILDING], fill=(150, 150, 155), outline=(70, 70, 75), width=4)
    d.line([(BUILDING[0], 180), (BUILDING[2], 180)], fill=(70, 70, 75), width=3)
    d.rectangle([*STAGE], fill=(55, 55, 65), outline=(20, 20, 25), width=4)
    for ax, ay in TENTS:
        d.polygon(
            [(ax, ay), (ax - 35, ay + 56), (ax + 35, ay + 56)],
            fill=(235, 235, 240),
            outline=(40, 60, 140),
        )
    fx1, fy1, fx2, fy2 = FENCE
    for seg in [
        ((fx1, fy1), (fx2, fy1)),
        ((fx1, fy2), (fx2, fy2)),
        ((fx1, fy1), (fx1, fy2)),
        ((fx2, fy1), (fx2, fy2)),
    ]:
        d.line([seg[0], seg[1]], fill=(120, 82, 45), width=6)
    for cx, cy, r in TREES:
        d.ellipse(
            [cx - r, cy - r, cx + r, cy + r], fill=(38, 105, 42), outline=(22, 70, 26), width=3
        )
    for i, (x, y) in enumerate(CARS):
        d.rectangle(
            [x, y, x + 44, y + 18],
            fill=[(200, 60, 50), (50, 90, 200), (230, 200, 60), (240, 240, 240)][i],
        )
    return img


def save(name: str, img: Image.Image, gt: dict, mask: np.ndarray) -> None:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    img.save(SAMPLES / name)
    Image.fromarray(mask).save(SAMPLES / name.replace(".png", "_gtmask.png"))
    (SAMPLES / name.replace(".png", "_gt.json")).write_text(
        json.dumps(gt, indent=2), encoding="utf-8"
    )
    print(f"wrote {name} (+ _gt.json, _gtmask.png)")


def main() -> None:
    save("spike_plan_synth.png", draw_plan(hand=False), plan_gt(), plan_walkable())
    save("spike_plan_hand.png", draw_plan(hand=True), plan_gt(), plan_walkable())
    save("spike_aerial_synth.png", draw_aerial(), aerial_gt(), aerial_walkable())


if __name__ == "__main__":
    main()
