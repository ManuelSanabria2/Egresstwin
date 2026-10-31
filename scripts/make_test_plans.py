"""Generate varied synthetic test floor plans in data/samples/test_plans/.

Same visual language as the spike assets (dark walls, grey door slabs,
green EXIT labels, red dimension line) so the perception pipeline reads
them the same way. All project-generated, MIT, registered in
data/ASSETS.md. No network, no API calls.

    python scripts/make_test_plans.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "samples" / "test_plans"
WALL = (30, 30, 30)
DOOR = (200, 200, 200)
GREEN = (0, 130, 0)
RED = (200, 0, 0)


def base(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w, h), "white")
    return img, ImageDraw.Draw(img)


def wall(d: ImageDraw.ImageDraw, box: list[int]) -> None:
    d.rectangle(box, fill=WALL)


def door(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    d.rectangle([x, y, x + w, y + h], fill=DOOR, outline=WALL, width=2)


def dimension(d: ImageDraw.ImageDraw, x1: int, x2: int, y: int, label: str) -> None:
    d.line([(x1, y), (x2, y)], fill=RED, width=2)
    d.line([(x1, y - 6), (x1, y + 6)], fill=RED, width=2)
    d.line([(x2, y - 6), (x2, y + 6)], fill=RED, width=2)
    d.text(((x1 + x2) // 2 - 20, y + 6), label, fill=RED)


def office() -> None:
    img, d = base(900, 650)
    wall(d, [60, 60, 840, 68])
    wall(d, [60, 60, 68, 520])
    wall(d, [60, 512, 340, 520])
    wall(d, [420, 512, 840, 520])
    wall(d, [832, 60, 840, 300])
    wall(d, [832, 380, 840, 520])
    wall(d, [446, 60, 454, 240])
    wall(d, [446, 320, 454, 520])
    wall(d, [454, 286, 620, 294])
    wall(d, [700, 286, 840, 294])
    wall(d, [240, 68, 248, 220])
    wall(d, [240, 300, 248, 512])
    for x in (140, 540, 640):
        d.rectangle([x, 140, x + 90, 200], fill=(170, 190, 220), outline=WALL, width=2)
    d.rectangle([120, 380, 200, 460], fill=(120, 120, 120))
    door(d, 340, 452, 80, 68)
    door(d, 752, 300, 88, 80)
    door(d, 360, 240, 86, 80)
    door(d, 620, 246, 80, 80)
    door(d, 160, 220, 80, 80)
    d.text((348, 556), "EXIT", fill=GREEN)
    d.text((760, 330), "EXIT", fill=GREEN)
    d.text((70, 76), "OPEN OFFICE", fill=WALL)
    d.text((470, 76), "MEETING", fill=WALL)
    dimension(d, 60, 840, 580, "15.6 m")
    img.save(OUT / "test_office.png")


def classroom() -> None:
    img, d = base(820, 600)
    wall(d, [60, 60, 760, 68])
    wall(d, [60, 60, 68, 500])
    wall(d, [60, 492, 300, 500])
    wall(d, [380, 492, 760, 500])
    wall(d, [752, 60, 760, 240])
    wall(d, [752, 320, 760, 500])
    for row in range(3):
        for col in range(4):
            x, y = 120 + col * 150, 130 + row * 110
            d.rectangle([x, y, x + 100, y + 60], fill=(210, 180, 140), outline=WALL, width=2)
    d.rectangle([300, 76, 520, 110], fill=(60, 110, 70))
    d.text((330, 86), "BOARD", fill=(255, 255, 255))
    door(d, 300, 432, 80, 68)
    door(d, 672, 240, 88, 80)
    d.text((308, 530), "EXIT", fill=GREEN)
    d.text((676, 270), "EXIT", fill=GREEN)
    dimension(d, 60, 760, 540, "14.0 m")
    img.save(OUT / "test_classroom.png")


def restaurant() -> None:
    img, d = base(900, 650)
    wall(d, [60, 60, 840, 68])
    wall(d, [60, 60, 68, 540])
    wall(d, [60, 532, 260, 540])
    wall(d, [340, 532, 840, 540])
    wall(d, [832, 60, 840, 340])
    wall(d, [832, 420, 840, 540])
    wall(d, [560, 68, 568, 300])
    wall(d, [560, 380, 568, 532])
    wall(d, [568, 296, 700, 304])
    wall(d, [568, 60, 840, 68])
    for row in range(3):
        for col in range(3):
            x, y = 110 + col * 140, 120 + row * 120
            d.ellipse([x, y, x + 70, y + 70], fill=(190, 160, 120), outline=WALL, width=2)
    d.rectangle([600, 120, 800, 260], fill=(150, 150, 150), outline=WALL, width=2)
    d.text((640, 180), "KITCHEN", fill=WALL)
    d.rectangle([100, 420, 480, 470], fill=(170, 120, 80), outline=WALL, width=2)
    d.text((240, 435), "BAR", fill=WALL)
    door(d, 260, 472, 80, 68)
    door(d, 752, 340, 88, 80)
    door(d, 480, 300, 80, 80)
    d.text((268, 566), "EXIT", fill=GREEN)
    d.text((756, 372), "EXIT", fill=GREEN)
    dimension(d, 60, 840, 590, "15.6 m")
    img.save(OUT / "test_restaurant.png")


def shop() -> None:
    img, d = base(900, 650)
    wall(d, [60, 60, 840, 68])
    wall(d, [60, 60, 68, 540])
    wall(d, [60, 532, 380, 540])
    wall(d, [460, 532, 840, 540])
    wall(d, [832, 60, 840, 540])
    for row in range(4):
        y = 120 + row * 100
        wall(d, [120, y, 500, y + 14])
    wall(d, [560, y, 780, y + 14])
    d.rectangle([640, 460, 800, 520], outline=WALL, width=3)
    d.text((660, 482), "STOCK", fill=WALL)
    d.rectangle([100, 460, 260, 500], fill=(170, 190, 220), outline=WALL, width=2)
    d.text((130, 472), "CHECKOUT", fill=WALL)
    door(d, 380, 472, 80, 68)
    d.text((388, 566), "EXIT", fill=GREEN)
    dimension(d, 60, 840, 590, "15.6 m")
    img.save(OUT / "test_shop.png")


def apartment() -> None:
    img, d = base(760, 620)
    wall(d, [80, 60, 680, 68])
    wall(d, [80, 60, 88, 520])
    wall(d, [80, 512, 300, 520])
    wall(d, [380, 512, 680, 520])
    wall(d, [672, 60, 680, 280])
    wall(d, [672, 360, 680, 520])
    wall(d, [376, 68, 384, 260])
    wall(d, [376, 340, 384, 512])
    wall(d, [88, 286, 220, 294])
    wall(d, [300, 286, 376, 294])
    wall(d, [384, 286, 540, 294])
    wall(d, [620, 286, 672, 294])
    d.rectangle([110, 90, 290, 190], fill=(180, 200, 230), outline=WALL, width=2)
    d.text((150, 130), "BEDROOM", fill=WALL)
    d.rectangle([430, 90, 640, 190], fill=(230, 200, 170), outline=WALL, width=2)
    d.text((480, 130), "LIVING", fill=WALL)
    d.rectangle([110, 330, 300, 470], fill=(190, 220, 190), outline=WALL, width=2)
    d.text((160, 390), "KITCHEN", fill=WALL)
    d.rectangle([450, 330, 640, 420], fill=(200, 200, 200), outline=WALL, width=2)
    d.text((500, 365), "BATH", fill=WALL)
    door(d, 300, 452, 80, 68)
    door(d, 592, 280, 88, 80)
    door(d, 220, 246, 80, 80)
    door(d, 540, 246, 80, 80)
    d.text((308, 548), "EXIT", fill=GREEN)
    d.text((596, 306), "EXIT", fill=GREEN)
    dimension(d, 80, 680, 560, "12.0 m")
    img.save(OUT / "test_apartment.png")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (office, classroom, restaurant, shop, apartment):
        make()
        print(OUT / f"test_{make.__name__}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
