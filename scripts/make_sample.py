"""Generate the synthetic sample floor-plan image in data/samples/."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "door_sample.png"


def main() -> None:
    """Draw a tiny synthetic plan: room outline plus a door opening."""
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 300, 220], outline="black", width=4)
    draw.rectangle([120, 18, 180, 26], fill="white")  # door opening
    draw.rectangle([120, 20, 180, 90], outline="black", width=2)  # door slab
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
