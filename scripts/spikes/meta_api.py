"""Spike-only Meta Model API helpers (P01). Not production code.

The production client (src/egresstwin/meta/client.py) only does text chat
and a provisional SAM transport, so the documented wire formats are
implemented here for the spike:
  * Muse Spark 1.3 vision + structured output via Responses
    (text.format json_schema; grounding grid 0-1000, converted to px).
  * SAM 3.1 via Responses streaming: one concept per request, or one
    box-prompt request; metadata include_confidence="true" (string).
    Output wire grammar parsed by regex; masks decoded with the
    official meta-sam-parser when installed.
Every call is cached under runs/spikes/cache/ and charged (estimate)
to runs/spikes/costs_spike.jsonl. Set SPIKE_NO_CACHE=1 to force calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from egresstwin.meta.costs import estimate_sam_image_cost, estimate_text_cost  # noqa: E402

CACHE = ROOT / "runs" / "spikes" / "cache"
LEDGER = ROOT / "runs" / "spikes" / "costs_spike.jsonl"
SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"

SPARK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "image_type": {"type": "string", "enum": ["floor_plan", "aerial_photo", "other"]},
        "exits_or_doors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "x_min", "y_min", "x_max", "y_max", "confidence"],
            },
        },
        "dimension_annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "x_min": {"type": "integer"},
                    "y_min": {"type": "integer"},
                    "x_max": {"type": "integer"},
                    "y_max": {"type": "integer"},
                },
                "required": ["text", "x_min", "y_min", "x_max", "y_max"],
            },
        },
        "scale_annotation": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": [
        "image_type",
        "exits_or_doors",
        "dimension_annotations",
        "scale_annotation",
        "confidence",
    ],
}

SPARK_PROMPT = (
    "You are an object grounding expert. Analyse this image for an evacuation "
    "digital twin. Classify image_type as floor_plan, aerial_photo or other. "
    "List every exit, doorway or door in exits_or_doors; do not miss any. "
    "List every visible dimension/scale annotation (e.g. '13.6 m') in "
    "dimension_annotations with its exact text, and copy the one that gives "
    "the plan scale into scale_annotation (null if none). All coordinates "
    "are integers on a normalized 0-1000 grid, (0,0) top-left. "
    "confidence is your overall confidence from 0 to 1."
)


def _client() -> Any:
    from openai import OpenAI

    key = os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set; real spike calls need it")
    return OpenAI(base_url="https://api.meta.ai/v1", api_key=key, timeout=120.0, max_retries=0)


def _data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _cached(key_payload: dict[str, Any]) -> tuple[Path, dict[str, Any] | None]:
    raw = json.dumps(key_payload, sort_keys=True).encode()
    file = CACHE / (hashlib.sha256(raw).hexdigest() + ".json")
    if file.is_file() and not os.environ.get("SPIKE_NO_CACHE"):
        return file, json.loads(file.read_text(encoding="utf-8"))
    return file, None


def _charge(kind: str, model: str, cost: float, detail: dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "kind": kind,
                    "model": model,
                    "cost_usd": round(cost, 8),
                    **detail,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def spark_read_image(path: Path) -> dict[str, Any]:
    """Task 1: Spark vision with structured output. Boxes returned in px."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    key = {
        "kind": "spark",
        "model": SPARK_MODEL,
        "image": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    file, hit = _cached(key)
    if hit is not None:
        return {**hit, "cached": True, "cost_usd": 0.0}
    resp = _client().responses.create(
        model=SPARK_MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": SPARK_PROMPT},
                    {"type": "input_image", "image_url": _data_url(path)},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "PerceptionRead",
                "strict": True,
                "schema": SPARK_SCHEMA,
            }
        },
    )
    data = json.loads(resp.output_text)
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    cost = estimate_text_cost(SPARK_MODEL, usage["input_tokens"], usage["output_tokens"])
    for item in data.get("exits_or_doors", []):  # normalized 0-1000 -> pixels
        item["box_px"] = [
            round(item["x_min"] * w / 1000),
            round(item["y_min"] * h / 1000),
            round((item["x_max"] - item["x_min"]) * w / 1000),
            round((item["y_max"] - item["y_min"]) * h / 1000),
        ]
    out = {"model": SPARK_MODEL, "image_size": {"w": w, "h": h}, "usage": usage, **data}
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    _charge("spark_vision", SPARK_MODEL, cost, {"image": path.name, "usage": usage})
    return {**out, "cached": False, "cost_usd": cost}


_BOX_RE = re.compile(r"<\|box;([^|]*)\|>")
_MASK_RE = re.compile(r"<\|mask;([^|]*)\|>")
_ID_RE = re.compile(r"(\d+)\s*$")


def _fields(raw: str) -> dict[str, str]:
    return dict(kv.split("=", 1) for kv in raw.split(";") if "=" in kv)


def parse_sam_wire(text: str) -> list[dict[str, Any]]:
    """Parse SAM output_text lines into box/mask records (docs grammar)."""
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
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


def decode_mask(rec: dict[str, Any]) -> Any:
    """Decode one mask record to a 0/1 numpy raster, or None if unavailable."""
    try:
        from meta_sam_parser import SegmentationMask, decode_mask_to_raster
    except ImportError:
        return None
    enc = "one_bit" if rec["mask_enc"] == "!" else "lossless"
    mask = SegmentationMask(
        encoding=enc, payload=rec["mask_payload"], width=rec["mask_w"], height=rec["mask_h"]
    )
    import numpy as np

    try:
        raw = decode_mask_to_raster(mask)
    except Exception:  # noqa: BLE001 - one bad mask must not kill the spike
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(rec["mask_h"], rec["mask_w"])


def build_box_prompt(boxes: list[list[int]], w: int, h: int) -> str:
    """Box-prompt input_text per docs grammar (x2/y2 inclusive)."""
    parts = []
    for i, (x, y, bw, bh) in enumerate(boxes, start=1):
        parts.append(f"{i}<|box;x1={x};y1={y};x2={x + bw - 1};y2={y + bh - 1};w={w};h={h}|>")
    return "<0f>" + ",".join(parts)


def sam_segment(path: Path, prompt_text: str) -> dict[str, Any]:
    """Tasks 2/3: one SAM 3.1 request (concept phrase OR box prompts)."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    key = {
        "kind": "sam",
        "model": SAM_MODEL,
        "prompt": prompt_text,
        "image": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    file, hit = _cached(key)
    if hit is not None:
        hit["records"] = parse_sam_wire(hit.get("raw", ""))  # re-parse with current parser
        return {**hit, "cached": True, "cost_usd": 0.0}
    stream = _client().responses.create(
        model=SAM_MODEL,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": _data_url(path)},
                ],
            }
        ],
        stream=True,
        metadata={"include_confidence": "true"},
    )
    raw = "".join(ev.delta for ev in stream if ev.type == "response.output_text.delta")
    records = parse_sam_wire(raw)
    cost = estimate_sam_image_cost(1)
    out = {
        "model": SAM_MODEL,
        "prompt": prompt_text,
        "image_size": {"w": w, "h": h},
        "raw": raw,
        "records": records,
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    _charge("sam_segment", SAM_MODEL, cost, {"image": path.name, "prompt": prompt_text[:60]})
    return {**out, "cached": False, "cost_usd": cost}
