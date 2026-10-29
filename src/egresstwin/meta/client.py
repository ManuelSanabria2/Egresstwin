"""Single entry point for all Meta Model API access.

Wraps the OpenAI SDK pointed at the Meta base URL with retries and
backoff, timeout, disk cache by input hash, cost logging to
runs/costs.jsonl, budget guard, and a fixture-backed mock mode.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from egresstwin.config import Settings, get_settings
from egresstwin.meta.costs import (
    CostTracker,
    estimate_sam_image_cost,
    estimate_text_cost,
)

BASE_URL = "https://api.meta.ai/v1"
_FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "meta"


@dataclass
class MetaResult:
    """Normalized result of a Meta call."""

    kind: str
    model: str
    text: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    cached: bool = False
    usage: dict[str, Any] = field(default_factory=dict)


def _hash_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class MetaClient:
    """Meta Model API client. All project API access goes through here."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sdk_client: Any = None,
        fixtures_dir: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(self.settings.cache_dir)
        self.runs_dir = Path(self.settings.runs_dir)
        self.fixtures_dir = fixtures_dir or _FIXTURES
        self._sdk = sdk_client
        self._sleep = sleep
        self.tracker = CostTracker(
            budget_usd=self.settings.meta_budget_usd,
            spent_usd=self._load_spent(),
        )

    # -- public API -----------------------------------------------------

    def chat_text(
        self, prompt: str, *, system: str | None = None, model: str | None = None
    ) -> MetaResult:
        """One text call to Muse Spark. Returns text plus charged cost."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        model = model or self.settings.meta_spark_model
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"kind": "chat_text", "model": model, "messages": messages}
        hit = self._cache_hit(payload)
        if hit is not None:
            return hit
        self._precheck_text(model, prompt)
        started = time.perf_counter()
        if self.settings.mock:
            raw = self._load_fixture("chat_text.json")
            text = str(raw.get("text", ""))
            usage = dict(raw.get("usage", {}))
        else:
            text, usage = self._with_retries(lambda: self._real_chat(model, messages))
        cost = estimate_text_cost(
            model,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            int(usage.get("cached_input_tokens", 0)),
        )
        result = MetaResult(kind="chat_text", model=model, text=text, usage=usage)
        self._settle(payload, result, cost, time.perf_counter() - started)
        return result

    def vision_structured(
        self,
        image_path: str | Path,
        prompt: str,
        schema: dict[str, Any],
        *,
        schema_name: str = "StructuredRead",
        model: str | None = None,
    ) -> MetaResult:
        """One Muse Spark vision call with strict JSON-schema output."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        model = model or self.settings.meta_spark_model
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {
            "kind": "vision_structured",
            "model": model,
            "prompt": prompt,
            "schema_name": schema_name,
            "schema": schema,
            "image_sha256": image_hash,
        }
        hit = self._cache_hit(payload)
        if hit is not None:
            return hit
        self._precheck_text(model, prompt)
        started = time.perf_counter()
        if self.settings.mock:
            raw = self._load_fixture("plan_reader.json")
            data = dict(raw.get("data", raw))
            usage = dict(raw.get("usage", {"input_tokens": 120, "output_tokens": 40}))
            text = json.dumps(data, ensure_ascii=False)
        else:
            data, text, usage = self._with_retries(
                lambda: self._real_vision(model, path, prompt, schema, schema_name)
            )
        cost = estimate_text_cost(
            model,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            int(usage.get("cached_input_tokens", 0)),
        )
        result = MetaResult(
            kind="vision_structured", model=model, text=text, data=data, usage=usage
        )
        self._settle(payload, result, cost, time.perf_counter() - started)
        return result

    def responses_turn(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
        previous_response_id: str | None = None,
        model: str | None = None,
    ) -> MetaResult:
        """One Muse Spark Responses turn with function calling.

        Follows INSTRUCCIONES section 5: tool_choice is only "auto",
        previous_response_id chains turns, and a turn may carry
        several function_call items. Cost is settled through the
        usual cache/budget/log path (payloads with tool outputs are
        not disk-cached: they are conversational state).
        """
        model = model or self.settings.meta_spark_model
        self._precheck_text(model, instructions)
        started = time.perf_counter()
        if self.settings.mock:
            raw = self._load_fixture("agent_turn.json")
            data = dict(raw.get("data", raw))
            usage = dict(raw.get("usage", {"input_tokens": 120, "output_tokens": 40}))
        else:
            data, usage = self._with_retries(
                lambda: self._real_responses_turn(
                    model, input_items, tools, instructions, previous_response_id
                )
            )
        cost = estimate_text_cost(
            model,
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
            int(usage.get("cached_input_tokens", 0)),
        )
        result = MetaResult(
            kind="responses_turn", model=model,
            text=data.get("text"), data=data, usage=usage,
        )
        # Conversational turns are charged and logged but not cached.
        self.tracker.charge(cost)
        result.cost_usd = cost
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": result.kind, "model": model,
            "cost_usd": round(cost, 8),
            "spent_usd": round(self.tracker.spent_usd, 8),
            "elapsed_s": round(time.perf_counter() - started, 4),
            "mock": bool(self.settings.mock), "usage": result.usage,
        }
        with (self.runs_dir / "costs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result

    def segment_image_boxes(
        self,
        image_path: str | Path,
        boxes: list[list[int]],
        *,
        model: str | None = None,
    ) -> MetaResult:
        """One SAM 3.1 box-prompt request seeded by pixel boxes [x, y, w, h]."""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        if not boxes:
            return MetaResult(
                kind="segment_image_boxes",
                model=model or self.settings.meta_sam_model,
                data={"segments": [], "records": [], "raw": ""},
            )
        model = model or self.settings.meta_sam_model
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {
            "kind": "segment_image_boxes",
            "model": model,
            "boxes": boxes,
            "image_sha256": image_hash,
        }
        hit = self._cache_hit(payload)
        if hit is not None:
            return hit
        self.tracker.check(estimate_sam_image_cost(1))
        started = time.perf_counter()
        if self.settings.mock:
            data: dict[str, Any] = {
                "segments": [
                    {
                        "label": "door",
                        "confidence": 0.97,
                        "box": {"x": int(b[0]), "y": int(b[1]), "w": int(b[2]), "h": int(b[3])},
                    }
                    for b in boxes
                ],
                "records": [],
                "raw": "",
            }
            usage = {"images": 1}
        else:
            data, usage = self._with_retries(lambda: self._real_sam_boxes(model, path, boxes))
        cost = estimate_sam_image_cost(int(usage.get("images", 1)))
        result = MetaResult(kind="segment_image_boxes", model=model, data=data, usage=usage)
        self._settle(payload, result, cost, time.perf_counter() - started)
        return result

    def segment_image(
        self, image_path: str | Path, concept: str, *, model: str | None = None
    ) -> MetaResult:
        """One SAM segmentation request for a single concept (e.g. 'door')."""
        concept = concept.strip()
        if not concept:
            raise ValueError("concept must be a short noun phrase, not empty")
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {path}")
        model = model or self.settings.meta_sam_model
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = {
            "kind": "segment_image",
            "model": model,
            "concept": concept,
            "image_sha256": image_hash,
        }
        hit = self._cache_hit(payload)
        if hit is not None:
            return hit
        self.tracker.check(estimate_sam_image_cost(1))
        started = time.perf_counter()
        if self.settings.mock:
            raw = self._load_fixture(f"sam_{concept}.json", fallback="sam_door.json")
            data: dict[str, Any] = dict(raw)
            usage = dict(raw.get("usage", {"images": 1}))
        else:
            data, usage = self._with_retries(
                lambda: self._real_segment(model, path, concept)
            )
        data["segments"] = [
            seg
            for seg in data.get("segments", [])
            if float(seg.get("confidence", 1.0)) >= self.settings.sam_confidence_threshold
        ]
        data.setdefault("concept", concept)
        cost = estimate_sam_image_cost(int(usage.get("images", 1)))
        result = MetaResult(
            kind="segment_image", model=model, data=data, usage=usage
        )
        self._settle(payload, result, cost, time.perf_counter() - started)
        return result

    # -- internals --------------------------------------------------------

    def _load_spent(self) -> float:
        log = self.runs_dir / "costs.jsonl"
        if not log.is_file():
            return 0.0
        total = 0.0
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line).get("cost_usd", 0.0))
        return total

    def _precheck_text(self, model: str, prompt: str) -> None:
        rough_input = max(1, len(prompt) // 4)
        self.tracker.check(estimate_text_cost(model, rough_input, 1024))

    def _cache_hit(self, payload: dict[str, Any]) -> MetaResult | None:
        file = self.cache_dir / f"{_hash_payload(payload)}.json"
        if not file.is_file():
            return None
        raw = json.loads(file.read_text(encoding="utf-8"))
        return MetaResult(
            kind=str(raw["kind"]),
            model=str(raw["model"]),
            text=raw.get("text"),
            data=dict(raw.get("data", {})),
            cost_usd=0.0,
            cached=True,
            usage=dict(raw.get("usage", {})),
        )

    def _settle(
        self, payload: dict[str, Any], result: MetaResult, cost: float, elapsed: float
    ) -> None:
        self.tracker.charge(cost)
        result.cost_usd = cost
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{_hash_payload(payload)}.json"
        cache_file.write_text(
            json.dumps(
                {
                    "kind": result.kind,
                    "model": result.model,
                    "text": result.text,
                    "data": result.data,
                    "usage": result.usage,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "kind": result.kind,
            "model": result.model,
            "cost_usd": round(cost, 8),
            "spent_usd": round(self.tracker.spent_usd, 8),
            "elapsed_s": round(elapsed, 4),
            "mock": bool(self.settings.mock),
            "usage": result.usage,
        }
        with (self.runs_dir / "costs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load_fixture(self, name: str, fallback: str | None = None) -> dict[str, Any]:
        file = self.fixtures_dir / name
        if not file.is_file() and fallback:
            file = self.fixtures_dir / fallback
        if not file.is_file():
            return {"text": "", "segments": [], "usage": {}}
        return dict(json.loads(file.read_text(encoding="utf-8")))

    def _sdk_client(self) -> Any:
        if self._sdk is None:
            key = self.settings.model_api_key
            if not key:
                raise ValueError("MODEL_API_KEY is not set (real mode needs it)")
            from openai import OpenAI

            options: dict[str, Any] = {
                "api_key": key,
                "base_url": BASE_URL,
                "timeout": self.settings.meta_timeout_s,
                "max_retries": 0,
            }
            self._sdk = OpenAI(**options)
        return self._sdk

    def _with_retries(self, fn: Callable[[], Any]) -> Any:
        attempts = self.settings.meta_max_retries + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - re-raised after backoff
                last_exc = exc
                if attempt < attempts - 1:
                    self._sleep(0.5 * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def _real_chat(
        self, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, Any]]:
        response = self._sdk_client().chat.completions.create(
            model=model, messages=messages
        )
        text = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage: dict[str, Any] = {
            "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "cached_input_tokens": 0,
        }
        details = getattr(usage_obj, "prompt_tokens_details", None)
        if details is not None:
            usage["cached_input_tokens"] = int(
                getattr(details, "cached_tokens", 0) or 0
            )
        return text, usage

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def _real_vision(
        self,
        model: str,
        path: Path,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        """Responses API vision call (documented transport, spike P01)."""
        resp = self._sdk_client().responses.create(
            model=model,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": self._data_url(path)},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
        )
        text = str(getattr(resp, "output_text", "") or "")
        data = json.loads(text)
        usage_obj = getattr(resp, "usage", None)
        usage = {
            "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
            "cached_input_tokens": 0,
        }
        return dict(data), text, usage

    def _real_responses_turn(
        self,
        model: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        instructions: str,
        previous_response_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "tools": tools,
            "tool_choice": "auto",
            "input": input_items,
        }
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        resp = self._sdk_client().responses.create(**kwargs)
        calls: list[dict[str, Any]] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                calls.append(
                    {
                        "name": str(getattr(item, "name", "")),
                        "arguments": str(getattr(item, "arguments", "") or ""),
                        "call_id": str(
                            getattr(item, "call_id", "") or getattr(item, "id", "")
                        ),
                    }
                )
        usage_obj = getattr(resp, "usage", None)
        usage = {
            "input_tokens": int(getattr(usage_obj, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "output_tokens", 0) or 0),
            "cached_input_tokens": 0,
        }
        data = {
            "response_id": str(getattr(resp, "id", "") or ""),
            "function_calls": calls,
            "text": str(getattr(resp, "output_text", "") or ""),
        }
        return data, usage

    def _real_sam_stream(self, model: str, path: Path, prompt_text: str) -> str:
        """SAM 3.1 Responses streaming call; returns raw wire text."""
        stream = self._sdk_client().responses.create(
            model=model,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": self._data_url(path)},
                    ],
                }
            ],
            stream=True,
            metadata={"include_confidence": "true"},
        )
        return "".join(
            str(getattr(ev, "delta", "") or "")
            for ev in stream
            if getattr(ev, "type", "") == "response.output_text.delta"
        )

    def _real_segment(
        self, model: str, path: Path, concept: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """One concept per request (INSTRUCCIONES section 5). Zero is valid."""
        raw = self._real_sam_stream(model, path, concept)
        return {"raw": raw, "segments": [], "concept": concept}, {"images": 1}

    def _real_sam_boxes(
        self, model: str, path: Path, boxes: list[list[int]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
        parts = []
        for i, (x, y, bw, bh) in enumerate(boxes, start=1):
            parts.append(
                f"{i}<|box;x1={int(x)};y1={int(y)};x2={int(x + bw - 1)};"
                f"y2={int(y + bh - 1)};w={w};h={h}|>"
            )
        raw = self._real_sam_stream(model, path, "<0f>" + ",".join(parts))
        return {"raw": raw, "segments": [], "records": []}, {"images": 1}
