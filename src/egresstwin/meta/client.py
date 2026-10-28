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

    def _real_segment(
        self, model: str, path: Path, concept: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # Provisional transport pending verification against dev.meta.ai:
        # SAM has no OpenAI-SDK endpoint, so request structured JSON from
        # the vision model. Documented in PROGRESS.md as an open issue.
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": (
                    f"Segment every '{concept}' in this image. Reply with JSON "
                    '{"segments": [{"label": str, "confidence": float, '
                    '"box": {"x": int, "y": int, "w": int, "h": int}}]}. '
                    f"data:image/png;base64,{encoded}"
                ),
            }
        ]
        text, _usage = self._real_chat(model, messages)  # type: ignore[arg-type]
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"segments": []}
        return dict(data), {"images": 1}
