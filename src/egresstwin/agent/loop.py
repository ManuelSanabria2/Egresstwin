"""Muse Spark tool-calling loop over the Responses API.

One turn may carry several function_call items: all are executed and
each gets its function_call_output. Tool arguments are validated
with Pydantic before dispatch. submit_report is checked by the
deterministic verifier; a rejected report comes back as a tool
result and the agent gets up to max_corrections re-submissions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from egresstwin.agent.report import AgentReport, Objective
from egresstwin.agent.tools import TOOL_ARG_MODELS, AgentTools, tool_schemas
from egresstwin.agent.trace import TraceWriter, new_run_id
from egresstwin.agent.verifier import VerifierConfig, verify_report
from egresstwin.layout.schema import Layout
from egresstwin.parallel.store import ScenarioStore

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


@dataclass
class AgentRunResult:
    status: str  # VERIFIED | PARTIALLY_VERIFIED | FAILED
    run_id: str
    trace_path: Path
    report: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    total_cost_usd: float = 0.0
    turns: int = 0
    scenarios_run: int = 0
    reason: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


def load_system_prompt() -> str:
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "You are EgressTwin. Use tools for every number and end with submit_report."


class AgentLoop:
    """Run the optimizer loop for one layout."""

    def __init__(
        self,
        layout: Layout,
        *,
        objective: Objective = "minimize_t95",
        model_client: Any = None,
        tools: AgentTools | None = None,
        max_turns: int = 25,
        max_scenarios: int = 60,
        budget_usd: float = 1.0,
        max_corrections: int = 2,
        max_changes: int = 3,
        user_constraints: str = "",
        runs_dir: str | Path = "runs",
        run_id: str | None = None,
        seed: int = 42,
        verifier_config: VerifierConfig | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.layout = layout
        self.objective = objective
        self.max_turns = max_turns
        self.budget_usd = budget_usd
        self.max_corrections = max_corrections
        self.user_constraints = user_constraints
        self.on_event = on_event
        self.run_id = run_id or new_run_id()
        self.trace = TraceWriter(self.run_id, runs_dir)
        store = ScenarioStore(path=self.trace.run_dir / "scenarios.json")
        self.tools = tools or AgentTools(
            layout, store=store, seed=seed, max_scenarios=max_scenarios
        )
        self.verifier_config = verifier_config or VerifierConfig(max_changes=max_changes)
        if model_client is None:
            from egresstwin.meta.client import MetaClient

            model_client = MetaClient()
        self.model_client = model_client
        self.system_prompt = load_system_prompt()

    def _emit(self, event: dict[str, Any]) -> None:
        entry = self.trace.write(event)
        if self.on_event is not None:
            self.on_event(entry)

    def _dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tools = self.tools
        if name == "get_layout_summary":
            return tools.get_layout_summary()
        if name == "get_baseline":
            return tools.get_baseline(**args)
        if name == "run_scenarios":
            return tools.run_scenarios(**args)
        if name == "get_hotspots":
            return tools.get_hotspots(**args)
        if name == "stress_test":
            return tools.stress_test(**args)
        if name == "compare":
            return tools.compare(**args)
        raise ValueError(f"unknown tool {name!r}")

    def run(self) -> AgentRunResult:
        schemas = tool_schemas()
        user_text = (
            f"Objective for this run: {self.objective}. "
            f"Maximum changes per recommendation: "
            f"{self.verifier_config.max_changes}. "
            f"Additional user constraints: {self.user_constraints or 'none stated'}. "
            "Explore the layout, demonstrate any improvement with "
            "simulations and an N-1 stress test, then call submit_report."
        )
        input_items: list[dict[str, Any]] = [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": user_text}]}
        ]
        previous_id: str | None = None
        total_cost = 0.0
        turns = 0
        submit_attempts = 0
        final_report: AgentReport | None = None
        final_verification: dict[str, Any] | None = None
        status = "FAILED"
        reason = "no submit_report received"
        self._emit({"event": "run_start", "run_id": self.run_id, "objective": self.objective})

        for turn in range(1, self.max_turns + 1):
            turns = turn
            if total_cost > self.budget_usd:
                reason = f"run budget exceeded: USD {total_cost:.4f} > {self.budget_usd:.2f}"
                break
            started = time.perf_counter()
            result = self.model_client.responses_turn(
                input_items=input_items,
                tools=schemas,
                instructions=self.system_prompt,
                previous_response_id=previous_id,
            )
            elapsed = time.perf_counter() - started
            total_cost += float(getattr(result, "cost_usd", 0.0) or 0.0)
            data = dict(getattr(result, "data", {}) or {})
            previous_id = str(data.get("response_id") or previous_id or "") or previous_id
            calls = list(data.get("function_calls", []))
            self._emit(
                {
                    "event": "turn", "turn": turn, "response_id": previous_id,
                    "model": getattr(result, "model", ""),
                    "usage": dict(getattr(result, "usage", {}) or {}),
                    "cost_usd": round(float(getattr(result, "cost_usd", 0.0) or 0.0), 8),
                    "duration_s": round(elapsed, 4),
                    "n_function_calls": len(calls),
                    "text": str(data.get("text", ""))[:300],
                }
            )
            if not calls:
                input_items = [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text":
                                  "Call a tool next. End the run with submit_report."}]}
                ]
                continue

            outputs: list[dict[str, Any]] = []
            finished = False
            for call in calls:
                name = str(call.get("name", ""))
                call_id = str(call.get("call_id", ""))
                raw_args = call.get("arguments", {})
                tool_started = time.perf_counter()
                verification_dict: dict[str, Any] | None = None
                try:
                    args_obj = (
                        json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    )
                    model_cls = TOOL_ARG_MODELS.get(name)
                    if model_cls is None:
                        raise ValueError(f"unknown tool {name!r}")
                    validated = model_cls.model_validate(args_obj)
                    args = validated.model_dump(mode="python")
                    self._emit(
                        {"event": "tool_call", "turn": turn, "tool": name,
                         "call_id": call_id, "arguments": args}
                    )
                    if name == "submit_report":
                        submit_attempts += 1
                        report_arg = args["report"]
                        final_report = (
                            report_arg
                            if isinstance(report_arg, AgentReport)
                            else AgentReport.model_validate(report_arg)
                        )
                        verification = verify_report(
                            final_report, self.layout, self.tools.store,
                            config=self.verifier_config,
                        )
                        verification_dict = verification.as_dict()
                        final_verification = verification_dict
                        payload = {"tool": name, "verification": verification_dict}
                        if verification.status == "VERIFIED":
                            status, reason, finished = "VERIFIED", "report verified", True
                        elif submit_attempts > self.max_corrections:
                            status = verification.status
                            reason = "correction budget exhausted"
                            finished = True
                        else:
                            payload["instruction"] = (
                                "Report not verified. Fix the failed claims "
                                "with tool evidence and submit again."
                            )
                    else:
                        payload = self._dispatch(name, args)
                except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
                duration = time.perf_counter() - tool_started
                summary = json.dumps(payload, ensure_ascii=False, default=str)
                self._emit(
                    {"event": "tool_result", "turn": turn, "tool": name,
                     "call_id": call_id, "duration_s": round(duration, 4),
                     "summary": summary[:600],
                     **({"verification_status": verification_dict["status"]}
                        if verification_dict else {})}
                )
                outputs.append(
                    {"type": "function_call_output", "call_id": call_id,
                     "output": summary}
                )
                if finished:
                    break
            if finished:
                break
            input_items = outputs
        else:
            reason = f"max_turns={self.max_turns} reached without a verified report"

        if final_verification is not None and status == "FAILED" and submit_attempts:
            status = str(final_verification["status"])
        run_result = AgentRunResult(
            status=status,
            run_id=self.run_id,
            trace_path=self.trace.path,
            report=final_report.model_dump(mode="json") if final_report else None,
            verification=final_verification,
            total_cost_usd=round(total_cost, 8),
            turns=turns,
            scenarios_run=self.tools.scenarios_run,
            reason=reason,
            events=list(self.trace.events),
        )
        if final_report is not None:
            (self.trace.run_dir / "report.json").write_text(
                json.dumps(run_result.report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        if final_verification is not None:
            (self.trace.run_dir / "verification.json").write_text(
                json.dumps(final_verification, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        self._emit(
            {"event": "run_done", "status": status, "reason": reason,
             "total_cost_usd": run_result.total_cost_usd, "turns": turns,
             "scenarios_run": self.tools.scenarios_run}
        )
        run_result.events = list(self.trace.events)
        return run_result
