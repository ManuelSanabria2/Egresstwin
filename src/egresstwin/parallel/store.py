"""Scenario result store: per-replica metrics and mean/CIs per scenario."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

METRIC_KEYS = ("t95_s", "total_time_s", "mean_egress_s", "peak_density_per_m2")


@dataclass
class ReplicaMetrics:
    """Metrics of one simulated replica."""

    seed: int
    t95_s: float
    total_time_s: float
    mean_egress_s: float
    peak_density_per_m2: float
    n_people: int
    n_exited: int
    hotspots: list[dict[str, Any]] = field(default_factory=list)
    flow_per_exit: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ScenarioRecord:
    """A scenario spec plus every replica run under it."""

    scenario_id: str
    name: str
    modifications: list[dict[str, Any]] = field(default_factory=list)
    blocked_exit_ids: list[str] = field(default_factory=list)
    exit_ids: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    replicas: list[ReplicaMetrics] = field(default_factory=list)

    def values(self, metric: str) -> np.ndarray:
        return np.array([float(getattr(r, metric)) for r in self.replicas], dtype=float)

    def aggregate(self, metric: str) -> dict[str, float]:
        return aggregate_values(self.values(metric))


def aggregate_values(values: np.ndarray) -> dict[str, float]:
    """Mean, sample std and 95% t confidence interval of the mean."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    mean = float(vals.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_low": mean, "ci_high": mean, "n": 1}
    std = float(vals.std(ddof=1))
    half = float(stats.t.ppf(0.975, n - 1)) * std / float(np.sqrt(n))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "n": n}


def welch_compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Welch t-test of a vs b plus Cohen's d (positive means a is larger)."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    aa, bb = aa[np.isfinite(aa)], bb[np.isfinite(bb)]
    out: dict[str, float] = {
        "mean_a": float(aa.mean()) if len(aa) else float("nan"),
        "mean_b": float(bb.mean()) if len(bb) else float("nan"),
        "diff_mean_b_minus_a": (
            float(bb.mean() - aa.mean()) if len(aa) and len(bb) else float("nan")
        ),
        "p_value": 1.0,
        "cohens_d": 0.0,
    }
    if len(aa) >= 2 and len(bb) >= 2 and aa.std(ddof=1) == 0 and bb.std(ddof=1) == 0:
        # Degenerate (censored) samples: identical constants or a
        # clean separation; scipy's t-test is numerically meaningless.
        out["p_value"] = 0.0 if aa.mean() != bb.mean() else 1.0
        out["cohens_d"] = 0.0 if aa.mean() == bb.mean() else float(
            np.sign(bb.mean() - aa.mean()) * 10.0
        )
    elif len(aa) >= 2 and len(bb) >= 2:
        result = stats.ttest_ind(aa, bb, equal_var=False)
        out["p_value"] = float(result.pvalue) if np.isfinite(result.pvalue) else 1.0
        pooled = float(np.sqrt((aa.var(ddof=1) + bb.var(ddof=1)) / 2.0))
        if pooled > 0:
            out["cohens_d"] = float((bb.mean() - aa.mean()) / pooled)
    return out


class ScenarioStore:
    """In-memory scenario records, optionally mirrored to a JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.records: dict[str, ScenarioRecord] = {}
        self.stress: dict[tuple[str, str], ScenarioRecord] = {}
        self.baseline_id: str | None = None
        self.path = Path(path) if path is not None else None

    def add(self, record: ScenarioRecord) -> ScenarioRecord:
        self.records[record.scenario_id] = record
        self._flush()
        return record

    def get(self, scenario_id: str) -> ScenarioRecord | None:
        return self.records.get(scenario_id)

    def add_stress(self, scenario_id: str, blocked_exit_id: str, record: ScenarioRecord) -> None:
        self.stress[(scenario_id, blocked_exit_id)] = record
        self.records[record.scenario_id] = record
        self._flush()

    def _flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for sid, rec in self.records.items():
            payload[sid] = {
                "name": rec.name,
                "modifications": rec.modifications,
                "blocked_exit_ids": rec.blocked_exit_ids,
                "exit_ids": rec.exit_ids,
                "events": rec.events,
                "aggregates": {k: rec.aggregate(k) for k in METRIC_KEYS},
                "replicas": [
                    {k: getattr(r, k) for k in ("seed", *METRIC_KEYS, "n_people", "n_exited")}
                    for r in rec.replicas
                ],
            }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
