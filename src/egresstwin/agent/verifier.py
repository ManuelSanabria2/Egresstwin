"""Deterministic verifier for agent reports. No LLM is involved.

Every claimed number is re-read from the scenario store, improvement
is tested with a Welch t-test plus confidence-interval separation,
the N-1 stress results are re-checked against the baseline, and the
modifications are re-applied to prove feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import TypeAdapter

from egresstwin.agent.report import OBJECTIVE_METRIC, AgentReport
from egresstwin.layout.schema import Layout
from egresstwin.parallel.store import METRIC_KEYS, ScenarioStore, welch_compare
from egresstwin.scenarios.apply import apply_modifications
from egresstwin.scenarios.spec import Modification

_MODS_ADAPTER = TypeAdapter(list[Modification])


@dataclass
class VerifierConfig:
    tolerance: float = 0.01  # claimed numbers must match within 1%
    p_threshold: float = 0.05
    require_ci_separation: bool = True
    n1_max_worsening_pct: float = 20.0  # vs the baseline T95
    max_changes: int | None = None


@dataclass
class ClaimCheck:
    claim: str
    passed: bool
    detail: str


@dataclass
class VerificationResult:
    status: str  # VERIFIED | PARTIALLY_VERIFIED | FAILED
    checks: list[ClaimCheck] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": [
                {"claim": c.claim, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
            "n_passed": sum(c.passed for c in self.checks),
            "n_checks": len(self.checks),
        }


def _close(claimed: float, actual: float, tolerance: float) -> bool:
    if actual == 0:
        return abs(claimed) <= tolerance
    return abs(claimed - actual) / abs(actual) <= tolerance


def verify_report(
    report: AgentReport,
    layout: Layout,
    store: ScenarioStore,
    *,
    config: VerifierConfig | None = None,
) -> VerificationResult:
    """Check every claim of a report against the scenario store."""
    cfg = config or VerifierConfig()
    checks: list[ClaimCheck] = []

    def add(claim: str, passed: bool, detail: str) -> None:
        checks.append(ClaimCheck(claim=claim, passed=passed, detail=detail))

    baseline = store.get(report.baseline_id)
    add(
        "baseline_exists",
        baseline is not None,
        f"baseline {report.baseline_id!r} "
        + ("found in store" if baseline else "not found in store; run get_baseline"),
    )
    if baseline is None:
        return VerificationResult(status="FAILED", checks=checks)

    metric = OBJECTIVE_METRIC[report.objective]
    base_agg = baseline.aggregate(metric)

    if not report.recommendations:
        add(
            "honest_no_improvement",
            True,
            "report recommends nothing; no improvement claim to check",
        )
        return VerificationResult(status="VERIFIED", checks=checks)

    rec_fully_verified = 0
    for rec in report.recommendations:
        tag = f"[{rec.scenario_id}]"
        rec_start = len(checks)
        record = store.get(rec.scenario_id)
        add(f"{tag} scenario_exists", record is not None,
            "found in store" if record else "not found; run it with run_scenarios")
        if record is None:
            continue

        # Every claimed number must match the store within tolerance.
        for key, claimed in rec.claimed_metrics.items():
            if key not in METRIC_KEYS:
                add(f"{tag} claim {key}", False, f"unknown metric {key!r}")
                continue
            actual = record.aggregate(key)["mean"]
            add(
                f"{tag} claim {key}",
                _close(float(claimed), actual, cfg.tolerance),
                f"claimed {claimed:g}, store mean {actual:.6g} "
                f"(tolerance {cfg.tolerance:.0%})",
            )

        # Claimed modifications must be the simulated ones, and feasible.
        claimed_mods = [m.model_dump(mode="json") for m in rec.modifications]
        add(
            f"{tag} modifications_match_scenario",
            claimed_mods == record.modifications,
            "report modifications equal the simulated spec"
            if claimed_mods == record.modifications
            else f"report {claimed_mods} != simulated {record.modifications}",
        )
        if cfg.max_changes is not None:
            add(
                f"{tag} within_change_limit",
                len(rec.modifications) <= cfg.max_changes,
                f"{len(rec.modifications)} changes vs limit {cfg.max_changes}",
            )
        try:
            mods = _MODS_ADAPTER.validate_python(rec.modifications)
            apply_modifications(layout, mods)
            add(f"{tag} modifications_feasible", True, "apply_modifications succeeds")
        except Exception as exc:  # noqa: BLE001 - feasibility detail for the agent
            add(f"{tag} modifications_feasible", False, str(exc))

        # Significant improvement on the objective metric.
        cmp_ = welch_compare(baseline.values(metric), record.values(metric))
        rec_agg = record.aggregate(metric)
        improved = cmp_["diff_mean_b_minus_a"] < 0
        significant = cmp_["p_value"] < cfg.p_threshold
        ci_sep = rec_agg["ci_high"] < base_agg["ci_low"]
        passed_sig = improved and significant and (ci_sep or not cfg.require_ci_separation)
        add(
            f"{tag} significant_improvement_{metric}",
            passed_sig,
            f"baseline mean {base_agg['mean']:.4g} CI[{base_agg['ci_low']:.4g},"
            f"{base_agg['ci_high']:.4g}] vs scenario mean {rec_agg['mean']:.4g} "
            f"CI[{rec_agg['ci_low']:.4g},{rec_agg['ci_high']:.4g}], "
            f"Welch p={cmp_['p_value']:.4g}, Cohen d={cmp_['cohens_d']:.3g}",
        )

        # N-1 robustness: stressed T95 no worse than baseline + X%.
        stress_claims = {
            c.blocked_exit_id: c for c in report.robustness if c.scenario_id == rec.scenario_id
        }
        base_t95 = baseline.aggregate("t95_s")["mean"]
        for exit_id in record.exit_ids:
            stored = store.stress.get((rec.scenario_id, exit_id))
            claim = stress_claims.get(exit_id)
            if stored is None or claim is None:
                add(
                    f"{tag} n_minus_1[{exit_id}]",
                    False,
                    "no stress result in store and report; run stress_test",
                )
                continue
            stored_mean = stored.aggregate("t95_s")["mean"]
            numbers_ok = _close(claim.claimed_t95_s, stored_mean, cfg.tolerance)
            worsening = 100.0 * (stored_mean - base_t95) / base_t95 if base_t95 else 0.0
            robust = worsening <= cfg.n1_max_worsening_pct
            add(
                f"{tag} n_minus_1[{exit_id}]",
                numbers_ok and robust,
                f"claimed T95 {claim.claimed_t95_s:g}, stored {stored_mean:.6g}; "
                f"worsening vs baseline {worsening:.2f}% "
                f"(limit {cfg.n1_max_worsening_pct:g}%)",
            )
        if all(c.passed for c in checks[rec_start:]):
            rec_fully_verified += 1

    failed = [c for c in checks if not c.passed]
    substantive_pass = any(
        c.passed and any(k in c.claim for k in ("claim ", "significant", "feasible", "n_minus_1"))
        for c in checks
    )
    if not failed:
        status = "VERIFIED"
    elif rec_fully_verified > 0 or substantive_pass:
        status = "PARTIALLY_VERIFIED"
    else:
        status = "FAILED"
    return VerificationResult(status=status, checks=checks)
