"""Simulator verification against standard test cases.

Cases live in cases.py; the generated report and status file live in
docs/verification/ (see scripts/run_verification.py).
"""

from egresstwin.validation.cases import ALL_CASES, CaseResult, run_all
from egresstwin.validation.status import load_verification_status

__all__ = ["ALL_CASES", "CaseResult", "load_verification_status", "run_all"]
