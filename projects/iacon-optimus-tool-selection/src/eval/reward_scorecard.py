"""Scorecard for Strategy 1's RLCD reward: a score plus diagnostic
feedback, not just pass/fail.

Usage:

    # Score the reference implementation in src/train/reward.py (sanity check)
    python -m src.eval.reward_scorecard

    # Score Ameya's actual implementation once it exists, e.g. a function
    # `compute_reward(confidence, outcome)` in src/train/trainer.py
    python -m src.eval.reward_scorecard --impl src.train.trainer:compute_reward

`--impl` takes `module.path:function_name` and must point at a function
with signature `(confidence: float, outcome: int) -> float`. If Ameya's
real function takes a different shape (e.g. a dict, or extra context
args), wrap it in a one-line adapter function and point `--impl` at the
adapter instead of changing this script.

Three checks, weighted into one 0-100 score:
  - golden      (50%): exact reproduction of the 5 worked-example rows
                        from Ameya's doc
  - boundary    (25%): known edge cases (confidence 0/1 vs outcome 0/1,
                        the 0.5 midpoint)
  - properties  (25%): the structural guarantees that make this a
                        *proper* scoring rule at all (never positive,
                        zero only at an exact match, symmetric, bounded)

A failing case prints expected vs. actual plus a best-guess diagnosis
(sign flipped, not squared, scaled by a constant, etc.) so a failure
tells you what to go check, not just that something's wrong.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from typing import Callable

from src.train.reward import GOLDEN_CASES, GoldenCase


RewardFn = Callable[[float, int], float]


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _load_impl(spec: str | None) -> RewardFn:
    if spec is None:
        from src.train.reward import proper_scoring_reward

        return proper_scoring_reward
    module_path, _, func_name = spec.partition(":")
    if not func_name:
        raise ValueError(f"--impl must look like 'module.path:function_name', got {spec!r}")
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _diagnose(expected: float, actual: float) -> str:
    """Best-effort guess at what's wrong, given one bad (expected, actual) pair."""
    if math.isclose(actual, -expected, abs_tol=1e-6) and expected != 0:
        return "sign is flipped — check for a missing/extra negative sign"
    if expected != 0 and math.isclose(actual / expected, round(actual / expected), rel_tol=1e-3):
        ratio = actual / expected
        if abs(ratio - 1) > 1e-6:
            return f"off by a constant factor of ~{ratio:.3g} — check for an extra multiplier"
    if abs(actual) > abs(expected) * 1.5 or abs(actual) < abs(expected) * 0.5:
        return "magnitude is off — check squaring (confidence - outcome)**2 vs abs(confidence - outcome)"
    return "value doesn't match expected — check the formula against reward = -(confidence - outcome)**2"


def check_golden(fn: RewardFn) -> list[CheckResult]:
    results = []
    for case in GOLDEN_CASES:
        actual = fn(case.confidence, case.outcome)
        ok = math.isclose(actual, case.expected_reward, abs_tol=1e-4)
        detail = (
            f"expected {case.expected_reward}, got {actual}"
            if ok
            else f"expected {case.expected_reward}, got {actual} — {_diagnose(case.expected_reward, actual)}"
        )
        results.append(CheckResult(f"golden:{case.case_id}", ok, detail))
    return results


def check_boundary(fn: RewardFn) -> list[CheckResult]:
    cases = [
        ("confidence=1.0, outcome=1 -> 0.0", 1.0, 1, 0.0),
        ("confidence=0.0, outcome=0 -> 0.0", 0.0, 0, 0.0),
        ("confidence=1.0, outcome=0 -> -1.0 (worst case)", 1.0, 0, -1.0),
        ("confidence=0.0, outcome=1 -> -1.0 (worst case)", 0.0, 1, -1.0),
        ("confidence=0.5, outcome=1 -> -0.25 (max uncertainty)", 0.5, 1, -0.25),
        ("confidence=0.5, outcome=0 -> -0.25 (max uncertainty)", 0.5, 0, -0.25),
    ]
    results = []
    for label, conf, outcome, expected in cases:
        actual = fn(conf, outcome)
        ok = math.isclose(actual, expected, abs_tol=1e-4)
        detail = f"got {actual}" if ok else f"got {actual} — {_diagnose(expected, actual)}"
        results.append(CheckResult(label, ok, detail))
    return results


def check_properties(fn: RewardFn) -> list[CheckResult]:
    results = []

    never_positive = all(
        fn(c, o) <= 1e-9 for c in [0.0, 0.25, 0.5, 0.75, 1.0] for o in (0, 1)
    )
    results.append(
        CheckResult(
            "never_positive",
            never_positive,
            "reward should never exceed 0" if not never_positive else "ok",
        )
    )

    zero_only_at_match = all(
        (abs(fn(c, o)) < 1e-9) == math.isclose(c, o)
        for c in [0.0, 0.3, 0.5, 0.7, 1.0]
        for o in (0, 1)
    )
    results.append(
        CheckResult(
            "zero_only_at_exact_match",
            zero_only_at_match,
            "reward should be exactly 0 iff confidence == outcome" if not zero_only_at_match else "ok",
        )
    )

    symmetric = all(
        math.isclose(fn(o + d, o), fn(o - d, o), abs_tol=1e-9)
        for o in (0, 1)
        for d in [0.05, 0.2, 0.4]
    )
    results.append(
        CheckResult(
            "symmetric_around_outcome",
            symmetric,
            "being off by delta above vs below the outcome should cost the same" if not symmetric else "ok",
        )
    )

    bounded = all(
        fn(c, o) >= -1.0 - 1e-9 for c in [0.0, 0.3, 0.5, 0.8, 1.0] for o in (0, 1)
    )
    results.append(
        CheckResult(
            "bounded_below_by_minus_one",
            bounded,
            "reward should never go below -1.0 for confidence/outcome in [0,1]" if not bounded else "ok",
        )
    )

    return results


def score(fn: RewardFn) -> tuple[float, dict[str, list[CheckResult]]]:
    groups = {
        "golden": check_golden(fn),
        "boundary": check_boundary(fn),
        "properties": check_properties(fn),
    }
    weights = {"golden": 0.50, "boundary": 0.25, "properties": 0.25}
    total = 0.0
    for name, results in groups.items():
        if not results:
            continue
        pass_rate = sum(r.passed for r in results) / len(results)
        total += pass_rate * weights[name]
    return total * 100, groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a Strategy 1 reward implementation")
    parser.add_argument(
        "--impl",
        default=None,
        help="module.path:function_name to test (defaults to the reference impl in src/train/reward.py)",
    )
    args = parser.parse_args(argv)

    fn = _load_impl(args.impl)
    overall, groups = score(fn)

    any_failed = False
    for group_name, results in groups.items():
        print(f"\n[{group_name}]")
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"  {mark}  {r.name}: {r.detail}")
            any_failed = any_failed or not r.passed

    print(f"\nScore: {overall:.1f}/100")
    if overall == 100:
        print("All checks pass — matches the published formula exactly.")
    elif overall >= 75:
        print("Close — check the FAIL lines above, likely a small formula bug.")
    else:
        print("Significant mismatch — re-check the formula against reward = -(confidence - outcome)**2.")

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
