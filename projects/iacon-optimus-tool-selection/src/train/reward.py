"""Strategy 1 RLCD reward: proper scoring rule (negative Brier score).

Source: Ameya's "Fine-Tuning using RLCD: Reward Strategies" doc, Idea 1.

    reward = -(confidence - outcome)^2

This is the reward strategy the team picked (Strategy 1, not the learned
auxiliary-reward model in Idea 2). Ameya owns the actual training-loop
integration; this module exists so the formula has one, tested, canonical
home — `tests/test_reward.py` and `src/eval/reward_scorecard.py` both
import `proper_scoring_reward` and `GOLDEN_CASES` from here.

----------------------------------------------------------------------
NEEDS CONFIRMATION FROM AMEYA
----------------------------------------------------------------------
This file is a proposed interface, not a confirmed one. Pia doesn't yet
know the real module/function name Ameya's implementation will use. Two
things need to happen once his code exists:

  1. If his function lives somewhere else / is named differently, either
     re-export it from here (`from ameya_module import real_fn as
     proper_scoring_reward`) or point reward_scorecard.py's `--impl` flag
     at it directly — see that file's docstring.
  2. Confirm `outcome` is really an int in {0, 1} (as the doc's table
     implies) and not, say, a bool or a float probability — the formula
     behaves the same either way since Python treats True/False as 1/0,
     but the type hint below should match what his code actually passes.
----------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


def proper_scoring_reward(confidence: float, outcome: int) -> float:
    """Strategy 1 reward: -(confidence - outcome)^2.

    `confidence` is the model's reported probability (0-1) that its
    decision was correct. `outcome` is 1 if the decision was actually
    correct, 0 if wrong. Reward is a strictly proper scoring rule: its
    expected value is maximized only when `confidence` equals the true
    probability of being correct, so it's maximized (= 0) exactly when
    confidence == outcome, and gets more negative the further apart they
    are.
    """
    return -((confidence - outcome) ** 2)


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    confidence: float
    outcome: int
    expected_reward: float


# The five worked-example rows from Ameya's doc, reward values exactly as
# published there. These are the ground truth for test_reward.py's golden
# tests and for the scorecard — if an implementation doesn't reproduce
# these to within floating-point tolerance, something is wrong.
GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase("Case #1023", confidence=0.92, outcome=1, expected_reward=-0.0064),
    GoldenCase("Case #1024", confidence=0.60, outcome=1, expected_reward=-0.16),
    GoldenCase("Case #1025", confidence=0.55, outcome=0, expected_reward=-0.3025),
    GoldenCase("Case #1026", confidence=0.90, outcome=1, expected_reward=-0.01),
    GoldenCase("Case #1027", confidence=0.85, outcome=0, expected_reward=-0.7225),
]
