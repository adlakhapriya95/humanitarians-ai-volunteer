"""Tests for Strategy 1's RLCD reward (proper scoring rule).

Scope, per decisions.md: reward-formula correctness only. This does not
test training-loop integration (trainer.py's RL/CE blend) or run a
calibration regression on toy data — those are explicitly out of scope
for this pass; see docs/test_plan.md.
"""

import math

import pytest

from src.train.reward import GOLDEN_CASES, proper_scoring_reward


# ---------------------------------------------------------------------
# 1. Golden values — must match Ameya's worked-example table exactly
# ---------------------------------------------------------------------


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[c.case_id for c in GOLDEN_CASES])
def test_golden_case_matches_doc(case):
    actual = proper_scoring_reward(case.confidence, case.outcome)
    assert actual == pytest.approx(case.expected_reward, abs=1e-4)


# ---------------------------------------------------------------------
# 2. Boundary / edge cases not in the worked table
# ---------------------------------------------------------------------


def test_perfect_confident_correct_is_zero():
    # confidence == outcome exactly -> reward is the maximum possible, 0
    assert proper_scoring_reward(1.0, 1) == pytest.approx(0.0)


def test_perfect_confident_wrong_is_worst_case():
    # confidence == 1 but outcome == 0 -> maximally wrong, reward == -1
    assert proper_scoring_reward(1.0, 0) == pytest.approx(-1.0)


def test_perfect_confident_wrong_the_other_way():
    # confidence == 0 but outcome == 1 -> also maximally wrong
    assert proper_scoring_reward(0.0, 1) == pytest.approx(-1.0)


def test_zero_confidence_correctly_wrong_is_zero():
    # confidence == outcome == 0 -> also the maximum, 0
    assert proper_scoring_reward(0.0, 0) == pytest.approx(0.0)


def test_maximum_uncertainty_midpoint():
    # confidence == 0.5 is equidistant from both outcomes
    assert proper_scoring_reward(0.5, 1) == pytest.approx(-0.25)
    assert proper_scoring_reward(0.5, 0) == pytest.approx(-0.25)


# ---------------------------------------------------------------------
# 3. Properties the formula must hold regardless of specific values
#    (these are what makes it a *strictly proper* scoring rule, which is
#    the entire justification in Ameya's doc for why this reward pushes
#    the model toward calibration at all)
# ---------------------------------------------------------------------


def test_reward_is_never_positive():
    for conf in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]:
        for outcome in (0, 1):
            assert proper_scoring_reward(conf, outcome) <= 0.0


def test_reward_is_zero_only_at_exact_match():
    for conf in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        for outcome in (0, 1):
            reward = proper_scoring_reward(conf, outcome)
            is_exact_match = math.isclose(conf, outcome)
            assert (reward == 0.0) == is_exact_match


def test_reward_is_symmetric_around_outcome():
    # Being off by delta in either direction should cost the same.
    for outcome in (0, 1):
        for delta in [0.05, 0.2, 0.4]:
            above = proper_scoring_reward(outcome + delta, outcome)
            below = proper_scoring_reward(outcome - delta, outcome)
            assert above == pytest.approx(below)


def test_reward_strictly_worsens_as_confidence_moves_away_from_outcome():
    # Monotonicity: further from the truth -> strictly more negative.
    outcome = 1
    confidences = [1.0, 0.9, 0.7, 0.4, 0.0]  # strictly decreasing, moving away from outcome=1
    rewards = [proper_scoring_reward(c, outcome) for c in confidences]
    for earlier, later in zip(rewards, rewards[1:]):
        assert later < earlier  # strictly worse (more negative) each step


def test_reward_bounded_below_by_minus_one():
    # confidence and outcome are both in [0, 1], so the squared gap can't
    # exceed 1, so reward can't go below -1.
    for conf in [0.0, 0.3, 0.5, 0.8, 1.0]:
        for outcome in (0, 1):
            assert proper_scoring_reward(conf, outcome) >= -1.0
