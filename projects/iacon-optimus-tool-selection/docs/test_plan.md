# Test plan: Strategy 1 RLCD reward (proper scoring rule)

Owner: Pia. Covers testing only, not implementation, that's Ameya's.
Source doc: Ameya's "Fine-Tuning using RLCD: Reward Strategies," Idea 1.

## Scope

The team picked Strategy 1: `reward = -(confidence - outcome)^2`. This
plan tests that formula's correctness only. Explicitly out of scope for
this pass (can be a follow-up once Ameya's training-loop integration
exists):

- How the reward feeds into trainer.py's RL/CE loss blend
- A calibration regression test (does training on synthetic data actually
  reduce Brier score / improve ECE over time)

## What "testing it" means here

Two things, not one:

1. **`tests/test_reward.py`** — a normal pytest suite, pass/fail, that
   runs alongside the existing 33 tests (`pytest tests/`).
2. **`src/eval/reward_scorecard.py`** — a scorecard tool. Instead of just
   pass/fail, it gives a 0-100 score and, for anything that fails, a
   best-guess diagnosis (sign flipped, missing the square, scaled by a
   constant, etc.) so a failure points at what to go check, not just that
   something's wrong. This is the thing to actually run while iterating:

   ```
   python -m src.eval.reward_scorecard --impl <his_module>:<his_function>
   ```

Both import their test cases from one shared source of truth,
`src/train/reward.py`, so the pytest suite and the scorecard can never
drift out of sync with each other.

## Interface Ameya's code needs to match

```python
def proper_scoring_reward(confidence: float, outcome: int) -> float:
    ...
```

`confidence` is the model's reported probability (0-1) that its decision
was correct. `outcome` is 1 if correct, 0 if wrong.

**This is a proposed interface, not a confirmed one** — I don't yet know
what Ameya is actually naming his function or where it'll live. Until he
tells me:

- `src/train/reward.py` has a reference implementation under this name,
  used as the default when you run the scorecard with no `--impl` flag.
- Once his real function exists, point `--impl` at it directly
  (`module.path:function_name`). If the signature doesn't match exactly
  (e.g. it takes a dict, or extra context args), wrap it in a one-line
  adapter and point `--impl` at the adapter instead of changing the test
  files.

## Test coverage

**Golden values** — the 5 rows from Ameya's worked-example table,
reproduced exactly (floating-point tolerance 1e-4):

| case | confidence | outcome | expected reward |
|---|---|---|---|
| #1023 | 0.92 | correct (1) | -0.0064 |
| #1024 | 0.60 | correct (1) | -0.16 |
| #1025 | 0.55 | wrong (0) | -0.3025 |
| #1026 | 0.90 | correct (1) | -0.01 |
| #1027 | 0.85 | wrong (0) | -0.7225 |

**Boundary cases** — not in the table, but implied by the formula:

- confidence=1.0 & outcome=1, or confidence=0.0 & outcome=0 → reward = 0
  (perfectly calibrated)
- confidence=1.0 & outcome=0, or confidence=0.0 & outcome=1 → reward = -1
  (worst possible case)
- confidence=0.5 (either outcome) → reward = -0.25 (maximum uncertainty)

**Properties** — structural guarantees that make this a *strictly proper*
scoring rule at all (Ameya's doc's whole justification for why this
reward pushes the model toward calibration rests on these holding):

- Reward is never positive
- Reward is exactly 0 if and only if confidence == outcome
- Symmetric: being off by delta above vs. below the outcome costs the same
- Strictly worsens (more negative) the further confidence drifts from
  outcome
- Bounded below by -1 (since confidence, outcome ∈ [0, 1])

## How to run it

```bash
cd projects/iacon-optimus-tool-selection

# Unit tests (pass/fail, part of the regular suite)
python -m pytest tests/test_reward.py -v

# Scorecard against the reference implementation (sanity check, should be 100/100)
python -m src.eval.reward_scorecard

# Scorecard against Ameya's real implementation, once it exists
python -m src.eval.reward_scorecard --impl src.train.trainer:compute_reward
```

## Open items

- Need Ameya's actual module/function name (or file path) to point the
  scorecard at his real implementation instead of the reference one.
- Confirm `outcome` really arrives as an int in `{0, 1}` and not a bool or
  a float — doesn't change the math, but the type hint should match
  reality.
- Once the training-loop integration exists, extend scope to cover that
  and the calibration regression check (both currently out of scope, see
  above).
