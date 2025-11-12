# xalgo-platform-delta — Delta algorithms (summary)

This document summarizes the core "delta" algorithms used across the xalgo platform (focused on the math and traffic-light/risk code). It is intended as a short, navigable reference so engineers can quickly find and understand the main computations used to convert model inputs and gradients into risk/ stake decisions.

## Scope and purpose
- Describe the weighted mapping algorithm (get_weighted_mapping) and how it is used by traffic-light / InstanceRisk to compute max stakes.
- Point to the primary implementation locations and tests.
- Explain inputs, outputs, thresholds and the fallback behavior.

## Key files
- Python implementation:
  - `xalgo-platform-schemas/xalgo_schemas/delta_unit/math.py` (get_weighted_mapping, z_score_threshold)
  - `xalgo-platform-schemas/xalgo_schemas/delta_unit/traffic_light.py` (InstanceRiskV1, calc_max_stakes_euro)
  - `xalgo-platform-schemas/xalgo_schemas/tests/test_instance_risk_v1.py` (unit tests and numeric expectations)
- Rust implementation (parity):
  - `xalgo-platform-schemas/xalgo-schemas-rs/src/delta_unit/math.rs`
  - `xalgo-platform-schemas/xalgo-schemas-rs/src/delta_unit/traffic_light.rs`

## High-level flow (how the math is used)
1. Models compute logit probabilities, sigmas and gradients for each outcome (per odd / per row).
2. `InstanceRiskV1.calc_max_stakes_euro` prepares inputs and calls `get_weighted_mapping` with:
   - gradients (matrix: n_odds x n_parameters)
   - sigma_wise (per-parameter array)
   - sigma_shared (per-parameter array)
   - z_factors (per-parameter scaling derived from probs and sample size)
   - parameter_turnovers (per-parameter turnover values)
3. `get_weighted_mapping` returns a normalized mapping per odd (directional weight for each parameter).
4. The mapping drives energy allocation (w_map squared), splitting positive/negative contributions and combining with parameter bankrolls and tl_scale to produce `odds_bankroll`.
5. Kelly-scaling and constraints are applied to compute final max stakes per odd.

## get_weighted_mapping — algorithm summary
Purpose: produce a per-row (per-odds) directional mapping over parameters that captures which model parameters are driving the prediction and scale them by significance and parameter turnover.

Inputs (names as in code):
- gradients: np.ndarray, shape (n_rows, n_parameters)
- sigma_wise: np.ndarray, shape (n_parameters,)
- sigma_shared: np.ndarray, shape (n_parameters,)
- z_factors: np.ndarray, shape (n_parameters, 1) or (n_parameters,)
- parameter_turnovers: np.ndarray, shape (n_parameters,)

Steps (short):
1. sigma_tot = sqrt(sigma_wise^2 + sigma_shared^2)
2. Mapping = gradients * sigma_tot (element-wise broadcasting)
3. z_score = z_factors * mapping
4. Compute three z-score thresholds via z_score_threshold(outcome_false_positive_prob, parameter_length):
   - fallback: outcome_false_positive_prob = 0.01
   - significant: 0.001
   - turnover: 0.0001
   These thresholds are used to decide whether a signal is directional/significant and which parameters count for turnover selection.
5. mapping_significant = mapping * (|z_score| > significant_threshold)
6. Identify main_parameter_turnover from parameters where |z_score| > turnover_threshold; if none, use parameter(s) with max |z_score|. Convert to a per-parameter turnover weight (main/param_turnover) capped at 1.
7. weighted_mapping = mapping_significant * turnover_weight
8. If weighted_mapping has zero norm for a row, fall back to a one-hot vector selecting the parameter with the largest |mapping|.
9. If the maximum |z_score| for a row is below the fallback threshold, zero out the mapping (no direction).
10. Normalize mapping_combined so each row has L2-norm 1 (except rows zeroed out).

Output:
- mapping_combined: np.ndarray, shape (n_rows, n_parameters) with directional weights in [-1,1]. Rows may be all zeros (no direction) or a normalized vector indicating parameter contributions.

Behavioral notes:
- The algorithm emphasizes significant parameters (via z-scores) and reduces influence of high-turnover parameters unless they are the main driver.
- Fallback ensures every row points somewhere when a direction is present, otherwise it becomes neutral.
- There is a Rust implementation; keep both in sync when updating logic.

## InstanceRiskV1.calc_max_stakes_euro — how mapping is consumed
- w_map = get_weighted_mapping(...)
- energy = w_map^2
- Split energy into e_pos (w_map < 0 masked) and e_neg (w_map > 0 masked) to account for directionality relative to plus/minus bankrolls.
- Compute parameter_bankroll from fast_stakes and trade settings (bankroll_base, stake_scale), capped by max_bankroll.
- odds_bankroll = (parameter_bankroll * tl_scale.plus) @ e_pos.T + (parameter_bankroll * tl_scale.minus) @ e_neg.T
- Apply safety overrides for unmapped rows and null gradient cases.
- Compute Kelly scale from logit_probs, logit_sigmas and odds, apply it to odds_bankroll to get raw stakes.
- Apply trade limits (max_stake, max_win) and rounding if requested.

See `InstanceRiskV1.calc_max_stakes_euro` for implementation details and many unit tests that assert numeric behavior.

## Where to look next (links / quick pointers)
- Math implementation and primary tests:
  - `xalgo-platform-schemas/xalgo_schemas/delta_unit/math.py`
  - `xalgo-platform-schemas/xalgo_schemas/tests/test_instance_risk_v1.py`
- Traffic-light usage and calc_max_stakes_euro:
  - `xalgo-platform-schemas/xalgo_schemas/delta_unit/traffic_light.py`
- Producer/consumer flow (next doc section to add):
  - Ingest mappers (e.g., `components/soccer_gateway/ingest/.../feed_mappers/opta.py`)
  - Occurrence/state managers and parameter-update producers (search for `UpdateOccurrence`, `parameter_update` in `xalgo-platform-delta/applications`)

## Notes and maintenance
- Keep Python + Rust implementations aligned. Tests exist in both language variants (see `tests/delta_unit.rs`).
- Threshold constants are tuned via z_score_threshold and expect `num_rows`/z_factors to reflect the effective sample size.
- When changing the mapping behavior, run `xalgo_schemas` tests and the Rust tests to validate numeric parity.

---
Generated: a focused summary suitable as a starting `delta.md`. Expand later with diagrams and the end-to-end ingestion/occurrence -> parameter update traces.
