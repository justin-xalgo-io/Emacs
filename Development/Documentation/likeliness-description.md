l
# Likeliness (per-player) — description and implementation notes

This document explains how "likeliness" values are computed, normalized, and consumed in the soccer simulation code (applications/soccer-model-full/soccer_model_full/core_model).

It summarises the math implemented in `player_flow.py`, how playtime and subs are treated, the exact inputs and outputs of the key function `likeliness_per_period`, and where the results are used for sampling. This aims to help engineers understand, test, and maintain likeliness-related logic.

## Location
- Implementation: `applications/soccer-model-full/soccer_model_full/core_model/player_flow.py`
- Feature header names and layout: `applications/soccer-model-full/soccer_model_full/core_model/constants.py` (`PLAYER_FEATURES_HEADER` and `PLAYER_FEATURES_HEADER_GOAL_ONLY`)
- Tests / fixtures: `applications/soccer-model-full/tests/*` and various JSON fixtures (many `pla_<id>_*_likeliness` keys)

## Conceptual overview

A model produces per-player scores (logits or log-probabilities) for events (goals, shots, fouls, cards, etc.). The code converts those raw per-player scores into per-player probabilities that are valid distributions over players for sampling occurrences in a match, split across first and second halves. The conversion accounts for each player's expected playtime in each half and reserves probability mass for a "sub" bucket (players not on pitch or residual mass).

Key ideas:
- Raw model outputs for a given event (e.g., `goal_likeliness`) are passed as log-space scores per player.
- The sub-slot occupies index 0 and is handled explicitly. Player feature arrays include an explicit sub slot at the front.
- Log-scores are exponentiated and normalised across non-sub players, then scaled by per-half playtimes.
- Remaining probability mass (1 - sum(player_probs_scaled)) is assigned to the sub bucket (non-negative clamp).
- The function returns log-probabilities (logits) for each player including the sub at index 0 for each half. Downstream sampling expects logits.

## The key function: likeliness_per_period

Signature (conceptual):
- Inputs:
  - log_probs: Tensor shaped [n_sims, num_players_total] (includes a leading column expected for "sub" at index 0)
  - playtimes_fh: Tensor shaped [n_sims, num_players_total] — first-half playtime fractions (0..1), sub slot included
  - playtimes_sh: Tensor shaped [n_sims, num_players_total] — second-half playtime fractions (0..1)

- Outputs:
  - log_probs_with_sub_fh: Tensor shaped [n_sims, num_players_total] — log-probabilities for first half (sub at index 0)
  - log_probs_with_sub_sh: Tensor shaped [n_sims, num_players_total] — log-probabilities for second half

Algorithm (exact steps as implemented):
1. Convert logits to linear space for non-sub players:
   - Take slice skipping the sub slot: `raw = log_probs[:, NUMBER_OF_SUBS:]`.
   - Compute `probs = exp(raw)`.
2. Normalise across players (non-sub) with numerical safety:
   - denom = max(sum(probs, axis=1, keepdims=True) + 1e-20, 1.0)
   - probs = probs / denom
   - Note: This applies a lower bound of 1.0 on denom. In practice if the model scores are such that the sum of exp is < 1.0, the code will divide by 1.0 (i.e., not inflate the small sum). This is an explicit design choice in the code base.
3. Weight probabilities by playtime for each half:
   - probs_fh = probs * playtimes_fh[:, NUMBER_OF_SUBS:]
   - probs_sh = probs * playtimes_sh[:, NUMBER_OF_SUBS:]
4. Compute residual (sub) probability for each half:
   - sub_prob_fh = max(1.0 - sum(probs_fh, axis=1, keepdims=True), 0.0)
   - sub_prob_sh = max(1.0 - sum(probs_sh, axis=1, keepdims=True), 0.0)
   - This ensures the total mass per simulation per half is at most 1.0 and non-negative.
5. Reassemble final distributions including the sub slot at index 0:
   - probs_with_sub_fh = concat([sub_prob_fh, probs_fh], axis=1)
   - probs_with_sub_sh = concat([sub_prob_sh, probs_sh], axis=1)
6. Return logits (logs) of those probabilities:
   - return log(probs_with_sub_fh), log(probs_with_sub_sh)

Why return logs?
- Downstream sampling utilities in `player_flow.py` call `tf.random.categorical` or sampling helpers that expect logits. Returning log(...) keeps numerical stability and matches downstream expectations.

## Playtime construction (where playtimes come from)
- Playtime is computed earlier in `team_player_flow` / `team_player_goals_only` from the `player_features` array:
  - `playtimes = exp(player_features[:, :, PLAYER_FEATURES_HEADER.index("play_time")] + log(2.0))`
  - This maps feature scalars into a 0..2.0 range (2.0 corresponds to full 90 minutes). Implementation then clips/caps per half:
    - `playtimes_fh = min(playtimes, 1.0)`
    - `playtimes_sh = min(max(playtimes - 1.0, 0.0), 1.0)`
  - So a playtime feature value is encoded such that `playtimes`=2 → 90 minutes, `playtimes`=1 → full first half (45m), and fractional values proportionally indicate minutes per half.

## Sub slot and NUMBER_OF_SUBS
- The code expects a fixed number of "sub" positions (`NUMBER_OF_SUBS`). The arrays used for sampling place those sub slots at the start of the player axis. The slicing `[:, NUMBER_OF_SUBS:]` consistently ignores these sub slots when converting logits to linear probabilities, then re-attaches a single aggregated sub probability as the first column of the returned arrays.

## Sampling usage
- After `likeliness_per_period` returns logits for each half, the code uses those logits with multinomial samplers:
  - `sample_multinomial(counts, logits)` repeats logits and samples categorical draws to produce per-player occurrence counts.
  - For joint distributions (e.g., goal + assist), `team_player_flow` constructs joint matrices from `goal_likeliness` and `assist_likeliness` and then logs them before sampling with `sample_joint_dist`.

## Edge cases and behavior notes
- denom floor 1.0: If the sum of exp(raw) < 1.0 then `denom` uses 1.0; this prevents inflating tiny sums. That design decision affects very small logits (very negative) — players won't get scaled up.
- Residual clamp: `sub_prob` is clamped with `max(..., 0.0)` to avoid negative mass from rounding errors or over-scaling.
- Playtime zero: If a player has playtime 0 for a half, their per-half probability becomes zero and mass flows to sub.
- Numerical safety: `+ 1e-20` is used before divisions to avoid division by zero.

## Pointers for maintainers
- If you change the ordering of features in `PLAYER_FEATURES_HEADER`, update consumer indices in `player_flow.py`.
- If the semantics of the playtime feature change, confirm the encoding `exp(feature + log(2.0))` remains valid for the new scale.
- If you want alternate normalization (e.g., don't floor denom at 1.0), adjust `denom = tf.maximum(..., 1.0)` accordingly — but update any tests that assume current behaviour.

## Quick examples (conceptual)
- Given 3 players with raw log-scores (excluding sub) of [-2.0, 0.0, -1.0] and playtimes_fh = [1.0, 1.0, 0.5], the function will:
  - exp -> [0.1353, 1.0, 0.3679]
  - sum = 1.5032 -> denom = max(1.5032 + 1e-20, 1.0) -> 1.5032
  - probs = [0.090, 0.665, 0.244]
  - probs_fh = probs * playtimes_fh -> [0.090, 0.665, 0.122]
  - sub_prob_fh = max(1.0 - sum(probs_fh), 0) = max(1 - 0.877, 0) = 0.123
  - final probs_with_sub_fh = [0.123, 0.090, 0.665, 0.122]
  - returns log(...) of that vector

## References in codebase
- `applications/soccer-model-full/soccer_model_full/core_model/player_flow.py` — main implementation
- `applications/soccer-model-full/soccer_model_full/core_model/constants.py` — `PLAYER_FEATURES_HEADER`
- Tests in `applications/soccer-model-full/tests/` — many tests exercise similar logic (look for `goal_likeliness`, `shot_likeliness` references)

---

If you want, I can:
- Add a minimal unit test in `applications/soccer-model-full/tests/` that checks `likeliness_per_period` behavior on a small synthetic tensor (happy path + edge case where sum(exp) < 1.0).
- Trace upstream to show exactly where the `log_probs` are produced by the model wrapper.
- Convert this document into part of repo docs (e.g., in `documentations/` or `applications/soccer-model-full/README.md`).
