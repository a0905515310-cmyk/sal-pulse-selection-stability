# Stage 4 Selector and 200-Cycle Kernel Validation Report

Verdict: **PASS**

## A. Baseline

- Stage 1-3 regression: PASS.
- Baseline Stage 1-3 tests: 147.
- Current complete test count: 219.
- StudyConfig changed: false.

## B. CandidateView

- Fields: `observed_toa_s`, `observed_width_s`, `tie_rank_hi`, `tie_rank_lo`.
- It contains no Source, source state, TrueTOA, TrueWidth, condition, scene, or physical event identity.
- All arrays are aligned one-dimensional read-only arrays.
- Every selector returns only `selected_pos`; source classification remains outside the selector.

## C. Five Selectors

- FIRST: argmin of observed TOA; width is not read.
- LAST: argmax of observed TOA; width is not read.
- T: argmin of `q_t = ((observed_toa - ref_toa) / sigma_dt)^2`; width is not read.
- W: argmin of `q_width = ((observed_width - ref_width) / sigma_dwidth)^2`; TOA deviation is not read.
- TW: argmin of exactly `D = q_t + q_width`, with no empirical weight, hard feature gate, or rejection threshold.

## D. Scales

- `sigma_dt = sqrt(2) * SigmaTOA`.
- `sigma_dwidth = SigmaWidth`.
- Both scales remain fixed for every condition, method, cycle, and prior selection outcome; no dynamic re-estimation occurs.

## E. Tie

- TieRank is used only when the primary float64 values are exactly equal; no tolerance or approximate tie is used.
- Every method chooses the minimum unsigned 128-bit `(tie_rank_hi, tie_rank_lo)` lexicographic rank.
- A complete 128-bit collision raises `TieRankCollisionError` and stops; array position is never a fallback.

## F. Gate

- Every gate is `[a,b)` with `a = ref_toa - 5 us` and `b = ref_toa + 5 us`.
- G and F membership uses true TOA only.
- H membership uses the frozen Stage 3 `hprf_index_range` resolver and signed n.
- Observed TOA is never used for a second membership filter.

## G. MethodState

- Five independent MethodState objects begin at the one shared `initial_ref_toa_s`.
- Their reference TOAs can naturally diverge after different selections.
- N does not terminate a trajectory; the next cycle is always evaluated.
- Reference width is not state and is read directly from the same-cycle encoding reference.

## H. Reference Update

- Selection: `next_ref_toa = selected_observed_toa + DeltaT`.
- N: `next_ref_toa = current_ref_toa + DeltaT`.
- The update function has no Source or selected-width input.

## I. State

- SourceStateCode is uint8: `0=N`, `1=G`, `2=H`, `3=F`.
- H and F remain distinct Stage 4 source codes; no downstream C/E/N metric is computed here.

## J. HPRF

- Per-Method x per-Cycle H count hard assertions executed: 42000.
- Every HPRF/composite Method x Cycle was asserted non-N.

## K. NI

- All five NI source-state trajectories are exactly identical within each validation Condition-Repeat.
- All five NI reference-TOA trajectories are exactly identical within each validation Condition-Repeat.

## L. Stage Boundary

Stage 5 was NOT executed.
RepeatMetrics were NOT computed.
Pilot was NOT executed.
Formal was NOT executed.
Bootstrap was NOT executed.
No paper result was generated.
No paper result figures were generated.

## Validation Checks

- selector.candidate_view_arrays_read_only: PASS
- selector.candidate_view_fields_exact: PASS
- selector.exact_tie_minimum_tierank: PASS
- selector.near_tie_primary_wins: PASS
- selector.no_approximate_tie_call: PASS
- selector.selector_first: PASS
- selector.selector_last: PASS
- selector.selector_permutation_invariance: PASS
- selector.selector_signatures_source_blind: PASS
- selector.selector_t: PASS
- selector.selector_tw: PASS
- selector.selector_tw_score_exact: PASS
- selector.selector_w: PASS
- selector.selectors_return_selected_pos: PASS
- selector.source_permutation_invariance: PASS
- selector.tierank_collision_stop: PASS
- kernel.condition_count_19: PASS
- kernel.cycle_count_200: PASS
- kernel.delta_t_consumed_199: PASS
- kernel.forbidden_random_api_scan: PASS
- kernel.forbidden_result_directories_absent: PASS
- kernel.half_open_gate: PASS
- kernel.hprf_composite_no_n: PASS
- kernel.hprf_per_cycle_count: PASS
- kernel.hprf_uses_stage3_resolver: PASS
- kernel.method_state_initialization: PASS
- kernel.method_state_minimal: PASS
- kernel.n_continuation: PASS
- kernel.ni_trajectory_identity: PASS
- kernel.observed_toa_not_regated: PASS
- kernel.physical_namespace_isolation: PASS
- kernel.reference_toa_divergence: PASS
- kernel.reference_update_source_blind: PASS
- kernel.retired_identifier_scan: PASS
- kernel.selection_reference_update: PASS
- kernel.source_state_mapping: PASS
- kernel.source_state_schema: PASS
- kernel.stage3_input_K_200: PASS
- kernel.stage3_input_condition_count_19: PASS
- kernel.stage3_input_encoded_duration: PASS
- kernel.stage3_input_encoding_reference_exact: PASS
- kernel.stage3_input_interval_count_199: PASS
- kernel.stage3_input_method_count_5: PASS
- kernel.stage3_input_reference_keys_exact: PASS
- kernel.stage3_input_study_config_exact: PASS
- kernel.stage5_identifier_scan: PASS
- kernel.true_toa_gate_membership: PASS
- kernel.validation_repeat_count: PASS

## Blocking Issues

None
