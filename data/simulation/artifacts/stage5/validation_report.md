# Stage 5 Metrics and RepeatMetrics Validation Report

Verdict: **PASS**

## A. Baseline

- Stage 1-4 regression: PASS.
- Frozen Stage 1-4 baseline tests: 219 / 219 passed.
- Current complete tests: 283 passed, 0 failed.
- StudyConfig changed: false.
- Selector changed: false.
- The only kernel edit is a public thin wrapper; the private 200-cycle kernel source hash is unchanged and both entry paths are array-identical.

## B. State mapping

- Stage 4 SourceStateCode `0=N` maps to evaluation state N.
- Stage 4 SourceStateCode `1=G` maps to evaluation state C.
- Stage 4 SourceStateCode `2=H` and `3=F` both map to evaluation state E.
- No selector, observed TOA, observed width, score, or TieRank is read to reclassify C/E/N.

## C. RepeatMetrics schema

- Key: ConditionOrdinal, ConditionCode, MethodOrdinal, MethodCode, RepeatID.
- Counts: N_C, N_E, N_N; all nine C/E/N transitions; N_Cdot, N_Edot, N_Ndot; N_run_NC, Sum_L_NC_obs, N_open_NC; N_E_H, N_E_F.
- EndState is exactly one of C, E, N. All count fields are integers.

## D. Count conservation

- Per Repeat: `N_C + N_E + N_N = 200`.
- Per Repeat: all nine adjacent transition counts sum to `199`.

## E. Conditional denominators

- `N_Cdot = N_CC + N_CE + N_CN`.
- `N_Edot = N_EC + N_EE + N_EN`.
- `N_Ndot = N_NC + N_NE + N_NN`.
- Each origin count equals its cycle count minus the indicator that EndState is that origin.

## F. NC runs

- E and N jointly form maximal non-correct runs; N does not split an NC run.
- Per Repeat: `Sum_L_NC_obs = N_E + N_N`.

## G. Right-open

- A run still E/N at cycle 200 is retained in N_run_NC and Sum_L_NC_obs and contributes one to N_open_NC.
- No cycle 201 is created. `N_open_NC = I(EndState in {E,N})`.

## H. Core metrics

- `P_cor = sum N_C / (R * 200)`.
- `P_C|C = sum N_CC / sum N_Cdot`.
- `P_C|E = sum N_EC / sum N_Edot`; the numerator is E-to-C, never C-to-E.
- `mean L_NC = sum Sum_L_NC_obs / sum N_run_NC`.

## I. Support metrics

- `P_E|C = sum N_CE / sum N_Cdot`; `P_N|C = sum N_CN / sum N_Cdot`.
- `P_E|E = sum N_EE / sum N_Edot`; `P_N|E = sum N_EN / sum N_Edot`.
- `P_N = sum N_N / (R * 200)`.
- `P_open_NC = sum N_open_NC / sum N_run_NC`.
- `P_NC_end = sum N_open_NC / R`; it is not P_open_NC.

## J. Undefined metrics

- Frozen reasons: C_DENOM_ZERO, E_DENOM_ZERO, NO_NC_RUN, E_COUNT_ZERO.
- Undefined metrics use `defined=false`, `value=null`, and an explicit reason; they are never zero-filled or silently stored as NaN.

## K. Aggregation

- Every conditional metric uses a ratio of total numerator count to total denominator count across Repeats.
- Per-Repeat conditional probabilities are not averaged.

## L. Sources

- `N_E_H + N_E_F = N_E` for every Repeat.
- NI has no E; HPRF E comes only from H; IDF E comes only from F; composite E retains H/F source contributions for support interpretation.

## M. H300

- Integer width-level difference maps to WIDTH_STRATUM_0NS, WIDTH_STRATUM_10NS, or WIDTH_STRATUM_GE20NS.
- Each Method-Repeat conserves 200 cycles and C+E equals cycle count in every stratum; H300 N is a hard error.
- C-to-E uses the current cycle k width stratum for transition k to k+1.
- L_NC is never stratified because a continuous NC run can cross width strata.

## N. Stage boundary

Stage 6 was NOT executed.
Pilot R=200 was NOT executed.
Formal-R decision was NOT executed.
Formal was NOT executed.
Bootstrap was NOT executed.
No paper result was generated.
No paper result figure was generated.

The PILOT random namespace was used only for fixed RepeatIDs 1, 2, and 17 in a 57 Condition-Repeat implementation validation; this is not the Pilot R=200 study.

## Machine checks

- baseline_219_passed: PASS
- complete_pytest_count_correct: PASS
- complete_pytest_passed: PASS
- forbidden_execution_directories_absent: PASS
- forbidden_random_api_scan: PASS
- frozen_science_files_unchanged: PASS
- h300_no_lnc_stratification: PASS
- private_cycle_kernel_source_unchanged: PASS
- public_wrapper_array_equivalence: PASS
- public_wrapper_signature: PASS
- selector_hash_unchanged: PASS
- stage1_artifact_pass: PASS
- stage2_artifact_pass: PASS
- stage3_artifact_pass: PASS
- stage4_artifact_pass: PASS
- stage5_metric_tests_passed: PASS
- stage5_repeat_metrics_tests_passed: PASS
- stage6_plus_definition_scan: PASS
- study_config_hash_unchanged: PASS
- synthetic_all_c_metric_values: PASS
- synthetic_all_nine_transitions: PASS
- synthetic_h300_c_to_e_current_cycle_anchor: PASS
- synthetic_h300_fixed_width_strata: PASS
- synthetic_h300_no_lnc_stratification: PASS
- synthetic_h_f_source_conservation: PASS
- synthetic_n_does_not_split_nc: PASS
- synthetic_nc_sum_length_conservation: PASS
- synthetic_origin_transition_conservation: PASS
- synthetic_p_open_nc_distinct_from_p_nc_end: PASS
- synthetic_private_cycle_kernel_source_unchanged: PASS
- synthetic_repeat_metrics_schema: PASS
- synthetic_right_open_nc_retained: PASS
- synthetic_source_state_mapping: PASS
- synthetic_streaming_reference_identity: PASS
- synthetic_total_count_ratio: PASS
- synthetic_transition_total_199: PASS
- synthetic_undefined_metric_encoding: PASS
- validation_aggregate_group_count: PASS
- validation_aggregate_total_count_formulas: PASS
- validation_c_transition_probability_sum: PASS
- validation_composite_source_probability_sum: PASS
- validation_composite_source_relation: PASS
- validation_condition_count_19: PASS
- validation_e_transition_probability_sum: PASS
- validation_end_state_origin_identity: PASS
- validation_full_transition_count: PASS
- validation_h300_c_e_conservation: PASS
- validation_h300_c_to_e_conservation: PASS
- validation_h300_cycle_conservation: PASS
- validation_h300_method_shared_cycle_counts: PASS
- validation_h300_rows_only_h300: PASS
- validation_h300_unique_key: PASS
- validation_h_f_source_conservation: PASS
- validation_hprf_composite_no_n: PASS
- validation_hprf_composite_p_n_zero: PASS
- validation_hprf_source_relation: PASS
- validation_idf_source_relation: PASS
- validation_method_count_5: PASS
- validation_nc_right_open: PASS
- validation_nc_sum_length_conservation: PASS
- validation_ni_aggregate_metric_identity: PASS
- validation_ni_five_method_repeat_metrics_identity: PASS
- validation_repeat_metrics_unique_key: PASS
- validation_state_count_conservation: PASS
- validation_trajectory_arrays_read_only: PASS
- validation_transition_conservation: PASS
- validation_validation_condition_repeat_count: PASS
- validation_validation_h300_row_count: PASS
- validation_validation_repeat_metrics_row_count: PASS

## Blocking issues

None
