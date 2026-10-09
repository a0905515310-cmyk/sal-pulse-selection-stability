# Stage 7 Validation Report

## Baseline gate

- Unique Stage 6 ZIP SHA256: PASS.
- Unmodified Stage 1-6 regression baseline: 316 passed.
- Stage 1-6 artifact status gate: PASS.

## Frozen SHA gate

- All 19 frozen science/configuration file hashes: PASS.
- Science contract changed: no.

## Stage 8 contract lock

- Machine-readable decision contract was locked before Pilot RepeatID 1.
- The post-Pilot SHA256 exactly matches the pre-Pilot SHA256.
- No Stage 8 R decision was executed.

## Combined execution preflight

- Serial reference, fresh parallel+chunk, and interrupted parallel+chunk+resume are exactly equal for H300, F2, and HF300-2.
- Preflight RunKind is VALIDATION and is stored outside the formal Pilot chunk directory.

## Pilot execution completeness and chunk integrity

- Namespace=PILOT, RepeatID=1..200, 19 Conditions, and 5 Methods completed.
- 190/190 chunks passed identity, schema, file-hash, DataSHA256, and full-load validation.
- Resume eligibility requires every frozen identity and data hash; invalid chunks are recomputed as a whole.

## Global row counts and scientific conservation

- pilot_repeat_metrics.csv: 19000 rows.
- pilot_aggregated_metrics.csv: 95 rows.
- pilot_h300_width_strata.csv: 3000 rows.
- stage8_decision_input.csv: 7200 rows.
- State, transition, run-length, and error-source conservation: PASS.
- NI five-method identity: PASS.
- HPRF and Composite N-state assertions: PASS.
- Stage 8 input scope, method scope, RepeatID completeness, and ordering: PASS.

## Tests and performance

- Full pytest: 355 passed, 0 failed.
- Pilot wall time: 1303.438243 seconds.
- Detailed engineering timing is recorded in performance.json.

## STOP status

- Formal execution: no.
- Bootstrap execution: no.
- Paper-result generation: no.
- Stage 8 R decision: no.
- Stop after Stage 7: yes.
