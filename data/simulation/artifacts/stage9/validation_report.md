# Stage 9 Validation Report

## Identity and frozen-boundary gates

- Stage 8 baseline ZIP SHA256 `a440cae9bbef5474bf30d8abc32078646bf76219910a6a37e31863065e233a3b`: PASS.
- Stage 8 R decision, StudyConfig, decision contract, and decision input frozen SHA256 gates: PASS.
- All 19 independently frozen science/configuration hashes: PASS.
- Stage 1-8 baseline tree byte identity, excluding only the permitted README/pyproject version boundary: PASS.
- Frozen scientific implementation changed: no.

## Preformal lock and execution identity

- Stage 10 Bootstrap scope contract was locked before the first official FORMAL Repeat.
- Namespace=FORMAL, FormalR=2000, RepeatID=1..2000, workers=2, ChunkSize=20, resume=true, diagnostic=false, H cache=true, ORDER_A: PASS.
- The FORMAL representative preflight used validation-only RepeatIDs and was excluded from the statistical dataset.

## Chunk and global data gates

- 1900/1900 chunk manifests passed strict inventory, identity, SHA256, DataSHA256, row-key, and full-load validation.
- `formal_repeat_metrics.csv`: 190000 rows with the exact 19 x 5 x 2000 key universe.
- `formal_aggregated_metrics.csv`: 95 point-estimate/support rows; no Bootstrap interval or p-value fields.
- `formal_h300_width_strata.csv`: 30000 rows; each Method-Repeat conserves 200 cycles across the three frozen strata.
- Repeat, transition, origin, EndState, run-length, and error-source conservation: PASS.
- NI full five-method identity-at-computation and all-C counts: PASS.
- HPRF and Composite no-N gates: PASS.
- Mutable RNG state and full Formal chunk trajectory persistence: absent.

## Preregistered audit

- Primary snapshots: 57/57 for RepeatIDs 1, 1000, and 2000 across all 19 Conditions.
- Independent diagnostic reruns: 57/57; 285 RepeatMetrics and 57000 CycleDiagnostic rows.
- Primary/rerun RepeatMetrics, source state, reference TOA, selected observed TOA (equal_nan), selection mask, and H300 strata: exact PASS.
- Audit reruns were excluded from the formal point-estimate population.

## Regression handling

- Complete pytest raw return code: 1.
- Complete pytest parsed result: 380 passed, 1 failed, 0 errors.
- Authorized exception code: `LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE`.
- Authorized exception node: `tests/test_stage8_decision.py::test_stage1_through_stage7_byte_identity_and_frozen_science_gate`.
- Reason: Stage8 boundary test rejects taskbook-authorized Stage9 additions.
- Failure difference scope: taskbook-authorized Stage9 additions only.
- Frozen Stage 8 modified: False.
- Stage 8 ZIP identity independently verified: True.
- Frozen science hashes independently verified: True.
- Other test failures: 0.
- Legacy external archive exclusions: 0.
- Remaining regression suite: 380 passed, 0 failed, 0 errors.
- Raw complete pytest output, including the one failure, is preserved in `test_report.txt`; this report does not claim full-suite PASS.

## Timing and STOP boundary

- Official Formal invocation wall time: 12512.058801 seconds.
- Independent audit wall time: 39.860264 seconds.
- ProjectedPrecisionWarning remains true; Stage 9 does not claim that it disappeared.
- Bootstrap, confidence intervals, p-values, paper figures, outcome-direction rules, condition dropping, and parameter changes: not executed.
- STOP after Stage 9: yes.
