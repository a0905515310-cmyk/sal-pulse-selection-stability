# Stage 10 validation report

## Identity and frozen boundary

- Unique Stage 9 baseline ZIP SHA256: PASS.
- Stage 9 status and Formal input table identities: PASS.
- Stage 10 canonical science-contract SHA256: PASS.
- Direct Stage 9 ZIP versus current Stage 1-9 byte-identity gate: PASS.
- No frozen Stage 1-9 file was modified or deleted outside README/pyproject version boundaries.

## Bootstrap execution integrity

- Namespace: BOOTSTRAP; independent unit: whole Repeat.
- Scale: 18 interference Conditions x 2000 Bootstrap replicates x 2000 draws.
- Workers: 1; vectorization block size: 50; persistent resume unit: Condition.
- Condition outputs reused after strict validation: 0; wholly recomputed: 18.
- Five methods share one index vector for every Condition/BootstrapID.
- Method is absent from RandomAddress; vector Philox and frozen scalar Philox are bitwise exact.
- Rejection sampling acceptance limit: 4294966000.
- Audit index shape: [18, 3, 2000]; primary and independent rerun exact: True.
- NI was checked for 5 x 2000 completeness and identity and was not bootstrapped.

## Statistics and outputs

- Estimator: ratio of resampled sums for all four metrics.
- Undefined denominator handling: NaN plus false mask; no redraw and no zero imputation.
- Paired contrasts: TW-T and TW-W only, with TW minus comparator direction for every metric.
- CI: 95% percentile Bootstrap, NumPy linear 0.025/0.975 quantiles.
- Point estimates: independently reconstructed from the Stage 9 Formal mother table, not Bootstrap means.
- Individual distribution shape: [18, 5, 4, 2000] and summary rows: 360.
- Paired distribution shape: [18, 2, 4, 2000] and summary rows: 144.
- Stage 8 precision audit rows: 216; failure to meet a target was not used as a Stage 10 failure condition.

## Regression boundary

- Full pytest raw result: 404 passed, 2 failed, 0 errors.
- Failed-node set equals the two authorized forward-compatibility boundary nodes exactly.
- After exact deselection: 404 passed, 0 failed, 0 errors.
- Raw failure output is preserved in `test_report.txt`; the report does not relabel full pytest as green.

## Prohibited actions and STOP

- No Stage 9 Formal rerun, physical simulation, p-value, paper figure, paper result conclusion, condition drop, parameter change, outcome-direction PASS rule, or post-Formal contrast addition occurred.
- ProjectedPrecisionWarning remains true.
- Stage 11 was not executed. STOP after automatic Stage 10 packaging.
