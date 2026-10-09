# Stage 8 Validation Report

## Identity gates

- Unique Stage 7 baseline ZIP SHA256: PASS.
- Stage 7 PASS status and STOP boundary: PASS.
- All 19 frozen science/configuration SHA256 values: PASS at Stage 8 start and end.
- Full Stage 1-7 tree changed only at the explicitly allowed README/pyproject boundary.
- Stage 8 decision contract and decision input SHA256 values: PASS.

## Decision input

- Rows: 7200 (18 Conditions x 2 Methods x 200 Repeats).
- Unique key universe: Condition 1..18, Method T/TW, RepeatID 1..200.
- Independent statistical unit: Repeat.
- Estimator: ratio of sums; P_C_given_E uses N_EC/N_Edot.

## Precision cells and frozen oracle

- Individual precision cells: 144.
- T-TW paired precision cells: 72.
- Individual RequiredR > 1000: 11.
- Paired RequiredR > 1000: 9.
- Individual RequiredR > 2000: 4.
- Paired RequiredR > 2000: 3.
- Maximum individual RequiredR: 41967.
- Maximum paired RequiredR: 42840.
- H100/TW/P_C_given_E and H100 paired/P_C_given_E critical oracles: PASS.

## Automatic decision and prohibited actions

- Frozen ALL-rule R1000 eligibility: false.
- Formal R selected for later stages: 2000.
- ProjectedPrecisionWarning: true.
- Outcome direction, manual override, Condition dropping, and parameter changes: none.
- FORMAL execution, Bootstrap, confidence intervals, p-values, and paper results: not run.
- Full pytest: 372 passed, 0 failed.
- Stop after Stage 8: yes.
