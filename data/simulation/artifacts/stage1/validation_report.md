# Stage 1 Validation Report

## Verdict

**PASS.** Stage 1 static contract is implemented and validated. No blocking implementation issue was found, and no scientific contract was modified. Execution stops at Stage 1.

## Static contract checks

- Study configuration: one `artifacts/config/study_config.json`.
- Physical conditions: 19, with ordinals 0–18 in the frozen order.
- Methods: 5, ordered FIRST, LAST, T, W, TW.
- Observation cycles: K = 200.
- Reference-width count: 200.
- Encoded-interval count: 199.
- Sum of the 199 encoded intervals: 29.5308825000000 s (float64 representation of the frozen 29.5308825 s value).
- Gate width identity: 10 us = 2 × 5 us.
- LFSR full-period check: 65535 nonzero states and return to the initial state.
- Active machine-file retirement scan: PASS; no retired active identifiers found.
- StudyConfig SHA-256: `e4ad227d9fe619b98a12208b957699a24d51ef5a6533fa6eabf790e9f171f72b`.

## Data types frozen in Stage 1

- Time and frequency quantities: float64.
- Cycle and HPRF pulse indices: signed integer representation (`int64` in this project).
- Source-state code and LFSR bits: uint8.
- LFSR state integer and encoded interval level: uint16.
- Width level: uint8.

## Non-blocking source-version warning

The uploaded file `第7项第二层_四类数值场景具体参数裁决_稳定性主线重构版(2).md` still contains the retired 25-condition branch. This is not used as the active Stage 1 machine definition because the user-provided handoff priority, the later third-layer formal contract, and item 8 explicitly freeze the 19-condition system. No scientific parameter was silently changed in the implementation.

## Deliberately not executed

- Stage 2 random-addressing layer;
- G/H/F physical event generation;
- selectors and 200-cycle physical kernel;
- Pilot;
- Formal simulation;
- Bootstrap;
- paper result figures.
