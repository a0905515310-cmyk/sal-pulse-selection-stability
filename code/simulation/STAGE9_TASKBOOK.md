# Stage 9 — FORMAL R=2000 sealed execution contract

## Sole responsibility

Stage 9 executes the frozen Stage 1-8 scientific contract with
`Namespace=FORMAL` and the Stage 8-selected `Rformal=2000` for exactly 19
Conditions and five Methods. It produces RepeatMetrics, H300 width-stratum
support data, preregistered audit trajectories, and aggregate point estimates.
It does not execute Bootstrap, construct confidence intervals, compute p-values,
draw paper figures, or use the direction of any method contrast as a PASS rule.

## Frozen identity

- Stage 8 ZIP SHA256:
  `a440cae9bbef5474bf30d8abc32078646bf76219910a6a37e31863065e233a3b`
- Stage 8 R-decision SHA256:
  `13a8c9ec8a7debb0c0d3c64ac790efb12fa4f444320ec77ff5812c5e7a48c70c`
- StudyConfig SHA256:
  `4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb`
- Stage 8 decision-contract SHA256:
  `1991d8cec0a91438ccd95a5e96ed02cfd4d27285e9e2fa4eba047f9938e285e6`
- Stage 8 decision-input SHA256:
  `655ffd9f03a7a7824662c9a7a1a2281ed620c73019239236e386fb15eac0eb7b`

The decision must state `Rformal=2000`,
`FormalRSource=STAGE8_FROZEN_ALL_RULE`, and
`ProjectedPrecisionWarning=true`, with Formal, Bootstrap, condition dropping,
parameter changes, outcome-direction use, and manual override all false.

## Official execution

- Namespace: `FORMAL` (ID 2)
- RepeatIDs: 1 through 2000
- Conditions: the frozen ordinals 0 through 18
- Methods: FIRST, LAST, T, W, TW in `ORDER_A`
- Workers: 2
- ChunkSize: 20
- Resume: true
- Diagnostic: false in the primary run
- H cache: enabled
- Expected chunks: 1900
- Expected Method-Repeat rows: 190000
- Expected Method-cycles: 38000000

No method identity is part of a RandomAddress, no mutable RNG state is
persisted, and no scientific-core source file is modified.

## Storage and Resume

Every primary chunk stores `chunk_repeat_metrics.csv` and a manifest committed
last. H300 additionally stores `chunk_h300_width_strata.csv`. A chunk containing
RepeatID 1, 1000, or 2000 stores one five-Method trajectory snapshot. Normal
chunks never store full state or trajectory arrays. Resume requires an exact
manifest identity, exact file inventory, per-file SHA256, aggregate DataSHA256,
complete row-key set, conservation, and scenario-specific gates. Any failure
causes whole-chunk recomputation; partial row repair is forbidden.

## Preformal Stage 10 scope lock

Before official FORMAL execution, Stage 9 freezes the later Bootstrap scope:
B=2000, Repeat as the independent unit, the four core metrics, individual
intervals for all five Methods, primary contrast TW-T, secondary ablation TW-W,
interference Conditions 1-18, and NI consistency only. No post-Formal contrast,
condition, or parameter may be added based on outcomes. This lock does not run
Bootstrap and does not claim the Stage 8 half-width target for TW-W.

## Global gates

The disk-reassembled outputs must contain exactly 190000 RepeatMetrics rows, 95
aggregate point-estimate/support rows, and 30000 H300 stratum rows. Keys must be
complete and unique. Every RepeatMetrics row must conserve 200 states and 199
transitions, origin denominators, EndState, non-correct run length, and H/F error
sources. NI must be all-C with full five-Method identity at initial computation.
H100-H500 and all composite HF Conditions must have `N_N=0`. Every H300
Method-Repeat must sum to 200 cycles over the three frozen strata.

## Independent audit

After all chunks pass global validation, RepeatIDs 1, 1000, and 2000 are rerun
for all 19 Conditions using FORMAL, workers=1, diagnostics enabled, H cache
enabled, and ORDER_A. The 285 RepeatMetrics rows, source states, reference TOAs,
selected observed TOAs with equal-NaN semantics, selection masks, and H300 strata
must exactly match the primary data. Exactly 57000 CycleDiagnostic rows are
required, and none enters the formal point-estimate population.

## PASS and STOP

PASS means identity, execution completeness, storage integrity, conservation,
scenario constraints, and exact audit reproducibility all pass. It does not mean
TW outperforms T or W, any interval excludes zero, or projected precision is
adequate. After the complete Stage 9 ZIP is produced, execution stops. Stage 10
requires separate independent acceptance.

## Authorized legacy boundary compatibility amendment

The complete pytest raw result is expected to retain exactly one failure at:

`tests/test_stage8_decision.py::test_stage1_through_stage7_byte_identity_and_frozen_science_gate`

with code `LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE`. This historical test
enumerates the closed Stage 8 delivery boundary and therefore rejects
taskbook-authorized Stage 9 additions. The exception is valid only when the raw
failure is retained, the difference is additions-only and entirely within the
Stage 9 allowlist, the Stage 8 ZIP and all frozen science hashes independently
pass, the nine Stage 9 tests pass, and every other test has zero failures. No
Stage 1-8 source or historical test may be edited, skipped, marked xfail, or
given a Stage 9 whitelist. The exception and all supporting booleans must be
recorded explicitly in the final status, manifest, test report, and validation
report.
