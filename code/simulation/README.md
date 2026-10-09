# SAL pulse-selection stability — Stage 10

Stage 1 static contract: **PASS**.

Stage 2 counter-based random layer: **PASS**. It uses Philox-4x32-10 and
logical-address deterministic generation.

Retained from Stage 1:

- one machine-readable study configuration;
- 19 physical conditions and five pulse-selection methods;
- SI-unit numeric constants and data-type declarations;
- deterministic 16-stage LFSR state generation;
- 200 reference widths and 199 encoded pulse intervals;
- static validation and active-file retirement scan.

Implemented in Stage 2:

- fixed `PILOT`, `FORMAL`, and `BOOTSTRAP` namespaces;
- 13 frozen variable families;
- immutable five-field `RandomAddress` with signed-int64 `EventIndex`;
- canonical scalar Philox-4x32-10 with cached SHA-256 key derivation;
- fixed uniform and Box-Muller cosine transforms;
- 128-bit TieRank construction and collision-stop helper;
- 28 fixed golden vectors plus order/chunk invariance tests.

Implemented in Stage 3:

- one shared physical world per Condition-Repeat;
- 200 true and observed G events plus one shared initial synchronization;
- optional F events with frozen true delay and independently addressed observations;
- repeat-level HPRF phase and discrete true width;
- on-demand H observations for signed pulse indices, including negative indices;
- strict half-open HPRF interval indexing with eight-ULP integer snapping;
- cache-invariant H event identity and strict 100-500 kHz count checks.

Implemented in Stage 4:

- one source-blind `CandidateView` shared by five frozen selectors;
- FIRST, LAST, T, W, and TW with exact float64 ties resolved by the minimum
  unsigned 128-bit TieRank;
- strict 10 us half-open gates whose G/F membership uses true TOA and whose H
  membership uses the frozen Stage 3 signed-index resolver;
- five independent MethodState references over one shared physical world;
- source-blind reference updates, including continued propagation after N;
- complete 200-cycle Condition-Repeat trajectories with uint8 SourceStateCode;
- per-Method, per-Cycle HPRF count assertions, HPRF/composite no-N assertions,
  and NI trajectory-identity assertions.

Implemented in Stage 5:

- the frozen SourceStateCode mapping `0→N`, `1→C`, and `2/3→E`;
- integer RepeatMetrics rows for every Condition-Method-Repeat, including all
  nine adjacent C/E/N transitions and their three origin denominators;
- streaming and independent reference reconstruction of state, transition,
  source, EndState, and maximal non-correct-run counts;
- retention of a cycle-200 right-open NC run without inventing cycle 201;
- total-count aggregation for P_cor, P_C|C, P_C|E, mean L_NC, and all frozen
  support metrics, with explicit null/reason encoding for zero denominators;
- H/F error-source contributions and H300 integer width-difference strata,
  with C-to-E anchored to the current cycle and no L_NC stratification;
- fixed-ID, 19-condition implementation validation artifacts only.

Implemented in Stage 6:

- behavior-neutral Method execution-order, local H-cache, and diagnostic controls;
- exact default-path verification against a locked pre-refactor Stage 5 oracle;
- canonical serial and Windows-spawn-compatible process-parallel runners whose
  minimum scientific task remains one complete Condition-Repeat;
- RepeatChunk execution with MethodOrdinal-stable assembly and RepeatID sorting;
- atomic chunk data writes with the completed manifest committed last;
- configuration, code-version, code-fingerprint, identity, per-file SHA256, and
  aggregate DataSHA256 validation before a completed chunk can be reused;
- whole-chunk resume and deterministic recomputation of missing, incomplete,
  corrupted, or manifest-incompatible chunks without persisted RNG state;
- exact serial/parallel/chunk-size/resume validation using Stage 6-only
  RepeatID values 10001 and above.

Implemented in Stage 7:

- a machine-readable Stage 8 automatic Formal-R decision contract that is
  locked before the first formal Pilot repeat and verified again afterward;
- an exact serial versus fresh parallel+chunk versus interrupted
  parallel+chunk+resume preflight for H300, F2, and HF300-2;
- a storage identity dedicated to `RunKind=PILOT`, `Namespace=PILOT`, and
  `PilotR=200`, with atomic files, full manifest/hash validation, and
  whole-chunk recomputation;
- the fixed 19-condition, five-method, RepeatID 1..200 Pilot with workers=2,
  ChunkSize=20, resume enabled, diagnostics disabled, and H cache enabled;
- deterministic global RepeatMetrics, total-count-ratio aggregates, H300 width
  strata, and the scoped Stage 8 decision-input CSV;
- hard conservation, NI identity, HPRF/composite no-N, frozen-hash, and
  Stage 7 STOP-boundary validation.

Implemented in Stage 8:

- byte-identity and SHA256 gates for the unique Stage 7 baseline, all frozen
  science/configuration files, the pre-Pilot decision contract, and the
  7200-row Pilot decision input;
- ratio-of-sums estimates and repeat-level influence variances for 144
  Condition-Method-Metric individual precision cells;
- strict RepeatID pairing of T and TW influences for 72 paired precision cells;
- the frozen variance upper-bound multiplier and projected half-width targets
  at R=1000 and R=2000;
- the pre-registered ALL-rule automatic decision, yielding one Formal R for all
  later Conditions and Methods, plus an explicit projected-precision warning;
- independent oracle checks for all cell counts, fail counts, exceedance counts,
  maxima, and both critical H100 P_C|E cells;
- a hard Stage 8 STOP boundary: no FORMAL run, Bootstrap, confidence interval,
  p-value, outcome-direction rule, manual override, or paper result.

Implemented in Stage 9:

- independent SHA256 and byte-boundary verification of the unique Stage 8 ZIP,
  frozen R decision, StudyConfig, decision contract, and decision input;
- a Stage 10 Bootstrap-scope contract locked before any official FORMAL Repeat;
- the fixed `Namespace=FORMAL`, R=2000 execution for all 19 Conditions and five
  Methods using workers=2, ChunkSize=20, resume enabled, diagnostics disabled,
  H cache enabled, and ORDER_A;
- 1900 atomic chunks containing only RepeatMetrics, H300 strata where relevant,
  and snapshots for preregistered RepeatIDs 1, 1000, and 2000;
- strict whole-chunk Resume eligibility based on identity, schema, file SHA256,
  DataSHA256, complete key sets, conservation, NI identity, and no-N gates;
- disk-only global reassembly into 190000 RepeatMetrics rows, 95 aggregate
  point-estimate/support rows, and 30000 H300 width-stratum rows;
- 57 independent diagnostic audit reruns with exact primary/rerun trajectory,
  RepeatMetrics, and H300-stratum equality plus 57000 CycleDiagnostic rows;
- a hard Stage 9 STOP boundary: no Bootstrap, confidence interval, p-value,
  paper figure, condition dropping, parameter change, or outcome-direction gate.

Golden vectors are created only by the explicit one-time command:

```bash
python scripts/build_stage2.py --generate-golden-vectors
```

Normal Stage 2 execution verifies the existing vectors and refuses implicit
replacement:

```bash
python scripts/build_stage2.py
```

Run the Stage 3 physical-event acceptance build:

```bash
python scripts/build_stage3.py
```

Run the Stage 4 selector and 200-cycle-kernel acceptance build:

```bash
python scripts/build_stage4.py
```

Run the Stage 5 metrics and RepeatMetrics acceptance build:

```bash
python scripts/build_stage5.py
```

Run the Stage 6 reproducibility and execution-architecture acceptance build:

```bash
python scripts/build_stage6.py
```

Run the fixed-order Stage 7 Pilot R=200 build:

```bash
python scripts/build_stage7.py
```

Run the Stage 8 automatic Formal-R decision only after the frozen Stage 7 ZIP
is present beside the project directory:

```bash
python scripts/build_stage8.py
```

Run the fixed Stage 9 FORMAL R=2000 build only after the frozen Stage 8 ZIP is
present beside the project directory. The command is resumable at whole-chunk
granularity and stops after producing the independently auditable Stage 9 ZIP:

```bash
python scripts/build_stage9.py
```

Run the fixed Stage 10 B=2000 whole-Repeat paired Bootstrap only from the
frozen Stage 9 ZIP baseline. The command reads the Stage 9 Formal mother table,
uses one shared RepeatID vector for all five methods within each
Condition/BootstrapID, persists complete Condition units, audits the two
authorized historical forward-boundary failures, automatically creates the
complete Stage 10 ZIP, and then stops:

```bash
python scripts/build_stage10.py
```

Stage 10 uses only the `BOOTSTRAP` namespace and rejection-mapped Philox words.
It computes ratio-of-resampled-sums estimates, the pre-registered `TW-T` and
`TW-W` contrasts, and linear 95% percentile intervals. Zero denominators remain
NaN/undefined without redraw. NI is consistency-only. No Formal rerun, physical
simulation, p-value, paper figure, paper conclusion, parameter change, outcome-
direction gate, additional contrast, or Stage 11 execution is permitted.

Run the complete regression suite:

```bash
python -m pytest
```

Stage 1: **PASS**. Stage 2: **PASS**. Stage 3: **PASS**. Stage 4: **PASS**.
Stage 5: **PASS**. Stage 6: **PASS**. Stage 7: **PASS**. Stage 8: **PASS**.
Stage 9 execution status is recorded in `artifacts/stage9/status.json` only after
all 1900 chunks, global conservation gates, and 57 exact audit reruns pass. The
frozen ProjectedPrecisionWarning remains true. Stage 9 does not execute
Bootstrap, confidence intervals, p-values, figures, or paper-result generation.
Stage 10 execution status is recorded in `artifacts/stage10/status.json` only
after all 18 Condition distributions, the independent 108000-index audit,
360/144/216-row output gates, frozen-tree identity checks, and regression gates
pass. Stage 8 precision-target misses are reported but cannot fail Stage 10.

Run Stage 11 only from the frozen `sal_stability_stage10_20260825.zip` beside
the project directory. The command verifies all frozen source identities,
exports complete graph-ready CSV evidence without recomputing inference,
creates the evidence registry and manifest, packages the complete Stage 11
project, validates the ZIP, and then stops:

```bash
python scripts/build_stage11.py
```

Stage 11 performs no Formal rerun, Bootstrap, RNG, new CI, p-value, new
contrast, Condition selection, parameter change, physical simulation, paper
figure rendering, or paper-result conclusion writing. Stage 11 is the final
code stage; no Stage 12 is created or executed.
