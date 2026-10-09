# Stage 6 End-to-End Reproducibility and Execution Validation

## A. Baseline

- Stage 1-5 regression: PASS (baseline 283 tests retained; full suite 316/316 PASS).

## B. Oracle

- Stage 5 oracle generated before any kernel refactor: YES.
- Oracle mode: VERIFY; overwrite prohibited.
- New default path exact array/metrics/H300-strata match: PASS.

## C. Source leakage

- Synthetic CandidateView with externally permuted source metadata left all five selected_pos values unchanged: PASS.

## D. Method order

- ORDER_A, ORDER_B and ORDER_C were exact for H300, F2 and HF300-2, RepeatID 10001..10010: PASS.

## E. Cache

- H cache ON/OFF was exact for H300 and HF300-2, RepeatID 10001..10010: PASS.

## F. Diagnostic

- diagnostic OFF/ON was exact for all three validation conditions, RepeatID 10001..10003: PASS.
- Source metadata was logged only after selected_pos returned.

## G. Parallel

- Serial and spawn-compatible ProcessPool workers=2 were exact for 60 Condition-Repeat units: PASS.
- Reversed completion collection order assembled to the same canonical output: PASS.

## H. Chunk

- ChunkSize 1, 5, 7 and 20 were exact against the clean serial reference: PASS.
- Duplicate/overlapping RepeatID chunks were rejected: PASS.

## I. Resume

- HF300-2 ChunkSize=5 interruption after two of four chunks resumed exactly: PASS.
- Two completed chunks were proven skipped; two missing chunks were computed: PASS.
- A second resume skipped all four valid chunks without computation: PASS.

## J. Corruption

- DataSHA mismatch, Completed=false, CodeVersion mismatch and StudyConfig manifest mismatch each invalidated and recomputed one complete chunk: PASS.

## K. RNG

- rng.py SHA256: 36bedc40407912a68f34a2ccac090549e8b61e3d2c59f255904d34ac3fe1b70b (UNCHANGED).
- rng_test_vectors.csv SHA256: 6f88198eca9ceaf882e0458d87b9986435316715fa91fd09617e0a3878db2274 (UNCHANGED).
- No global, loop-counter or thread RNG state was stored.

## L. Storage

- Data files used temporary-file, close/fsync and os.replace writes; manifest was committed last with Completed=true.
- Identity, individual file SHA256 and aggregate DataSHA256 were validated before reuse.
- chunk_state.npy shape was (5, R_chunk, 200), dtype uint8; full trajectories were retained in an additional NPZ.

## M. Performance

- Serial wall seconds: 32.823694.
- Parallel wall seconds (workers=2): 18.146526.
- Serial seconds per Condition-Repeat: 0.547062.
- Parallel seconds per Condition-Repeat: 0.302442.
- Projected Pilot serial seconds: 2078.834.
- Projected Pilot parallel seconds: 1149.280.
- Projection only; not a scientific result.

## N. Stage boundary

- Stage 7 was NOT executed.
- Pilot R=200 was NOT executed.
- R decision was NOT executed.
- Formal was NOT executed.
- Bootstrap was NOT executed.
- No paper result was generated.
