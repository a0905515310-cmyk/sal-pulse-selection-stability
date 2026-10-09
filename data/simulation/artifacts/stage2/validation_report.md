# Stage 2 Random Layer Validation Report

Verdict: **PASS**

## 1. Current PRNG

Philox-4x32-10. The scalar implementation passed all three official Random123 known-answer vectors.

## 2. Master Seed

- MasterSeedIdentifier: `SAL_STABILITY_MASTER_SEED_V1`
- MasterSeedSHA256: `da7aa37fe1058887dd108bb419c38ba3be40922b088a3586002b1483316bdee4`

## 3. Key Derivation

The two Philox key words are derived only from `MasterSeed + Namespace + VariableFamily` by SHA-256 and are cached per namespace/family pair.

## 4. Counter Layout

`(ConditionOrdinal_u32, RepeatID_u32, EventIndex_low32_twos_complement, EventIndex_high32_twos_complement)`

## 5. Signed EventIndex

`EventIndex=-1` encodes as `Ctr2=0xFFFFFFFF` and `Ctr3=0xFFFFFFFF`. The full signed-int64 boundary tests passed.

## 6. Uniform Transform

`word0` is transformed as `(x+0.5)/2^32`, producing a float64 strictly inside `(0,1)`.

## 7. Normal Transform

`word0 -> u1` and `word1 -> u2`; the returned value is the cosine component of the fixed Box-Muller transform.

## 8. TieRank

`word0/word1 -> TieRankHi` and `word2/word3 -> TieRankLo`. Both halves are unsigned 64-bit integers. A complete 128-bit collision raises and has no fallback ordering.

## 9. Method Isolation

Method is structurally absent from RandomAddress. The physical random APIs accept only a `RandomAddress`; the five future methods therefore cannot alter the address.

## 10. Namespace Isolation

`PILOT != FORMAL != BOOTSTRAP` for otherwise identical address fields; the fixed isolation sample produced three distinct raw blocks.

## 11. Golden Vectors

- Current mode: `VERIFY_GOLDEN_VECTORS`
- Creation record: the file was explicitly created once with `--generate-golden-vectors`; normal execution defaults to verification and refuses implicit replacement.
- Vector count: 28
- All verified: true

## 12. Stage 1 Regression

- Stage 1 tests: 8/8 passed
- Stage 2 tests: 52/52 passed
- Full pytest: 60/60 passed
- Frozen Stage 1 scientific config hash preserved: true

## 13. Static and Invariance Checks

- box_muller_fixed_slots: PASS
- chunk_invariance: PASS
- condition_isolation: PASS
- forbidden_random_api_scan: PASS
- frozen_files_unchanged: PASS
- golden_condition_boundaries: PASS
- golden_count_at_least_20: PASS
- golden_event_examples: PASS
- golden_families_complete: PASS
- golden_large_negative_h_index: PASS
- golden_large_positive_h_index: PASS
- golden_namespaces_complete: PASS
- golden_repeat_examples: PASS
- golden_vectors_verified: PASS
- key_derivation_cached: PASS
- key_excludes_counter_fields: PASS
- legacy_scan: PASS
- method_absent_from_rng_api: PASS
- namespace_ids: PASS
- namespace_isolation: PASS
- negative_event_order_invariance: PASS
- order_invariance: PASS
- other_addresses_do_not_mutate_result: PASS
- philox_official_known_answers: PASS
- philox_round_count_10: PASS
- random_address_fields: PASS
- raw_words_uint32: PASS
- repeat_calls_are_stateless: PASS
- rng_spec_matches_config: PASS
- science_contract_unchanged: PASS
- serial_parallel_invariance: PASS
- signed_event_index: PASS
- signed_event_index_bounds: PASS
- stage1_artifact_status: PASS
- stage1_static_contract: PASS
- tierank_128_layout: PASS
- tierank_collision_stops: PASS
- uniform_open_interval_and_formula: PASS
- variable_family_ids: PASS
- variable_family_isolation: PASS

## 14. Unexecuted Content

Stage 3 not executed.
Pilot not executed.
Formal not executed.
Bootstrap not executed.
No paper result generated.

## 15. Blocking Issues

None
