from __future__ import annotations

import inspect

import numpy as np

from sal_stability_stage1.bootstrap import (
    ACCEPTANCE_LIMIT,
    BOOTSTRAP_AUDIT_IDS,
    GOLDEN_REPEAT_IDS,
    bootstrap_event_index,
    bootstrap_random_address,
    bootstrap_repeat_id_scalar,
    bootstrap_repeat_indices,
    map_uint32_words_to_repeat_ids,
    philox_raw_words_vectorized,
    validate_scalar_vector_and_golden_rng,
)
from sal_stability_stage1.rng import (
    RandomNamespace,
    VariableFamily,
    counter_words,
    raw_words,
)


def test_bootstrap_enum_identity_address_layout_and_uniqueness():
    assert int(RandomNamespace.BOOTSTRAP) == 3
    assert int(VariableFamily.BOOTSTRAP_INDEX) == 13
    addresses = [
        bootstrap_random_address(condition, bootstrap, draw, retry)
        for condition in (1, 18)
        for bootstrap in (1, 1000, 2000)
        for draw in (1, 2, 2000)
        for retry in (0, 1)
    ]
    counters = [counter_words(address) for address in addresses]
    assert len(counters) == len(set(counters))
    assert bootstrap_event_index(1, 0) == 0
    assert bootstrap_event_index(2, 0) == 1 << 32
    assert bootstrap_event_index(2000, 1) == (1999 << 32) | 1


def test_vectorized_philox_all_four_words_match_frozen_scalar_with_retries():
    cases = [(1, 1, 1, 0), (3, 71, 123, 1), (18, 2000, 2000, 2)]
    for condition, bootstrap, draw, retry in cases:
        vector = philox_raw_words_vectorized(
            condition,
            np.asarray([bootstrap], dtype=np.uint32),
            np.asarray([draw], dtype=np.uint32),
            np.asarray([retry], dtype=np.uint32),
        )[0]
        scalar = np.asarray(
            raw_words(bootstrap_random_address(condition, bootstrap, draw, retry)),
            dtype=np.uint32,
        )
        assert np.array_equal(vector, scalar)


def test_five_frozen_golden_repeat_id_groups_are_exact_scalar_and_vector():
    report = validate_scalar_vector_and_golden_rng()
    assert report == {
        "ScalarVectorRawWordCheckCount": 50,
        "GoldenVectorGroupCount": 5,
        "GoldenRepeatIDCount": 50,
        "Passed": True,
    }
    for (condition, bootstrap), expected in GOLDEN_REPEAT_IDS.items():
        actual = bootstrap_repeat_indices(condition, [bootstrap], draw_count=10)[0]
        assert tuple(int(value) for value in actual) == expected
        assert tuple(
            bootstrap_repeat_id_scalar(condition, bootstrap, draw)
            for draw in range(1, 11)
        ) == expected


def test_rejection_sampling_mapping_has_no_modulo_without_acceptance():
    words = np.asarray(
        [0, 1, ACCEPTANCE_LIMIT - 1, ACCEPTANCE_LIMIT, (1 << 32) - 1],
        dtype=np.uint64,
    )
    repeat_ids, accepted = map_uint32_words_to_repeat_ids(words)
    assert accepted.tolist() == [True, True, True, False, False]
    assert repeat_ids[:3].tolist() == [1, 2, 2000]
    assert repeat_ids[3:].tolist() == [0, 0]


def test_method_cannot_enter_address_and_audit_primary_rerun_is_exact():
    assert "method" not in " ".join(
        inspect.signature(bootstrap_random_address).parameters
    ).lower()
    assert BOOTSTRAP_AUDIT_IDS == (1, 1000, 2000)
    primary = np.stack(
        [bootstrap_repeat_indices(condition, BOOTSTRAP_AUDIT_IDS) for condition in range(1, 19)]
    )
    rerun = np.stack(
        [bootstrap_repeat_indices(condition, BOOTSTRAP_AUDIT_IDS) for condition in range(1, 19)]
    )
    assert primary.shape == (18, 3, 2000)
    assert primary.dtype == np.uint16
    assert primary.size == 108000
    assert np.array_equal(primary, rerun)
