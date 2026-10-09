import hashlib
import inspect
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

import sal_stability_stage1.rng as rng_module
from sal_stability_stage1.build import (
    ROOT,
    build_rng_spec,
    build_study_config,
    canonical_json_bytes,
    scan_active_files,
    sha256_bytes,
)
from sal_stability_stage1.contracts import METHODS
from sal_stability_stage1.rng import (
    MASK32,
    MASK64,
    MASTER_SEED_BYTES,
    MASTER_SEED_IDENTIFIER,
    MASTER_SEED_SHA256,
    RandomAddress,
    RandomNamespace,
    TieRankCollisionError,
    VariableFamily,
    assert_unique_tieranks,
    counter_words,
    derive_philox_key,
    encode_event_index,
    mulhilo32,
    philox4x32_10,
    random_normal,
    random_tierank,
    random_uniform,
    raw_words,
    uniform_from_uint32,
)
from sal_stability_stage1.stage2 import (
    FROZEN_STAGE1_CONFIG_SHA256,
    GOLDEN_ADDRESSES,
    GOLDEN_FIELDS,
    GOLDEN_PATH,
    generate_golden_vectors,
    golden_coverage_checks,
    read_golden_vectors,
    scan_forbidden_random_apis,
    verify_golden_vectors,
)


def make_address(
    namespace=RandomNamespace.FORMAL,
    condition_ordinal=2,
    repeat_id=1,
    variable_family=VariableFamily.G_TOA_ERROR,
    event_index=1,
):
    return RandomAddress(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        variable_family=variable_family,
        event_index=event_index,
    )


def evaluate(addresses):
    return {address: raw_words(address) for address in addresses}


def fixed_permutation(addresses):
    def key(address):
        payload = (
            f"{int(address.namespace)}:{address.condition_ordinal}:"
            f"{address.repeat_id}:{int(address.variable_family)}:{address.event_index}"
        ).encode("ascii")
        return hashlib.sha256(payload).digest()

    return sorted(addresses, key=key)


def test_namespace_and_variable_family_ids_are_frozen():
    assert {member.name: int(member) for member in RandomNamespace} == {
        "PILOT": 1,
        "FORMAL": 2,
        "BOOTSTRAP": 3,
    }
    assert {member.name: int(member) for member in VariableFamily} == {
        "INITIAL_SYNC": 1,
        "G_TOA_ERROR": 2,
        "G_WIDTH_ERROR": 3,
        "H_PHASE": 4,
        "H_TRUE_WIDTH_LEVEL": 5,
        "H_TOA_ERROR": 6,
        "H_WIDTH_ERROR": 7,
        "F_TOA_ERROR": 8,
        "F_WIDTH_ERROR": 9,
        "TIE_G": 10,
        "TIE_H": 11,
        "TIE_F": 12,
        "BOOTSTRAP_INDEX": 13,
    }


def test_random_address_has_exactly_five_immutable_fields():
    expected = [
        "namespace",
        "condition_ordinal",
        "repeat_id",
        "variable_family",
        "event_index",
    ]
    assert [field.name for field in fields(RandomAddress)] == expected
    address = make_address()
    with pytest.raises(FrozenInstanceError):
        address.repeat_id = 2


@pytest.mark.parametrize(
    ("keyword", "value", "error_type"),
    [
        ("namespace", 0, ValueError),
        ("namespace", True, TypeError),
        ("condition_ordinal", -1, ValueError),
        ("condition_ordinal", 19, ValueError),
        ("condition_ordinal", False, TypeError),
        ("repeat_id", 0, ValueError),
        ("repeat_id", MASK32 + 1, ValueError),
        ("repeat_id", True, TypeError),
        ("variable_family", 0, ValueError),
        ("variable_family", True, TypeError),
        ("event_index", 1 << 63, ValueError),
        ("event_index", -(1 << 63) - 1, ValueError),
        ("event_index", False, TypeError),
    ],
)
def test_random_address_rejects_invalid_fields(keyword, value, error_type):
    arguments = {
        "namespace": RandomNamespace.FORMAL,
        "condition_ordinal": 2,
        "repeat_id": 1,
        "variable_family": VariableFamily.G_TOA_ERROR,
        "event_index": 1,
    }
    arguments[keyword] = value
    with pytest.raises(error_type):
        RandomAddress(**arguments)


def test_repeat_id_is_uint32_not_hard_coded_to_2000():
    assert make_address(repeat_id=2001).repeat_id == 2001
    assert make_address(repeat_id=MASK32).repeat_id == MASK32


def test_mulhilo32_uses_explicit_uint32_halves():
    low, high = mulhilo32(0xFFFFFFFF, 0xFFFFFFFF)
    assert (low, high) == (0x00000001, 0xFFFFFFFE)
    with pytest.raises(ValueError):
        mulhilo32(-1, 1)


@pytest.mark.parametrize(
    ("counter", "key", "expected"),
    [
        (
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0x00000000, 0x00000000),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        ),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_philox_matches_official_random123_known_answers(counter, key, expected):
    assert philox4x32_10(counter, key) == expected


def test_master_seed_and_key_derivation_are_exact_and_cached():
    assert MASTER_SEED_IDENTIFIER == "SAL_STABILITY_MASTER_SEED_V1"
    assert MASTER_SEED_BYTES == hashlib.sha256(
        b"SAL_STABILITY_MASTER_SEED_V1"
    ).digest()
    assert MASTER_SEED_SHA256 == MASTER_SEED_BYTES.hex()
    derive_philox_key.cache_clear()
    expected_digest = hashlib.sha256(
        MASTER_SEED_BYTES
        + int(RandomNamespace.FORMAL).to_bytes(4, "little", signed=False)
        + int(VariableFamily.G_TOA_ERROR).to_bytes(4, "little", signed=False)
    ).digest()
    expected = (
        int.from_bytes(expected_digest[0:4], "little"),
        int.from_bytes(expected_digest[4:8], "little"),
    )
    assert derive_philox_key(
        RandomNamespace.FORMAL, VariableFamily.G_TOA_ERROR
    ) == expected
    assert derive_philox_key(
        RandomNamespace.FORMAL, VariableFamily.G_TOA_ERROR
    ) == expected
    info = derive_philox_key.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_same_address_is_exactly_deterministic_for_100_calls():
    address = make_address(variable_family=VariableFamily.TIE_G, event_index=-1)
    expected = (
        raw_words(address),
        random_uniform(address),
        random_normal(address),
        random_tierank(address),
    )
    for _ in range(100):
        assert raw_words(address) == expected[0]
        assert random_uniform(address) == expected[1]
        assert random_normal(address) == expected[2]
        assert random_tierank(address) == expected[3]


def test_namespaces_are_isolated_for_otherwise_identical_address_fields():
    addresses = [make_address(namespace=namespace) for namespace in RandomNamespace]
    assert len(set(addresses)) == 3
    assert len({raw_words(address) for address in addresses}) == 3
    assert raw_words(addresses[0]) != raw_words(addresses[1])


def test_variable_families_are_key_isolated():
    keys = {
        derive_philox_key(RandomNamespace.FORMAL, family)
        for family in VariableFamily
    }
    blocks = {
        raw_words(make_address(variable_family=family)) for family in VariableFamily
    }
    assert len(keys) == len(VariableFamily)
    assert len(blocks) == len(VariableFamily)
    assert raw_words(
        make_address(variable_family=VariableFamily.INITIAL_SYNC)
    ) != raw_words(make_address(variable_family=VariableFamily.G_TOA_ERROR))


def test_condition_isolation_changes_counter_not_key():
    h100 = make_address(condition_ordinal=1)
    h200 = make_address(condition_ordinal=2)
    assert counter_words(h100) != counter_words(h200)
    assert derive_philox_key(h100.namespace, h100.variable_family) == derive_philox_key(
        h200.namespace, h200.variable_family
    )
    assert raw_words(h100) != raw_words(h200)


def test_method_is_absent_from_address_and_physical_rng_signatures():
    functions = (
        counter_words,
        raw_words,
        random_uniform,
        random_normal,
        random_tierank,
    )
    assert all(list(inspect.signature(function).parameters) == ["address"] for function in functions)
    source = Path(inspect.getsourcefile(rng_module)).read_text(encoding="utf-8").lower()
    assert "method" not in source
    address = make_address(variable_family=VariableFamily.TIE_G)
    expected = (raw_words(address), random_uniform(address), random_normal(address), random_tierank(address))
    for method_order in (METHODS, tuple(reversed(METHODS))):
        for outer_method in method_order:
            assert outer_method.Code in {"FIRST", "LAST", "T", "W", "TW"}
            assert (
                raw_words(address),
                random_uniform(address),
                random_normal(address),
                random_tierank(address),
            ) == expected


@pytest.mark.parametrize(
    ("event_index", "expected"),
    [
        (0, (0x00000000, 0x00000000)),
        (1, (0x00000001, 0x00000000)),
        (-1, (0xFFFFFFFF, 0xFFFFFFFF)),
        (-2, (0xFFFFFFFE, 0xFFFFFFFF)),
        ((1 << 63) - 1, (0xFFFFFFFF, 0x7FFFFFFF)),
        (-(1 << 63), (0x00000000, 0x80000000)),
    ],
)
def test_signed_event_index_two_complement_encoding(event_index, expected):
    assert encode_event_index(event_index) == expected
    address = make_address(event_index=event_index)
    assert counter_words(address)[2:] == expected


@pytest.mark.parametrize("event_index", [1 << 63, -(1 << 63) - 1])
def test_signed_event_index_rejects_out_of_range(event_index):
    with pytest.raises(ValueError):
        encode_event_index(event_index)


def test_raw_words_are_always_uint32_for_golden_addresses():
    for _, address in GOLDEN_ADDRESSES:
        words = raw_words(address)
        assert len(words) == 4
        assert all(0 <= word <= MASK32 for word in words)


def test_uniform_is_open_and_uses_exact_midpoint_formula():
    for _, address in GOLDEN_ADDRESSES:
        words = raw_words(address)
        expected = (words[0] + 0.5) / (1 << 32)
        assert 0.0 < random_uniform(address) < 1.0
        assert random_uniform(address) == expected
    assert 0.0 < uniform_from_uint32(0) < 1.0
    assert 0.0 < uniform_from_uint32(MASK32) < 1.0


def test_box_muller_always_uses_word0_word1_and_cosine_component():
    for _, address in GOLDEN_ADDRESSES:
        words = raw_words(address)
        u1 = (words[0] + 0.5) / (1 << 32)
        u2 = (words[1] + 0.5) / (1 << 32)
        expected = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        assert random_normal(address) == expected


def test_tierank_layout_range_and_unsigned_tuple_order():
    for family in (VariableFamily.TIE_G, VariableFamily.TIE_H, VariableFamily.TIE_F):
        address = make_address(variable_family=family, event_index=-2)
        w0, w1, w2, w3 = raw_words(address)
        expected = ((w0 << 32) | w1, (w2 << 32) | w3)
        assert random_tierank(address) == expected
        assert all(0 <= word <= MASK64 for word in expected)
    ranks = [(1, 5), (0, MASK64), (1, 4)]
    assert sorted(ranks) == [(0, MASK64), (1, 4), (1, 5)]


def test_tie_families_are_isolated_and_collision_stops_without_fallback():
    addresses = [
        make_address(variable_family=family, event_index=17)
        for family in (VariableFamily.TIE_G, VariableFamily.TIE_H, VariableFamily.TIE_F)
    ]
    ranks = [random_tierank(address) for address in addresses]
    assert len(set(ranks)) == 3
    assert_unique_tieranks(ranks)
    with pytest.raises(TieRankCollisionError):
        assert_unique_tieranks((ranks[0], ranks[0]))
    with pytest.raises(ValueError):
        random_tierank(make_address(variable_family=VariableFamily.G_TOA_ERROR))


def test_golden_vector_file_is_complete_and_exactly_regenerable():
    result = verify_golden_vectors(GOLDEN_PATH)
    assert result["passed"], result["issues"]
    assert result["count"] == len(GOLDEN_ADDRESSES) == 28
    assert all(golden_coverage_checks().values())
    header, rows = read_golden_vectors(GOLDEN_PATH)
    assert header == list(GOLDEN_FIELDS)
    assert len({row["VectorID"] for row in rows}) == len(rows)


def test_golden_generation_refuses_to_overwrite_the_regression_contract():
    with pytest.raises(FileExistsError):
        generate_golden_vectors(GOLDEN_PATH)


def test_each_golden_vector_recomputes_every_recorded_value():
    _, rows = read_golden_vectors(GOLDEN_PATH)
    for row in rows:
        address = RandomAddress(
            namespace=RandomNamespace(int(row["NamespaceID"])),
            condition_ordinal=int(row["ConditionOrdinal"]),
            repeat_id=int(row["RepeatID"]),
            variable_family=VariableFamily(int(row["VariableFamilyID"])),
            event_index=int(row["EventIndex"]),
        )
        assert address.namespace.name == row["Namespace"]
        assert address.variable_family.name == row["VariableFamily"]
        counter = counter_words(address)
        key = derive_philox_key(address.namespace, address.variable_family)
        words = raw_words(address)
        assert [int(row[f"Counter{slot}_hex"], 16) for slot in range(4)] == list(counter)
        assert [int(row[f"Key{slot}_hex"], 16) for slot in range(2)] == list(key)
        assert [int(row[f"Raw{slot}_hex"], 16) for slot in range(4)] == list(words)
        u0 = uniform_from_uint32(words[0])
        u1 = uniform_from_uint32(words[1])
        normal = random_normal(address)
        assert float(row["Uniform0"]) == u0
        assert float(row["Uniform1"]) == u1
        assert float(row["NormalZ"]) == normal
        assert float.fromhex(row["Uniform0_hex"]) == u0
        assert float.fromhex(row["Uniform1_hex"]) == u1
        assert float.fromhex(row["NormalZ_hex"]) == normal
        if address.variable_family in {
            VariableFamily.TIE_G,
            VariableFamily.TIE_H,
            VariableFamily.TIE_F,
        }:
            tie_hi, tie_lo = random_tierank(address)
            assert int(row["TieRankHi_hex"], 16) == tie_hi
            assert int(row["TieRankLo_hex"], 16) == tie_lo
        else:
            assert row["TieRankHi_hex"] == ""
            assert row["TieRankLo_hex"] == ""


def test_call_order_invariance_for_1000_addresses():
    addresses = [
        make_address(
            condition_ordinal=index % 19,
            repeat_id=(index % 2000) + 1,
            variable_family=VariableFamily.H_TOA_ERROR,
            event_index=index - 500,
        )
        for index in range(1000)
    ]
    expected = evaluate(addresses)
    assert evaluate(list(reversed(addresses))) == expected
    assert evaluate(fixed_permutation(addresses)) == expected


def test_serial_and_parallel_calls_match_for_every_address():
    addresses = [
        make_address(
            condition_ordinal=index % 19,
            repeat_id=(index % 2000) + 1,
            variable_family=VariableFamily.F_WIDTH_ERROR,
            event_index=index - 500,
        )
        for index in range(1000)
    ]
    serial = [raw_words(address) for address in addresses]
    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = list(executor.map(raw_words, reversed(addresses)))
    assert dict(zip(addresses, serial, strict=True)) == dict(
        zip(reversed(addresses), parallel, strict=True)
    )


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 1000])
def test_chunk_invariance(chunk_size):
    addresses = [
        make_address(
            condition_ordinal=index % 19,
            repeat_id=(index % 2000) + 1,
            variable_family=VariableFamily.H_WIDTH_ERROR,
            event_index=index - 500,
        )
        for index in range(1000)
    ]
    expected = evaluate(addresses)
    actual = {}
    for start in range(0, len(addresses), chunk_size):
        actual.update(evaluate(addresses[start : start + chunk_size]))
    assert actual == expected


def test_repeated_and_intervening_calls_do_not_consume_state():
    address_a = make_address(event_index=-1)
    address_b = make_address(event_index=200)
    first = raw_words(address_a)
    assert raw_words(address_a) == first
    assert raw_words(address_b) != first
    assert raw_words(address_a) == first
    for index in range(1000):
        raw_words(
            make_address(
                condition_ordinal=index % 19,
                repeat_id=index + 1,
                event_index=index - 500,
            )
        )
    assert raw_words(address_a) == first


def test_negative_event_index_generation_order_is_irrelevant():
    addresses = [
        make_address(
            condition_ordinal=3,
            variable_family=VariableFamily.H_TOA_ERROR,
            event_index=event_index,
        )
        for event_index in range(-100, 101)
    ]
    assert evaluate(addresses) == evaluate(list(reversed(addresses)))


def test_study_config_adds_only_rng_spec_to_frozen_stage1_config():
    config = build_study_config()
    assert config["RNGSpec"] == build_rng_spec()
    assert config["RNGSpec"]["MasterSeedIdentifier"] == MASTER_SEED_IDENTIFIER
    assert config["RNGSpec"]["MasterSeedSHA256"] == MASTER_SEED_SHA256
    stage1_view = dict(config)
    stage1_view.pop("RNGSpec")
    assert sha256_bytes(canonical_json_bytes(stage1_view)) == FROZEN_STAGE1_CONFIG_SHA256
    disk_config = json.loads(
        (ROOT / "artifacts" / "config" / "study_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert disk_config == config


def test_active_sources_pass_legacy_and_forbidden_rng_api_scans():
    source_paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
    ]
    active_paths = [ROOT / "artifacts" / "config" / "study_config.json", *source_paths]
    assert scan_active_files(active_paths) == []
    assert scan_forbidden_random_apis(source_paths) == []
