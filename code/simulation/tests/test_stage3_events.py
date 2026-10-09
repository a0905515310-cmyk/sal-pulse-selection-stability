from __future__ import annotations

import inspect
import json
from dataclasses import fields

import numpy as np
import pytest

from sal_stability_stage1.contracts import ConditionSpec
from sal_stability_stage1.events import (
    ForwardedEvents,
    GuidanceEvents,
    build_forwarded_events,
    build_guidance_events,
    build_initial_sync,
    condition_spec_from_config,
    physical_random_address,
)
from sal_stability_stage1.hprf import (
    HEventObservation,
    HPRFRepeatState,
    build_hprf_repeat_state,
    get_h_event_observation,
    hprf_index_range,
    snap_integer_ulp,
)
from sal_stability_stage1.rng import (
    RandomAddress,
    RandomNamespace,
    VariableFamily,
    random_normal,
    random_tierank,
)
from sal_stability_stage1.stage3 import (
    SharedPhysicalWorld,
    _physical_api_has_no_method,
    _write_json,
    build_shared_physical_world,
    load_stage3_inputs,
    shared_worlds_identical,
    validate_stage3_inputs,
)


@pytest.fixture(scope="session")
def frozen_inputs() -> tuple[dict, dict[str, np.ndarray]]:
    return load_stage3_inputs()


@pytest.fixture(scope="session")
def world_factory(frozen_inputs):
    config, reference = frozen_inputs
    cache: dict[tuple[RandomNamespace, int, int], SharedPhysicalWorld] = {}

    def get_world(
        ordinal: int,
        repeat_id: int = 1,
        namespace: RandomNamespace = RandomNamespace.FORMAL,
    ) -> SharedPhysicalWorld:
        key = (namespace, ordinal, repeat_id)
        if key not in cache:
            cache[key] = build_shared_physical_world(
                namespace=namespace,
                condition_ordinal=ordinal,
                repeat_id=repeat_id,
                study_config=config,
                encoding_reference=reference,
            )
        return cache[key]

    return get_world


def test_frozen_stage3_inputs_are_exact(frozen_inputs):
    config, reference = frozen_inputs
    checks = validate_stage3_inputs(config, reference)
    assert checks
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]


def test_machine_report_writer_normalizes_numpy_scalars(tmp_path):
    output = tmp_path / "report.json"
    _write_json(
        output,
        {"passed": np.bool_(True), "count": np.int64(3), "value": np.float64(1.5)},
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "passed": True,
        "count": 3,
        "value": 1.5,
    }


def test_condition_spec_values_come_from_study_config(frozen_inputs):
    config, _ = frozen_inputs
    for ordinal, raw in enumerate(config["ConditionSpecs"]):
        condition = condition_spec_from_config(config, ordinal)
        assert condition.ConditionOrdinal == raw["ConditionOrdinal"]
        assert condition.Code == raw["Code"]
        assert condition.Scene == raw["Scene"]
        assert condition.HPRF_Hz == raw["HPRF_Hz"]
        assert condition.FDelay_s == raw["FDelay_s"]


def test_guidance_has_200_one_based_events_and_required_dtypes(world_factory):
    guidance = world_factory(0).guidance_events
    assert np.array_equal(guidance.event_index, np.arange(1, 201, dtype=np.int64))
    assert guidance.event_index.dtype == np.int64
    for values in (
        guidance.true_toa_s,
        guidance.true_width_s,
        guidance.observed_toa_s,
        guidance.observed_width_s,
    ):
        assert values.shape == (200,)
        assert values.dtype == np.float64
        assert np.all(np.isfinite(values))
    assert guidance.tie_rank_hi.dtype == np.uint64
    assert guidance.tie_rank_lo.dtype == np.uint64


def test_guidance_true_sequence_is_frozen(world_factory, frozen_inputs):
    _, reference = frozen_inputs
    guidance = world_factory(0).guidance_events
    assert guidance.true_toa_s[0] == 0.0
    assert len(reference["delta_t_s"]) == 199
    assert np.allclose(
        np.diff(guidance.true_toa_s),
        reference["delta_t_s"],
        rtol=0.0,
        atol=4e-15,
    )
    assert np.isclose(
        guidance.true_toa_s[-1], 29.5308825, rtol=0.0, atol=2e-14
    )
    assert np.array_equal(guidance.true_width_s, reference["ref_width_s"])
    allowed_widths = 250e-9 + np.arange(16, dtype=np.float64) * 10e-9
    assert np.all(np.isin(guidance.true_width_s, allowed_widths))


@pytest.mark.parametrize("offset", [0, 36, 199])
def test_guidance_observations_use_stage2_addresses(
    offset, world_factory, frozen_inputs
):
    config, _ = frozen_inputs
    world = world_factory(18, repeat_id=17)
    guidance = world.guidance_events
    k = int(guidance.event_index[offset])
    toa_address = physical_random_address(
        RandomNamespace.FORMAL,
        18,
        17,
        VariableFamily.G_TOA_ERROR,
        k,
    )
    width_address = physical_random_address(
        RandomNamespace.FORMAL,
        18,
        17,
        VariableFamily.G_WIDTH_ERROR,
        k,
    )
    expected_toa = guidance.true_toa_s[offset] + float(
        config["SigmaTOA_s"]
    ) * random_normal(toa_address)
    expected_width = guidance.true_width_s[offset] + float(
        config["SigmaWidth_s"]
    ) * random_normal(width_address)
    assert guidance.observed_toa_s[offset] == expected_toa
    assert guidance.observed_width_s[offset] == expected_width


def test_guidance_true_values_do_not_change_with_repeat(world_factory):
    first = world_factory(18, repeat_id=1).guidance_events
    later = world_factory(18, repeat_id=200).guidance_events
    assert np.array_equal(first.true_toa_s, later.true_toa_s)
    assert np.array_equal(first.true_width_s, later.true_width_s)
    assert not np.array_equal(first.observed_toa_s, later.observed_toa_s)
    assert not np.array_equal(first.observed_width_s, later.observed_width_s)


def test_guidance_tieranks_use_one_based_event_addresses(world_factory):
    world = world_factory(7, repeat_id=2)
    offset = 199
    address = physical_random_address(
        RandomNamespace.FORMAL, 7, 2, VariableFamily.TIE_G, 200
    )
    expected_hi, expected_lo = random_tierank(address)
    assert int(world.guidance_events.tie_rank_hi[offset]) == expected_hi
    assert int(world.guidance_events.tie_rank_lo[offset]) == expected_lo


def test_initial_sync_is_one_shared_reference(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    world = world_factory(0, repeat_id=17)
    expected_error = float(config["SigmaSync_s"]) * random_normal(
        physical_random_address(
            RandomNamespace.FORMAL,
            0,
            17,
            VariableFamily.INITIAL_SYNC,
            0,
        )
    )
    assert world.initial_sync_error_s == expected_error
    assert (
        world.initial_ref_toa_s
        == world.guidance_events.true_toa_s[0] + world.initial_sync_error_s
    )
    assert np.isfinite(world.initial_ref_toa_s)


def test_initial_sync_and_first_guidance_error_addresses_are_distinct():
    sync = physical_random_address(
        RandomNamespace.FORMAL, 0, 1, VariableFamily.INITIAL_SYNC, 0
    )
    first_g = physical_random_address(
        RandomNamespace.FORMAL, 0, 1, VariableFamily.G_TOA_ERROR, 1
    )
    assert isinstance(sync, RandomAddress)
    assert sync != first_g
    assert (sync.variable_family, sync.event_index) == (
        VariableFamily.INITIAL_SYNC,
        0,
    )
    assert (first_g.variable_family, first_g.event_index) == (
        VariableFamily.G_TOA_ERROR,
        1,
    )


@pytest.mark.parametrize("ordinal", range(19))
def test_forwarded_presence_for_all_conditions(ordinal, world_factory, frozen_inputs):
    config, _ = frozen_inputs
    expected = config["ConditionSpecs"][ordinal]["Scene"] in ("IDF", "COMPOSITE")
    assert (world_factory(ordinal).forwarded_events is not None) == expected


def test_forwarded_true_relations_for_all_f_conditions_and_repeats(
    world_factory, frozen_inputs
):
    config, _ = frozen_inputs
    for ordinal, raw in enumerate(config["ConditionSpecs"]):
        if raw["Scene"] not in ("IDF", "COMPOSITE"):
            continue
        for repeat_id in (1, 2, 17, 200):
            world = world_factory(ordinal, repeat_id=repeat_id)
            forwarded = world.forwarded_events
            assert forwarded is not None
            assert np.array_equal(
                forwarded.true_toa_s,
                world.guidance_events.true_toa_s + float(raw["FDelay_s"]),
            )
            assert np.array_equal(
                forwarded.true_width_s, world.guidance_events.true_width_s
            )


def test_forwarded_observations_are_independently_addressed(
    world_factory, frozen_inputs
):
    config, _ = frozen_inputs
    world = world_factory(18, repeat_id=17)
    forwarded = world.forwarded_events
    assert forwarded is not None
    offset = 72
    k = int(forwarded.event_index[offset])
    f_toa = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.F_TOA_ERROR, k
    )
    g_toa = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.G_TOA_ERROR, k
    )
    f_width = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.F_WIDTH_ERROR, k
    )
    g_width = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.G_WIDTH_ERROR, k
    )
    assert f_toa != g_toa
    assert f_width != g_width
    assert forwarded.observed_toa_s[offset] == forwarded.true_toa_s[
        offset
    ] + float(config["SigmaTOA_s"]) * random_normal(f_toa)
    assert forwarded.observed_width_s[offset] == forwarded.true_width_s[
        offset
    ] + float(config["SigmaWidth_s"]) * random_normal(f_width)


def test_forwarded_tierank_is_not_guidance_tierank(world_factory):
    world = world_factory(18, repeat_id=17)
    forwarded = world.forwarded_events
    assert forwarded is not None
    k = 101
    g_address = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.TIE_G, k
    )
    f_address = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.TIE_F, k
    )
    assert g_address != f_address
    assert random_tierank(g_address) != random_tierank(f_address)
    assert (
        int(forwarded.tie_rank_hi[k - 1]),
        int(forwarded.tie_rank_lo[k - 1]),
    ) == random_tierank(f_address)


@pytest.mark.parametrize("ordinal", range(19))
def test_hprf_presence_for_all_conditions(ordinal, world_factory, frozen_inputs):
    config, _ = frozen_inputs
    expected = config["ConditionSpecs"][ordinal]["Scene"] in (
        "HPRF",
        "COMPOSITE",
    )
    assert (world_factory(ordinal).hprf_state is not None) == expected


def test_hprf_phase_and_width_are_repeat_level_and_discrete(
    world_factory, frozen_inputs
):
    config, _ = frozen_inputs
    allowed_widths = {float(250e-9 + level * 10e-9) for level in range(16)}
    for ordinal, raw in enumerate(config["ConditionSpecs"]):
        if raw["Scene"] not in ("HPRF", "COMPOSITE"):
            continue
        for repeat_id in (1, 2, 17, 200):
            state = world_factory(ordinal, repeat_id=repeat_id).hprf_state
            assert state is not None
            assert state.frequency_hz == raw["HPRF_Hz"]
            assert state.period_s == 1.0 / float(raw["HPRF_Hz"])
            assert 0.0 < state.phase_s < state.period_s
            assert 0 <= int(state.true_width_level) <= 15
            assert state.true_width_s == 250e-9 + int(state.true_width_level) * 10e-9
            assert float(state.true_width_s) in allowed_widths
            rebuilt = build_hprf_repeat_state(
                namespace=RandomNamespace.FORMAL,
                condition_ordinal=ordinal,
                repeat_id=repeat_id,
                condition=condition_spec_from_config(config, ordinal),
            )
            assert rebuilt == state


def _get_h(world, n, config, cache=None) -> HEventObservation:
    assert world.hprf_state is not None
    return get_h_event_observation(
        namespace=world.namespace,
        condition_ordinal=world.condition_ordinal,
        repeat_id=world.repeat_id,
        event_index_n=n,
        hprf_state=world.hprf_state,
        sigma_toa_s=float(config["SigmaTOA_s"]),
        sigma_width_s=float(config["SigmaWidth_s"]),
        cache=cache,
    )


@pytest.mark.parametrize("n", [-1000, -100, -2, -1, 0, 1, 2, 100, 1000])
def test_signed_h_event_indices_include_negative_values(
    n, world_factory, frozen_inputs
):
    config, _ = frozen_inputs
    world = world_factory(5, repeat_id=17)
    observation = _get_h(world, n, config)
    assert world.hprf_state is not None
    assert observation.event_index == n
    assert isinstance(observation.event_index, int)
    assert observation.true_toa_s == np.float64(
        world.hprf_state.phase_s + np.float64(n) * world.hprf_state.period_s
    )
    assert np.all(
        np.isfinite(
            [
                observation.true_toa_s,
                observation.true_width_s,
                observation.observed_toa_s,
                observation.observed_width_s,
            ]
        )
    )


def test_h_period_relation_across_zero(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    world = world_factory(5, repeat_id=17)
    assert world.hprf_state is not None
    events = [_get_h(world, n, config) for n in (-2, -1, 0, 1)]
    assert np.allclose(
        np.diff([event.true_toa_s for event in events]),
        world.hprf_state.period_s,
        rtol=0.0,
        atol=1e-20,
    )


def test_h_true_width_is_constant_across_indices(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    world = world_factory(18, repeat_id=200)
    events = [_get_h(world, n, config) for n in (-10, -1, 0, 1, 10, 100)]
    assert len({float(event.true_width_s) for event in events}) == 1
    assert all(event.true_width_s == world.hprf_state.true_width_s for event in events)


def test_h_event_identity_and_stage2_addresses(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    world = world_factory(18, repeat_id=17)
    n = -37
    first = _get_h(world, n, config)
    second = _get_h(world, n, config)
    assert first == second
    toa_address = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.H_TOA_ERROR, n
    )
    width_address = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.H_WIDTH_ERROR, n
    )
    tie_address = physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.TIE_H, n
    )
    assert first.observed_toa_s == first.true_toa_s + float(
        config["SigmaTOA_s"]
    ) * random_normal(toa_address)
    assert first.observed_width_s == first.true_width_s + float(
        config["SigmaWidth_s"]
    ) * random_normal(width_address)
    assert (int(first.tie_rank_hi), int(first.tie_rank_lo)) == random_tierank(
        tie_address
    )
    assert toa_address != physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.H_TOA_ERROR, n + 1
    )


def test_h_cache_on_and_off_are_identical(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    world = world_factory(18, repeat_id=17)
    indices = (-1000, -100, -2, -1, 0, 1, 2, 100, 1000)
    uncached = [_get_h(world, n, config) for n in indices]
    cache: dict[int, HEventObservation] = {}
    cached = [_get_h(world, n, config, cache) for n in indices]
    assert uncached == cached
    assert list(cache) == list(indices)
    for n, expected in zip(indices, uncached, strict=True):
        assert _get_h(world, n, config, cache) == expected


def test_hprf_index_range_is_strictly_half_open(world_factory):
    state = world_factory(5, repeat_id=17).hprf_state
    assert state is not None
    n = 23
    a = np.float64(state.phase_s + n * state.period_s)
    b = np.float64(state.phase_s + (n + 5) * state.period_s)
    n_min, n_max = hprf_index_range(a, b, state.phase_s, state.period_s)
    assert (n_min, n_max) == (n, n + 4)
    assert state.phase_s + n_min * state.period_s >= a
    assert state.phase_s + (n_max + 1) * state.period_s == b


def test_hprf_index_range_empty_interval(world_factory):
    state = world_factory(1).hprf_state
    assert state is not None
    n_min, n_max = hprf_index_range(
        state.phase_s, state.phase_s, state.phase_s, state.period_s
    )
    assert n_max < n_min


@pytest.mark.parametrize("steps", [0, 1, 4, 8])
def test_snap_integer_at_or_below_eight_ulp(steps):
    integer = np.float64(2048.0)
    value = integer
    for _ in range(steps):
        value = np.nextafter(value, np.inf)
    assert snap_integer_ulp(value) == integer


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_snap_integer_beyond_eight_ulp_does_not_snap(direction):
    integer = np.float64(2048.0)
    value = integer
    for _ in range(9):
        value = np.nextafter(value, direction)
    assert value != integer
    assert snap_integer_ulp(value) == value


def _property_start(state: HPRFRepeatState, sample_index: int) -> np.float64:
    group, offset = divmod(sample_index, 4)
    if offset == 0:
        return np.float64(state.phase_s + (group + 0.375) * state.period_s)
    if offset == 1:
        return np.float64(state.phase_s - (group + 0.625) * state.period_s)
    if offset == 2:
        integer_x = np.float64(10_000 + group)
        direction = np.inf if group % 2 else -np.inf
        near_integer_x = integer_x
        for _ in range(1 + group % 4):
            near_integer_x = np.nextafter(near_integer_x, direction)
        return np.float64(state.phase_s + near_integer_x * state.period_s)
    return np.float64(
        state.phase_s
        + (100_000_000 + 97 * group + 0.25) * state.period_s
    )


def test_hprf_strict_counts_for_20000_intervals(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    expected_by_frequency = {
        100_000: 1,
        200_000: 2,
        300_000: 3,
        400_000: 4,
        500_000: 5,
    }
    case_count = 0
    for ordinal in range(1, 6):
        expected_count = expected_by_frequency[
            int(config["ConditionSpecs"][ordinal]["HPRF_Hz"])
        ]
        for repeat_id in (1, 2, 17, 200):
            state = world_factory(ordinal, repeat_id=repeat_id).hprf_state
            assert state is not None
            for sample_index in range(1000):
                a = _property_start(state, sample_index)
                b = np.float64(a + 10e-6)
                n_min, n_max = hprf_index_range(
                    a, b, state.phase_s, state.period_s
                )
                assert max(0, n_max - n_min + 1) == expected_count
                case_count += 1
    assert case_count == 20_000


def test_composite_hprf_counts_follow_condition_frequency(
    world_factory, frozen_inputs
):
    config, _ = frozen_inputs
    for ordinal in range(10, 19):
        state = world_factory(ordinal, repeat_id=17).hprf_state
        assert state is not None
        a = np.float64(state.phase_s + 0.321 * state.period_s)
        n_min, n_max = hprf_index_range(
            a, np.float64(a + 10e-6), state.phase_s, state.period_s
        )
        expected = int(round(float(config["ConditionSpecs"][ordinal]["HPRF_Hz"]) * 10e-6))
        assert n_max - n_min + 1 == expected


def test_builders_use_condition_fields_not_code(frozen_inputs):
    config, reference = frozen_inputs
    guidance = build_guidance_events(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=10,
        repeat_id=1,
        encoding_reference=reference,
        sigma_toa_s=float(config["SigmaTOA_s"]),
        sigma_width_s=float(config["SigmaWidth_s"]),
    )
    synthetic = ConditionSpec(
        ConditionOrdinal=10,
        Code="opaque-code",
        Scene="COMPOSITE",
        HPRF_Hz=123_456.0,
        FDelay_s=7e-6,
    )
    forwarded = build_forwarded_events(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=10,
        repeat_id=1,
        condition=synthetic,
        guidance_events=guidance,
        sigma_toa_s=float(config["SigmaTOA_s"]),
        sigma_width_s=float(config["SigmaWidth_s"]),
    )
    state = build_hprf_repeat_state(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=10,
        repeat_id=1,
        condition=synthetic,
    )
    assert forwarded is not None
    assert np.array_equal(forwarded.true_toa_s, guidance.true_toa_s + 7e-6)
    assert state is not None
    assert state.frequency_hz == 123_456.0


def test_shared_physical_world_is_deterministic(frozen_inputs):
    config, reference = frozen_inputs
    first = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=18,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    second = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=18,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    assert shared_worlds_identical(first, second)


def test_pilot_and_formal_share_truth_but_not_random_world(world_factory):
    pilot = world_factory(18, repeat_id=17, namespace=RandomNamespace.PILOT)
    formal = world_factory(18, repeat_id=17, namespace=RandomNamespace.FORMAL)
    assert np.array_equal(
        pilot.guidance_events.true_toa_s, formal.guidance_events.true_toa_s
    )
    assert np.array_equal(
        pilot.guidance_events.true_width_s, formal.guidance_events.true_width_s
    )
    assert np.array_equal(
        pilot.forwarded_events.true_toa_s, formal.forwarded_events.true_toa_s
    )
    assert pilot.hprf_state.frequency_hz == formal.hprf_state.frequency_hz
    assert not np.array_equal(
        pilot.guidance_events.observed_toa_s,
        formal.guidance_events.observed_toa_s,
    )
    assert pilot.initial_sync_error_s != formal.initial_sync_error_s
    assert pilot.hprf_state.phase_s != formal.hprf_state.phase_s
    assert physical_random_address(
        RandomNamespace.PILOT, 18, 17, VariableFamily.H_PHASE, 0
    ) != physical_random_address(
        RandomNamespace.FORMAL, 18, 17, VariableFamily.H_PHASE, 0
    )


def test_bootstrap_cannot_build_a_physical_world(frozen_inputs):
    config, reference = frozen_inputs
    with pytest.raises(ValueError, match="BOOTSTRAP"):
        build_shared_physical_world(
            namespace=RandomNamespace.BOOTSTRAP,
            condition_ordinal=0,
            repeat_id=1,
            study_config=config,
            encoding_reference=reference,
        )


def test_all_physical_entry_points_reject_bootstrap(frozen_inputs):
    config, reference = frozen_inputs
    with pytest.raises(ValueError, match="BOOTSTRAP"):
        build_guidance_events(
            namespace=RandomNamespace.BOOTSTRAP,
            condition_ordinal=0,
            repeat_id=1,
            encoding_reference=reference,
            sigma_toa_s=float(config["SigmaTOA_s"]),
            sigma_width_s=float(config["SigmaWidth_s"]),
        )
    with pytest.raises(ValueError, match="BOOTSTRAP"):
        build_initial_sync(
            namespace=RandomNamespace.BOOTSTRAP,
            condition_ordinal=0,
            repeat_id=1,
            true_first_toa_s=0.0,
            sigma_sync_s=float(config["SigmaSync_s"]),
        )


def test_method_is_structurally_absent_from_physical_generation():
    assert _physical_api_has_no_method()
    functions = (
        build_guidance_events,
        build_initial_sync,
        build_forwarded_events,
        build_hprf_repeat_state,
        get_h_event_observation,
        build_shared_physical_world,
    )
    assert all(
        all("method" not in name.lower() for name in inspect.signature(fn).parameters)
        for fn in functions
    )
    for structure in (
        GuidanceEvents,
        ForwardedEvents,
        HPRFRepeatState,
        HEventObservation,
        SharedPhysicalWorld,
    ):
        assert all("method" not in field.name.lower() for field in fields(structure))


def test_hprf_state_does_not_store_a_pre_generated_event_train(world_factory):
    state = world_factory(5).hprf_state
    assert state is not None
    assert {field.name for field in fields(state)} == {
        "frequency_hz",
        "period_s",
        "phase_s",
        "true_width_level",
        "true_width_s",
    }
    assert all(not isinstance(getattr(state, field.name), np.ndarray) for field in fields(state))


def test_physical_event_arrays_are_read_only(world_factory):
    world = world_factory(18)
    event_arrays = [
        world.guidance_events.event_index,
        world.guidance_events.true_toa_s,
        world.guidance_events.true_width_s,
        world.guidance_events.observed_toa_s,
        world.guidance_events.observed_width_s,
        world.guidance_events.tie_rank_hi,
        world.guidance_events.tie_rank_lo,
        world.forwarded_events.event_index,
        world.forwarded_events.true_toa_s,
        world.forwarded_events.true_width_s,
        world.forwarded_events.observed_toa_s,
        world.forwarded_events.observed_width_s,
        world.forwarded_events.tie_rank_hi,
        world.forwarded_events.tie_rank_lo,
    ]
    assert all(not values.flags.writeable for values in event_arrays)
    with pytest.raises(ValueError):
        world.guidance_events.true_toa_s[0] = 1.0


def test_nonfinite_and_invalid_hprf_inputs_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        snap_integer_ulp(np.inf)
    with pytest.raises(ValueError, match="finite"):
        hprf_index_range(np.nan, 1.0, 0.1, 0.01)
    with pytest.raises(ValueError, match="positive"):
        hprf_index_range(0.0, 1.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="b >= a"):
        hprf_index_range(1.0, 0.0, 0.1, 0.01)


def test_h_event_index_must_be_signed_int64(world_factory, frozen_inputs):
    config, _ = frozen_inputs
    world = world_factory(5)
    with pytest.raises(TypeError, match="integer"):
        _get_h(world, 1.5, config)
    with pytest.raises(ValueError, match="signed int64"):
        _get_h(world, 1 << 63, config)
    with pytest.raises(ValueError, match="signed int64"):
        _get_h(world, -(1 << 63) - 1, config)
