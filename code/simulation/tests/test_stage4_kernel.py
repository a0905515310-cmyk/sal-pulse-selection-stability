from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

import sal_stability_stage1.kernel as kernel
from sal_stability_stage1.events import ForwardedEvents, GuidanceEvents
from sal_stability_stage1.hprf import HEventObservation, HPRFRepeatState
from sal_stability_stage1.kernel import (
    CycleCandidates,
    HPRFCountError,
    HPRFEmptyGateError,
    SOURCE_F,
    SOURCE_G,
    SOURCE_H,
    SOURCE_N,
    build_cycle_candidates,
    gate_bounds,
    in_half_open_gate,
    initialize_method_states,
    selected_source_state_code,
    simulate_condition_repeat,
    update_ref_toa,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.selectors import CandidateView
from sal_stability_stage1.stage3 import (
    SharedPhysicalWorld,
    build_shared_physical_world,
    load_stage3_inputs,
)


@pytest.fixture(scope="session")
def frozen_inputs():
    return load_stage3_inputs()


def _event_arrays(
    *,
    true_toa_s: np.ndarray,
    observed_toa_s: np.ndarray | None = None,
    observed_width_s: np.ndarray | None = None,
    tie_base: int = 1000,
    forwarded: bool = False,
):
    true_toa = np.asarray(true_toa_s, dtype=np.float64)
    if true_toa.shape != (200,):
        raise ValueError("synthetic true TOA must have 200 entries")
    observed_toa = (
        true_toa.copy()
        if observed_toa_s is None
        else np.asarray(observed_toa_s, dtype=np.float64)
    )
    observed_width = (
        np.full(200, 300e-9, dtype=np.float64)
        if observed_width_s is None
        else np.asarray(observed_width_s, dtype=np.float64)
    )
    event_type = ForwardedEvents if forwarded else GuidanceEvents
    return event_type(
        event_index=np.arange(1, 201, dtype=np.int64),
        true_toa_s=true_toa,
        true_width_s=np.full(200, 300e-9, dtype=np.float64),
        observed_toa_s=observed_toa,
        observed_width_s=observed_width,
        tie_rank_hi=np.arange(tie_base, tie_base + 200, dtype=np.uint64),
        tie_rank_lo=np.arange(200, dtype=np.uint64),
    )


def _synthetic_world(
    *,
    condition_ordinal: int = 0,
    condition_code: str = "NI",
    initial_ref_toa_s: float = 0.0,
    guidance_true_toa_s: np.ndarray | None = None,
    guidance_observed_toa_s: np.ndarray | None = None,
    guidance_observed_width_s: np.ndarray | None = None,
    forwarded_true_toa_s: np.ndarray | None = None,
    forwarded_observed_toa_s: np.ndarray | None = None,
    forwarded_observed_width_s: np.ndarray | None = None,
    hprf_state: HPRFRepeatState | None = None,
) -> SharedPhysicalWorld:
    guidance_true = (
        np.arange(200, dtype=np.float64)
        if guidance_true_toa_s is None
        else np.asarray(guidance_true_toa_s, dtype=np.float64)
    )
    guidance = _event_arrays(
        true_toa_s=guidance_true,
        observed_toa_s=guidance_observed_toa_s,
        observed_width_s=guidance_observed_width_s,
        tie_base=1000,
    )
    forwarded = None
    if forwarded_true_toa_s is not None:
        forwarded = _event_arrays(
            true_toa_s=np.asarray(forwarded_true_toa_s, dtype=np.float64),
            observed_toa_s=forwarded_observed_toa_s,
            observed_width_s=forwarded_observed_width_s,
            tie_base=2000,
            forwarded=True,
        )
    return SharedPhysicalWorld(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        repeat_id=1,
        initial_sync_error_s=np.float64(initial_ref_toa_s),
        initial_ref_toa_s=np.float64(initial_ref_toa_s),
        guidance_events=guidance,
        forwarded_events=forwarded,
        hprf_state=hprf_state,
    )


def _candidate_kwargs(world, ref_toa_s=0.0):
    return {
        "world": world,
        "cycle_index": 1,
        "ref_toa_s": ref_toa_s,
        "gate_half_width_s": 5e-6,
        "sigma_toa_s": 1e-6,
        "sigma_width_s": 1.5e-9,
        "h_cache": {},
    }


def test_gate_is_exactly_ten_microseconds_and_strictly_half_open(frozen_inputs):
    config, reference = frozen_inputs
    _, _, _, half_width, sigma_dt, sigma_dwidth = kernel._kernel_inputs(
        config, reference
    )
    a_s, b_s = gate_bounds(10.0, half_width)
    assert np.float64(2.0) * half_width == np.float64(config["GateFullWidth_s"])
    assert a_s == np.float64(np.float64(10.0) - half_width)
    assert b_s == np.float64(np.float64(10.0) + half_width)
    assert in_half_open_gate(a_s, a_s, b_s)
    assert in_half_open_gate(np.nextafter(b_s, a_s), a_s, b_s)
    assert not in_half_open_gate(b_s, a_s, b_s)
    assert sigma_dt == np.float64(np.sqrt(2.0)) * np.float64(config["SigmaTOA_s"])
    assert sigma_dwidth == np.float64(config["SigmaWidth_s"])


@pytest.mark.parametrize(
    ("true_toa", "observed_toa", "expected_count"),
    (
        (0.0, 100e-6, 1),
        (-5e-6, 100e-6, 1),
        (5e-6, 0.0, 0),
        (6e-6, 0.0, 0),
    ),
)
def test_guidance_membership_uses_true_not_observed_toa(
    true_toa, observed_toa, expected_count
):
    true_values = np.arange(200, dtype=np.float64) + 1.0
    observed_values = true_values.copy()
    true_values[0] = true_toa
    observed_values[0] = observed_toa
    world = _synthetic_world(
        guidance_true_toa_s=true_values,
        guidance_observed_toa_s=observed_values,
    )
    candidates = build_cycle_candidates(**_candidate_kwargs(world))
    assert len(candidates.view) == expected_count
    if expected_count:
        assert candidates.view.observed_toa_s[0] == observed_toa
        assert candidates.source_state_code[0] == SOURCE_G


@pytest.mark.parametrize(
    ("true_toa", "observed_toa", "expected_count"),
    (
        (0.0, 100e-6, 1),
        (-5e-6, 100e-6, 1),
        (5e-6, 0.0, 0),
        (6e-6, 0.0, 0),
    ),
)
def test_forwarded_membership_uses_true_not_observed_toa(
    true_toa, observed_toa, expected_count
):
    guidance_true = np.arange(200, dtype=np.float64) + 1.0
    forwarded_true = np.arange(200, dtype=np.float64) + 2.0
    forwarded_observed = forwarded_true.copy()
    forwarded_true[0] = true_toa
    forwarded_observed[0] = observed_toa
    world = _synthetic_world(
        condition_ordinal=6,
        condition_code="F1",
        guidance_true_toa_s=guidance_true,
        forwarded_true_toa_s=forwarded_true,
        forwarded_observed_toa_s=forwarded_observed,
    )
    candidates = build_cycle_candidates(**_candidate_kwargs(world))
    assert len(candidates.view) == expected_count
    if expected_count:
        assert candidates.view.observed_toa_s[0] == observed_toa
        assert candidates.source_state_code[0] == SOURCE_F


def test_h_membership_uses_index_resolver_and_never_observed_toa_regating(monkeypatch):
    guidance_true = np.arange(200, dtype=np.float64) + 1.0
    h_state = HPRFRepeatState(
        frequency_hz=np.float64(200_000.0),
        period_s=np.float64(5e-6),
        phase_s=np.float64(2.5e-6),
        true_width_level=np.uint8(5),
        true_width_s=np.float64(300e-9),
    )
    world = _synthetic_world(guidance_true_toa_s=guidance_true, hprf_state=h_state)

    def observed_far_outside(**kwargs):
        n = kwargs["event_index_n"]
        return HEventObservation(
            event_index=n,
            true_toa_s=np.float64(h_state.phase_s + n * h_state.period_s),
            true_width_s=h_state.true_width_s,
            observed_toa_s=np.float64(0.1 + n * 1e-3),
            observed_width_s=np.float64(300e-9),
            tie_rank_hi=np.uint64(3000 + n),
            tie_rank_lo=np.uint64(4000 + n),
        )

    monkeypatch.setattr(kernel, "get_h_event_observation", observed_far_outside)
    candidates = build_cycle_candidates(**_candidate_kwargs(world))
    assert candidates.event_index.tolist() == [-1, 0]
    assert candidates.source_state_code.tolist() == [2, 2]
    assert np.all(np.abs(candidates.view.observed_toa_s) > 5e-6)


def test_candidate_construction_order_is_g_then_signed_h_then_f(monkeypatch):
    guidance_true = np.arange(200, dtype=np.float64) + 1.0
    guidance_true[0] = 0.0
    forwarded_true = np.arange(200, dtype=np.float64) + 2.0
    forwarded_true[0] = 1e-6
    h_state = HPRFRepeatState(
        frequency_hz=np.float64(200_000.0),
        period_s=np.float64(5e-6),
        phase_s=np.float64(2.5e-6),
        true_width_level=np.uint8(5),
        true_width_s=np.float64(300e-9),
    )
    world = _synthetic_world(
        condition_ordinal=10,
        condition_code="HF200-1",
        guidance_true_toa_s=guidance_true,
        forwarded_true_toa_s=forwarded_true,
        hprf_state=h_state,
    )

    def h_observation(**kwargs):
        n = kwargs["event_index_n"]
        return HEventObservation(
            event_index=n,
            true_toa_s=np.float64(h_state.phase_s + n * h_state.period_s),
            true_width_s=h_state.true_width_s,
            observed_toa_s=np.float64(n * 1e-6),
            observed_width_s=h_state.true_width_s,
            tie_rank_hi=np.uint64(3000 + n),
            tie_rank_lo=np.uint64(0),
        )

    monkeypatch.setattr(kernel, "get_h_event_observation", h_observation)
    candidates = build_cycle_candidates(**_candidate_kwargs(world))
    assert candidates.source_state_code.tolist() == [1, 2, 2, 3]
    assert candidates.event_index.tolist() == [1, -1, 0, 1]


def test_source_state_classification_occurs_from_outer_metadata_after_selection():
    empty = CycleCandidates(
        CandidateView([], [], [], []),
        [],
        [],
        np.float64(-5e-6),
        np.float64(5e-6),
    )
    assert selected_source_state_code(empty, None) == SOURCE_N
    candidates = CycleCandidates(
        CandidateView([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1, 2, 3], [0, 0, 0]),
        [SOURCE_G, SOURCE_H, SOURCE_F],
        [1, -7, 1],
        np.float64(-5e-6),
        np.float64(5e-6),
    )
    assert selected_source_state_code(candidates, 0) == SOURCE_G
    assert selected_source_state_code(candidates, 1) == SOURCE_H
    assert selected_source_state_code(candidates, 2) == SOURCE_F


def test_five_method_states_are_independent_and_width_is_not_state(frozen_inputs):
    config, reference = frozen_inputs
    world = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=0,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    states = initialize_method_states(world, config)
    assert len(states) == 5
    assert len({id(state) for state in states}) == 5
    assert all(state.ref_toa_s == world.initial_ref_toa_s for state in states)
    assert [field.name for field in fields(type(states[0]))] == ["ref_toa_s"]


def test_reference_update_is_source_blind_for_selection_and_n():
    signature = inspect.signature(update_ref_toa)
    assert tuple(signature.parameters) == (
        "current_ref_toa_s",
        "selected_observed_toa_s",
        "delta_t_s",
    )
    assert update_ref_toa(10.0, 3.0, 0.5) == np.float64(3.5)
    assert update_ref_toa(10.0, None, 0.5) == np.float64(10.5)
    assert all("source" not in name.lower() for name in signature.parameters)
    assert all("width" not in name.lower() for name in signature.parameters)


def _tracking_world(reference, *, condition_ordinal=6, condition_code="F1"):
    delta = np.asarray(reference["delta_t_s"], dtype=np.float64)
    true_toa = np.empty(200, dtype=np.float64)
    true_toa[0] = 0.0
    np.cumsum(delta, out=true_toa[1:])
    guidance_observed = true_toa.copy()
    guidance_observed[0] = -1e-6
    forwarded_true = true_toa + np.float64(1e-6)
    forwarded_observed = forwarded_true.copy()
    forwarded_observed[0] = 2e-6
    guidance_width = np.asarray(reference["ref_width_s"], dtype=np.float64) + 80e-9
    forwarded_width = np.asarray(reference["ref_width_s"], dtype=np.float64) - 70e-9
    return _synthetic_world(
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        guidance_true_toa_s=true_toa,
        guidance_observed_toa_s=guidance_observed,
        guidance_observed_width_s=guidance_width,
        forwarded_true_toa_s=forwarded_true,
        forwarded_observed_toa_s=forwarded_observed,
        forwarded_observed_width_s=forwarded_width,
    )


def test_one_world_is_built_once_and_first_last_references_naturally_diverge(
    frozen_inputs, monkeypatch
):
    config, reference = frozen_inputs
    world = _tracking_world(reference)
    calls = []

    def one_world_builder(**kwargs):
        calls.append(kwargs)
        return world

    monkeypatch.setattr(kernel, "build_shared_physical_world", one_world_builder)
    trajectory = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    assert len(calls) == 1
    delta_1 = np.float64(reference["delta_t_s"][0])
    assert trajectory.ref_toa_before_s[0, 1] == np.float64(-1e-6) + delta_1
    assert trajectory.ref_toa_before_s[1, 1] == np.float64(2e-6) + delta_1
    assert trajectory.ref_toa_before_s[0, 1] != trajectory.ref_toa_before_s[1, 1]


def test_n_continues_through_all_200_cycles_and_consumes_only_199_deltas(
    frozen_inputs, monkeypatch
):
    config, reference = frozen_inputs
    delta = np.asarray(reference["delta_t_s"], dtype=np.float64)
    true_toa = np.empty(200, dtype=np.float64)
    true_toa[0] = 1.0
    true_toa[1:] = 1.0 + np.cumsum(delta)
    world = _synthetic_world(guidance_true_toa_s=true_toa)
    monkeypatch.setattr(kernel, "build_shared_physical_world", lambda **kwargs: world)
    trajectory = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=0,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    assert trajectory.source_state_code.shape == (5, 200)
    assert trajectory.ref_toa_before_s.shape == (5, 200)
    assert np.all(trajectory.source_state_code == SOURCE_N)
    assert not np.any(trajectory.has_selection)
    assert np.all(np.isnan(trajectory.selected_observed_toa_s))
    assert trajectory.delta_t_consumed_count == 199
    expected_refs = np.empty(200, dtype=np.float64)
    expected_refs[0] = 0.0
    expected_refs[1:] = np.cumsum(delta)
    for method_row in trajectory.ref_toa_before_s:
        np.testing.assert_array_equal(method_row, expected_refs)


def test_real_ni_has_exact_five_method_state_and_reference_identity(frozen_inputs):
    config, reference = frozen_inputs
    trajectory = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=0,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    for method_index in range(1, 5):
        np.testing.assert_array_equal(
            trajectory.source_state_code[0],
            trajectory.source_state_code[method_index],
        )
        np.testing.assert_array_equal(
            trajectory.ref_toa_before_s[0],
            trajectory.ref_toa_before_s[method_index],
        )


@pytest.mark.parametrize("condition_ordinal", tuple(range(1, 6)) + tuple(range(10, 19)))
def test_real_hprf_and_composite_conditions_check_every_method_cycle_and_never_n(
    frozen_inputs, condition_ordinal
):
    config, reference = frozen_inputs
    trajectory = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=condition_ordinal,
        repeat_id=2,
        study_config=config,
        encoding_reference=reference,
    )
    assert trajectory.hprf_count_check_count == 5 * 200
    assert trajectory.hprf_no_n_check_count == 5 * 200
    assert np.all(trajectory.source_state_code != SOURCE_N)


def test_wrong_h_count_is_a_hard_stop(frozen_inputs, monkeypatch):
    config, reference = frozen_inputs
    world = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=2,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    monkeypatch.setattr(kernel, "hprf_index_range", lambda *args: (0, -1))
    with pytest.raises(HPRFCountError, match="per-method per-cycle"):
        kernel._simulate_shared_world(
            world=world,
            study_config=config,
            encoding_reference=reference,
        )


def test_hprf_n_is_a_hard_stop_even_after_count_passes(frozen_inputs, monkeypatch):
    config, reference = frozen_inputs
    world = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=2,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    monkeypatch.setattr(kernel, "selected_source_state_code", lambda *args: SOURCE_N)
    with pytest.raises(HPRFEmptyGateError, match="produced N"):
        kernel._simulate_shared_world(
            world=world,
            study_config=config,
            encoding_reference=reference,
        )


def test_reference_width_is_read_from_encoding_each_cycle_not_selected_width(
    frozen_inputs, monkeypatch
):
    config, reference = frozen_inputs
    world = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=2,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    captured_w_references = []
    original = kernel.select_for_method

    def capture(method_code, view, **kwargs):
        if method_code == "W":
            captured_w_references.append(kwargs["ref_width_s"])
        return original(method_code, view, **kwargs)

    monkeypatch.setattr(kernel, "select_for_method", capture)
    kernel._simulate_shared_world(
        world=world,
        study_config=config,
        encoding_reference=reference,
    )
    np.testing.assert_array_equal(captured_w_references, reference["ref_width_s"])


def test_trajectory_schema_dtype_values_masks_and_read_only_arrays(frozen_inputs):
    config, reference = frozen_inputs
    trajectory = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    assert trajectory.method_codes == ("FIRST", "LAST", "T", "W", "TW")
    assert trajectory.source_state_code.dtype == np.uint8
    assert trajectory.ref_toa_before_s.dtype == np.float64
    assert trajectory.selected_observed_toa_s.dtype == np.float64
    assert trajectory.has_selection.dtype == np.bool_
    assert set(np.unique(trajectory.source_state_code)).issubset({0, 1, 2, 3})
    assert np.array_equal(
        trajectory.has_selection,
        trajectory.source_state_code != SOURCE_N,
    )
    assert all(
        not values.flags.writeable
        for values in (
            trajectory.source_state_code,
            trajectory.ref_toa_before_s,
            trajectory.selected_observed_toa_s,
            trajectory.has_selection,
        )
    )


def test_pilot_and_formal_kernel_worlds_are_isolated_not_forced_identical(frozen_inputs):
    config, reference = frozen_inputs
    pilot = simulate_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    formal = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    assert not np.array_equal(pilot.ref_toa_before_s, formal.ref_toa_before_s)


def test_invalid_method_identity_is_rejected_before_execution(frozen_inputs):
    config, reference = frozen_inputs
    changed = dict(config)
    changed["MethodSpecs"] = [dict(item) for item in config["MethodSpecs"]]
    changed["MethodSpecs"][0]["Code"] = "NOT_FIRST"
    with pytest.raises(ValueError, match="frozen FIRST/LAST/T/W/TW"):
        kernel._kernel_inputs(changed, reference)
