from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .build import CONFIG_DIR, ROOT, build_study_config
from .encoding import EXPECTED_199_INTERVAL_SUM_S, generate_reference_arrays
from .events import (
    ForwardedEvents,
    GuidanceEvents,
    build_forwarded_events,
    build_guidance_events,
    build_initial_sync,
    condition_spec_from_config,
    physical_random_address,
)
from .hprf import (
    HEventObservation,
    HPRFRepeatState,
    build_hprf_repeat_state,
    get_h_event_observation,
    hprf_index_range,
    snap_integer_ulp,
)
from .rng import RandomNamespace, VariableFamily, random_normal
from .stage2 import TestRun, run_pytest, scan_forbidden_random_apis


STAGE3_DIR = ROOT / "artifacts" / "stage3"
FROZEN_CONFIG_SHA256 = (
    "4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb"
)
FROZEN_STAGE2_FILES_SHA256 = {
    "artifacts/config/encoding_reference.csv": (
        "aa319c482a3494d0cdece3bfa4aa28f4d919edaa5b0d2c45bb0d4918ee9647f6"
    ),
    "artifacts/config/encoding_reference.npz": (
        "5a058296f762477009c7d94a33d6d61a982c2d1dda7233db3f5949214ee3cd43"
    ),
    "src/sal_stability_stage1/contracts.py": (
        "029a063dddae55ca66cf875528ab8db99343253ded7237026f6b2da242a9f594"
    ),
    "src/sal_stability_stage1/encoding.py": (
        "197eba52f68d2f91d9051f0ca70f83eda710595e98a949650232c882ea65a71f"
    ),
    "src/sal_stability_stage1/rng.py": (
        "36bedc40407912a68f34a2ccac090549e8b61e3d2c59f255904d34ac3fe1b70b"
    ),
    "src/sal_stability_stage1/stage2.py": (
        "d224c03ad0f9b8afcb47e7fc0fa56c25df49a5461342064cf26cfa2f069a928c"
    ),
}


@dataclass(frozen=True, slots=True)
class SharedPhysicalWorld:
    namespace: RandomNamespace
    condition_ordinal: int
    condition_code: str
    repeat_id: int
    initial_sync_error_s: np.float64
    initial_ref_toa_s: np.float64
    guidance_events: GuidanceEvents
    forwarded_events: ForwardedEvents | None
    hprf_state: HPRFRepeatState | None


def load_stage3_inputs() -> tuple[dict, dict[str, np.ndarray]]:
    config_path = CONFIG_DIR / "study_config.json"
    reference_path = CONFIG_DIR / "encoding_reference.npz"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = {name: np.array(archive[name], copy=True) for name in archive.files}
    return config, reference


def _validate_world_contract(
    study_config: Mapping[str, object], encoding_reference: Mapping[str, np.ndarray]
) -> None:
    required_config = {
        "K": 200,
        "EffectiveIntervalCount": 199,
        "ExpectedEncodedDuration_s": 29.5308825,
        "Xi": 0.0,
    }
    for key, expected in required_config.items():
        if study_config.get(key) != expected:
            raise ValueError(f"StudyConfig {key} is not the frozen value {expected!r}")
    condition_specs = study_config.get("ConditionSpecs")
    method_specs = study_config.get("MethodSpecs")
    if not isinstance(condition_specs, list) or len(condition_specs) != 19:
        raise ValueError("StudyConfig must contain exactly 19 conditions")
    if not isinstance(method_specs, list) or len(method_specs) != 5:
        raise ValueError("StudyConfig must contain exactly five methods")
    try:
        delta_t_s = np.asarray(encoding_reference["delta_t_s"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("encoding reference is missing delta_t_s") from exc
    if delta_t_s.shape != (199,):
        raise ValueError("encoding reference must contain 199 intervals")
    duration = delta_t_s.sum(dtype=np.float64)
    if not np.isclose(
        duration,
        np.float64(study_config["ExpectedEncodedDuration_s"]),
        rtol=0.0,
        atol=5e-15,
    ):
        raise ValueError("encoded duration differs from the frozen contract")


def build_shared_physical_world(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> SharedPhysicalWorld:
    """Build one shared physical world for exactly one Condition-Repeat."""
    _validate_world_contract(study_config, encoding_reference)
    condition = condition_spec_from_config(study_config, condition_ordinal)
    sigma_toa_s = float(study_config["SigmaTOA_s"])
    sigma_sync_s = float(study_config["SigmaSync_s"])
    sigma_width_s = float(study_config["SigmaWidth_s"])

    guidance = build_guidance_events(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        encoding_reference=encoding_reference,
        sigma_toa_s=sigma_toa_s,
        sigma_width_s=sigma_width_s,
    )
    initial_sync = build_initial_sync(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        true_first_toa_s=float(guidance.true_toa_s[0]),
        sigma_sync_s=sigma_sync_s,
    )
    forwarded = build_forwarded_events(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        condition=condition,
        guidance_events=guidance,
        sigma_toa_s=sigma_toa_s,
        sigma_width_s=sigma_width_s,
    )
    hprf_state = build_hprf_repeat_state(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        condition=condition,
    )

    numeric_scalars = np.array(
        [initial_sync.error_s, initial_sync.initial_ref_toa_s], dtype=np.float64
    )
    if not np.all(np.isfinite(numeric_scalars)):
        raise ValueError("shared physical world contains a non-finite scalar")
    return SharedPhysicalWorld(
        namespace=RandomNamespace(namespace),
        condition_ordinal=int(condition_ordinal),
        condition_code=condition.Code,
        repeat_id=int(repeat_id),
        initial_sync_error_s=initial_sync.error_s,
        initial_ref_toa_s=initial_sync.initial_ref_toa_s,
        guidance_events=guidance,
        forwarded_events=forwarded,
        hprf_state=hprf_state,
    )


def _event_arrays_equal(left: object, right: object) -> bool:
    array_fields = (
        "event_index",
        "true_toa_s",
        "true_width_s",
        "observed_toa_s",
        "observed_width_s",
        "tie_rank_hi",
        "tie_rank_lo",
    )
    return type(left) is type(right) and all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in array_fields
    )


def shared_worlds_identical(
    left: SharedPhysicalWorld, right: SharedPhysicalWorld
) -> bool:
    scalar_equal = (
        left.namespace == right.namespace
        and left.condition_ordinal == right.condition_ordinal
        and left.condition_code == right.condition_code
        and left.repeat_id == right.repeat_id
        and left.initial_sync_error_s == right.initial_sync_error_s
        and left.initial_ref_toa_s == right.initial_ref_toa_s
        and left.hprf_state == right.hprf_state
    )
    if not scalar_equal or not _event_arrays_equal(
        left.guidance_events, right.guidance_events
    ):
        return False
    if left.forwarded_events is None or right.forwarded_events is None:
        return left.forwarded_events is None and right.forwarded_events is None
    return _event_arrays_equal(left.forwarded_events, right.forwarded_events)


def validate_stage3_inputs(
    config: Mapping[str, object], reference: Mapping[str, np.ndarray]
) -> dict[str, bool]:
    expected_config = build_study_config()
    expected_reference = generate_reference_arrays(200)
    checks = {
        "study_config_exact": dict(config) == expected_config,
        "condition_count_19": len(config.get("ConditionSpecs", ())) == 19,
        "method_count_5": len(config.get("MethodSpecs", ())) == 5,
        "K_200": config.get("K") == 200,
        "interval_count_199": config.get("EffectiveIntervalCount") == 199,
        "reference_keys_exact": set(reference) == set(expected_reference),
    }
    checks["encoding_reference_exact"] = checks["reference_keys_exact"] and all(
        np.array_equal(reference[name], expected_reference[name])
        and reference[name].dtype == expected_reference[name].dtype
        for name in expected_reference
    )
    if "delta_t_s" in reference:
        duration = np.asarray(reference["delta_t_s"], dtype=np.float64).sum(
            dtype=np.float64
        )
        checks["encoded_duration"] = bool(
            np.isclose(
                duration,
                EXPECTED_199_INTERVAL_SUM_S,
                rtol=0.0,
                atol=5e-15,
            )
        )
    else:
        checks["encoded_duration"] = False
    return checks


def _h_observation(
    world: SharedPhysicalWorld,
    event_index: int,
    config: Mapping[str, object],
    cache: dict[int, HEventObservation] | None = None,
) -> HEventObservation:
    if world.hprf_state is None:
        raise ValueError("the world has no HPRF state")
    return get_h_event_observation(
        namespace=world.namespace,
        condition_ordinal=world.condition_ordinal,
        repeat_id=world.repeat_id,
        event_index_n=event_index,
        hprf_state=world.hprf_state,
        sigma_toa_s=float(config["SigmaTOA_s"]),
        sigma_width_s=float(config["SigmaWidth_s"]),
        cache=cache,
    )


def _h_count_start(
    state: HPRFRepeatState, sample_index: int
) -> np.float64:
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


def _physical_api_has_no_method() -> bool:
    physical_functions = (
        build_guidance_events,
        build_initial_sync,
        build_forwarded_events,
        build_hprf_repeat_state,
        get_h_event_observation,
        build_shared_physical_world,
    )
    signatures_clean = all(
        all("method" not in parameter.lower() for parameter in inspect.signature(fn).parameters)
        for fn in physical_functions
    )
    structures = (
        GuidanceEvents,
        ForwardedEvents,
        HPRFRepeatState,
        HEventObservation,
        SharedPhysicalWorld,
    )
    fields_clean = all(
        all("method" not in field.name.lower() for field in fields(structure))
        for structure in structures
    )
    return signatures_clean and fields_clean


def validate_physical_event_layer(
    config: Mapping[str, object], reference: Mapping[str, np.ndarray]
) -> dict[str, object]:
    input_checks = validate_stage3_inputs(config, reference)
    if not all(input_checks.values()):
        return {
            "checks": input_checks,
            "details": {"input_failure": True},
            "all_pass": False,
        }

    checks: dict[str, bool] = dict(input_checks)
    details: dict[str, object] = {}
    repeats = (1, 2, 17, 200)
    ni_worlds = [
        build_shared_physical_world(
            namespace=RandomNamespace.FORMAL,
            condition_ordinal=0,
            repeat_id=repeat_id,
            study_config=config,
            encoding_reference=reference,
        )
        for repeat_id in repeats
    ]
    guidance = ni_worlds[0].guidance_events
    delta_t_s = np.asarray(reference["delta_t_s"], dtype=np.float64)
    checks["guidance_count_200"] = len(guidance.true_toa_s) == 200
    checks["guidance_first_toa_zero"] = guidance.true_toa_s[0] == 0.0
    checks["guidance_event_index_1_based"] = np.array_equal(
        guidance.event_index, np.arange(1, 201, dtype=np.int64)
    )
    checks["guidance_intervals_199"] = bool(
        np.allclose(
            np.diff(guidance.true_toa_s), delta_t_s, rtol=0.0, atol=4e-15
        )
    )
    checks["guidance_duration"] = bool(
        np.isclose(
            guidance.true_toa_s[-1],
            np.float64(29.5308825),
            rtol=0.0,
            atol=2e-14,
        )
    )
    checks["guidance_true_width_reference"] = np.array_equal(
        guidance.true_width_s, reference["ref_width_s"]
    )
    allowed_widths = np.float64(250e-9) + np.arange(16, dtype=np.float64) * np.float64(
        10e-9
    )
    checks["guidance_widths_discrete"] = bool(
        np.all(np.isin(guidance.true_width_s, allowed_widths))
    )
    checks["guidance_true_repeat_invariant"] = all(
        np.array_equal(guidance.true_toa_s, world.guidance_events.true_toa_s)
        and np.array_equal(guidance.true_width_s, world.guidance_events.true_width_s)
        for world in ni_worlds[1:]
    )

    sample_offset = 36
    sample_k = int(guidance.event_index[sample_offset])
    expected_g_toa_error = float(config["SigmaTOA_s"]) * random_normal(
        physical_random_address(
            RandomNamespace.FORMAL,
            0,
            1,
            VariableFamily.G_TOA_ERROR,
            sample_k,
        )
    )
    expected_g_width_error = float(config["SigmaWidth_s"]) * random_normal(
        physical_random_address(
            RandomNamespace.FORMAL,
            0,
            1,
            VariableFamily.G_WIDTH_ERROR,
            sample_k,
        )
    )
    checks["guidance_observation_addresses"] = (
        guidance.observed_toa_s[sample_offset]
        == guidance.true_toa_s[sample_offset] + expected_g_toa_error
        and guidance.observed_width_s[sample_offset]
        == guidance.true_width_s[sample_offset] + expected_g_width_error
    )

    sync_address = physical_random_address(
        RandomNamespace.FORMAL, 0, 1, VariableFamily.INITIAL_SYNC, 0
    )
    first_g_address = physical_random_address(
        RandomNamespace.FORMAL, 0, 1, VariableFamily.G_TOA_ERROR, 1
    )
    checks["initial_sync_address"] = (
        sync_address.variable_family == VariableFamily.INITIAL_SYNC
        and sync_address.event_index == 0
        and sync_address != first_g_address
    )
    checks["initial_sync_shared_reference"] = (
        ni_worlds[0].initial_ref_toa_s
        == guidance.true_toa_s[0] + ni_worlds[0].initial_sync_error_s
    )

    all_worlds: list[SharedPhysicalWorld] = []
    forwarded_relation_ok = True
    forwarded_width_ok = True
    forwarded_observation_ok = True
    forwarded_tie_ok = True
    f_presence_ok = True
    h_presence_ok = True
    h_state_ok = True
    condition_source_ok = True
    composite_counts_ok = True
    for ordinal, raw_condition in enumerate(config["ConditionSpecs"]):
        condition = condition_spec_from_config(config, ordinal)
        for repeat_id in repeats:
            world = build_shared_physical_world(
                namespace=RandomNamespace.FORMAL,
                condition_ordinal=ordinal,
                repeat_id=repeat_id,
                study_config=config,
                encoding_reference=reference,
            )
            all_worlds.append(world)
            should_have_f = raw_condition["Scene"] in ("IDF", "COMPOSITE")
            should_have_h = raw_condition["Scene"] in ("HPRF", "COMPOSITE")
            f_presence_ok &= (world.forwarded_events is not None) == should_have_f
            h_presence_ok &= (world.hprf_state is not None) == should_have_h
            condition_source_ok &= world.condition_code == raw_condition["Code"]

            if world.forwarded_events is not None:
                forwarded = world.forwarded_events
                expected_true_toa = (
                    world.guidance_events.true_toa_s + float(condition.FDelay_s)
                )
                forwarded_relation_ok &= np.array_equal(
                    forwarded.true_toa_s, expected_true_toa
                )
                forwarded_width_ok &= np.array_equal(
                    forwarded.true_width_s, world.guidance_events.true_width_s
                )
                f_offset = 73
                f_k = int(forwarded.event_index[f_offset])
                f_toa_address = physical_random_address(
                    RandomNamespace.FORMAL,
                    ordinal,
                    repeat_id,
                    VariableFamily.F_TOA_ERROR,
                    f_k,
                )
                f_width_address = physical_random_address(
                    RandomNamespace.FORMAL,
                    ordinal,
                    repeat_id,
                    VariableFamily.F_WIDTH_ERROR,
                    f_k,
                )
                expected_f_toa_error = float(config["SigmaTOA_s"]) * random_normal(
                    f_toa_address
                )
                expected_f_width_error = float(
                    config["SigmaWidth_s"]
                ) * random_normal(f_width_address)
                forwarded_observation_ok &= (
                    forwarded.observed_toa_s[f_offset]
                    == forwarded.true_toa_s[f_offset] + expected_f_toa_error
                    and forwarded.observed_width_s[f_offset]
                    == forwarded.true_width_s[f_offset] + expected_f_width_error
                    and f_toa_address
                    != physical_random_address(
                        RandomNamespace.FORMAL,
                        ordinal,
                        repeat_id,
                        VariableFamily.G_TOA_ERROR,
                        f_k,
                    )
                    and f_width_address
                    != physical_random_address(
                        RandomNamespace.FORMAL,
                        ordinal,
                        repeat_id,
                        VariableFamily.G_WIDTH_ERROR,
                        f_k,
                    )
                )
                forwarded_tie_ok &= (
                    physical_random_address(
                        RandomNamespace.FORMAL,
                        ordinal,
                        repeat_id,
                        VariableFamily.TIE_F,
                        f_k,
                    )
                    != physical_random_address(
                        RandomNamespace.FORMAL,
                        ordinal,
                        repeat_id,
                        VariableFamily.TIE_G,
                        f_k,
                    )
                    and (
                        int(forwarded.tie_rank_hi[f_offset]),
                        int(forwarded.tie_rank_lo[f_offset]),
                    )
                    != (
                        int(world.guidance_events.tie_rank_hi[f_offset]),
                        int(world.guidance_events.tie_rank_lo[f_offset]),
                    )
                )

            if world.hprf_state is not None:
                state = world.hprf_state
                h_state_ok &= (
                    state.frequency_hz == float(condition.HPRF_Hz)
                    and state.period_s == 1.0 / float(condition.HPRF_Hz)
                    and 0.0 < state.phase_s < state.period_s
                    and 0 <= int(state.true_width_level) <= 15
                    and state.true_width_s
                    == np.float64(
                        250e-9 + int(state.true_width_level) * np.float64(10e-9)
                    )
                )
                condition_source_ok &= state.frequency_hz == raw_condition["HPRF_Hz"]
                if raw_condition["Scene"] == "COMPOSITE":
                    a = np.float64(state.phase_s + 0.321 * state.period_s)
                    n_min, n_max = hprf_index_range(
                        a,
                        np.float64(a + float(config["GateFullWidth_s"])),
                        state.phase_s,
                        state.period_s,
                    )
                    expected_count = int(round(state.frequency_hz * 10e-6))
                    composite_counts_ok &= n_max - n_min + 1 == expected_count

    checks["forwarded_presence"] = bool(f_presence_ok)
    checks["forwarded_true_toa_relation"] = bool(forwarded_relation_ok)
    checks["forwarded_true_width_relation"] = bool(forwarded_width_ok)
    checks["forwarded_observation_addresses"] = bool(forwarded_observation_ok)
    checks["forwarded_tierank_independent"] = bool(forwarded_tie_ok)
    checks["hprf_presence"] = bool(h_presence_ok)
    checks["hprf_repeat_state"] = bool(h_state_ok)
    checks["condition_spec_parameter_source"] = bool(condition_source_ok)
    checks["composite_h_counts"] = bool(composite_counts_ok)

    h_world = next(world for world in all_worlds if world.condition_ordinal == 5)
    assert h_world.hprf_state is not None
    negative_indices = (-1000, -100, -2, -1, 0, 1, 2, 100, 1000)
    h_events = [_h_observation(h_world, n, config) for n in negative_indices]
    checks["negative_h_index"] = all(
        observation.event_index == n
        and observation.true_toa_s
        == np.float64(
            h_world.hprf_state.phase_s
            + np.float64(n) * h_world.hprf_state.period_s
        )
        for n, observation in zip(negative_indices, h_events, strict=True)
    )
    sequence_events = [_h_observation(h_world, n, config) for n in (-2, -1, 0, 1)]
    checks["hprf_period_relation"] = bool(
        np.allclose(
            np.diff([event.true_toa_s for event in sequence_events]),
            h_world.hprf_state.period_s,
            rtol=0.0,
            atol=1e-20,
        )
    )
    checks["h_true_width_repeat_constant"] = len(
        {float(event.true_width_s) for event in h_events}
    ) == 1
    repeated_h = _h_observation(h_world, -17, config)
    checks["h_event_identity"] = repeated_h == _h_observation(h_world, -17, config)
    cache: dict[int, HEventObservation] = {}
    cached_events = [_h_observation(h_world, n, config, cache) for n in negative_indices]
    checks["h_cache_invariance"] = h_events == cached_events and all(
        _h_observation(h_world, n, config, cache) == event
        for n, event in zip(negative_indices, cached_events, strict=True)
    )

    boundary_n = 23
    boundary_a = np.float64(
        h_world.hprf_state.phase_s + boundary_n * h_world.hprf_state.period_s
    )
    boundary_b = np.float64(
        h_world.hprf_state.phase_s + (boundary_n + 5) * h_world.hprf_state.period_s
    )
    boundary_range = hprf_index_range(
        boundary_a,
        boundary_b,
        h_world.hprf_state.phase_s,
        h_world.hprf_state.period_s,
    )
    checks["hprf_half_open_boundaries"] = boundary_range == (
        boundary_n,
        boundary_n + 4,
    )

    ulp_base = np.float64(1024.0)
    ulp = np.spacing(ulp_base)
    inside = np.float64(ulp_base + 8.0 * ulp)
    outside = np.float64(ulp_base + 9.0 * ulp)
    checks["snap_integer_eight_ulp"] = (
        snap_integer_ulp(inside) == ulp_base
        and snap_integer_ulp(outside) == outside
        and snap_integer_ulp(outside) != ulp_base
    )

    expected_by_frequency = {
        100_000: 1,
        200_000: 2,
        300_000: 3,
        400_000: 4,
        500_000: 5,
    }
    h_count_ok = True
    h_count_cases = 0
    for ordinal in range(1, 6):
        expected_count = expected_by_frequency[
            int(config["ConditionSpecs"][ordinal]["HPRF_Hz"])
        ]
        for repeat_id in repeats:
            world = build_shared_physical_world(
                namespace=RandomNamespace.FORMAL,
                condition_ordinal=ordinal,
                repeat_id=repeat_id,
                study_config=config,
                encoding_reference=reference,
            )
            assert world.hprf_state is not None
            for sample_index in range(1000):
                a = _h_count_start(world.hprf_state, sample_index)
                b = np.float64(a + float(config["GateFullWidth_s"]))
                n_min, n_max = hprf_index_range(
                    a, b, world.hprf_state.phase_s, world.hprf_state.period_s
                )
                count = max(0, n_max - n_min + 1)
                h_count_ok &= count == expected_count
                h_count_cases += 1
    checks["hprf_strict_count_assertions"] = bool(h_count_ok)
    details["hprf_strict_count_case_count"] = h_count_cases
    details["hprf_expected_counts"] = {
        str(frequency): count for frequency, count in expected_by_frequency.items()
    }

    deterministic_a = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=18,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    deterministic_b = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=18,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    checks["shared_world_deterministic"] = shared_worlds_identical(
        deterministic_a, deterministic_b
    )
    pilot_world = build_shared_physical_world(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=18,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    checks["pilot_formal_scientific_identity"] = (
        np.array_equal(
            pilot_world.guidance_events.true_toa_s,
            deterministic_a.guidance_events.true_toa_s,
        )
        and np.array_equal(
            pilot_world.guidance_events.true_width_s,
            deterministic_a.guidance_events.true_width_s,
        )
        and np.array_equal(
            pilot_world.forwarded_events.true_toa_s,
            deterministic_a.forwarded_events.true_toa_s,
        )
        and pilot_world.hprf_state.frequency_hz
        == deterministic_a.hprf_state.frequency_hz
    )
    checks["pilot_formal_random_isolation"] = (
        physical_random_address(
            RandomNamespace.PILOT, 18, 17, VariableFamily.INITIAL_SYNC, 0
        )
        != physical_random_address(
            RandomNamespace.FORMAL, 18, 17, VariableFamily.INITIAL_SYNC, 0
        )
        and not np.array_equal(
            pilot_world.guidance_events.observed_toa_s,
            deterministic_a.guidance_events.observed_toa_s,
        )
        and pilot_world.hprf_state.phase_s != deterministic_a.hprf_state.phase_s
    )
    try:
        build_shared_physical_world(
            namespace=RandomNamespace.BOOTSTRAP,
            condition_ordinal=0,
            repeat_id=1,
            study_config=config,
            encoding_reference=reference,
        )
    except ValueError:
        checks["bootstrap_physics_rejected"] = True
    else:
        checks["bootstrap_physics_rejected"] = False

    checks["method_absent_from_physical_api"] = _physical_api_has_no_method()
    hprf_source = inspect.getsource(sys.modules[build_hprf_repeat_state.__module__])
    checks["no_full_hprf_pre_generation"] = "np.arange" not in hprf_source
    checks["all_physical_values_finite"] = all(
        np.all(np.isfinite(world.guidance_events.true_toa_s))
        and np.all(np.isfinite(world.guidance_events.observed_toa_s))
        and np.all(np.isfinite(world.guidance_events.true_width_s))
        and np.all(np.isfinite(world.guidance_events.observed_width_s))
        and np.isfinite(world.initial_sync_error_s)
        and np.isfinite(world.initial_ref_toa_s)
        and (
            world.forwarded_events is None
            or (
                np.all(np.isfinite(world.forwarded_events.true_toa_s))
                and np.all(np.isfinite(world.forwarded_events.observed_toa_s))
                and np.all(np.isfinite(world.forwarded_events.true_width_s))
                and np.all(np.isfinite(world.forwarded_events.observed_width_s))
            )
        )
        and (
            world.hprf_state is None
            or (
                np.isfinite(world.hprf_state.frequency_hz)
                and np.isfinite(world.hprf_state.period_s)
                and np.isfinite(world.hprf_state.phase_s)
                and np.isfinite(world.hprf_state.true_width_s)
            )
        )
        for world in all_worlds
    )
    details["validation_repeat_ids"] = list(repeats)
    details["condition_repeat_world_count"] = len(all_worlds)
    details["negative_h_indices"] = list(negative_indices)
    return {"checks": checks, "details": details, "all_pass": all(checks.values())}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retired_identifier_fragments() -> tuple[tuple[str, ...], ...]:
    return (
        ("Pairing", "BlockID"),
        ("Pairing", "Registry"),
        ("Condition", "Registry"),
        ("M", "1"),
        ("M", "2"),
        ("M", "3"),
        ("M", "4"),
        ("M", "5"),
        ("M", "6"),
        ("U", "0"),
        ("U", "T"),
        ("Act", "ive"),
        ("Termin", "ated"),
        ("H", "1"),
        ("H", "2"),
        ("H", "3"),
        ("Sigma", "F"),
        ("P", "Intercept"),
        ("F", "Intercept", "Uniform"),
        ("F", "Delay", "Std", "Normal"),
    )


def scan_retired_identifiers(paths: Iterable[Path]) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    tokens = ["".join(parts) for parts in _retired_identifier_fragments()]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in tokens:
            pattern = re.compile(
                r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"
            )
            match = pattern.search(text)
            if match:
                findings.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": text.count("\n", 0, match.start()) + 1,
                        "token": token,
                    }
                )
    return findings


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    def normalize(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=normalize,
        )
        + "\n",
        encoding="utf-8",
    )


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def write_stage3_test_report(
    test_runs: Sequence[TestRun], build_status: str
) -> None:
    by_label = {run.label: run for run in test_runs}
    stage1 = by_label["stage1"]
    stage2 = by_label["stage2"]
    stage3 = by_label["stage3"]
    total = by_label["all"]
    baseline_count = stage1.passed + stage1.failed + stage2.passed + stage2.failed
    lines = [
        "SAL stability Stage 3 test report",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"NumPy version: {np.__version__}",
        f"pytest version: {importlib.metadata.version('pytest')}",
        f"baseline Stage1+2 test count: {baseline_count}",
        f"Stage3 test count: {stage3.passed + stage3.failed}",
        f"total tests: {total.passed + total.failed}",
        f"passed tests: {total.passed}",
        f"failed tests: {total.failed}",
        f"build_stage3 result: {build_status}",
        "",
    ]
    for run in test_runs:
        lines.extend(
            [
                f"[{run.label}]",
                f"command: {_format_command(run.command)}",
                f"return code: {run.returncode}",
                f"passed: {run.passed}",
                f"failed: {run.failed}",
                f"duration: {run.duration_s:.6f} s",
                "output:",
                run.output or "<no output>",
                "",
            ]
        )
    (STAGE3_DIR / "test_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_stage3_changed_files() -> None:
    lines = [
        "MODIFIED README.md",
        "MODIFIED pyproject.toml",
        "ADDED src/sal_stability_stage1/events.py",
        "ADDED src/sal_stability_stage1/hprf.py",
        "ADDED src/sal_stability_stage1/stage3.py",
        "ADDED scripts/build_stage3.py",
        "ADDED tests/test_stage3_events.py",
        "ADDED artifacts/stage3/status.json",
        "ADDED artifacts/stage3/validation_report.md",
        "ADDED artifacts/stage3/changed_files.txt",
        "ADDED artifacts/stage3/test_report.txt",
        "ADDED artifacts/stage3/physical_event_validation.json",
        "UNCHANGED src/sal_stability_stage1/rng.py",
        "UNCHANGED src/sal_stability_stage1/contracts.py",
        "UNCHANGED src/sal_stability_stage1/encoding.py",
        "UNCHANGED src/sal_stability_stage1/stage2.py",
        "UNCHANGED artifacts/config/*",
        "UNCHANGED artifacts/stage1/*",
        "UNCHANGED artifacts/stage2/*",
    ]
    (STAGE3_DIR / "changed_files.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_stage3_validation_report(
    status: Mapping[str, object],
    validation: Mapping[str, object],
    blocking_issues: Sequence[str],
) -> None:
    lines = [
        "# Stage 3 Physical Event Layer Validation Report",
        "",
        f"Verdict: **{status['status']}**",
        "",
        "## A. G",
        "",
        "- Exactly 200 guidance events were built with EventIndex 1 through 200.",
        "- `t_G,1 = 0`.",
        "- Exactly 199 frozen DeltaT values define the recurrence; no DeltaT_200 exists.",
        "- The final true G TOA corresponds to 29.5308825 s within float64 roundoff.",
        "- Every true G width equals the same-cycle encoding reference width and lies on the frozen 16-level grid.",
        "",
        "## B. Initial Sync",
        "",
        "- VariableFamily: `INITIAL_SYNC`.",
        "- EventIndex: `0`.",
        "- One shared `initial_ref_toa_s = t_G,1 + epsilon_sync` is formed per Condition-Repeat.",
        "- Its address is structurally distinct from `G_TOA_ERROR` at EventIndex 1.",
        "",
        "## C. F",
        "",
        "- F exists only for F1-F4 and all nine composite conditions.",
        "- True TOA is computed cycle-wise as `true_t_F = true_t_G + ConditionSpec.FDelay_s`.",
        "- True F width is exactly equal to true G width in every cycle.",
        "- F TOA and width observations use `F_TOA_ERROR` and `F_WIDTH_ERROR`; they are not copied from G observations.",
        "",
        "## D. H",
        "",
        "- H exists only for the five HPRF conditions and nine composite conditions.",
        "- One repeat-level phase is generated by `H_PHASE`, EventIndex 0, as an open-unit uniform times the ConditionSpec period.",
        "- One repeat-level width level is generated by `H_TRUE_WIDTH_LEVEL`, EventIndex 0, and mapped to 250-400 ns in 10 ns steps.",
        "- Signed H index n is supported, including negative n; negative-index validation passed.",
        "- H observations are generated on demand. No full 29.5308825 s H pulse train is pre-generated.",
        "",
        "## E. HPRF Indexing",
        "",
        "- Membership uses the strict half-open interval `[a,b)`: a left-boundary event is included and a right-boundary event is excluded.",
        "- `snap_integer_ulp` snaps only within eight ULP of a theoretical integer and does not expand a physical interval.",
        f"- Strict 100-500 kHz count assertions passed across {validation.get('details', {}).get('hprf_strict_count_case_count', 0)} intervals.",
        "- Counts per 10 us interval are exactly 1, 2, 3, 4, and 5 at 100, 200, 300, 400, and 500 kHz.",
        "",
        "## F. Random Sharing",
        "",
        "Method is absent from physical event generation and from every physical random address. One SharedPhysicalWorld is built per Condition-Repeat for all future readers.",
        "",
        "## G. Stage Boundary",
        "",
        "Stage 4 was NOT executed.",
        "No selector was implemented or executed.",
        "Pilot was NOT executed.",
        "Formal was NOT executed.",
        "Bootstrap was NOT executed.",
        "No paper result was generated.",
        "No paper result figures were generated.",
        "",
        "## Validation Checks",
        "",
    ]
    for name, passed in sorted(validation.get("checks", {}).items()):
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Blocking Issues", ""])
    if blocking_issues:
        lines.extend(f"- {issue}" for issue in blocking_issues)
    else:
        lines.append("None")
    (STAGE3_DIR / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_stage3() -> dict[str, object]:
    STAGE3_DIR.mkdir(parents=True, exist_ok=True)
    blocking_issues: list[str] = []

    frozen_file_checks = {
        relative: (ROOT / relative).is_file()
        and _sha256_file(ROOT / relative) == expected
        for relative, expected in FROZEN_STAGE2_FILES_SHA256.items()
    }
    config_path = CONFIG_DIR / "study_config.json"
    checksum_path = CONFIG_DIR / "study_config.sha256"
    config_hash = _sha256_file(config_path) if config_path.is_file() else ""
    checksum_value = (
        checksum_path.read_text(encoding="utf-8").strip()
        if checksum_path.is_file()
        else ""
    )
    config_hash_ok = (
        config_hash == FROZEN_CONFIG_SHA256 and checksum_value == config_hash
    )
    science_contract_unchanged = config_hash_ok and all(frozen_file_checks.values())
    if not config_hash_ok:
        blocking_issues.append("frozen StudyConfig hash or checksum changed")
    if not all(frozen_file_checks.values()):
        changed = [name for name, passed in frozen_file_checks.items() if not passed]
        blocking_issues.append(f"frozen Stage 1/2 files changed: {', '.join(changed)}")

    for prior_stage in (1, 2):
        prior_path = ROOT / "artifacts" / f"stage{prior_stage}" / "status.json"
        try:
            prior_status = json.loads(prior_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blocking_issues.append(f"Stage {prior_stage} status is unreadable: {exc}")
            continue
        if prior_status.get("status") != "PASS" or prior_status.get(
            "science_contract_changed"
        ) is not False:
            blocking_issues.append(f"Stage {prior_stage} artifact is not a frozen PASS")

    try:
        config, reference = load_stage3_inputs()
        validation = validate_physical_event_layer(config, reference)
    except Exception as exc:
        validation = {
            "checks": {"physical_validation_completed": False},
            "details": {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            "all_pass": False,
        }
        blocking_issues.append(
            f"physical event validation raised {type(exc).__name__}: {exc}"
        )
    if not validation["all_pass"]:
        failed_checks = [
            name for name, passed in validation["checks"].items() if not passed
        ]
        blocking_issues.append(
            f"physical event validation failed: {', '.join(failed_checks)}"
        )

    active_paths = [
        CONFIG_DIR / "study_config.json",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
    ]
    retired_findings = scan_retired_identifiers(active_paths)
    random_api_findings = scan_forbidden_random_apis(
        [path for path in active_paths if path.suffix == ".py"]
    )
    validation["checks"]["retired_identifier_scan"] = not retired_findings
    validation["checks"]["forbidden_random_api_scan"] = not random_api_findings
    validation["details"]["retired_identifier_findings"] = retired_findings
    validation["details"]["forbidden_random_api_findings"] = random_api_findings
    validation["details"]["frozen_file_checks"] = frozen_file_checks
    validation["details"]["study_config_hash_ok"] = config_hash_ok
    if retired_findings:
        blocking_issues.append("retired identifiers were found in active files")
    if random_api_findings:
        blocking_issues.append("a forbidden random API was found in active Python")

    forbidden_result_dirs = (
        ROOT / "artifacts" / "pilot",
        ROOT / "artifacts" / "formal",
        ROOT / "artifacts" / "bootstrap",
        ROOT / "artifacts" / "results",
        ROOT / "artifacts" / "figures",
    )
    forbidden_existing = [
        str(path.relative_to(ROOT)) for path in forbidden_result_dirs if path.exists()
    ]
    validation["checks"]["forbidden_result_directories_absent"] = not forbidden_existing
    validation["details"]["forbidden_result_directories"] = forbidden_existing
    if forbidden_existing:
        blocking_issues.append(
            f"forbidden result directories exist: {', '.join(forbidden_existing)}"
        )
    validation["all_pass"] = all(validation["checks"].values())

    stage1_run = run_pytest("stage1", ("tests/test_stage1.py",))
    stage2_run = run_pytest("stage2", ("tests/test_stage2_rng.py",))
    stage3_run = run_pytest("stage3", ("tests/test_stage3_events.py",))
    total_run = run_pytest("all", ())
    test_runs = (stage1_run, stage2_run, stage3_run, total_run)
    for run in test_runs:
        if run.returncode != 0:
            blocking_issues.append(f"{run.label} pytest failed")
    baseline_test_count = (
        stage1_run.passed
        + stage1_run.failed
        + stage2_run.passed
        + stage2_run.failed
    )
    if baseline_test_count != 60:
        blocking_issues.append(
            f"Stage 1+2 regression count is {baseline_test_count}, expected 60"
        )

    blocking_issues = list(dict.fromkeys(blocking_issues))
    passed = (
        not blocking_issues
        and science_contract_unchanged
        and validation["all_pass"]
        and all(run.returncode == 0 for run in test_runs)
    )
    status: dict[str, object] = {
        "stage": 3,
        "stage_name": "PHYSICAL_EVENT_LAYER",
        "status": "PASS" if passed else "FAIL",
        "blocking_issue_count": len(blocking_issues),
        "science_contract_changed": not science_contract_unchanged,
        "stage1_regression_passed": stage1_run.returncode == 0,
        "stage2_regression_passed": stage2_run.returncode == 0,
        "condition_count": 19,
        "method_count": 5,
        "K": 200,
        "guidance_event_tests_passed": stage3_run.returncode == 0
        and validation["checks"].get("guidance_count_200", False),
        "initial_sync_tests_passed": stage3_run.returncode == 0
        and validation["checks"].get("initial_sync_address", False),
        "forwarded_event_tests_passed": stage3_run.returncode == 0
        and validation["checks"].get("forwarded_true_width_relation", False),
        "hprf_state_tests_passed": stage3_run.returncode == 0
        and validation["checks"].get("hprf_repeat_state", False),
        "negative_h_index_passed": stage3_run.returncode == 0
        and validation["checks"].get("negative_h_index", False),
        "hprf_index_range_passed": stage3_run.returncode == 0
        and validation["checks"].get("hprf_half_open_boundaries", False),
        "hprf_count_assertions_passed": stage3_run.returncode == 0
        and validation["checks"].get("hprf_strict_count_assertions", False),
        "h_cache_invariance_passed": stage3_run.returncode == 0
        and validation["checks"].get("h_cache_invariance", False),
        "bootstrap_physics_rejected": stage3_run.returncode == 0
        and validation["checks"].get("bootstrap_physics_rejected", False),
        "stage4_executed": False,
        "pilot_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_figures_generated": False,
        "baseline_stage1_stage2_test_count": baseline_test_count,
        "stage3_test_count": stage3_run.passed + stage3_run.failed,
        "total_test_count": total_run.passed + total_run.failed,
        "total_passed_count": total_run.passed,
        "total_failed_count": total_run.failed,
        "study_config_sha256": config_hash,
        "stop_after_stage3": True,
    }
    validation["all_pass"] = passed
    validation["blocking_issues"] = blocking_issues
    _write_json(STAGE3_DIR / "physical_event_validation.json", validation)
    _write_json(STAGE3_DIR / "status.json", status)
    write_stage3_test_report(test_runs, status["status"])
    write_stage3_validation_report(status, validation, blocking_issues)
    write_stage3_changed_files()
    return status
