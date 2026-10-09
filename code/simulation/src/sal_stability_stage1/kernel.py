from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .diagnostics import CycleDiagnostic
from .events import condition_spec_from_config
from .hprf import (
    HEventCache,
    get_h_event_observation,
    hprf_index_range,
)
from .rng import RandomNamespace
from .selectors import (
    CandidateView,
    select_first,
    select_last,
    select_t,
    select_tw,
    select_w,
)
from .stage3 import SharedPhysicalWorld, build_shared_physical_world


SOURCE_N = np.uint8(0)
SOURCE_G = np.uint8(1)
SOURCE_H = np.uint8(2)
SOURCE_F = np.uint8(3)
SOURCE_CODES = frozenset((0, 1, 2, 3))

METHOD_IDENTITIES = (
    (0, "FIRST"),
    (1, "LAST"),
    (2, "T"),
    (3, "W"),
    (4, "TW"),
)
DEFAULT_METHOD_EXECUTION_ORDER = (0, 1, 2, 3, 4)

EXPECTED_H_COUNT_BY_FREQUENCY = {
    100_000.0: 1,
    200_000.0: 2,
    300_000.0: 3,
    400_000.0: 4,
    500_000.0: 5,
}


class HPRFCountError(RuntimeError):
    pass


class HPRFEmptyGateError(RuntimeError):
    pass


class NIIdentityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CycleCandidates:
    """Selector view plus aligned source metadata retained only by the kernel."""

    view: CandidateView
    source_state_code: np.ndarray
    event_index: np.ndarray
    gate_a_s: np.float64
    gate_b_s: np.float64
    h_index_min: int | None = None
    h_index_max: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.view, CandidateView):
            raise TypeError("view must be a CandidateView")
        source = _readonly_1d(
            self.source_state_code,
            np.dtype(np.uint8),
            "source_state_code",
        )
        event = _readonly_1d(
            self.event_index,
            np.dtype(np.int64),
            "event_index",
        )
        if len(source) != len(self.view) or len(event) != len(self.view):
            raise ValueError("outer metadata must align exactly with CandidateView")
        if any(int(value) not in (1, 2, 3) for value in source):
            raise ValueError("candidate source_state_code values must be G, H, or F")
        gate_a = _finite_scalar(self.gate_a_s, "gate_a_s")
        gate_b = _finite_scalar(self.gate_b_s, "gate_b_s")
        if gate_b <= gate_a:
            raise ValueError("candidate gate must have positive width")
        if (self.h_index_min is None) != (self.h_index_max is None):
            raise ValueError("H index bounds must either both exist or both be absent")
        if self.h_index_min is not None:
            h_min = _integer(self.h_index_min, "h_index_min")
            h_max = _integer(self.h_index_max, "h_index_max")
            object.__setattr__(self, "h_index_min", h_min)
            object.__setattr__(self, "h_index_max", h_max)
        object.__setattr__(self, "source_state_code", source)
        object.__setattr__(self, "event_index", event)
        object.__setattr__(self, "gate_a_s", gate_a)
        object.__setattr__(self, "gate_b_s", gate_b)

    @property
    def h_count(self) -> int:
        return int(np.count_nonzero(self.source_state_code == SOURCE_H))


@dataclass(slots=True)
class MethodState:
    ref_toa_s: np.float64

    def __post_init__(self) -> None:
        self.ref_toa_s = _finite_scalar(self.ref_toa_s, "ref_toa_s")


@dataclass(frozen=True, slots=True)
class ConditionRepeatTrajectory:
    condition_ordinal: int
    condition_code: str
    repeat_id: int
    namespace: RandomNamespace
    method_codes: tuple[str, ...]
    source_state_code: np.ndarray
    ref_toa_before_s: np.ndarray
    selected_observed_toa_s: np.ndarray
    has_selection: np.ndarray
    delta_t_consumed_count: int
    hprf_count_check_count: int
    hprf_no_n_check_count: int

    def __post_init__(self) -> None:
        condition_ordinal = _integer(self.condition_ordinal, "condition_ordinal")
        repeat_id = _integer(self.repeat_id, "repeat_id")
        try:
            namespace = RandomNamespace(_integer(self.namespace, "namespace"))
        except ValueError as exc:
            raise ValueError("trajectory namespace is invalid") from exc
        method_codes = tuple(str(code) for code in self.method_codes)
        if method_codes != tuple(code for _, code in METHOD_IDENTITIES):
            raise ValueError("trajectory method identity is not the frozen 5-method order")

        source = _readonly_2d(
            self.source_state_code,
            np.dtype(np.uint8),
            (5, 200),
            "source_state_code",
        )
        reference = _readonly_2d(
            self.ref_toa_before_s,
            np.dtype(np.float64),
            (5, 200),
            "ref_toa_before_s",
        )
        selected = _readonly_2d(
            self.selected_observed_toa_s,
            np.dtype(np.float64),
            (5, 200),
            "selected_observed_toa_s",
        )
        has_selection = _readonly_2d(
            self.has_selection,
            np.dtype(np.bool_),
            (5, 200),
            "has_selection",
        )
        if any(int(value) not in SOURCE_CODES for value in source.flat):
            raise ValueError("trajectory source_state_code contains an invalid value")
        if not np.all(np.isfinite(reference)):
            raise ValueError("trajectory reference TOAs must be finite")
        if not np.array_equal(has_selection, source != SOURCE_N):
            raise ValueError("has_selection must agree with source_state_code")
        if np.any(~np.isfinite(selected[has_selection])):
            raise ValueError("selected TOAs must be finite where a selection exists")
        if np.any(~np.isnan(selected[~has_selection])):
            raise ValueError("selected TOAs must be NaN exactly where no selection exists")
        if _integer(self.delta_t_consumed_count, "delta_t_consumed_count") != 199:
            raise ValueError("a 200-cycle trajectory must consume exactly 199 DeltaT values")
        if _integer(self.hprf_count_check_count, "hprf_count_check_count") < 0:
            raise ValueError("hprf_count_check_count cannot be negative")
        if _integer(self.hprf_no_n_check_count, "hprf_no_n_check_count") < 0:
            raise ValueError("hprf_no_n_check_count cannot be negative")

        object.__setattr__(self, "condition_ordinal", condition_ordinal)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "method_codes", method_codes)
        object.__setattr__(self, "source_state_code", source)
        object.__setattr__(self, "ref_toa_before_s", reference)
        object.__setattr__(self, "selected_observed_toa_s", selected)
        object.__setattr__(self, "has_selection", has_selection)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _finite_scalar(value: object, field_name: str) -> np.float64:
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _readonly_1d(values: object, dtype: np.dtype, field_name: str) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    result.setflags(write=False)
    return result


def _readonly_2d(
    values: object,
    dtype: np.dtype,
    shape: tuple[int, int],
    field_name: str,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if result.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    result.setflags(write=False)
    return result


def _method_codes(study_config: Mapping[str, object]) -> tuple[str, ...]:
    raw_specs = study_config.get("MethodSpecs")
    if not isinstance(raw_specs, (list, tuple)) or len(raw_specs) != 5:
        raise ValueError("StudyConfig must contain exactly five MethodSpecs")
    identities: list[tuple[int, str]] = []
    for raw in raw_specs:
        if not isinstance(raw, Mapping):
            raise TypeError("each MethodSpec must be a mapping")
        identities.append(
            (
                _integer(raw.get("MethodOrdinal"), "MethodOrdinal"),
                str(raw.get("Code")),
            )
        )
    if tuple(identities) != METHOD_IDENTITIES:
        raise ValueError("MethodSpecs must be the frozen FIRST/LAST/T/W/TW ordinals")
    return tuple(code for _, code in identities)


def _validated_method_execution_order(
    method_execution_order: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(method_execution_order, (str, bytes)):
        raise TypeError("method_execution_order must be a sequence of ordinals")
    try:
        raw_order = tuple(method_execution_order)
    except TypeError as exc:
        raise TypeError("method_execution_order must be a sequence of ordinals") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in raw_order):
        raise TypeError("method_execution_order ordinals cannot be bool")
    order = tuple(
        _integer(value, "method_execution_order ordinal") for value in raw_order
    )
    if len(order) != len(METHOD_IDENTITIES) or set(order) != set(
        DEFAULT_METHOD_EXECUTION_ORDER
    ):
        raise ValueError("method_execution_order must be a permutation of 0..4")
    return order


def _kernel_inputs(
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.float64, np.float64, np.float64]:
    required = {
        "K": 200,
        "EffectiveIntervalCount": 199,
        "ExpectedEncodedDuration_s": 29.5308825,
        "GateFullWidth_s": 10e-6,
        "GateHalfWidth_s": 5e-6,
        "SigmaTOA_s": 1e-6,
        "SigmaSync_s": 1e-6,
        "SigmaWidth_s": 1.5e-9,
        "Xi": 0.0,
    }
    for name, expected in required.items():
        if study_config.get(name) != expected:
            raise ValueError(f"StudyConfig {name} is not the frozen value {expected!r}")
    if len(study_config.get("ConditionSpecs", ())) != 19:
        raise ValueError("StudyConfig must contain exactly 19 conditions")
    method_codes = _method_codes(study_config)

    try:
        delta_t_s = np.asarray(encoding_reference["delta_t_s"], dtype=np.float64)
        ref_width_s = np.asarray(
            encoding_reference["ref_width_s"],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(f"encoding reference is missing {exc.args[0]}") from exc
    if delta_t_s.shape != (199,):
        raise ValueError("encoding reference must contain exactly 199 DeltaT values")
    if ref_width_s.shape != (200,):
        raise ValueError("encoding reference must contain exactly 200 reference widths")
    if not np.all(np.isfinite(delta_t_s)) or np.any(delta_t_s <= 0.0):
        raise ValueError("DeltaT values must be finite and positive")
    if not np.all(np.isfinite(ref_width_s)):
        raise ValueError("reference widths must be finite")

    half_width = np.float64(study_config["GateHalfWidth_s"])
    full_width = np.float64(study_config["GateFullWidth_s"])
    if np.float64(2.0) * half_width != full_width:
        raise ValueError("two GateHalfWidth values must equal GateFullWidth in float64")
    sigma_dt = np.float64(math.sqrt(2.0)) * np.float64(
        study_config["SigmaTOA_s"]
    )
    sigma_dwidth = np.float64(study_config["SigmaWidth_s"])
    return (
        method_codes,
        delta_t_s,
        ref_width_s,
        half_width,
        sigma_dt,
        sigma_dwidth,
    )


def gate_bounds(ref_toa_s: float, gate_half_width_s: float) -> tuple[np.float64, np.float64]:
    reference = _finite_scalar(ref_toa_s, "ref_toa_s")
    half_width = _finite_scalar(gate_half_width_s, "gate_half_width_s")
    if half_width <= 0.0:
        raise ValueError("gate_half_width_s must be positive")
    a = np.float64(reference - half_width)
    b = np.float64(reference + half_width)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("gate bounds are not a finite positive-width interval")
    return a, b


def in_half_open_gate(true_toa_s: float, a_s: float, b_s: float) -> bool:
    true_toa = _finite_scalar(true_toa_s, "true_toa_s")
    a = _finite_scalar(a_s, "a_s")
    b = _finite_scalar(b_s, "b_s")
    if b <= a:
        raise ValueError("gate requires b > a")
    return bool(a <= true_toa < b)


def _append_array_event(
    *,
    source_code: np.uint8,
    offset: int,
    events: object,
    observed_toa: list[np.float64],
    observed_width: list[np.float64],
    tie_hi: list[np.uint64],
    tie_lo: list[np.uint64],
    sources: list[np.uint8],
    event_indices: list[int],
) -> None:
    observed_toa.append(np.float64(events.observed_toa_s[offset]))
    observed_width.append(np.float64(events.observed_width_s[offset]))
    tie_hi.append(np.uint64(events.tie_rank_hi[offset]))
    tie_lo.append(np.uint64(events.tie_rank_lo[offset]))
    sources.append(source_code)
    event_indices.append(int(events.event_index[offset]))


def build_cycle_candidates(
    *,
    world: SharedPhysicalWorld,
    cycle_index: int,
    ref_toa_s: float,
    gate_half_width_s: float,
    sigma_toa_s: float,
    sigma_width_s: float,
    h_cache: HEventCache | None,
) -> CycleCandidates:
    """Build G/H/F candidates in a true-TOA gate; observations are never re-gated."""
    if not isinstance(world, SharedPhysicalWorld):
        raise TypeError("world must be a SharedPhysicalWorld")
    k = _integer(cycle_index, "cycle_index")
    if not 1 <= k <= 200:
        raise ValueError("cycle_index must lie in 1..200")
    if h_cache is not None and not isinstance(h_cache, dict):
        raise TypeError("h_cache must be a Condition-Repeat local dict or None")
    sigma_toa = _finite_scalar(sigma_toa_s, "sigma_toa_s")
    sigma_width = _finite_scalar(sigma_width_s, "sigma_width_s")
    if sigma_toa < 0.0 or sigma_width < 0.0:
        raise ValueError("observation sigmas must be nonnegative")

    offset = k - 1
    guidance = world.guidance_events
    if len(guidance.event_index) != 200 or int(guidance.event_index[offset]) != k:
        raise ValueError("guidance events do not provide the current one-based cycle")
    a_s, b_s = gate_bounds(ref_toa_s, gate_half_width_s)

    observed_toa: list[np.float64] = []
    observed_width: list[np.float64] = []
    tie_hi: list[np.uint64] = []
    tie_lo: list[np.uint64] = []
    sources: list[np.uint8] = []
    event_indices: list[int] = []

    if in_half_open_gate(guidance.true_toa_s[offset], a_s, b_s):
        _append_array_event(
            source_code=SOURCE_G,
            offset=offset,
            events=guidance,
            observed_toa=observed_toa,
            observed_width=observed_width,
            tie_hi=tie_hi,
            tie_lo=tie_lo,
            sources=sources,
            event_indices=event_indices,
        )

    h_index_min: int | None = None
    h_index_max: int | None = None
    if world.hprf_state is not None:
        h_index_min, h_index_max = hprf_index_range(
            a_s,
            b_s,
            world.hprf_state.phase_s,
            world.hprf_state.period_s,
        )
        for event_index_n in range(h_index_min, h_index_max + 1):
            observation = get_h_event_observation(
                namespace=world.namespace,
                condition_ordinal=world.condition_ordinal,
                repeat_id=world.repeat_id,
                event_index_n=event_index_n,
                hprf_state=world.hprf_state,
                sigma_toa_s=sigma_toa,
                sigma_width_s=sigma_width,
                cache=h_cache,
            )
            observed_toa.append(np.float64(observation.observed_toa_s))
            observed_width.append(np.float64(observation.observed_width_s))
            tie_hi.append(np.uint64(observation.tie_rank_hi))
            tie_lo.append(np.uint64(observation.tie_rank_lo))
            sources.append(SOURCE_H)
            event_indices.append(int(observation.event_index))

    forwarded = world.forwarded_events
    if forwarded is not None:
        if len(forwarded.event_index) != 200 or int(forwarded.event_index[offset]) != k:
            raise ValueError("forwarded events do not provide the current one-based cycle")
        if in_half_open_gate(forwarded.true_toa_s[offset], a_s, b_s):
            _append_array_event(
                source_code=SOURCE_F,
                offset=offset,
                events=forwarded,
                observed_toa=observed_toa,
                observed_width=observed_width,
                tie_hi=tie_hi,
                tie_lo=tie_lo,
                sources=sources,
                event_indices=event_indices,
            )

    view = CandidateView(observed_toa, observed_width, tie_hi, tie_lo)
    return CycleCandidates(
        view=view,
        source_state_code=sources,
        event_index=event_indices,
        gate_a_s=a_s,
        gate_b_s=b_s,
        h_index_min=h_index_min,
        h_index_max=h_index_max,
    )


def selected_source_state_code(
    candidates: CycleCandidates,
    selected_pos: int | None,
) -> np.uint8:
    if not isinstance(candidates, CycleCandidates):
        raise TypeError("candidates must be CycleCandidates")
    if selected_pos is None:
        if len(candidates.view) != 0:
            raise ValueError("only an empty candidate set can have no selected_pos")
        return SOURCE_N
    position = _integer(selected_pos, "selected_pos")
    if not 0 <= position < len(candidates.view):
        raise IndexError("selected_pos is outside CandidateView")
    return np.uint8(candidates.source_state_code[position])


def initialize_method_states(
    world: SharedPhysicalWorld,
    study_config: Mapping[str, object],
) -> tuple[MethodState, ...]:
    _method_codes(study_config)
    initial_reference = _finite_scalar(world.initial_ref_toa_s, "initial_ref_toa_s")
    return tuple(MethodState(initial_reference) for _ in METHOD_IDENTITIES)


def update_ref_toa(
    current_ref_toa_s: float,
    selected_observed_toa_s: float | None,
    delta_t_s: float,
) -> np.float64:
    """Advance a reference without accepting or inspecting pulse-source metadata."""
    current = _finite_scalar(current_ref_toa_s, "current_ref_toa_s")
    delta = _finite_scalar(delta_t_s, "delta_t_s")
    if delta <= 0.0:
        raise ValueError("delta_t_s must be positive")
    base = (
        current
        if selected_observed_toa_s is None
        else _finite_scalar(selected_observed_toa_s, "selected_observed_toa_s")
    )
    updated = np.float64(base + delta)
    if not np.isfinite(updated):
        raise ValueError("updated reference TOA is not finite")
    return updated


def select_for_method(
    method_code: str,
    view: CandidateView,
    *,
    ref_toa_s: float,
    ref_width_s: float,
    sigma_dt_s: float,
    sigma_dwidth_s: float,
) -> int:
    if method_code == "FIRST":
        return select_first(view)
    if method_code == "LAST":
        return select_last(view)
    if method_code == "T":
        return select_t(view, ref_toa_s, sigma_dt_s)
    if method_code == "W":
        return select_w(view, ref_width_s, sigma_dwidth_s)
    if method_code == "TW":
        return select_tw(
            view,
            ref_toa_s,
            ref_width_s,
            sigma_dt_s,
            sigma_dwidth_s,
        )
    raise ValueError(f"unsupported frozen method code: {method_code!r}")


def _expected_h_count(world: SharedPhysicalWorld) -> int:
    if world.hprf_state is None:
        return 0
    frequency = float(world.hprf_state.frequency_hz)
    try:
        return EXPECTED_H_COUNT_BY_FREQUENCY[frequency]
    except KeyError as exc:
        raise ValueError(f"unsupported frozen HPRF frequency: {frequency!r}") from exc


def _hprf_diagnostic(
    world: SharedPhysicalWorld,
    candidates: CycleCandidates,
    method_code: str,
    cycle_index: int,
    ref_toa_s: float,
) -> str:
    return (
        f"Condition={world.condition_code}, Repeat={world.repeat_id}, "
        f"Method={method_code}, Cycle={cycle_index}, "
        f"ref_toa={float(ref_toa_s)!r}, a={float(candidates.gate_a_s)!r}, "
        f"b={float(candidates.gate_b_s)!r}, n_min={candidates.h_index_min!r}, "
        f"n_max={candidates.h_index_max!r}, H_count={candidates.h_count}"
    )


def _simulate_shared_world(
    *,
    world: SharedPhysicalWorld,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> ConditionRepeatTrajectory:
    (
        method_codes,
        delta_t_s,
        ref_width_s,
        gate_half_width_s,
        sigma_dt_s,
        sigma_dwidth_s,
    ) = _kernel_inputs(study_config, encoding_reference)
    condition = condition_spec_from_config(study_config, world.condition_ordinal)
    if world.condition_code != condition.Code:
        raise ValueError("world condition code disagrees with StudyConfig")

    states = initialize_method_states(world, study_config)
    h_cache: dict = {}
    source_state_code = np.empty((5, 200), dtype=np.uint8)
    ref_toa_before_s = np.empty((5, 200), dtype=np.float64)
    selected_observed_toa_s = np.full((5, 200), np.nan, dtype=np.float64)
    has_selection = np.zeros((5, 200), dtype=np.bool_)
    hprf_count_check_count = 0
    hprf_no_n_check_count = 0
    expected_h_count = _expected_h_count(world)

    for cycle_offset in range(200):
        cycle_index = cycle_offset + 1
        cycle_ref_width_s = np.float64(ref_width_s[cycle_offset])
        for method_offset, method_code in enumerate(method_codes):
            state = states[method_offset]
            current_ref_toa_s = np.float64(state.ref_toa_s)
            ref_toa_before_s[method_offset, cycle_offset] = current_ref_toa_s
            candidates = build_cycle_candidates(
                world=world,
                cycle_index=cycle_index,
                ref_toa_s=current_ref_toa_s,
                gate_half_width_s=gate_half_width_s,
                sigma_toa_s=float(study_config["SigmaTOA_s"]),
                sigma_width_s=float(study_config["SigmaWidth_s"]),
                h_cache=h_cache,
            )

            if world.hprf_state is not None:
                hprf_count_check_count += 1
                if candidates.h_count != expected_h_count:
                    raise HPRFCountError(
                        "HPRF per-method per-cycle count mismatch: "
                        + _hprf_diagnostic(
                            world,
                            candidates,
                            method_code,
                            cycle_index,
                            current_ref_toa_s,
                        )
                        + f", expected={expected_h_count}"
                    )

            selected_pos: int | None
            selected_toa: np.float64 | None
            if len(candidates.view) == 0:
                selected_pos = None
                selected_toa = None
            else:
                selected_pos = select_for_method(
                    method_code,
                    candidates.view,
                    ref_toa_s=current_ref_toa_s,
                    ref_width_s=cycle_ref_width_s,
                    sigma_dt_s=sigma_dt_s,
                    sigma_dwidth_s=sigma_dwidth_s,
                )
                selected_toa = np.float64(
                    candidates.view.observed_toa_s[selected_pos]
                )
                selected_observed_toa_s[method_offset, cycle_offset] = selected_toa
                has_selection[method_offset, cycle_offset] = True

            state_code = selected_source_state_code(candidates, selected_pos)
            source_state_code[method_offset, cycle_offset] = state_code
            if world.hprf_state is not None:
                hprf_no_n_check_count += 1
                if state_code == SOURCE_N:
                    raise HPRFEmptyGateError(
                        "HPRF/composite cycle produced N: "
                        + _hprf_diagnostic(
                            world,
                            candidates,
                            method_code,
                            cycle_index,
                            current_ref_toa_s,
                        )
                    )

            if cycle_offset < 199:
                state.ref_toa_s = update_ref_toa(
                    current_ref_toa_s,
                    selected_toa,
                    delta_t_s[cycle_offset],
                )

    if condition.Scene == "NI":
        if not all(
            np.array_equal(source_state_code[0], source_state_code[index])
            and np.array_equal(ref_toa_before_s[0], ref_toa_before_s[index])
            for index in range(1, 5)
        ):
            raise NIIdentityError(
                "NI method state or reference trajectories are not exactly identical"
            )

    return ConditionRepeatTrajectory(
        condition_ordinal=world.condition_ordinal,
        condition_code=world.condition_code,
        repeat_id=world.repeat_id,
        namespace=world.namespace,
        method_codes=method_codes,
        source_state_code=source_state_code,
        ref_toa_before_s=ref_toa_before_s,
        selected_observed_toa_s=selected_observed_toa_s,
        has_selection=has_selection,
        delta_t_consumed_count=199,
        hprf_count_check_count=hprf_count_check_count,
        hprf_no_n_check_count=hprf_no_n_check_count,
    )


def _simulate_shared_world_controlled(
    *,
    world: SharedPhysicalWorld,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    method_execution_order: Sequence[int] = DEFAULT_METHOD_EXECUTION_ORDER,
    h_cache_enabled: bool = True,
    diagnostic: bool = False,
    diagnostic_collector: list[CycleDiagnostic] | None = None,
) -> ConditionRepeatTrajectory:
    (
        method_codes,
        delta_t_s,
        ref_width_s,
        gate_half_width_s,
        sigma_dt_s,
        sigma_dwidth_s,
    ) = _kernel_inputs(study_config, encoding_reference)
    condition = condition_spec_from_config(study_config, world.condition_ordinal)
    if world.condition_code != condition.Code:
        raise ValueError("world condition code disagrees with StudyConfig")
    execution_order = _validated_method_execution_order(method_execution_order)
    if type(h_cache_enabled) is not bool:
        raise TypeError("h_cache_enabled must be bool")
    if type(diagnostic) is not bool:
        raise TypeError("diagnostic must be bool")
    if diagnostic_collector is not None and not isinstance(diagnostic_collector, list):
        raise TypeError("diagnostic_collector must be a list or None")
    if not diagnostic and diagnostic_collector is not None:
        raise ValueError("diagnostic_collector requires diagnostic=True")
    active_diagnostics = diagnostic_collector if diagnostic_collector is not None else []

    states = initialize_method_states(world, study_config)
    h_cache: dict | None = {} if h_cache_enabled else None
    source_state_code = np.empty((5, 200), dtype=np.uint8)
    ref_toa_before_s = np.empty((5, 200), dtype=np.float64)
    selected_observed_toa_s = np.full((5, 200), np.nan, dtype=np.float64)
    has_selection = np.zeros((5, 200), dtype=np.bool_)
    hprf_count_check_count = 0
    hprf_no_n_check_count = 0
    expected_h_count = _expected_h_count(world)

    for cycle_offset in range(200):
        cycle_index = cycle_offset + 1
        cycle_ref_width_s = np.float64(ref_width_s[cycle_offset])
        for method_ordinal in execution_order:
            method_code = method_codes[method_ordinal]
            state = states[method_ordinal]
            current_ref_toa_s = np.float64(state.ref_toa_s)
            ref_toa_before_s[method_ordinal, cycle_offset] = current_ref_toa_s
            candidates = build_cycle_candidates(
                world=world,
                cycle_index=cycle_index,
                ref_toa_s=current_ref_toa_s,
                gate_half_width_s=gate_half_width_s,
                sigma_toa_s=float(study_config["SigmaTOA_s"]),
                sigma_width_s=float(study_config["SigmaWidth_s"]),
                h_cache=h_cache,
            )

            if world.hprf_state is not None:
                hprf_count_check_count += 1
                if candidates.h_count != expected_h_count:
                    raise HPRFCountError(
                        "HPRF per-method per-cycle count mismatch: "
                        + _hprf_diagnostic(
                            world,
                            candidates,
                            method_code,
                            cycle_index,
                            current_ref_toa_s,
                        )
                        + f", expected={expected_h_count}"
                    )

            selected_pos: int | None
            selected_toa: np.float64 | None
            if len(candidates.view) == 0:
                selected_pos = None
                selected_toa = None
            else:
                selected_pos = select_for_method(
                    method_code,
                    candidates.view,
                    ref_toa_s=current_ref_toa_s,
                    ref_width_s=cycle_ref_width_s,
                    sigma_dt_s=sigma_dt_s,
                    sigma_dwidth_s=sigma_dwidth_s,
                )
                selected_toa = np.float64(
                    candidates.view.observed_toa_s[selected_pos]
                )
                selected_observed_toa_s[method_ordinal, cycle_offset] = selected_toa
                has_selection[method_ordinal, cycle_offset] = True

            state_code = selected_source_state_code(candidates, selected_pos)
            source_state_code[method_ordinal, cycle_offset] = state_code
            if world.hprf_state is not None:
                hprf_no_n_check_count += 1
                if state_code == SOURCE_N:
                    raise HPRFEmptyGateError(
                        "HPRF/composite cycle produced N: "
                        + _hprf_diagnostic(
                            world,
                            candidates,
                            method_code,
                            cycle_index,
                            current_ref_toa_s,
                        )
                    )

            if cycle_offset < 199:
                state.ref_toa_s = update_ref_toa(
                    current_ref_toa_s,
                    selected_toa,
                    delta_t_s[cycle_offset],
                )
            if diagnostic:
                active_diagnostics.append(
                    CycleDiagnostic(
                        Namespace=world.namespace.name,
                        ConditionOrdinal=world.condition_ordinal,
                        ConditionCode=world.condition_code,
                        RepeatID=world.repeat_id,
                        Cycle=cycle_index,
                        MethodOrdinal=method_ordinal,
                        MethodCode=method_code,
                        GateLeft=float(candidates.gate_a_s),
                        GateRight=float(candidates.gate_b_s),
                        CandidateCount=len(candidates.view),
                        SelectedPos=selected_pos,
                        SelectedSourceAfterSelection=int(state_code),
                        RefTOABefore=float(current_ref_toa_s),
                        RefTOAAfter=float(state.ref_toa_s),
                    )
                )

    if condition.Scene == "NI":
        if not all(
            np.array_equal(source_state_code[0], source_state_code[index])
            and np.array_equal(ref_toa_before_s[0], ref_toa_before_s[index])
            for index in range(1, 5)
        ):
            raise NIIdentityError(
                "NI method state or reference trajectories are not exactly identical"
            )

    return ConditionRepeatTrajectory(
        condition_ordinal=world.condition_ordinal,
        condition_code=world.condition_code,
        repeat_id=world.repeat_id,
        namespace=world.namespace,
        method_codes=method_codes,
        source_state_code=source_state_code,
        ref_toa_before_s=ref_toa_before_s,
        selected_observed_toa_s=selected_observed_toa_s,
        has_selection=has_selection,
        delta_t_consumed_count=199,
        hprf_count_check_count=hprf_count_check_count,
        hprf_no_n_check_count=hprf_no_n_check_count,
    )


def simulate_shared_world(
    *,
    world: SharedPhysicalWorld,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> ConditionRepeatTrajectory:
    """Public, behavior-neutral entry point for an already-built shared world."""
    return _simulate_shared_world(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )


def simulate_shared_world_controlled(
    *,
    world: SharedPhysicalWorld,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    method_execution_order: Sequence[int] = DEFAULT_METHOD_EXECUTION_ORDER,
    h_cache_enabled: bool = True,
    diagnostic: bool = False,
    diagnostic_collector: list[CycleDiagnostic] | None = None,
) -> ConditionRepeatTrajectory:
    """Stage 6 execution controls over one already-built physical world."""
    return _simulate_shared_world_controlled(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
        method_execution_order=method_execution_order,
        h_cache_enabled=h_cache_enabled,
        diagnostic=diagnostic,
        diagnostic_collector=diagnostic_collector,
    )


def simulate_condition_repeat(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> ConditionRepeatTrajectory:
    """Execute one complete 200-cycle trajectory over one shared physical world."""
    world = build_shared_physical_world(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )
    return simulate_shared_world(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )


def simulate_condition_repeat_controlled(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    method_execution_order: Sequence[int] = DEFAULT_METHOD_EXECUTION_ORDER,
    h_cache_enabled: bool = True,
    diagnostic: bool = False,
    diagnostic_collector: list[CycleDiagnostic] | None = None,
) -> ConditionRepeatTrajectory:
    """Build one physical world and execute it with Stage 6 controls."""
    world = build_shared_physical_world(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )
    return simulate_shared_world_controlled(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
        method_execution_order=method_execution_order,
        h_cache_enabled=h_cache_enabled,
        diagnostic=diagnostic,
        diagnostic_collector=diagnostic_collector,
    )


__all__ = [
    "ConditionRepeatTrajectory",
    "CycleCandidates",
    "DEFAULT_METHOD_EXECUTION_ORDER",
    "HPRFCountError",
    "HPRFEmptyGateError",
    "METHOD_IDENTITIES",
    "MethodState",
    "NIIdentityError",
    "SOURCE_F",
    "SOURCE_G",
    "SOURCE_H",
    "SOURCE_N",
    "build_cycle_candidates",
    "gate_bounds",
    "in_half_open_gate",
    "initialize_method_states",
    "select_for_method",
    "selected_source_state_code",
    "simulate_condition_repeat",
    "simulate_condition_repeat_controlled",
    "simulate_shared_world",
    "simulate_shared_world_controlled",
    "update_ref_toa",
]
