from __future__ import annotations

import math
import operator
from collections.abc import MutableMapping
from dataclasses import dataclass

import numpy as np

from .contracts import ConditionSpec
from .events import (
    HPRF_SCENES,
    physical_random_address,
    validate_condition_spec,
    validate_physical_namespace,
)
from .rng import RandomNamespace, VariableFamily, random_normal, random_tierank, random_uniform


SIGNED_INT64_MIN = -(1 << 63)
SIGNED_INT64_MAX = (1 << 63) - 1
H_WIDTH_BASE_S = np.float64(250e-9)
H_WIDTH_STEP_S = np.float64(10e-9)


@dataclass(frozen=True, slots=True)
class HPRFRepeatState:
    frequency_hz: np.float64
    period_s: np.float64
    phase_s: np.float64
    true_width_level: np.uint8
    true_width_s: np.float64


@dataclass(frozen=True, slots=True)
class HEventObservation:
    event_index: int
    true_toa_s: np.float64
    true_width_s: np.float64
    observed_toa_s: np.float64
    observed_width_s: np.float64
    tie_rank_hi: np.uint64
    tie_rank_lo: np.uint64


HEventCache = MutableMapping[int, HEventObservation]


def _signed_int64(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc
    if not SIGNED_INT64_MIN <= result <= SIGNED_INT64_MAX:
        raise ValueError(f"{field_name} must be a signed int64")
    return result


def _finite_nonnegative(value: object, field_name: str) -> np.float64:
    result = np.float64(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return result


def _validate_state(state: HPRFRepeatState) -> HPRFRepeatState:
    if not isinstance(state, HPRFRepeatState):
        raise TypeError("hprf_state must be an HPRFRepeatState")
    values = np.array(
        [state.frequency_hz, state.period_s, state.phase_s, state.true_width_s],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("HPRFRepeatState contains a non-finite value")
    if state.frequency_hz <= 0.0 or state.period_s <= 0.0:
        raise ValueError("HPRF frequency and period must be positive")
    if not 0.0 < state.phase_s < state.period_s:
        raise ValueError("HPRF phase must lie strictly inside (0, period)")
    level = int(state.true_width_level)
    if not 0 <= level <= 15:
        raise ValueError("H true width level must be in [0, 15]")
    expected_width = np.float64(H_WIDTH_BASE_S + level * H_WIDTH_STEP_S)
    if state.true_width_s != expected_width:
        raise ValueError("H true width must be the discrete width for its level")
    return state


def build_hprf_repeat_state(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    condition: ConditionSpec,
) -> HPRFRepeatState | None:
    """Build the one phase and one discrete H width for a Condition-Repeat."""
    namespace = validate_physical_namespace(namespace)
    condition = validate_condition_spec(condition)
    if condition.Scene not in HPRF_SCENES:
        return None
    if condition.ConditionOrdinal != int(condition_ordinal):
        raise ValueError("condition ordinal and ConditionSpec disagree")

    frequency_hz = np.float64(condition.HPRF_Hz)
    period_s = np.float64(1.0) / frequency_hz
    phase_address = physical_random_address(
        namespace,
        condition_ordinal,
        repeat_id,
        VariableFamily.H_PHASE,
        0,
    )
    phase_s = np.float64(random_uniform(phase_address)) * period_s

    width_address = physical_random_address(
        namespace,
        condition_ordinal,
        repeat_id,
        VariableFamily.H_TRUE_WIDTH_LEVEL,
        0,
    )
    true_width_level = np.uint8(
        math.floor(16.0 * random_uniform(width_address))
    )
    true_width_s = np.float64(
        H_WIDTH_BASE_S + int(true_width_level) * H_WIDTH_STEP_S
    )
    return _validate_state(
        HPRFRepeatState(
            frequency_hz=frequency_hz,
            period_s=period_s,
            phase_s=phase_s,
            true_width_level=true_width_level,
            true_width_s=true_width_s,
        )
    )


def snap_integer_ulp(x: float) -> np.float64:
    """Snap only float64 representation noise within eight ULP of an integer."""
    value = np.float64(x)
    if not np.isfinite(value):
        raise ValueError("x must be finite")
    nearest_integer = round(float(value))
    with np.errstate(over="ignore", invalid="ignore"):
        ulp = abs(float(np.spacing(value)))
    if math.isfinite(ulp) and abs(float(value) - nearest_integer) <= 8.0 * ulp:
        return np.float64(nearest_integer)
    return value


def hprf_index_range(
    a: float, b: float, phi_h: float, period_h: float
) -> tuple[int, int]:
    """Return inclusive signed-index bounds for H events in the interval [a, b)."""
    a_s = np.float64(a)
    b_s = np.float64(b)
    phase_s = np.float64(phi_h)
    period_s = np.float64(period_h)
    if not np.all(np.isfinite((a_s, b_s, phase_s, period_s))):
        raise ValueError("HPRF interval inputs must be finite")
    if period_s <= 0.0:
        raise ValueError("period_h must be positive")
    if b_s < a_s:
        raise ValueError("hprf_index_range requires b >= a")

    x_a = np.float64((a_s - phase_s) / period_s)
    x_b = np.float64((b_s - phase_s) / period_s)
    n_min = math.ceil(float(snap_integer_ulp(x_a)))
    n_max = math.ceil(float(snap_integer_ulp(x_b))) - 1
    _signed_int64(n_min, "n_min")
    _signed_int64(n_max, "n_max")
    return n_min, n_max


def get_h_event_observation(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    event_index_n: int,
    hprf_state: HPRFRepeatState,
    sigma_toa_s: float,
    sigma_width_s: float,
    cache: HEventCache | None = None,
) -> HEventObservation:
    """Generate one signed-index H observation on demand from its logical address."""
    namespace = validate_physical_namespace(namespace)
    state = _validate_state(hprf_state)
    n = _signed_int64(event_index_n, "event_index_n")
    sigma_toa = _finite_nonnegative(sigma_toa_s, "sigma_toa_s")
    sigma_width = _finite_nonnegative(sigma_width_s, "sigma_width_s")

    if cache is not None and n in cache:
        cached = cache[n]
        if not isinstance(cached, HEventObservation) or cached.event_index != n:
            raise ValueError("H cache contains an invalid entry")
        return cached

    true_toa_s = np.float64(state.phase_s + np.float64(n) * state.period_s)
    toa_address = physical_random_address(
        namespace,
        condition_ordinal,
        repeat_id,
        VariableFamily.H_TOA_ERROR,
        n,
    )
    width_address = physical_random_address(
        namespace,
        condition_ordinal,
        repeat_id,
        VariableFamily.H_WIDTH_ERROR,
        n,
    )
    tie_address = physical_random_address(
        namespace,
        condition_ordinal,
        repeat_id,
        VariableFamily.TIE_H,
        n,
    )
    observed_toa_s = np.float64(
        true_toa_s + sigma_toa * np.float64(random_normal(toa_address))
    )
    observed_width_s = np.float64(
        state.true_width_s
        + sigma_width * np.float64(random_normal(width_address))
    )
    tie_hi, tie_lo = random_tierank(tie_address)
    observation = HEventObservation(
        event_index=n,
        true_toa_s=true_toa_s,
        true_width_s=state.true_width_s,
        observed_toa_s=observed_toa_s,
        observed_width_s=observed_width_s,
        tie_rank_hi=np.uint64(tie_hi),
        tie_rank_lo=np.uint64(tie_lo),
    )
    numeric_values = np.array(
        [
            observation.true_toa_s,
            observation.true_width_s,
            observation.observed_toa_s,
            observation.observed_width_s,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("H observation contains a non-finite value")
    if cache is not None:
        cache[n] = observation
    return observation
