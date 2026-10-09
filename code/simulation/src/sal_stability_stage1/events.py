from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import ConditionSpec
from .encoding import WIDTH_BASE_S, WIDTH_STEP_S
from .rng import (
    RandomAddress,
    RandomNamespace,
    VariableFamily,
    random_normal,
    random_tierank,
)


PHYSICAL_NAMESPACES = frozenset((RandomNamespace.PILOT, RandomNamespace.FORMAL))
FORWARDED_SCENES = frozenset(("IDF", "COMPOSITE"))
HPRF_SCENES = frozenset(("HPRF", "COMPOSITE"))
VALID_SCENES = frozenset(("NI", "HPRF", "IDF", "COMPOSITE"))


@dataclass(frozen=True, slots=True)
class GuidanceEvents:
    event_index: np.ndarray
    true_toa_s: np.ndarray
    true_width_s: np.ndarray
    observed_toa_s: np.ndarray
    observed_width_s: np.ndarray
    tie_rank_hi: np.ndarray
    tie_rank_lo: np.ndarray


@dataclass(frozen=True, slots=True)
class ForwardedEvents:
    event_index: np.ndarray
    true_toa_s: np.ndarray
    true_width_s: np.ndarray
    observed_toa_s: np.ndarray
    observed_width_s: np.ndarray
    tie_rank_hi: np.ndarray
    tie_rank_lo: np.ndarray


@dataclass(frozen=True, slots=True)
class InitialSynchronization:
    error_s: np.float64
    initial_ref_toa_s: np.float64


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def validate_physical_namespace(namespace: RandomNamespace) -> RandomNamespace:
    """Normalize a namespace and reject the resampling-only namespace."""
    try:
        member = RandomNamespace(_integer(namespace, "namespace"))
    except ValueError as exc:
        raise ValueError(f"invalid namespace: {namespace!r}") from exc
    if member not in PHYSICAL_NAMESPACES:
        raise ValueError("BOOTSTRAP cannot create a physical event world")
    return member


def physical_random_address(
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    variable_family: VariableFamily,
    event_index: int,
) -> RandomAddress:
    """Create a Stage 2 address for a physical event without any method field."""
    return RandomAddress(
        namespace=validate_physical_namespace(namespace),
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        variable_family=variable_family,
        event_index=event_index,
    )


def validate_condition_spec(condition: ConditionSpec) -> ConditionSpec:
    if not isinstance(condition, ConditionSpec):
        raise TypeError("condition must be a ConditionSpec")
    if condition.Scene not in VALID_SCENES:
        raise ValueError(f"unsupported scene: {condition.Scene!r}")

    has_forwarded = condition.Scene in FORWARDED_SCENES
    has_hprf = condition.Scene in HPRF_SCENES
    if has_forwarded != (condition.FDelay_s is not None):
        raise ValueError("scene and FDelay_s are inconsistent")
    if has_hprf != (condition.HPRF_Hz is not None):
        raise ValueError("scene and HPRF_Hz are inconsistent")
    if condition.FDelay_s is not None:
        delay = float(condition.FDelay_s)
        if not math.isfinite(delay) or delay <= 0.0:
            raise ValueError("FDelay_s must be finite and positive")
    if condition.HPRF_Hz is not None:
        frequency = float(condition.HPRF_Hz)
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("HPRF_Hz must be finite and positive")
    return condition


def condition_spec_from_config(
    study_config: Mapping[str, object], condition_ordinal: int
) -> ConditionSpec:
    """Read a condition by ordinal from StudyConfig; codes are never parsed."""
    ordinal = _integer(condition_ordinal, "condition_ordinal")
    try:
        raw_specs = study_config["ConditionSpecs"]
    except (KeyError, TypeError) as exc:
        raise ValueError("StudyConfig is missing ConditionSpecs") from exc
    if not isinstance(raw_specs, (list, tuple)):
        raise TypeError("ConditionSpecs must be a list or tuple")

    matches: list[Mapping[str, object]] = []
    for raw in raw_specs:
        if not isinstance(raw, Mapping):
            raise TypeError("each ConditionSpecs entry must be a mapping")
        if _integer(raw.get("ConditionOrdinal"), "ConditionOrdinal") == ordinal:
            matches.append(raw)
    if len(matches) != 1:
        raise ValueError(
            f"condition ordinal {ordinal} must occur exactly once in StudyConfig"
        )

    raw = matches[0]
    try:
        hprf_hz = raw["HPRF_Hz"]
        f_delay_s = raw["FDelay_s"]
        condition = ConditionSpec(
            ConditionOrdinal=ordinal,
            Code=str(raw["Code"]),
            Scene=str(raw["Scene"]),
            HPRF_Hz=None if hprf_hz is None else float(hprf_hz),
            FDelay_s=None if f_delay_s is None else float(f_delay_s),
        )
    except KeyError as exc:
        raise ValueError(f"ConditionSpec is missing {exc.args[0]}") from exc
    return validate_condition_spec(condition)


def _finite_nonnegative(value: object, field_name: str) -> np.float64:
    result = np.float64(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and nonnegative")
    return result


def _finite_scalar(value: object, field_name: str) -> np.float64:
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _readonly_array(values: object, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _assert_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains a non-finite value")


def _reference_event_arrays(
    encoding_reference: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        delta_t_s = np.asarray(encoding_reference["delta_t_s"], dtype=np.float64)
        ref_width_s = np.asarray(
            encoding_reference["ref_width_s"], dtype=np.float64
        )
        width_level = np.asarray(encoding_reference["width_level"])
    except KeyError as exc:
        raise ValueError(f"encoding reference is missing {exc.args[0]}") from exc

    if delta_t_s.shape != (199,):
        raise ValueError("encoding reference must contain exactly 199 intervals")
    if ref_width_s.shape != (200,):
        raise ValueError("encoding reference must contain exactly 200 widths")
    if width_level.shape != (200,):
        raise ValueError("encoding reference must contain exactly 200 width levels")
    if not np.issubdtype(width_level.dtype, np.integer):
        raise TypeError("width_level must have an integer dtype")
    if np.any(width_level < 0) or np.any(width_level > 15):
        raise ValueError("reference width levels must be in [0, 15]")
    _assert_finite("delta_t_s", delta_t_s)
    _assert_finite("ref_width_s", ref_width_s)
    if np.any(delta_t_s <= 0.0):
        raise ValueError("all encoded intervals must be positive")

    expected_width_s = (
        np.float64(WIDTH_BASE_S)
        + width_level.astype(np.float64) * np.float64(WIDTH_STEP_S)
    )
    if not np.array_equal(ref_width_s, expected_width_s):
        raise ValueError("reference widths do not match their frozen 16 levels")
    return delta_t_s, ref_width_s, width_level


def build_guidance_events(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    encoding_reference: Mapping[str, np.ndarray],
    sigma_toa_s: float,
    sigma_width_s: float,
) -> GuidanceEvents:
    """Build the 200 true and observed guidance events for one repeat."""
    namespace = validate_physical_namespace(namespace)
    sigma_toa = _finite_nonnegative(sigma_toa_s, "sigma_toa_s")
    sigma_width = _finite_nonnegative(sigma_width_s, "sigma_width_s")
    delta_t_s, ref_width_s, _ = _reference_event_arrays(encoding_reference)

    event_index = np.arange(1, 201, dtype=np.int64)
    true_toa_s = np.empty(200, dtype=np.float64)
    true_toa_s[0] = np.float64(0.0)
    np.cumsum(delta_t_s, dtype=np.float64, out=true_toa_s[1:])
    true_width_s = np.array(ref_width_s, dtype=np.float64, copy=True)
    observed_toa_s = np.empty(200, dtype=np.float64)
    observed_width_s = np.empty(200, dtype=np.float64)
    tie_rank_hi = np.empty(200, dtype=np.uint64)
    tie_rank_lo = np.empty(200, dtype=np.uint64)

    for offset, event in enumerate(event_index):
        k = int(event)
        toa_address = physical_random_address(
            namespace,
            condition_ordinal,
            repeat_id,
            VariableFamily.G_TOA_ERROR,
            k,
        )
        width_address = physical_random_address(
            namespace,
            condition_ordinal,
            repeat_id,
            VariableFamily.G_WIDTH_ERROR,
            k,
        )
        tie_address = physical_random_address(
            namespace,
            condition_ordinal,
            repeat_id,
            VariableFamily.TIE_G,
            k,
        )
        observed_toa_s[offset] = true_toa_s[offset] + sigma_toa * np.float64(
            random_normal(toa_address)
        )
        observed_width_s[offset] = true_width_s[offset] + sigma_width * np.float64(
            random_normal(width_address)
        )
        tie_hi, tie_lo = random_tierank(tie_address)
        tie_rank_hi[offset] = np.uint64(tie_hi)
        tie_rank_lo[offset] = np.uint64(tie_lo)

    for name, values in (
        ("true guidance TOA", true_toa_s),
        ("observed guidance TOA", observed_toa_s),
        ("true guidance width", true_width_s),
        ("observed guidance width", observed_width_s),
    ):
        _assert_finite(name, values)

    return GuidanceEvents(
        event_index=_readonly_array(event_index, np.dtype(np.int64)),
        true_toa_s=_readonly_array(true_toa_s, np.dtype(np.float64)),
        true_width_s=_readonly_array(true_width_s, np.dtype(np.float64)),
        observed_toa_s=_readonly_array(observed_toa_s, np.dtype(np.float64)),
        observed_width_s=_readonly_array(
            observed_width_s, np.dtype(np.float64)
        ),
        tie_rank_hi=_readonly_array(tie_rank_hi, np.dtype(np.uint64)),
        tie_rank_lo=_readonly_array(tie_rank_lo, np.dtype(np.uint64)),
    )


def build_initial_sync(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    true_first_toa_s: float,
    sigma_sync_s: float,
) -> InitialSynchronization:
    """Build the one shared initial reference for a Condition-Repeat world."""
    true_first = _finite_scalar(true_first_toa_s, "true_first_toa_s")
    sigma_sync = _finite_nonnegative(sigma_sync_s, "sigma_sync_s")
    address = physical_random_address(
        namespace,
        condition_ordinal,
        repeat_id,
        VariableFamily.INITIAL_SYNC,
        0,
    )
    error_s = np.float64(sigma_sync * np.float64(random_normal(address)))
    initial_ref_toa_s = np.float64(true_first + error_s)
    if not np.isfinite(error_s) or not np.isfinite(initial_ref_toa_s):
        raise ValueError("initial synchronization produced a non-finite value")
    return InitialSynchronization(error_s, initial_ref_toa_s)


def build_forwarded_events(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    condition: ConditionSpec,
    guidance_events: GuidanceEvents,
    sigma_toa_s: float,
    sigma_width_s: float,
) -> ForwardedEvents | None:
    """Build forwarded events when the ConditionSpec declares an F source."""
    namespace = validate_physical_namespace(namespace)
    condition = validate_condition_spec(condition)
    if condition.Scene not in FORWARDED_SCENES:
        return None
    if condition.ConditionOrdinal != _integer(condition_ordinal, "condition_ordinal"):
        raise ValueError("condition ordinal and ConditionSpec disagree")

    sigma_toa = _finite_nonnegative(sigma_toa_s, "sigma_toa_s")
    sigma_width = _finite_nonnegative(sigma_width_s, "sigma_width_s")
    mu_f = _finite_scalar(condition.FDelay_s, "FDelay_s")
    if guidance_events.event_index.shape != (200,):
        raise ValueError("guidance_events must contain exactly 200 events")

    event_index = np.array(guidance_events.event_index, dtype=np.int64, copy=True)
    true_toa_s = np.asarray(guidance_events.true_toa_s, dtype=np.float64) + mu_f
    true_width_s = np.array(
        guidance_events.true_width_s, dtype=np.float64, copy=True
    )
    observed_toa_s = np.empty(200, dtype=np.float64)
    observed_width_s = np.empty(200, dtype=np.float64)
    tie_rank_hi = np.empty(200, dtype=np.uint64)
    tie_rank_lo = np.empty(200, dtype=np.uint64)

    for offset, event in enumerate(event_index):
        k = int(event)
        toa_address = physical_random_address(
            namespace,
            condition_ordinal,
            repeat_id,
            VariableFamily.F_TOA_ERROR,
            k,
        )
        width_address = physical_random_address(
            namespace,
            condition_ordinal,
            repeat_id,
            VariableFamily.F_WIDTH_ERROR,
            k,
        )
        tie_address = physical_random_address(
            namespace,
            condition_ordinal,
            repeat_id,
            VariableFamily.TIE_F,
            k,
        )
        observed_toa_s[offset] = true_toa_s[offset] + sigma_toa * np.float64(
            random_normal(toa_address)
        )
        observed_width_s[offset] = true_width_s[offset] + sigma_width * np.float64(
            random_normal(width_address)
        )
        tie_hi, tie_lo = random_tierank(tie_address)
        tie_rank_hi[offset] = np.uint64(tie_hi)
        tie_rank_lo[offset] = np.uint64(tie_lo)

    for name, values in (
        ("true forwarded TOA", true_toa_s),
        ("observed forwarded TOA", observed_toa_s),
        ("true forwarded width", true_width_s),
        ("observed forwarded width", observed_width_s),
    ):
        _assert_finite(name, values)
    if not np.array_equal(true_width_s, guidance_events.true_width_s):
        raise AssertionError("true forwarded widths must equal true guidance widths")

    return ForwardedEvents(
        event_index=_readonly_array(event_index, np.dtype(np.int64)),
        true_toa_s=_readonly_array(true_toa_s, np.dtype(np.float64)),
        true_width_s=_readonly_array(true_width_s, np.dtype(np.float64)),
        observed_toa_s=_readonly_array(observed_toa_s, np.dtype(np.float64)),
        observed_width_s=_readonly_array(
            observed_width_s, np.dtype(np.float64)
        ),
        tie_rank_hi=_readonly_array(tie_rank_hi, np.dtype(np.uint64)),
        tie_rank_lo=_readonly_array(tie_rank_lo, np.dtype(np.uint64)),
    )
