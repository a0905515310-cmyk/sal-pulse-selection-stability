from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .rng import (
    MASK32,
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_ROUNDS,
    PHILOX_W0,
    PHILOX_W1,
    RandomAddress,
    RandomNamespace,
    VariableFamily,
    derive_philox_key,
    raw_words,
)


BOOTSTRAP_B = 2000
BOOTSTRAP_DRAW_COUNT = 2000
BOOTSTRAP_BLOCK_SIZE = 50
BOOTSTRAP_AUDIT_IDS = (1, 1000, 2000)
FORMAL_REPEAT_COUNT = 2000
K = 200

UINT32_SIZE = 1 << 32
ACCEPTANCE_LIMIT = (UINT32_SIZE // FORMAL_REPEAT_COUNT) * FORMAL_REPEAT_COUNT

METHOD_CODES = ("FIRST", "LAST", "T", "W", "TW")
METHOD_ORDINALS = tuple(range(len(METHOD_CODES)))
METRIC_CODES = ("P_cor", "P_C_given_C", "P_C_given_E", "Mean_L_NC")
CONTRAST_CODES = ("TW-T", "TW-W")
COMPARATOR_METHOD_ORDINALS = (2, 3)
TW_METHOD_ORDINAL = 4

STATISTIC_FIELDS = (
    "N_C",
    "N_CC",
    "N_Cdot",
    "N_EC",
    "N_Edot",
    "Sum_L_NC_obs",
    "N_run_NC",
)
STATISTIC_INDEX = {name: index for index, name in enumerate(STATISTIC_FIELDS)}

POINT_UNDEFINED_REASONS = (
    "",
    "C_DENOM_ZERO",
    "E_DENOM_ZERO",
    "NO_NC_RUN",
)
CI_UNDEFINED_REASON = "INSUFFICIENT_DEFINED_BOOTSTRAP_REPLICATES"

GOLDEN_REPEAT_IDS: dict[tuple[int, int], tuple[int, ...]] = {
    (1, 1): (1143, 1364, 408, 200, 1923, 1139, 1575, 944, 709, 1143),
    (1, 1000): (35, 141, 1994, 838, 686, 1074, 1381, 1338, 405, 1308),
    (1, 2000): (1302, 1139, 1444, 1613, 847, 1917, 1400, 1244, 1760, 1907),
    (18, 1): (341, 245, 1697, 143, 943, 1662, 428, 1766, 403, 1615),
    (18, 2000): (1623, 1043, 1810, 1183, 1335, 673, 1554, 1254, 1010, 785),
}


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _integer_array(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.dtype(bool) or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{field_name} must contain integers")
    if np.any(array < minimum) or np.any(array > maximum):
        raise ValueError(f"{field_name} must lie in [{minimum}, {maximum}]")
    return array


def bootstrap_event_index(draw_position: int, retry: int = 0) -> int:
    draw = _strict_integer(draw_position, "draw_position")
    retry_value = _strict_integer(retry, "retry")
    if not 1 <= draw <= BOOTSTRAP_DRAW_COUNT:
        raise ValueError("draw_position must lie in 1..2000")
    if not 0 <= retry_value <= MASK32:
        raise ValueError("retry must be a uint32")
    return ((draw - 1) << 32) | retry_value


def bootstrap_random_address(
    condition_ordinal: int,
    bootstrap_id: int,
    draw_position: int,
    retry: int = 0,
) -> RandomAddress:
    condition = _strict_integer(condition_ordinal, "condition_ordinal")
    bootstrap = _strict_integer(bootstrap_id, "bootstrap_id")
    if not 1 <= condition <= 18:
        raise ValueError("inferential Bootstrap condition_ordinal must lie in 1..18")
    if not 1 <= bootstrap <= BOOTSTRAP_B:
        raise ValueError("bootstrap_id must lie in 1..2000")
    return RandomAddress(
        namespace=RandomNamespace.BOOTSTRAP,
        condition_ordinal=condition,
        repeat_id=bootstrap,
        variable_family=VariableFamily.BOOTSTRAP_INDEX,
        event_index=bootstrap_event_index(draw_position, retry),
    )


def bootstrap_repeat_id_scalar(
    condition_ordinal: int,
    bootstrap_id: int,
    draw_position: int,
) -> int:
    retry = 0
    while True:
        word = raw_words(
            bootstrap_random_address(
                condition_ordinal,
                bootstrap_id,
                draw_position,
                retry,
            )
        )[0]
        if word < ACCEPTANCE_LIMIT:
            return (word % FORMAL_REPEAT_COUNT) + 1
        retry += 1
        if retry > MASK32:
            raise ArithmeticError("Bootstrap rejection retry overflowed uint32")


def map_uint32_words_to_repeat_ids(words: object) -> tuple[np.ndarray, np.ndarray]:
    array = _integer_array(words, "words", minimum=0, maximum=MASK32).astype(
        np.uint64, copy=False
    )
    accepted = array < np.uint64(ACCEPTANCE_LIMIT)
    repeat_ids = np.zeros(array.shape, dtype=np.uint16)
    repeat_ids[accepted] = (
        (array[accepted] % np.uint64(FORMAL_REPEAT_COUNT)) + np.uint64(1)
    ).astype(np.uint16)
    return repeat_ids, accepted


def philox_raw_words_vectorized(
    condition_ordinal: int,
    bootstrap_ids: object,
    draw_positions: object,
    retries: object = 0,
) -> np.ndarray:
    """Return vectorized Philox words for the frozen Bootstrap address layout."""
    condition = _strict_integer(condition_ordinal, "condition_ordinal")
    if not 1 <= condition <= 18:
        raise ValueError("condition_ordinal must lie in 1..18")
    bootstrap_array = _integer_array(
        bootstrap_ids, "bootstrap_ids", minimum=1, maximum=BOOTSTRAP_B
    )
    draw_array = _integer_array(
        draw_positions,
        "draw_positions",
        minimum=1,
        maximum=BOOTSTRAP_DRAW_COUNT,
    )
    retry_array = _integer_array(retries, "retries", minimum=0, maximum=MASK32)
    bootstrap_broadcast, draw_broadcast, retry_broadcast = np.broadcast_arrays(
        bootstrap_array, draw_array, retry_array
    )
    shape = bootstrap_broadcast.shape

    c0 = np.full(shape, condition, dtype=np.uint32)
    c1 = bootstrap_broadcast.astype(np.uint32, copy=True)
    c2 = retry_broadcast.astype(np.uint32, copy=True)
    c3 = (draw_broadcast - 1).astype(np.uint32, copy=True)
    k0, k1 = derive_philox_key(
        RandomNamespace.BOOTSTRAP, VariableFamily.BOOTSTRAP_INDEX
    )

    mask64 = np.uint64(MASK32)
    for round_index in range(PHILOX_ROUNDS):
        product0 = np.uint64(PHILOX_M0) * c0.astype(np.uint64)
        product1 = np.uint64(PHILOX_M1) * c2.astype(np.uint64)
        lo0 = (product0 & mask64).astype(np.uint32)
        hi0 = (product0 >> np.uint64(32)).astype(np.uint32)
        lo1 = (product1 & mask64).astype(np.uint32)
        hi1 = (product1 >> np.uint64(32)).astype(np.uint32)
        c0, c1, c2, c3 = (
            hi1 ^ c1 ^ np.uint32(k0),
            lo1,
            hi0 ^ c3 ^ np.uint32(k1),
            lo0,
        )
        if round_index + 1 < PHILOX_ROUNDS:
            k0 = (k0 + PHILOX_W0) & MASK32
            k1 = (k1 + PHILOX_W1) & MASK32

    return np.stack((c0, c1, c2, c3), axis=-1)


def bootstrap_repeat_indices(
    condition_ordinal: int,
    bootstrap_ids: Sequence[int] | np.ndarray,
    *,
    draw_count: int = BOOTSTRAP_DRAW_COUNT,
) -> np.ndarray:
    condition = _strict_integer(condition_ordinal, "condition_ordinal")
    if not 1 <= condition <= 18:
        raise ValueError("NI and non-inference conditions cannot enter Bootstrap")
    draws = _strict_integer(draw_count, "draw_count")
    if not 1 <= draws <= BOOTSTRAP_DRAW_COUNT:
        raise ValueError("draw_count must lie in 1..2000")
    bootstrap_array = _integer_array(
        bootstrap_ids, "bootstrap_ids", minimum=1, maximum=BOOTSTRAP_B
    ).reshape(-1)
    if bootstrap_array.size == 0:
        raise ValueError("bootstrap_ids cannot be empty")
    if np.unique(bootstrap_array).size != bootstrap_array.size:
        raise ValueError("bootstrap_ids must be unique")

    bootstrap_matrix = bootstrap_array[:, None]
    draw_matrix = np.arange(1, draws + 1, dtype=np.uint32)[None, :]
    retries = np.zeros((bootstrap_array.size, draws), dtype=np.uint32)
    result = np.zeros(retries.shape, dtype=np.uint16)
    pending = np.ones(retries.shape, dtype=bool)
    while np.any(pending):
        words = philox_raw_words_vectorized(
            condition, bootstrap_matrix, draw_matrix, retries
        )[..., 0]
        accepted_now = pending & (words < np.uint32(ACCEPTANCE_LIMIT))
        result[accepted_now] = (
            (words[accepted_now].astype(np.uint64) % FORMAL_REPEAT_COUNT) + 1
        ).astype(np.uint16)
        pending &= ~accepted_now
        if np.any(pending):
            if np.any(retries[pending] == np.uint32(MASK32)):
                raise ArithmeticError("Bootstrap rejection retry overflowed uint32")
            retries[pending] += np.uint32(1)
    return result


def validate_scalar_vector_and_golden_rng() -> dict[str, object]:
    scalar_vector_checks = 0
    for condition, bootstrap_id in GOLDEN_REPEAT_IDS:
        draws = np.arange(1, 11, dtype=np.uint32)
        vector_words = philox_raw_words_vectorized(
            condition,
            np.full(10, bootstrap_id, dtype=np.uint32),
            draws,
            np.zeros(10, dtype=np.uint32),
        )
        vector_ids = bootstrap_repeat_indices(
            condition, [bootstrap_id], draw_count=10
        )[0]
        scalar_ids: list[int] = []
        for offset, draw in enumerate(range(1, 11)):
            address = bootstrap_random_address(condition, bootstrap_id, draw, 0)
            scalar = np.asarray(raw_words(address), dtype=np.uint32)
            if not np.array_equal(vector_words[offset], scalar):
                raise AssertionError(
                    f"vector Philox mismatch at C={condition}, B={bootstrap_id}, D={draw}"
                )
            scalar_vector_checks += 1
            scalar_ids.append(bootstrap_repeat_id_scalar(condition, bootstrap_id, draw))
        expected = GOLDEN_REPEAT_IDS[(condition, bootstrap_id)]
        if tuple(int(value) for value in vector_ids) != expected:
            raise AssertionError(
                f"golden Bootstrap RepeatID mismatch at C={condition}, B={bootstrap_id}"
            )
        if tuple(scalar_ids) != expected:
            raise AssertionError(
                f"scalar golden Bootstrap RepeatID mismatch at C={condition}, B={bootstrap_id}"
            )
    return {
        "ScalarVectorRawWordCheckCount": scalar_vector_checks,
        "GoldenVectorGroupCount": len(GOLDEN_REPEAT_IDS),
        "GoldenRepeatIDCount": sum(len(values) for values in GOLDEN_REPEAT_IDS.values()),
        "Passed": True,
    }


@dataclass(frozen=True, slots=True)
class BootstrapMetricBlock:
    individual_values: np.ndarray
    individual_defined: np.ndarray
    paired_values: np.ndarray
    paired_defined: np.ndarray
    statistic_sums: np.ndarray


@dataclass(frozen=True, slots=True)
class ConditionBootstrapResult:
    condition_ordinal: int
    bootstrap_ids: np.ndarray
    individual_values: np.ndarray
    individual_defined: np.ndarray
    paired_values: np.ndarray
    paired_defined: np.ndarray
    audit_ids: np.ndarray
    audit_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class PointEstimateResult:
    values: np.ndarray
    defined: np.ndarray
    sum_a: np.ndarray
    sum_b: np.ndarray
    positive_denominator_repeat_count: np.ndarray
    undefined_reason: np.ndarray


def _validate_condition_statistics(statistics: object) -> np.ndarray:
    array = np.asarray(statistics)
    expected = (len(METHOD_CODES), len(STATISTIC_FIELDS), FORMAL_REPEAT_COUNT)
    if array.shape != expected:
        raise ValueError(f"condition statistics must have shape {expected}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("condition statistics must be integer counts")
    if np.any(array < 0):
        raise ValueError("condition statistics cannot contain negative counts")
    return array.astype(np.int64, copy=False)


def compute_bootstrap_metric_block(
    statistics: object,
    repeat_ids: object,
) -> BootstrapMetricBlock:
    counts = _validate_condition_statistics(statistics)
    index_array = _integer_array(
        repeat_ids,
        "repeat_ids",
        minimum=1,
        maximum=FORMAL_REPEAT_COUNT,
    )
    if index_array.ndim != 2 or index_array.shape[1] != BOOTSTRAP_DRAW_COUNT:
        raise ValueError("repeat_ids must have shape (BootstrapBlock, 2000)")
    zero_based = index_array.astype(np.int64, copy=False) - 1
    selected = np.take(counts, zero_based, axis=2)
    sums = selected.sum(axis=-1, dtype=np.int64)
    block_count = index_array.shape[0]

    individual_values = np.full(
        (len(METHOD_CODES), len(METRIC_CODES), block_count),
        np.nan,
        dtype=np.float64,
    )
    individual_defined = np.zeros(individual_values.shape, dtype=bool)

    n_c = sums[:, STATISTIC_INDEX["N_C"], :]
    individual_values[:, 0, :] = n_c / float(BOOTSTRAP_DRAW_COUNT * K)
    individual_defined[:, 0, :] = True

    ratio_fields = (
        ("N_CC", "N_Cdot"),
        ("N_EC", "N_Edot"),
        ("Sum_L_NC_obs", "N_run_NC"),
    )
    for metric_index, (numerator_name, denominator_name) in enumerate(
        ratio_fields, start=1
    ):
        numerator = sums[:, STATISTIC_INDEX[numerator_name], :]
        denominator = sums[:, STATISTIC_INDEX[denominator_name], :]
        defined = denominator > 0
        np.divide(
            numerator,
            denominator,
            out=individual_values[:, metric_index, :],
            where=defined,
        )
        individual_defined[:, metric_index, :] = defined

    paired_values = np.full(
        (len(CONTRAST_CODES), len(METRIC_CODES), block_count),
        np.nan,
        dtype=np.float64,
    )
    paired_defined = np.zeros(paired_values.shape, dtype=bool)
    for contrast_index, comparator_index in enumerate(COMPARATOR_METHOD_ORDINALS):
        defined = (
            individual_defined[TW_METHOD_ORDINAL]
            & individual_defined[comparator_index]
        )
        paired_defined[contrast_index] = defined
        np.subtract(
            individual_values[TW_METHOD_ORDINAL],
            individual_values[comparator_index],
            out=paired_values[contrast_index],
            where=defined,
        )

    if not np.array_equal(np.isnan(individual_values), ~individual_defined):
        raise AssertionError("individual NaN/defined mask invariant failed")
    if not np.array_equal(np.isnan(paired_values), ~paired_defined):
        raise AssertionError("paired NaN/defined mask invariant failed")
    return BootstrapMetricBlock(
        individual_values=individual_values,
        individual_defined=individual_defined,
        paired_values=paired_values,
        paired_defined=paired_defined,
        statistic_sums=sums,
    )


def run_condition_bootstrap(
    statistics: object,
    *,
    condition_ordinal: int,
    bootstrap_ids: Iterable[int] = range(1, BOOTSTRAP_B + 1),
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    audit_ids: Sequence[int] = BOOTSTRAP_AUDIT_IDS,
) -> ConditionBootstrapResult:
    counts = _validate_condition_statistics(statistics)
    condition = _strict_integer(condition_ordinal, "condition_ordinal")
    if not 1 <= condition <= 18:
        raise ValueError("NI never enters the inferential Bootstrap engine")
    ids = _integer_array(
        tuple(bootstrap_ids), "bootstrap_ids", minimum=1, maximum=BOOTSTRAP_B
    ).reshape(-1)
    if ids.size == 0 or np.unique(ids).size != ids.size:
        raise ValueError("bootstrap_ids must be a non-empty unique sequence")
    block = _strict_integer(block_size, "block_size")
    if block < 1:
        raise ValueError("block_size must be positive")
    audit = _integer_array(
        tuple(audit_ids), "audit_ids", minimum=1, maximum=BOOTSTRAP_B
    ).reshape(-1)
    if np.unique(audit).size != audit.size:
        raise ValueError("audit_ids must be unique")
    missing_audit = sorted(set(int(value) for value in audit) - set(int(value) for value in ids))
    if missing_audit:
        raise ValueError(f"audit_ids are absent from bootstrap_ids: {missing_audit}")

    individual_values = np.full(
        (len(METHOD_CODES), len(METRIC_CODES), ids.size), np.nan, dtype=np.float64
    )
    individual_defined = np.zeros(individual_values.shape, dtype=bool)
    paired_values = np.full(
        (len(CONTRAST_CODES), len(METRIC_CODES), ids.size), np.nan, dtype=np.float64
    )
    paired_defined = np.zeros(paired_values.shape, dtype=bool)
    audit_indices = np.zeros(
        (audit.size, BOOTSTRAP_DRAW_COUNT), dtype=np.uint16
    )
    audit_position = {int(value): index for index, value in enumerate(audit)}

    for start in range(0, ids.size, block):
        stop = min(start + block, ids.size)
        block_ids = ids[start:stop]
        indices = bootstrap_repeat_indices(condition, block_ids)
        metrics = compute_bootstrap_metric_block(counts, indices)
        individual_values[:, :, start:stop] = metrics.individual_values
        individual_defined[:, :, start:stop] = metrics.individual_defined
        paired_values[:, :, start:stop] = metrics.paired_values
        paired_defined[:, :, start:stop] = metrics.paired_defined
        for local_index, bootstrap_id in enumerate(block_ids):
            target = audit_position.get(int(bootstrap_id))
            if target is not None:
                audit_indices[target] = indices[local_index]

    if audit.size and np.any(audit_indices == 0):
        raise AssertionError("an audit Bootstrap index was not captured from the primary run")
    if not np.array_equal(np.isnan(individual_values), ~individual_defined):
        raise AssertionError("condition individual NaN/defined mask invariant failed")
    if not np.array_equal(np.isnan(paired_values), ~paired_defined):
        raise AssertionError("condition paired NaN/defined mask invariant failed")
    return ConditionBootstrapResult(
        condition_ordinal=condition,
        bootstrap_ids=ids.astype(np.int16, copy=False),
        individual_values=individual_values,
        individual_defined=individual_defined,
        paired_values=paired_values,
        paired_defined=paired_defined,
        audit_ids=audit.astype(np.int16, copy=False),
        audit_indices=audit_indices,
    )


def formal_point_estimates(statistics: object) -> PointEstimateResult:
    counts = _validate_condition_statistics(statistics)
    sums = counts.sum(axis=2, dtype=np.int64)
    values = np.full((len(METHOD_CODES), len(METRIC_CODES)), np.nan, dtype=np.float64)
    defined = np.zeros(values.shape, dtype=bool)
    sum_a = np.zeros(values.shape, dtype=np.int64)
    sum_b = np.zeros(values.shape, dtype=np.int64)
    positive = np.zeros(values.shape, dtype=np.int64)
    reasons = np.full(values.shape, "", dtype="<U64")

    sum_a[:, 0] = sums[:, STATISTIC_INDEX["N_C"]]
    sum_b[:, 0] = FORMAL_REPEAT_COUNT * K
    positive[:, 0] = FORMAL_REPEAT_COUNT
    values[:, 0] = sum_a[:, 0] / sum_b[:, 0]
    defined[:, 0] = True

    ratio_fields = (
        ("N_CC", "N_Cdot", "C_DENOM_ZERO"),
        ("N_EC", "N_Edot", "E_DENOM_ZERO"),
        ("Sum_L_NC_obs", "N_run_NC", "NO_NC_RUN"),
    )
    for metric_index, (numerator_name, denominator_name, reason) in enumerate(
        ratio_fields, start=1
    ):
        numerator_rows = counts[:, STATISTIC_INDEX[numerator_name], :]
        denominator_rows = counts[:, STATISTIC_INDEX[denominator_name], :]
        sum_a[:, metric_index] = numerator_rows.sum(axis=1, dtype=np.int64)
        sum_b[:, metric_index] = denominator_rows.sum(axis=1, dtype=np.int64)
        positive[:, metric_index] = np.count_nonzero(denominator_rows > 0, axis=1)
        metric_defined = sum_b[:, metric_index] > 0
        defined[:, metric_index] = metric_defined
        reasons[~metric_defined, metric_index] = reason
        np.divide(
            sum_a[:, metric_index],
            sum_b[:, metric_index],
            out=values[:, metric_index],
            where=metric_defined,
        )
    if not np.array_equal(np.isnan(values), ~defined):
        raise AssertionError("formal point NaN/defined mask invariant failed")
    return PointEstimateResult(
        values=values,
        defined=defined,
        sum_a=sum_a,
        sum_b=sum_b,
        positive_denominator_repeat_count=positive,
        undefined_reason=reasons,
    )


def percentile_interval(
    values: object,
    defined: object,
) -> tuple[bool, str, float, float]:
    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    defined_array = np.asarray(defined, dtype=bool).reshape(-1)
    if value_array.shape != defined_array.shape:
        raise ValueError("values and defined must have the same shape")
    if not np.array_equal(np.isnan(value_array), ~defined_array):
        raise ValueError("undefined values must be NaN and only undefined values may be NaN")
    selected = value_array[defined_array]
    if selected.size < 2:
        return False, CI_UNDEFINED_REASON, np.nan, np.nan
    lower = float(np.quantile(selected, 0.025, method="linear"))
    upper = float(np.quantile(selected, 0.975, method="linear"))
    return True, "", lower, upper


__all__ = [
    "ACCEPTANCE_LIMIT",
    "BOOTSTRAP_AUDIT_IDS",
    "BOOTSTRAP_B",
    "BOOTSTRAP_BLOCK_SIZE",
    "BOOTSTRAP_DRAW_COUNT",
    "CI_UNDEFINED_REASON",
    "COMPARATOR_METHOD_ORDINALS",
    "CONTRAST_CODES",
    "ConditionBootstrapResult",
    "FORMAL_REPEAT_COUNT",
    "GOLDEN_REPEAT_IDS",
    "K",
    "METHOD_CODES",
    "METHOD_ORDINALS",
    "METRIC_CODES",
    "PointEstimateResult",
    "STATISTIC_FIELDS",
    "STATISTIC_INDEX",
    "TW_METHOD_ORDINAL",
    "UINT32_SIZE",
    "bootstrap_event_index",
    "bootstrap_random_address",
    "bootstrap_repeat_id_scalar",
    "bootstrap_repeat_indices",
    "compute_bootstrap_metric_block",
    "formal_point_estimates",
    "map_uint32_words_to_repeat_ids",
    "percentile_interval",
    "philox_raw_words_vectorized",
    "run_condition_bootstrap",
    "validate_scalar_vector_and_golden_rng",
]
