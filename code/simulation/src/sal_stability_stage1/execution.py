from __future__ import annotations

import multiprocessing
import operator
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from .diagnostics import CycleDiagnostic, canonical_diagnostics
from .events import condition_spec_from_config
from .kernel import (
    DEFAULT_METHOD_EXECUTION_ORDER,
    METHOD_IDENTITIES,
    SOURCE_N,
    simulate_shared_world_controlled,
)
from .metrics import RepeatMetrics, repeat_metrics_from_trajectory
from .rng import RandomNamespace
from .stage3 import build_shared_physical_world
from .stage5 import (
    H300_CONDITION_CODE,
    H300WidthStratumMetrics,
    WIDTH_STRATA,
    compute_h300_width_strata,
    level_strata_match_physical_width_difference,
)


ORDER_A = DEFAULT_METHOD_EXECUTION_ORDER
ORDER_B = (4, 3, 2, 1, 0)
ORDER_C = (2, 4, 0, 3, 1)


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _strict_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be bool")
    return value


def _method_order(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("method_execution_order must be a sequence of ordinals")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError("method_execution_order must be a sequence of ordinals") from exc
    order = tuple(
        _strict_integer(value, "method_execution_order ordinal") for value in raw
    )
    if len(order) != 5 or set(order) != set(DEFAULT_METHOD_EXECUTION_ORDER):
        raise ValueError("method_execution_order must be a complete permutation of 0..4")
    return order


def _repeat_ids(values: Iterable[int]) -> tuple[int, ...]:
    materialized = tuple(_strict_integer(value, "RepeatID") for value in values)
    if not materialized:
        raise ValueError("repeat_ids cannot be empty")
    if any(value < 1 for value in materialized):
        raise ValueError("RepeatID must be positive")
    if len(materialized) != len(set(materialized)):
        raise ValueError("repeat_ids cannot contain duplicates")
    return tuple(sorted(materialized))


def _readonly_array(
    values: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    field_name: str,
) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if result.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    workers: int = 1
    chunk_size: int = 5
    resume: bool = False
    diagnostic: bool = False
    h_cache_enabled: bool = True
    method_execution_order: tuple[int, ...] = DEFAULT_METHOD_EXECUTION_ORDER

    def __post_init__(self) -> None:
        workers = _strict_integer(self.workers, "workers")
        chunk_size = _strict_integer(self.chunk_size, "chunk_size")
        if workers < 1:
            raise ValueError("workers must be at least one")
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least one")
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "resume", _strict_bool(self.resume, "resume"))
        object.__setattr__(
            self,
            "diagnostic",
            _strict_bool(self.diagnostic, "diagnostic"),
        )
        object.__setattr__(
            self,
            "h_cache_enabled",
            _strict_bool(self.h_cache_enabled, "h_cache_enabled"),
        )
        object.__setattr__(
            self,
            "method_execution_order",
            _method_order(self.method_execution_order),
        )


@dataclass(frozen=True, slots=True)
class ConditionRepeatResult:
    namespace: RandomNamespace
    condition_ordinal: int
    condition_code: str
    repeat_id: int
    source_state_code: np.ndarray
    ref_toa_before_s: np.ndarray
    selected_observed_toa_s: np.ndarray
    has_selection: np.ndarray
    repeat_metrics: tuple[RepeatMetrics, ...]
    h300_width_strata: tuple[H300WidthStratumMetrics, ...]
    diagnostics: tuple[CycleDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        try:
            namespace = RandomNamespace(_strict_integer(self.namespace, "namespace"))
        except ValueError as exc:
            raise ValueError("namespace is invalid") from exc
        if namespace not in (RandomNamespace.PILOT, RandomNamespace.FORMAL):
            raise ValueError("execution supports only PILOT or FORMAL physical namespaces")
        condition_ordinal = _strict_integer(
            self.condition_ordinal, "condition_ordinal"
        )
        repeat_id = _strict_integer(self.repeat_id, "repeat_id")
        condition_code = str(self.condition_code)
        if not 0 <= condition_ordinal < 19:
            raise ValueError("condition_ordinal must lie in 0..18")
        if repeat_id < 1:
            raise ValueError("repeat_id must be positive")
        if not condition_code:
            raise ValueError("condition_code cannot be empty")

        source = _readonly_array(
            self.source_state_code,
            dtype=np.dtype(np.uint8),
            shape=(5, 200),
            field_name="source_state_code",
        )
        reference = _readonly_array(
            self.ref_toa_before_s,
            dtype=np.dtype(np.float64),
            shape=(5, 200),
            field_name="ref_toa_before_s",
        )
        selected = _readonly_array(
            self.selected_observed_toa_s,
            dtype=np.dtype(np.float64),
            shape=(5, 200),
            field_name="selected_observed_toa_s",
        )
        has_selection = _readonly_array(
            self.has_selection,
            dtype=np.dtype(np.bool_),
            shape=(5, 200),
            field_name="has_selection",
        )
        if not np.array_equal(has_selection, source != SOURCE_N):
            raise ValueError("has_selection disagrees with source_state_code")
        if np.any(~np.isfinite(reference)):
            raise ValueError("ref_toa_before_s must be finite")
        if np.any(~np.isfinite(selected[has_selection])):
            raise ValueError("selected TOAs must be finite where selected")
        if np.any(~np.isnan(selected[~has_selection])):
            raise ValueError("selected TOAs must be NaN where no selection exists")

        metrics = tuple(self.repeat_metrics)
        expected_metric_identity = tuple(
            (condition_ordinal, condition_code, method_ordinal, method_code, repeat_id)
            for method_ordinal, method_code in METHOD_IDENTITIES
        )
        actual_metric_identity = tuple(
            (
                row.ConditionOrdinal,
                row.ConditionCode,
                row.MethodOrdinal,
                row.MethodCode,
                row.RepeatID,
            )
            for row in metrics
        )
        if actual_metric_identity != expected_metric_identity:
            raise ValueError("RepeatMetrics identity or method order is not canonical")

        strata = tuple(self.h300_width_strata)
        if condition_code == H300_CONDITION_CODE:
            if len(strata) != 15:
                raise ValueError("H300 requires exactly 15 width-stratum rows")
        elif strata:
            raise ValueError("only H300 can contain width-stratum rows")
        if any(
            row.ConditionOrdinal != condition_ordinal
            or row.ConditionCode != condition_code
            or row.RepeatID != repeat_id
            for row in strata
        ):
            raise ValueError("width-stratum identity disagrees with result")

        diagnostics = canonical_diagnostics(self.diagnostics)
        if diagnostics and len(diagnostics) != 5 * 200:
            raise ValueError("diagnostic mode must produce one record per method-cycle")
        if any(
            record.Namespace != namespace.name
            or record.ConditionOrdinal != condition_ordinal
            or record.ConditionCode != condition_code
            or record.RepeatID != repeat_id
            for record in diagnostics
        ):
            raise ValueError("diagnostic identity disagrees with result")

        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "condition_ordinal", condition_ordinal)
        object.__setattr__(self, "condition_code", condition_code)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "source_state_code", source)
        object.__setattr__(self, "ref_toa_before_s", reference)
        object.__setattr__(self, "selected_observed_toa_s", selected)
        object.__setattr__(self, "has_selection", has_selection)
        object.__setattr__(self, "repeat_metrics", metrics)
        object.__setattr__(self, "h300_width_strata", strata)
        object.__setattr__(self, "diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class ConditionRunResult:
    namespace: RandomNamespace
    condition_ordinal: int
    condition_code: str
    repeat_ids: tuple[int, ...]
    source_state: np.ndarray
    ref_toa_before_s: np.ndarray
    selected_observed_toa_s: np.ndarray
    has_selection: np.ndarray
    repeat_metrics: tuple[RepeatMetrics, ...]
    h300_width_strata: tuple[H300WidthStratumMetrics, ...]
    diagnostics: tuple[CycleDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        try:
            namespace = RandomNamespace(_strict_integer(self.namespace, "namespace"))
        except ValueError as exc:
            raise ValueError("namespace is invalid") from exc
        condition_ordinal = _strict_integer(
            self.condition_ordinal, "condition_ordinal"
        )
        condition_code = str(self.condition_code)
        repeat_ids = _repeat_ids(self.repeat_ids)
        if tuple(self.repeat_ids) != repeat_ids:
            raise ValueError("repeat_ids must already be in ascending canonical order")
        repeat_count = len(repeat_ids)
        shape = (5, repeat_count, 200)
        source = _readonly_array(
            self.source_state,
            dtype=np.dtype(np.uint8),
            shape=shape,
            field_name="source_state",
        )
        reference = _readonly_array(
            self.ref_toa_before_s,
            dtype=np.dtype(np.float64),
            shape=shape,
            field_name="ref_toa_before_s",
        )
        selected = _readonly_array(
            self.selected_observed_toa_s,
            dtype=np.dtype(np.float64),
            shape=shape,
            field_name="selected_observed_toa_s",
        )
        has_selection = _readonly_array(
            self.has_selection,
            dtype=np.dtype(np.bool_),
            shape=shape,
            field_name="has_selection",
        )
        if not np.array_equal(has_selection, source != SOURCE_N):
            raise ValueError("has_selection disagrees with source_state")
        if np.any(~np.isfinite(reference)):
            raise ValueError("ref_toa_before_s must be finite")
        if np.any(~np.isfinite(selected[has_selection])):
            raise ValueError("selected TOAs must be finite where selected")
        if np.any(~np.isnan(selected[~has_selection])):
            raise ValueError("selected TOAs must be NaN where no selection exists")

        metrics = tuple(self.repeat_metrics)
        expected_metric_keys = tuple(
            (repeat_id, method_ordinal)
            for repeat_id in repeat_ids
            for method_ordinal, _ in METHOD_IDENTITIES
        )
        actual_metric_keys = tuple(
            (row.RepeatID, row.MethodOrdinal) for row in metrics
        )
        if actual_metric_keys != expected_metric_keys:
            raise ValueError("RepeatMetrics are not sorted by RepeatID and MethodOrdinal")
        if any(
            row.ConditionOrdinal != condition_ordinal
            or row.ConditionCode != condition_code
            for row in metrics
        ):
            raise ValueError("RepeatMetrics condition identity disagrees with result")

        strata = tuple(self.h300_width_strata)
        stratum_rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
        if condition_code == H300_CONDITION_CODE:
            expected_strata_keys = tuple(
                (repeat_id, method_ordinal, width_stratum)
                for repeat_id in repeat_ids
                for method_ordinal, _ in METHOD_IDENTITIES
                for width_stratum in WIDTH_STRATA
            )
            actual_strata_keys = tuple(
                (row.RepeatID, row.MethodOrdinal, row.WidthStratum) for row in strata
            )
            if actual_strata_keys != expected_strata_keys:
                raise ValueError("H300 width strata are not in canonical order")
        elif strata:
            raise ValueError("only H300 can contain width-stratum rows")
        if any(
            row.ConditionOrdinal != condition_ordinal
            or row.ConditionCode != condition_code
            or row.WidthStratum not in stratum_rank
            for row in strata
        ):
            raise ValueError("width-stratum identity disagrees with result")

        diagnostics = canonical_diagnostics(self.diagnostics)
        if diagnostics and len(diagnostics) != repeat_count * 5 * 200:
            raise ValueError("diagnostics must cover every requested method-cycle")
        if any(
            record.Namespace != namespace.name
            or record.ConditionOrdinal != condition_ordinal
            or record.ConditionCode != condition_code
            or record.RepeatID not in repeat_ids
            for record in diagnostics
        ):
            raise ValueError("diagnostic identity disagrees with result")

        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "condition_ordinal", condition_ordinal)
        object.__setattr__(self, "condition_code", condition_code)
        object.__setattr__(self, "repeat_ids", repeat_ids)
        object.__setattr__(self, "source_state", source)
        object.__setattr__(self, "ref_toa_before_s", reference)
        object.__setattr__(self, "selected_observed_toa_s", selected)
        object.__setattr__(self, "has_selection", has_selection)
        object.__setattr__(self, "repeat_metrics", metrics)
        object.__setattr__(self, "h300_width_strata", strata)
        object.__setattr__(self, "diagnostics", diagnostics)


def run_condition_repeat(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    execution_options: ExecutionOptions,
) -> ConditionRepeatResult:
    if not isinstance(execution_options, ExecutionOptions):
        raise TypeError("execution_options must be ExecutionOptions")
    condition_ordinal = _strict_integer(condition_ordinal, "condition_ordinal")
    repeat_id = _strict_integer(repeat_id, "repeat_id")
    if repeat_id < 1:
        raise ValueError("repeat_id must be positive")
    condition = condition_spec_from_config(study_config, condition_ordinal)

    world = build_shared_physical_world(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )
    diagnostic_records: list[CycleDiagnostic] = []
    trajectory = simulate_shared_world_controlled(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
        method_execution_order=execution_options.method_execution_order,
        h_cache_enabled=execution_options.h_cache_enabled,
        diagnostic=execution_options.diagnostic,
        diagnostic_collector=(diagnostic_records if execution_options.diagnostic else None),
    )
    metrics = repeat_metrics_from_trajectory(trajectory)
    strata: tuple[H300WidthStratumMetrics, ...] = ()
    if condition.Code == H300_CONDITION_CODE:
        if world.hprf_state is None:
            raise ValueError("H300 world is missing its HPRF repeat state")
        if not level_strata_match_physical_width_difference(
            h_true_width_level=world.hprf_state.true_width_level,
            reference_width_level=encoding_reference["width_level"],
            reference_width_s=encoding_reference["ref_width_s"],
        ):
            raise ValueError("H300 level strata disagree with physical widths")
        strata = compute_h300_width_strata(
            trajectory=trajectory,
            h_true_width_level=world.hprf_state.true_width_level,
            reference_width_level=encoding_reference["width_level"],
        )
    return ConditionRepeatResult(
        namespace=trajectory.namespace,
        condition_ordinal=trajectory.condition_ordinal,
        condition_code=trajectory.condition_code,
        repeat_id=trajectory.repeat_id,
        source_state_code=trajectory.source_state_code,
        ref_toa_before_s=trajectory.ref_toa_before_s,
        selected_observed_toa_s=trajectory.selected_observed_toa_s,
        has_selection=trajectory.has_selection,
        repeat_metrics=metrics,
        h300_width_strata=strata,
        diagnostics=tuple(diagnostic_records),
    )


def assemble_condition_repeat_results(
    results: Iterable[ConditionRepeatResult],
    *,
    expected_repeat_ids: Iterable[int] | None = None,
) -> ConditionRunResult:
    materialized = tuple(results)
    if not materialized:
        raise ValueError("at least one ConditionRepeatResult is required")
    if any(not isinstance(result, ConditionRepeatResult) for result in materialized):
        raise TypeError("all values must be ConditionRepeatResult instances")
    namespaces = {result.namespace for result in materialized}
    condition_ordinals = {result.condition_ordinal for result in materialized}
    condition_codes = {result.condition_code for result in materialized}
    if len(namespaces) != 1 or len(condition_ordinals) != 1 or len(condition_codes) != 1:
        raise ValueError("cannot assemble results from different conditions or namespaces")
    repeat_ids = tuple(result.repeat_id for result in materialized)
    if len(repeat_ids) != len(set(repeat_ids)):
        raise ValueError("cannot assemble duplicate RepeatID values")
    canonical_repeat_ids = tuple(sorted(repeat_ids))
    if expected_repeat_ids is not None:
        expected = _repeat_ids(expected_repeat_ids)
        if canonical_repeat_ids != expected:
            raise ValueError("assembled results contain a RepeatID gap or unexpected RepeatID")
    ordered = tuple(sorted(materialized, key=lambda result: result.repeat_id))

    source = np.stack(
        tuple(result.source_state_code for result in ordered), axis=1
    )
    reference = np.stack(
        tuple(result.ref_toa_before_s for result in ordered), axis=1
    )
    selected = np.stack(
        tuple(result.selected_observed_toa_s for result in ordered), axis=1
    )
    has_selection = np.stack(
        tuple(result.has_selection for result in ordered), axis=1
    )
    metrics = tuple(
        row for result in ordered for row in result.repeat_metrics
    )
    strata = tuple(
        row for result in ordered for row in result.h300_width_strata
    )
    diagnostics = canonical_diagnostics(
        record for result in ordered for record in result.diagnostics
    )
    first = ordered[0]
    return ConditionRunResult(
        namespace=first.namespace,
        condition_ordinal=first.condition_ordinal,
        condition_code=first.condition_code,
        repeat_ids=canonical_repeat_ids,
        source_state=source,
        ref_toa_before_s=reference,
        selected_observed_toa_s=selected,
        has_selection=has_selection,
        repeat_metrics=metrics,
        h300_width_strata=strata,
        diagnostics=diagnostics,
    )


def run_condition_repeats_serial(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_ids: Iterable[int],
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    execution_options: ExecutionOptions,
) -> ConditionRunResult:
    canonical_repeat_ids = _repeat_ids(repeat_ids)
    results = tuple(
        run_condition_repeat(
            namespace=namespace,
            condition_ordinal=condition_ordinal,
            repeat_id=repeat_id,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=execution_options,
        )
        for repeat_id in canonical_repeat_ids
    )
    return assemble_condition_repeat_results(
        results,
        expected_repeat_ids=canonical_repeat_ids,
    )


def _process_condition_repeat(
    namespace_value: int,
    condition_ordinal: int,
    repeat_id: int,
    study_config: dict[str, object],
    encoding_reference: dict[str, np.ndarray],
    execution_options: ExecutionOptions,
) -> ConditionRepeatResult:
    """Module-level spawn-safe worker for one scientific Condition-Repeat unit."""
    return run_condition_repeat(
        namespace=RandomNamespace(namespace_value),
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        study_config=study_config,
        encoding_reference=encoding_reference,
        execution_options=execution_options,
    )


def run_condition_repeats_parallel(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_ids: Iterable[int],
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    execution_options: ExecutionOptions,
) -> ConditionRunResult:
    if not isinstance(execution_options, ExecutionOptions):
        raise TypeError("execution_options must be ExecutionOptions")
    canonical_repeat_ids = _repeat_ids(repeat_ids)
    namespace_value = _strict_integer(namespace, "namespace")
    config_payload = dict(study_config)
    reference_payload = {
        name: np.array(values, copy=True)
        for name, values in encoding_reference.items()
    }
    spawn_context = multiprocessing.get_context("spawn")
    unordered: list[ConditionRepeatResult] = []
    with ProcessPoolExecutor(
        max_workers=execution_options.workers,
        mp_context=spawn_context,
    ) as executor:
        futures = tuple(
            executor.submit(
                _process_condition_repeat,
                namespace_value,
                condition_ordinal,
                repeat_id,
                config_payload,
                reference_payload,
                execution_options,
            )
            for repeat_id in canonical_repeat_ids
        )
        for future in as_completed(futures):
            unordered.append(future.result())
    return assemble_condition_repeat_results(
        unordered,
        expected_repeat_ids=canonical_repeat_ids,
    )


def scientific_results_equal(
    left: ConditionRunResult,
    right: ConditionRunResult,
) -> bool:
    if not isinstance(left, ConditionRunResult) or not isinstance(
        right, ConditionRunResult
    ):
        raise TypeError("both values must be ConditionRunResult instances")
    return bool(
        left.namespace == right.namespace
        and left.condition_ordinal == right.condition_ordinal
        and left.condition_code == right.condition_code
        and left.repeat_ids == right.repeat_ids
        and np.array_equal(left.source_state, right.source_state)
        and np.array_equal(left.ref_toa_before_s, right.ref_toa_before_s)
        and np.array_equal(
            left.selected_observed_toa_s,
            right.selected_observed_toa_s,
            equal_nan=True,
        )
        and np.array_equal(left.has_selection, right.has_selection)
        and left.repeat_metrics == right.repeat_metrics
        and left.h300_width_strata == right.h300_width_strata
    )


def assert_scientific_results_equal(
    left: ConditionRunResult,
    right: ConditionRunResult,
    *,
    label: str,
) -> None:
    if not scientific_results_equal(left, right):
        raise AssertionError(f"scientific execution results differ: {label}")


__all__ = [
    "ConditionRepeatResult",
    "ConditionRunResult",
    "ExecutionOptions",
    "ORDER_A",
    "ORDER_B",
    "ORDER_C",
    "assemble_condition_repeat_results",
    "assert_scientific_results_equal",
    "run_condition_repeat",
    "run_condition_repeats_parallel",
    "run_condition_repeats_serial",
    "scientific_results_equal",
]
