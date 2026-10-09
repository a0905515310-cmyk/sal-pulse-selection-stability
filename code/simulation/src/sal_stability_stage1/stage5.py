from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import inspect
import json
import operator
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .build import CONFIG_DIR, ROOT
from .kernel import (
    METHOD_IDENTITIES,
    SOURCE_F,
    SOURCE_G,
    SOURCE_H,
    SOURCE_N,
    ConditionRepeatTrajectory,
    simulate_condition_repeat,
    simulate_shared_world,
)
from .metrics import (
    C_DENOM_ZERO,
    E_COUNT_ZERO,
    E_DENOM_ZERO,
    NO_NC_RUN,
    REPEAT_COUNT_FIELDS,
    AggregatedMetrics,
    K,
    MetricValue,
    RepeatMetrics,
    aggregate_repeat_metrics,
    aggregated_metrics_to_dict,
    compute_repeat_metrics,
    reference_repeat_counts,
    repeat_metrics_field_names,
    repeat_metrics_from_trajectory,
    source_state_to_evaluation,
    streaming_repeat_counts,
)
from .rng import RandomNamespace
from .stage2 import TestRun, run_pytest, scan_forbidden_random_apis
from .stage3 import SharedPhysicalWorld, build_shared_physical_world, load_stage3_inputs


H300_CONDITION_ORDINAL = 3
H300_CONDITION_CODE = "H300"
WIDTH_STRATUM_0NS = "WIDTH_STRATUM_0NS"
WIDTH_STRATUM_10NS = "WIDTH_STRATUM_10NS"
WIDTH_STRATUM_GE20NS = "WIDTH_STRATUM_GE20NS"
WIDTH_STRATA = (
    WIDTH_STRATUM_0NS,
    WIDTH_STRATUM_10NS,
    WIDTH_STRATUM_GE20NS,
)
WIDTH_BASE_S = np.float64(250e-9)
WIDTH_STEP_S = np.float64(10e-9)

STAGE5_DIR = ROOT / "artifacts" / "stage5"
VALIDATION_REPEAT_IDS = (1, 2, 17)
FROZEN_CONFIG_SHA256 = (
    "4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb"
)
FROZEN_SELECTORS_SHA256 = (
    "bc285cb5163dc1c0631a860f312a45c61738d6502e85e9c9779e5b9ddafd5fc6"
)
FROZEN_STAGE4_SHA256 = (
    "a2c25cba787a657c5342773c7ddab0f907ded61a215c53c96222d90da9e3ffd9"
)
FROZEN_PRIVATE_KERNEL_SOURCE_SHA256 = (
    "e680f3852b1e7cbfa70f8b641e921c1e31695e2de5805fb1de6648c64dae5630"
)
FROZEN_SCIENCE_FILE_SHA256 = {
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
    "src/sal_stability_stage1/events.py": (
        "ce1c7a54cd3d2a2219a782dfe0720118bf8d4f19cbabec765bd03916bee1cab2"
    ),
    "src/sal_stability_stage1/hprf.py": (
        "be09cf938230c24c3b4dfa480a1a5ca63d24e15924bd76451b16ee050d6dd883"
    ),
    "src/sal_stability_stage1/stage3.py": (
        "59fbbf2cb6deedd2f42c2aad419c55b6df9f74231c9acb80a417377b5f516830"
    ),
    "src/sal_stability_stage1/selectors.py": FROZEN_SELECTORS_SHA256,
    "src/sal_stability_stage1/stage4.py": FROZEN_STAGE4_SHA256,
}
BASELINE_TEST_PATHS = (
    "tests/test_stage1.py",
    "tests/test_stage2_rng.py",
    "tests/test_stage3_events.py",
    "tests/test_stage4_selectors.py",
    "tests/test_stage4_kernel.py",
)


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _reference_width_levels(reference_width_level: object) -> np.ndarray:
    if not isinstance(reference_width_level, np.ndarray):
        raise TypeError("reference_width_level must be a numpy.ndarray")
    if reference_width_level.dtype != np.dtype(np.uint8):
        raise TypeError("reference_width_level must have dtype uint8")
    if reference_width_level.shape != (K,):
        raise ValueError("reference_width_level must have shape (200,)")
    if np.any(reference_width_level > np.uint8(15)):
        raise ValueError("reference_width_level must lie in 0..15")
    return reference_width_level


def width_strata_from_levels(
    *, h_true_width_level: object, reference_width_level: object
) -> np.ndarray:
    """Classify H300 cycles by integer width-level difference only."""
    h_level = _strict_integer(h_true_width_level, "h_true_width_level")
    if not 0 <= h_level <= 15:
        raise ValueError("h_true_width_level must lie in 0..15")
    reference_levels = _reference_width_levels(reference_width_level)
    level_difference = np.abs(reference_levels.astype(np.int16) - h_level)
    strata = np.empty(K, dtype=f"<U{len(WIDTH_STRATUM_GE20NS)}")
    strata[level_difference == 0] = WIDTH_STRATUM_0NS
    strata[level_difference == 1] = WIDTH_STRATUM_10NS
    strata[level_difference >= 2] = WIDTH_STRATUM_GE20NS
    strata.setflags(write=False)
    return strata


def level_strata_match_physical_width_difference(
    *,
    h_true_width_level: object,
    reference_width_level: object,
    reference_width_s: object,
) -> bool:
    """Validation oracle showing that discrete levels encode 10 ns width steps."""
    h_level = _strict_integer(h_true_width_level, "h_true_width_level")
    if not 0 <= h_level <= 15:
        raise ValueError("h_true_width_level must lie in 0..15")
    reference_levels = _reference_width_levels(reference_width_level)
    if not isinstance(reference_width_s, np.ndarray):
        raise TypeError("reference_width_s must be a numpy.ndarray")
    if reference_width_s.dtype != np.dtype(np.float64):
        raise TypeError("reference_width_s must have dtype float64")
    if reference_width_s.shape != (K,) or not np.all(np.isfinite(reference_width_s)):
        raise ValueError("reference_width_s must be a finite shape-(200,) array")
    expected_reference_width = (
        WIDTH_BASE_S
        + reference_levels.astype(np.float64) * WIDTH_STEP_S
    )
    if not np.array_equal(reference_width_s, expected_reference_width):
        return False

    h_true_width_s = np.float64(WIDTH_BASE_S + h_level * WIDTH_STEP_S)
    physical_step_difference = np.rint(
        np.abs(reference_width_s - h_true_width_s) / WIDTH_STEP_S
    ).astype(np.int16)
    level_difference = np.abs(reference_levels.astype(np.int16) - h_level)
    if not np.array_equal(physical_step_difference, level_difference):
        return False
    physical_strata = np.empty(K, dtype=f"<U{len(WIDTH_STRATUM_GE20NS)}")
    physical_strata[physical_step_difference == 0] = WIDTH_STRATUM_0NS
    physical_strata[physical_step_difference == 1] = WIDTH_STRATUM_10NS
    physical_strata[physical_step_difference >= 2] = WIDTH_STRATUM_GE20NS
    return bool(
        np.array_equal(
            physical_strata,
            width_strata_from_levels(
                h_true_width_level=h_level,
                reference_width_level=reference_levels,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class H300WidthStratumMetrics:
    ConditionOrdinal: int
    ConditionCode: str
    MethodOrdinal: int
    MethodCode: str
    RepeatID: int
    WidthStratum: str
    StratumCycleCount: int
    StratumCorrectCount: int
    StratumErrorCount: int
    Stratum_C_to_E_Count: int

    def __post_init__(self) -> None:
        condition_ordinal = _strict_integer(
            self.ConditionOrdinal, "ConditionOrdinal"
        )
        method_ordinal = _strict_integer(self.MethodOrdinal, "MethodOrdinal")
        repeat_id = _strict_integer(self.RepeatID, "RepeatID")
        if (
            condition_ordinal != H300_CONDITION_ORDINAL
            or str(self.ConditionCode) != H300_CONDITION_CODE
        ):
            raise ValueError("width-stratum metrics are defined only for H300")
        if not 0 <= method_ordinal < len(METHOD_IDENTITIES):
            raise ValueError("MethodOrdinal must lie in 0..4")
        if METHOD_IDENTITIES[method_ordinal] != (
            method_ordinal,
            str(self.MethodCode),
        ):
            raise ValueError("Method identity is not the frozen 5-method identity")
        if repeat_id < 1:
            raise ValueError("RepeatID must be positive")
        width_stratum = str(self.WidthStratum)
        if width_stratum not in WIDTH_STRATA:
            raise ValueError("WidthStratum is not one of the three frozen strata")
        object.__setattr__(self, "ConditionOrdinal", condition_ordinal)
        object.__setattr__(self, "ConditionCode", H300_CONDITION_CODE)
        object.__setattr__(self, "MethodOrdinal", method_ordinal)
        object.__setattr__(self, "MethodCode", str(self.MethodCode))
        object.__setattr__(self, "RepeatID", repeat_id)
        object.__setattr__(self, "WidthStratum", width_stratum)

        count_names = (
            "StratumCycleCount",
            "StratumCorrectCount",
            "StratumErrorCount",
            "Stratum_C_to_E_Count",
        )
        for name in count_names:
            count = _strict_integer(getattr(self, name), name)
            if count < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, count)
        if self.StratumCorrectCount + self.StratumErrorCount != self.StratumCycleCount:
            raise ValueError("H300 stratum C/E counts must conserve cycle count")
        if self.Stratum_C_to_E_Count > self.StratumCycleCount:
            raise ValueError("stratum C-to-E count cannot exceed its cycle count")


def compute_h300_width_strata(
    *,
    trajectory: ConditionRepeatTrajectory,
    h_true_width_level: object,
    reference_width_level: object,
) -> tuple[H300WidthStratumMetrics, ...]:
    """Compute 5 x 3 H300 support rows from one already-simulated trajectory."""
    if not isinstance(trajectory, ConditionRepeatTrajectory):
        raise TypeError("trajectory must be a ConditionRepeatTrajectory")
    if (
        trajectory.condition_ordinal != H300_CONDITION_ORDINAL
        or trajectory.condition_code != H300_CONDITION_CODE
    ):
        raise ValueError("H300 width strata cannot be computed for another condition")
    source = trajectory.source_state_code
    if np.any(source == SOURCE_N):
        raise ValueError("H300 trajectory contains forbidden N state")
    if np.any(source == SOURCE_F):
        raise ValueError("H300 trajectory contains forbidden F source")
    if np.any((source != SOURCE_G) & (source != SOURCE_H)):
        raise ValueError("H300 trajectory must contain only G and H sources")

    strata_by_cycle = width_strata_from_levels(
        h_true_width_level=h_true_width_level,
        reference_width_level=reference_width_level,
    )
    rows: list[H300WidthStratumMetrics] = []
    for method_ordinal, method_code in METHOD_IDENTITIES:
        method_source = source[method_ordinal]
        correct = method_source == SOURCE_G
        error = method_source == SOURCE_H
        c_to_e = correct[:-1] & error[1:]
        for width_stratum in WIDTH_STRATA:
            cycle_mask = strata_by_cycle == width_stratum
            rows.append(
                H300WidthStratumMetrics(
                    ConditionOrdinal=trajectory.condition_ordinal,
                    ConditionCode=trajectory.condition_code,
                    MethodOrdinal=method_ordinal,
                    MethodCode=method_code,
                    RepeatID=trajectory.repeat_id,
                    WidthStratum=width_stratum,
                    StratumCycleCount=int(np.count_nonzero(cycle_mask)),
                    StratumCorrectCount=int(np.count_nonzero(correct & cycle_mask)),
                    StratumErrorCount=int(np.count_nonzero(error & cycle_mask)),
                    Stratum_C_to_E_Count=int(
                        np.count_nonzero(c_to_e & cycle_mask[:-1])
                    ),
                )
            )

    result = tuple(rows)
    for method_ordinal, _ in METHOD_IDENTITIES:
        method_rows = tuple(
            row for row in result if row.MethodOrdinal == method_ordinal
        )
        if len(method_rows) != len(WIDTH_STRATA):
            raise AssertionError("H300 must produce exactly three rows per method")
        if sum(row.StratumCycleCount for row in method_rows) != K:
            raise AssertionError("H300 width strata must conserve all 200 cycles")
    shared_cycle_counts = tuple(
        row.StratumCycleCount for row in result[: len(WIDTH_STRATA)]
    )
    for method_ordinal in range(1, len(METHOD_IDENTITIES)):
        offset = method_ordinal * len(WIDTH_STRATA)
        if tuple(
            row.StratumCycleCount
            for row in result[offset : offset + len(WIDTH_STRATA)]
        ) != shared_cycle_counts:
            raise AssertionError("H300 stratum cycle counts must be shared by methods")
    return result


@dataclass(frozen=True, slots=True)
class ConditionRepeatMetricBundle:
    trajectory: ConditionRepeatTrajectory
    repeat_metrics: tuple[RepeatMetrics, ...]
    h300_width_strata: tuple[H300WidthStratumMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ConditionRepeatTrajectory):
            raise TypeError("trajectory must be a ConditionRepeatTrajectory")
        repeat_metrics = tuple(self.repeat_metrics)
        h300_width_strata = tuple(self.h300_width_strata)
        if len(repeat_metrics) != len(METHOD_IDENTITIES):
            raise ValueError("a metric bundle requires exactly five RepeatMetrics rows")
        expected_methods = tuple(
            (row.MethodOrdinal, row.MethodCode) for row in repeat_metrics
        )
        if expected_methods != METHOD_IDENTITIES:
            raise ValueError("RepeatMetrics rows are not in frozen method order")
        if any(
            row.ConditionOrdinal != self.trajectory.condition_ordinal
            or row.ConditionCode != self.trajectory.condition_code
            or row.RepeatID != self.trajectory.repeat_id
            for row in repeat_metrics
        ):
            raise ValueError("RepeatMetrics identity disagrees with trajectory")
        if self.trajectory.condition_code == H300_CONDITION_CODE:
            if len(h300_width_strata) != len(METHOD_IDENTITIES) * len(WIDTH_STRATA):
                raise ValueError("an H300 bundle requires exactly 15 stratum rows")
        elif h300_width_strata:
            raise ValueError("non-H300 bundles cannot contain width-stratum rows")
        object.__setattr__(self, "repeat_metrics", repeat_metrics)
        object.__setattr__(self, "h300_width_strata", h300_width_strata)


def metric_bundle_from_world(
    *,
    world: SharedPhysicalWorld,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> ConditionRepeatMetricBundle:
    """Simulate one supplied world once, then read metrics from its trajectory."""
    if not isinstance(world, SharedPhysicalWorld):
        raise TypeError("world must be a SharedPhysicalWorld")
    trajectory = simulate_shared_world(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )
    repeat_metrics = repeat_metrics_from_trajectory(trajectory)
    h300_width_strata: tuple[H300WidthStratumMetrics, ...] = ()
    if trajectory.condition_code == H300_CONDITION_CODE:
        if world.hprf_state is None:
            raise ValueError("H300 shared world is missing HPRFRepeatState")
        try:
            reference_width_level = encoding_reference["width_level"]
            reference_width_s = encoding_reference["ref_width_s"]
        except KeyError as exc:
            raise ValueError(
                f"encoding reference is missing {exc.args[0]}"
            ) from exc
        if not level_strata_match_physical_width_difference(
            h_true_width_level=world.hprf_state.true_width_level,
            reference_width_level=reference_width_level,
            reference_width_s=reference_width_s,
        ):
            raise ValueError(
                "H300 level strata disagree with the physical 10 ns width encoding"
            )
        h300_width_strata = compute_h300_width_strata(
            trajectory=trajectory,
            h_true_width_level=world.hprf_state.true_width_level,
            reference_width_level=reference_width_level,
        )
    return ConditionRepeatMetricBundle(
        trajectory=trajectory,
        repeat_metrics=repeat_metrics,
        h300_width_strata=h300_width_strata,
    )


def simulate_condition_repeat_with_metrics(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> ConditionRepeatMetricBundle:
    """Build exactly one physical world, simulate it once, and compute Stage 5 metrics."""
    world = build_shared_physical_world(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )
    return metric_bundle_from_world(
        world=world,
        study_config=study_config,
        encoding_reference=encoding_reference,
    )


def h300_width_strata_field_names() -> tuple[str, ...]:
    return tuple(field_info.name for field_info in fields(H300WidthStratumMetrics))


def _write_dataclass_csv(
    path: Path,
    rows: Sequence[object],
    field_names: Sequence[str],
    key_fields: Sequence[str],
) -> None:
    path = Path(path)
    materialized = tuple(rows)
    keys = [tuple(getattr(row, name) for name in key_fields) for row in materialized]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate row key while writing {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(field_names),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in materialized:
            writer.writerow(asdict(row))


def write_repeat_metrics_csv(path: Path, rows: Sequence[RepeatMetrics]) -> None:
    if any(not isinstance(row, RepeatMetrics) for row in rows):
        raise TypeError("repeat metrics CSV rows must all be RepeatMetrics")
    _write_dataclass_csv(
        Path(path),
        rows,
        tuple(field_info.name for field_info in fields(RepeatMetrics)),
        (
            "ConditionOrdinal",
            "ConditionCode",
            "MethodOrdinal",
            "MethodCode",
            "RepeatID",
        ),
    )


def write_h300_width_strata_csv(
    path: Path, rows: Sequence[H300WidthStratumMetrics]
) -> None:
    if any(not isinstance(row, H300WidthStratumMetrics) for row in rows):
        raise TypeError("H300 CSV rows must all be H300WidthStratumMetrics")
    _write_dataclass_csv(
        Path(path),
        rows,
        h300_width_strata_field_names(),
        (
            "ConditionOrdinal",
            "ConditionCode",
            "MethodOrdinal",
            "MethodCode",
            "RepeatID",
            "WidthStratum",
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: object) -> None:
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
            allow_nan=False,
            default=normalize,
        )
        + "\n",
        encoding="utf-8",
    )


def _skipped_test_run(label: str, reason: str) -> TestRun:
    return TestRun(
        label=label,
        command=(sys.executable, "-m", "pytest"),
        returncode=125,
        passed=0,
        failed=0,
        duration_s=0.0,
        output=f"SKIPPED: {reason}",
    )


def _synthetic_trajectory(
    source_state_code: np.ndarray,
    *,
    condition_ordinal: int = H300_CONDITION_ORDINAL,
    condition_code: str = H300_CONDITION_CODE,
    repeat_id: int = 1,
) -> ConditionRepeatTrajectory:
    source = np.asarray(source_state_code, dtype=np.uint8)
    if source.shape != (5, K):
        raise ValueError("synthetic source array must have shape (5, 200)")
    has_selection = source != SOURCE_N
    selected = np.where(has_selection, np.float64(0.0), np.nan)
    return ConditionRepeatTrajectory(
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        repeat_id=repeat_id,
        namespace=RandomNamespace.PILOT,
        method_codes=tuple(code for _, code in METHOD_IDENTITIES),
        source_state_code=source,
        ref_toa_before_s=np.zeros((5, K), dtype=np.float64),
        selected_observed_toa_s=selected,
        has_selection=has_selection,
        delta_t_consumed_count=K - 1,
        hprf_count_check_count=0,
        hprf_no_n_check_count=0,
    )


def _synthetic_repeat_metrics(
    source: np.ndarray,
    *,
    repeat_id: int = 1,
) -> RepeatMetrics:
    return compute_repeat_metrics(
        source_state_code_1d=source,
        condition_ordinal=0,
        condition_code="NI",
        method_ordinal=0,
        method_code="FIRST",
        repeat_id=repeat_id,
    )


def validate_synthetic_metrics() -> dict[str, object]:
    """Execute direct mathematical checks independent of pytest exit status."""
    checks: dict[str, bool] = {}
    checks["source_state_mapping"] = tuple(
        source_state_to_evaluation(code) for code in range(4)
    ) == ("N", "C", "E", "E")

    nine_source = np.asarray(
        [1, 1, 2, 1, 0, 2, 2, 0, 0, 1] + [1] * 190,
        dtype=np.uint8,
    )
    nine = _synthetic_repeat_metrics(nine_source)
    transition_names = (
        "N_CC",
        "N_CE",
        "N_CN",
        "N_EC",
        "N_EE",
        "N_EN",
        "N_NC",
        "N_NE",
        "N_NN",
    )
    checks["all_nine_transitions"] = all(
        getattr(nine, name) >= 1 for name in transition_names
    )
    checks["transition_total_199"] = sum(
        getattr(nine, name) for name in transition_names
    ) == (K - 1)
    checks["origin_transition_conservation"] = (
        nine.N_Cdot == nine.N_CC + nine.N_CE + nine.N_CN
        and nine.N_Edot == nine.N_EC + nine.N_EE + nine.N_EN
        and nine.N_Ndot == nine.N_NC + nine.N_NE + nine.N_NN
    )
    checks["streaming_reference_identity"] = streaming_repeat_counts(
        source_state_code_1d=nine_source
    ) == reference_repeat_counts(source_state_code_1d=nine_source)

    bridged = _synthetic_repeat_metrics(
        np.asarray([2, 2, 0, 2, 1] + [1] * 195, dtype=np.uint8)
    )
    checks["n_does_not_split_nc"] = (
        bridged.N_run_NC == 1
        and bridged.Sum_L_NC_obs == 4
        and bridged.N_open_NC == 0
    )
    right_open = _synthetic_repeat_metrics(
        np.asarray([1] * 197 + [2, 0, 2], dtype=np.uint8)
    )
    checks["right_open_nc_retained"] = (
        right_open.N_run_NC == 1
        and right_open.Sum_L_NC_obs == 3
        and right_open.N_open_NC == 1
        and right_open.EndState == "E"
    )
    checks["nc_sum_length_conservation"] = (
        right_open.Sum_L_NC_obs == right_open.N_E + right_open.N_N
    )

    all_c = _synthetic_repeat_metrics(np.ones(K, dtype=np.uint8))
    all_c_aggregate = aggregate_repeat_metrics((all_c,))
    checks["undefined_metric_encoding"] = (
        all_c_aggregate.P_C_given_E
        == MetricValue(None, False, E_DENOM_ZERO)
        and all_c_aggregate.Mean_L_NC
        == MetricValue(None, False, NO_NC_RUN)
        and all_c_aggregate.P_open_NC
        == MetricValue(None, False, NO_NC_RUN)
        and all_c_aggregate.P_E_H
        == MetricValue(None, False, E_COUNT_ZERO)
        and all_c_aggregate.P_E_F
        == MetricValue(None, False, E_COUNT_ZERO)
    )
    checks["all_c_metric_values"] = (
        all_c_aggregate.P_cor == MetricValue(1.0, True, None)
        and all_c_aggregate.P_C_given_C == MetricValue(1.0, True, None)
        and all_c_aggregate.P_N == MetricValue(0.0, True, None)
        and all_c_aggregate.P_NC_end == MetricValue(0.0, True, None)
    )

    ratio_one = _synthetic_repeat_metrics(
        np.asarray([2, 1] + [1] * 198, dtype=np.uint8), repeat_id=1
    )
    ratio_two = _synthetic_repeat_metrics(
        np.asarray(([2] * 10 + [1]) * 10 + [1] * 90, dtype=np.uint8),
        repeat_id=2,
    )
    ratio_aggregate = aggregate_repeat_metrics((ratio_one, ratio_two))
    checks["total_count_ratio"] = (
        ratio_aggregate.P_C_given_E.value == 11.0 / 101.0
        and ratio_aggregate.P_C_given_E.value != (1.0 + 0.1) / 2.0
    )

    open_ratio_source = np.asarray(
        [1, 2, 1] + [1] * 194 + [1, 2, 2], dtype=np.uint8
    )
    open_ratio = aggregate_repeat_metrics(
        (_synthetic_repeat_metrics(open_ratio_source),)
    )
    checks["p_open_nc_distinct_from_p_nc_end"] = (
        open_ratio.Total_N_run_NC == 2
        and open_ratio.Total_N_open_NC == 1
        and open_ratio.P_open_NC.value == 0.5
        and open_ratio.P_NC_end.value == 1.0
    )

    sources = _synthetic_repeat_metrics(
        np.resize(np.asarray([1, 2, 3], dtype=np.uint8), K)
    )
    checks["h_f_source_conservation"] = (
        sources.N_E_H + sources.N_E_F == sources.N_E
        and sources.N_E_H > 0
        and sources.N_E_F > 0
    )

    fixed_levels = np.full(K, 5, dtype=np.uint8)
    fixed_levels[:3] = [5, 4, 3]
    fixed_strata = width_strata_from_levels(
        h_true_width_level=np.uint8(5),
        reference_width_level=fixed_levels,
    )
    checks["h300_fixed_width_strata"] = fixed_strata[:3].tolist() == [
        WIDTH_STRATUM_0NS,
        WIDTH_STRATUM_10NS,
        WIDTH_STRATUM_GE20NS,
    ]

    h300_source = np.ones((5, K), dtype=np.uint8)
    h300_source[:, 1] = SOURCE_H
    controlled_trajectory = _synthetic_trajectory(h300_source)
    anchor_levels = np.full(K, 5, dtype=np.uint8)
    anchor_levels[1] = 3
    controlled_rows = compute_h300_width_strata(
        trajectory=controlled_trajectory,
        h_true_width_level=np.uint8(5),
        reference_width_level=anchor_levels,
    )
    checks["h300_c_to_e_current_cycle_anchor"] = all(
        next(
            row
            for row in controlled_rows
            if row.MethodOrdinal == method_ordinal
            and row.WidthStratum == WIDTH_STRATUM_0NS
        ).Stratum_C_to_E_Count
        == 1
        and next(
            row
            for row in controlled_rows
            if row.MethodOrdinal == method_ordinal
            and row.WidthStratum == WIDTH_STRATUM_GE20NS
        ).Stratum_C_to_E_Count
        == 0
        for method_ordinal in range(5)
    )
    h300_field_names = tuple(
        field_info.name for field_info in fields(H300WidthStratumMetrics)
    )
    checks["h300_no_lnc_stratification"] = not any(
        "L_NC" in name for name in h300_field_names
    )
    checks["repeat_metrics_schema"] = repeat_metrics_field_names() == (
        "ConditionOrdinal",
        "ConditionCode",
        "MethodOrdinal",
        "MethodCode",
        "RepeatID",
        "N_C",
        "N_E",
        "N_N",
        "N_CC",
        "N_CE",
        "N_CN",
        "N_EC",
        "N_EE",
        "N_EN",
        "N_NC",
        "N_NE",
        "N_NN",
        "N_Cdot",
        "N_Edot",
        "N_Ndot",
        "N_run_NC",
        "Sum_L_NC_obs",
        "N_open_NC",
        "EndState",
        "N_E_H",
        "N_E_F",
    )
    private_kernel_source = inspect.getsource(
        sys.modules[simulate_shared_world.__module__]._simulate_shared_world
    ).encode("utf-8")
    checks["private_cycle_kernel_source_unchanged"] = (
        hashlib.sha256(private_kernel_source).hexdigest()
        == FROZEN_PRIVATE_KERNEL_SOURCE_SHA256
    )
    return {
        "checks": checks,
        "details": {
            "nine_transition_counts": {
                name: getattr(nine, name) for name in transition_names
            },
            "total_ratio_expected": 11.0 / 101.0,
            "per_repeat_probability_mean_rejected": (1.0 + 0.1) / 2.0,
            "undefined_reason_codes": [
                C_DENOM_ZERO,
                E_DENOM_ZERO,
                NO_NC_RUN,
                E_COUNT_ZERO,
            ],
        },
        "all_pass": all(checks.values()),
    }


def _repeat_payload(row: RepeatMetrics) -> tuple[object, ...]:
    return tuple(getattr(row, name) for name in REPEAT_COUNT_FIELDS) + (
        row.EndState,
    )


def _aggregate_payload(metrics: AggregatedMetrics) -> dict[str, object]:
    payload = aggregated_metrics_to_dict(metrics)
    for identity_name in (
        "ConditionOrdinal",
        "ConditionCode",
        "MethodOrdinal",
        "MethodCode",
    ):
        payload.pop(identity_name)
    return payload


def run_small_scale_validation(
    config: Mapping[str, object],
    reference: Mapping[str, np.ndarray],
) -> tuple[
    tuple[RepeatMetrics, ...],
    tuple[H300WidthStratumMetrics, ...],
    tuple[AggregatedMetrics, ...],
    dict[str, object],
]:
    """Run only 19 conditions x fixed RepeatIDs for implementation validation."""
    repeat_rows: list[RepeatMetrics] = []
    h300_rows: list[H300WidthStratumMetrics] = []
    bundles: list[ConditionRepeatMetricBundle] = []
    for condition_ordinal in range(19):
        for repeat_id in VALIDATION_REPEAT_IDS:
            bundle = simulate_condition_repeat_with_metrics(
                namespace=RandomNamespace.PILOT,
                condition_ordinal=condition_ordinal,
                repeat_id=repeat_id,
                study_config=config,
                encoding_reference=reference,
            )
            bundles.append(bundle)
            repeat_rows.extend(bundle.repeat_metrics)
            h300_rows.extend(bundle.h300_width_strata)

    repeat_result = tuple(repeat_rows)
    h300_result = tuple(h300_rows)
    repeat_keys = [
        (
            row.ConditionOrdinal,
            row.ConditionCode,
            row.MethodOrdinal,
            row.MethodCode,
            row.RepeatID,
        )
        for row in repeat_result
    ]
    h300_keys = [
        (
            row.ConditionOrdinal,
            row.ConditionCode,
            row.MethodOrdinal,
            row.MethodCode,
            row.RepeatID,
            row.WidthStratum,
        )
        for row in h300_result
    ]

    groups: dict[tuple[int, str, int, str], list[RepeatMetrics]] = {}
    for row in repeat_result:
        key = (
            row.ConditionOrdinal,
            row.ConditionCode,
            row.MethodOrdinal,
            row.MethodCode,
        )
        groups.setdefault(key, []).append(row)
    aggregates = tuple(
        aggregate_repeat_metrics(groups[key]) for key in sorted(groups)
    )

    checks: dict[str, bool] = {
        "validation_condition_repeat_count": len(bundles) == 19 * 3,
        "validation_repeat_metrics_row_count": len(repeat_result) == 19 * 3 * 5,
        "validation_h300_row_count": len(h300_result) == 3 * 5 * 3,
        "repeat_metrics_unique_key": len(repeat_keys) == len(set(repeat_keys)),
        "h300_unique_key": len(h300_keys) == len(set(h300_keys)),
        "condition_count_19": {row.ConditionOrdinal for row in repeat_result}
        == set(range(19)),
        "method_count_5": {
            (row.MethodOrdinal, row.MethodCode) for row in repeat_result
        }
        == set(METHOD_IDENTITIES),
        "state_count_conservation": all(
            row.N_C + row.N_E + row.N_N == K for row in repeat_result
        ),
        "full_transition_count": all(
            row.N_Cdot + row.N_Edot + row.N_Ndot == K - 1
            for row in repeat_result
        ),
        "transition_conservation": all(
            row.N_Cdot == row.N_CC + row.N_CE + row.N_CN
            and row.N_Edot == row.N_EC + row.N_EE + row.N_EN
            and row.N_Ndot == row.N_NC + row.N_NE + row.N_NN
            for row in repeat_result
        ),
        "end_state_origin_identity": all(
            row.N_Cdot == row.N_C - int(row.EndState == "C")
            and row.N_Edot == row.N_E - int(row.EndState == "E")
            and row.N_Ndot == row.N_N - int(row.EndState == "N")
            for row in repeat_result
        ),
        "nc_sum_length_conservation": all(
            row.Sum_L_NC_obs == row.N_E + row.N_N for row in repeat_result
        ),
        "nc_right_open": all(
            row.N_open_NC in (0, 1)
            and row.N_open_NC == int(row.EndState in ("E", "N"))
            for row in repeat_result
        ),
        "h_f_source_conservation": all(
            row.N_E_H + row.N_E_F == row.N_E for row in repeat_result
        ),
        "trajectory_arrays_read_only": all(
            not bundle.trajectory.source_state_code.flags.writeable
            for bundle in bundles
        ),
    }

    ni_bundles = [
        bundle for bundle in bundles if bundle.trajectory.condition_code == "NI"
    ]
    checks["ni_five_method_repeat_metrics_identity"] = all(
        len({_repeat_payload(row) for row in bundle.repeat_metrics}) == 1
        and all(row.N_E == row.N_E_H == row.N_E_F == 0 for row in bundle.repeat_metrics)
        for bundle in ni_bundles
    )

    scene_by_ordinal = {
        int(raw["ConditionOrdinal"]): str(raw["Scene"])
        for raw in config["ConditionSpecs"]
    }
    hprf_rows = [
        row for row in repeat_result if scene_by_ordinal[row.ConditionOrdinal] == "HPRF"
    ]
    idf_rows = [
        row for row in repeat_result if scene_by_ordinal[row.ConditionOrdinal] == "IDF"
    ]
    composite_rows = [
        row
        for row in repeat_result
        if scene_by_ordinal[row.ConditionOrdinal] == "COMPOSITE"
    ]
    hprf_or_composite = hprf_rows + composite_rows
    checks["hprf_source_relation"] = all(
        row.N_E_H == row.N_E and row.N_E_F == 0 for row in hprf_rows
    )
    checks["idf_source_relation"] = all(
        row.N_E_H == 0 and row.N_E_F == row.N_E for row in idf_rows
    )
    checks["composite_source_relation"] = all(
        row.N_E_H + row.N_E_F == row.N_E for row in composite_rows
    )
    checks["hprf_composite_no_n"] = all(
        row.N_N == 0
        and row.N_CN == 0
        and row.N_EN == 0
        and row.N_NC == 0
        and row.N_NE == 0
        and row.N_NN == 0
        and row.N_Ndot == 0
        for row in hprf_or_composite
    )

    checks["aggregate_group_count"] = len(aggregates) == 19 * 5
    checks["aggregate_total_count_formulas"] = all(
        aggregate.P_cor.value == aggregate.Total_N_C / (aggregate.R * K)
        and aggregate.P_N.value == aggregate.Total_N_N / (aggregate.R * K)
        and aggregate.P_NC_end.value
        == aggregate.Total_N_open_NC / aggregate.R
        and (
            aggregate.P_C_given_C.value
            == aggregate.Total_N_CC / aggregate.Total_N_Cdot
            if aggregate.Total_N_Cdot > 0
            else aggregate.P_C_given_C.undefined_reason == C_DENOM_ZERO
        )
        and (
            aggregate.P_C_given_E.value
            == aggregate.Total_N_EC / aggregate.Total_N_Edot
            if aggregate.Total_N_Edot > 0
            else aggregate.P_C_given_E.undefined_reason == E_DENOM_ZERO
        )
        and (
            aggregate.Mean_L_NC.value
            == aggregate.Total_Sum_L_NC_obs / aggregate.Total_N_run_NC
            if aggregate.Total_N_run_NC > 0
            else aggregate.Mean_L_NC.undefined_reason == NO_NC_RUN
        )
        for aggregate in aggregates
    )
    checks["c_transition_probability_sum"] = all(
        np.isclose(
            aggregate.P_C_given_C.value
            + aggregate.P_E_given_C.value
            + aggregate.P_N_given_C.value,
            1.0,
            rtol=0.0,
            atol=1e-15,
        )
        if aggregate.Total_N_Cdot > 0
        else all(
            not metric.defined
            and metric.undefined_reason == C_DENOM_ZERO
            and metric.value is None
            for metric in (
                aggregate.P_C_given_C,
                aggregate.P_E_given_C,
                aggregate.P_N_given_C,
            )
        )
        for aggregate in aggregates
    )
    checks["e_transition_probability_sum"] = all(
        np.isclose(
            aggregate.P_C_given_E.value
            + aggregate.P_E_given_E.value
            + aggregate.P_N_given_E.value,
            1.0,
            rtol=0.0,
            atol=1e-15,
        )
        if aggregate.Total_N_Edot > 0
        else all(
            not metric.defined
            and metric.undefined_reason == E_DENOM_ZERO
            and metric.value is None
            for metric in (
                aggregate.P_C_given_E,
                aggregate.P_E_given_E,
                aggregate.P_N_given_E,
            )
        )
        for aggregate in aggregates
    )
    checks["hprf_composite_p_n_zero"] = all(
        aggregate.P_N == MetricValue(0.0, True, None)
        for aggregate in aggregates
        if scene_by_ordinal[aggregate.ConditionOrdinal] in ("HPRF", "COMPOSITE")
    )
    checks["composite_source_probability_sum"] = all(
        np.isclose(
            aggregate.P_E_H.value + aggregate.P_E_F.value,
            1.0,
            rtol=0.0,
            atol=1e-15,
        )
        if aggregate.Total_N_E > 0
        else (
            aggregate.P_E_H.undefined_reason == E_COUNT_ZERO
            and aggregate.P_E_F.undefined_reason == E_COUNT_ZERO
        )
        for aggregate in aggregates
        if scene_by_ordinal[aggregate.ConditionOrdinal] == "COMPOSITE"
    )
    ni_aggregates = [
        aggregate for aggregate in aggregates if aggregate.ConditionCode == "NI"
    ]
    checks["ni_aggregate_metric_identity"] = (
        len(ni_aggregates) == 5
        and len({_json_key(_aggregate_payload(item)) for item in ni_aggregates}) == 1
    )

    checks["h300_rows_only_h300"] = all(
        row.ConditionOrdinal == H300_CONDITION_ORDINAL
        and row.ConditionCode == H300_CONDITION_CODE
        for row in h300_result
    )
    checks["h300_cycle_conservation"] = all(
        sum(
            row.StratumCycleCount
            for row in h300_result
            if row.RepeatID == repeat_id and row.MethodOrdinal == method_ordinal
        )
        == K
        for repeat_id in VALIDATION_REPEAT_IDS
        for method_ordinal in range(5)
    )
    checks["h300_c_e_conservation"] = all(
        row.StratumCorrectCount + row.StratumErrorCount == row.StratumCycleCount
        for row in h300_result
    )
    checks["h300_method_shared_cycle_counts"] = all(
        len(
            {
                tuple(
                    row.StratumCycleCount
                    for row in h300_result
                    if row.RepeatID == repeat_id
                    and row.MethodOrdinal == method_ordinal
                )
                for method_ordinal in range(5)
            }
        )
        == 1
        for repeat_id in VALIDATION_REPEAT_IDS
    )
    checks["h300_c_to_e_conservation"] = all(
        sum(
            row.Stratum_C_to_E_Count
            for row in h300_result
            if row.RepeatID == metric_row.RepeatID
            and row.MethodOrdinal == metric_row.MethodOrdinal
        )
        == metric_row.N_CE
        for metric_row in repeat_result
        if metric_row.ConditionCode == H300_CONDITION_CODE
    )

    validation = {
        "checks": checks,
        "details": {
            "usage": "VALIDATION_ONLY",
            "namespace": "PILOT",
            "repeat_ids": list(VALIDATION_REPEAT_IDS),
            "repeat_id_count": len(VALIDATION_REPEAT_IDS),
            "validation_condition_repeat_count": len(bundles),
            "repeat_metrics_row_count": len(repeat_result),
            "h300_width_strata_row_count": len(h300_result),
            "aggregated_validation_group_count": len(aggregates),
            "condition_count": 19,
            "method_count": 5,
            "K": K,
        },
        "all_pass": all(checks.values()),
    }
    return repeat_result, h300_result, aggregates, validation


def _json_key(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _active_boundary_checks() -> tuple[dict[str, bool], dict[str, object]]:
    source_paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
    ]
    random_findings = scan_forbidden_random_apis(source_paths)
    forbidden_definitions = {
        "run_pilot",
        "compute_precheck",
        "decide_formal_r",
        "run_formal",
        "run_bootstrap",
        "export_results",
    }
    stage_boundary_findings: list[dict[str, object]] = []
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in forbidden_definitions:
                    stage_boundary_findings.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "line": node.lineno,
                            "definition": node.name,
                        }
                    )
    forbidden_directories = (
        ROOT / "artifacts" / "pilot",
        ROOT / "artifacts" / "formal",
        ROOT / "artifacts" / "bootstrap",
        ROOT / "artifacts" / "results",
        ROOT / "artifacts" / "figures",
    )
    existing_forbidden_directories = [
        str(path.relative_to(ROOT)) for path in forbidden_directories if path.exists()
    ]
    h300_fields = tuple(
        field_info.name for field_info in fields(H300WidthStratumMetrics)
    )
    checks = {
        "forbidden_random_api_scan": not random_findings,
        "stage6_plus_definition_scan": not stage_boundary_findings,
        "forbidden_execution_directories_absent": not existing_forbidden_directories,
        "h300_no_lnc_stratification": not any(
            "L_NC" in field_name for field_name in h300_fields
        ),
    }
    details = {
        "forbidden_random_api_findings": random_findings,
        "stage6_plus_definition_findings": stage_boundary_findings,
        "forbidden_execution_directories": existing_forbidden_directories,
        "h300_schema_fields": list(h300_fields),
    }
    return checks, details


def _validate_public_wrapper_equivalence(
    config: Mapping[str, object],
    reference: Mapping[str, np.ndarray],
) -> bool:
    world = build_shared_physical_world(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=H300_CONDITION_ORDINAL,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    from_world = simulate_shared_world(
        world=world,
        study_config=config,
        encoding_reference=reference,
    )
    legacy_entry = simulate_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=H300_CONDITION_ORDINAL,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    return (
        from_world.condition_ordinal == legacy_entry.condition_ordinal
        and from_world.condition_code == legacy_entry.condition_code
        and from_world.repeat_id == legacy_entry.repeat_id
        and from_world.namespace == legacy_entry.namespace
        and from_world.method_codes == legacy_entry.method_codes
        and np.array_equal(
            from_world.source_state_code, legacy_entry.source_state_code
        )
        and np.array_equal(
            from_world.ref_toa_before_s, legacy_entry.ref_toa_before_s
        )
        and np.array_equal(
            from_world.selected_observed_toa_s,
            legacy_entry.selected_observed_toa_s,
            equal_nan=True,
        )
        and np.array_equal(from_world.has_selection, legacy_entry.has_selection)
        and from_world.delta_t_consumed_count
        == legacy_entry.delta_t_consumed_count
        and from_world.hprf_count_check_count
        == legacy_entry.hprf_count_check_count
        and from_world.hprf_no_n_check_count
        == legacy_entry.hprf_no_n_check_count
    )


def _write_repeat_metrics_schema() -> None:
    metadata_integer_fields = {
        "ConditionOrdinal",
        "MethodOrdinal",
        "RepeatID",
    }
    string_fields = {"ConditionCode", "MethodCode"}
    schema_fields: list[dict[str, object]] = []
    for name in repeat_metrics_field_names():
        if name in metadata_integer_fields or name in REPEAT_COUNT_FIELDS:
            field_type: object = "integer"
        elif name in string_fields:
            field_type = "string"
        elif name == "EndState":
            field_type = {"type": "string", "enum": ["C", "E", "N"]}
        else:
            raise AssertionError(f"unclassified RepeatMetrics field: {name}")
        schema_fields.append({"name": name, "type": field_type})
    payload = {
        "schema_name": "SAL_STAGE5_REPEAT_METRICS_V1",
        "row_unit": "one Condition-Method-Repeat",
        "primary_key": [
            "ConditionOrdinal",
            "ConditionCode",
            "MethodOrdinal",
            "MethodCode",
            "RepeatID",
        ],
        "K": K,
        "effective_interval_count": K - 1,
        "fields": schema_fields,
        "counts_are_integer": True,
        "probabilities_are_not_stored_per_repeat": True,
    }
    _write_json(STAGE5_DIR / "repeat_metrics_schema.json", payload)


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def write_stage5_test_report(
    test_runs: Sequence[TestRun], build_status: str
) -> None:
    by_label = {run.label: run for run in test_runs}
    baseline = by_label["stage1_4_baseline"]
    metric_tests = by_label["stage5_metrics"]
    repeat_tests = by_label["stage5_repeat_metrics"]
    total = by_label["all"]
    lines = [
        "SAL stability Stage 5 test report",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"NumPy version: {np.__version__}",
        f"pytest version: {importlib.metadata.version('pytest')}",
        f"baseline Stage1-4 test count: {baseline.passed + baseline.failed}",
        f"Stage5 metric tests count: {metric_tests.passed + metric_tests.failed}",
        f"Stage5 RepeatMetrics tests count: {repeat_tests.passed + repeat_tests.failed}",
        f"total tests: {total.passed + total.failed}",
        f"passed: {total.passed}",
        f"failed: {total.failed}",
        f"build_stage5 result: {build_status}",
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
    (STAGE5_DIR / "test_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_stage5_changed_files() -> None:
    lines = [
        "MODIFIED README.md",
        "MODIFIED pyproject.toml",
        "MODIFIED src/sal_stability_stage1/kernel.py — INTERFACE-ONLY MODIFICATION: public simulate_shared_world thin wrapper; private 200-cycle kernel source unchanged",
        "ADDED src/sal_stability_stage1/metrics.py",
        "ADDED src/sal_stability_stage1/stage5.py",
        "ADDED scripts/build_stage5.py",
        "ADDED tests/test_stage5_metrics.py",
        "ADDED tests/test_stage5_repeat_metrics.py",
        "ADDED artifacts/stage5/status.json",
        "ADDED artifacts/stage5/validation_report.md",
        "ADDED artifacts/stage5/changed_files.txt",
        "ADDED artifacts/stage5/test_report.txt",
        "ADDED artifacts/stage5/metric_validation.json",
        "ADDED artifacts/stage5/repeat_metrics_validation.csv",
        "ADDED artifacts/stage5/h300_width_strata_validation.csv",
        "ADDED artifacts/stage5/repeat_metrics_schema.json",
        "ADDED artifacts/stage5/aggregated_metrics_validation.json",
        "UNCHANGED artifacts/config/study_config.json",
        "UNCHANGED artifacts/config/study_config.sha256",
        "UNCHANGED artifacts/config/encoding_reference.csv",
        "UNCHANGED artifacts/config/encoding_reference.npz",
        "UNCHANGED src/sal_stability_stage1/contracts.py",
        "UNCHANGED src/sal_stability_stage1/encoding.py",
        "UNCHANGED src/sal_stability_stage1/rng.py",
        "UNCHANGED src/sal_stability_stage1/events.py",
        "UNCHANGED src/sal_stability_stage1/hprf.py",
        "UNCHANGED src/sal_stability_stage1/stage3.py",
        "UNCHANGED src/sal_stability_stage1/selectors.py",
        "UNCHANGED src/sal_stability_stage1/stage4.py",
        "UNCHANGED artifacts/stage1/*",
        "UNCHANGED artifacts/stage2/*",
        "UNCHANGED artifacts/stage3/*",
        "UNCHANGED artifacts/stage4/*",
    ]
    (STAGE5_DIR / "changed_files.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_stage5_validation_report(
    status: Mapping[str, object],
    metric_validation: Mapping[str, object],
    blocking_issues: Sequence[str],
) -> None:
    lines = [
        "# Stage 5 Metrics and RepeatMetrics Validation Report",
        "",
        f"Verdict: **{status['status']}**",
        "",
        "## A. Baseline",
        "",
        f"- Stage 1-4 regression: {'PASS' if status['stage1_regression_passed'] and status['stage2_regression_passed'] and status['stage3_regression_passed'] and status['stage4_regression_passed'] else 'FAIL'}.",
        f"- Frozen Stage 1-4 baseline tests: {status['baseline_stage1_stage2_stage3_stage4_test_count']} / 219 passed.",
        f"- Current complete tests: {status['total_passed_count']} passed, {status['total_failed_count']} failed.",
        f"- StudyConfig changed: {str(status['study_config_changed']).lower()}.",
        f"- Selector changed: {str(status['selector_changed']).lower()}.",
        "- The only kernel edit is a public thin wrapper; the private 200-cycle kernel source hash is unchanged and both entry paths are array-identical.",
        "",
        "## B. State mapping",
        "",
        "- Stage 4 SourceStateCode `0=N` maps to evaluation state N.",
        "- Stage 4 SourceStateCode `1=G` maps to evaluation state C.",
        "- Stage 4 SourceStateCode `2=H` and `3=F` both map to evaluation state E.",
        "- No selector, observed TOA, observed width, score, or TieRank is read to reclassify C/E/N.",
        "",
        "## C. RepeatMetrics schema",
        "",
        "- Key: ConditionOrdinal, ConditionCode, MethodOrdinal, MethodCode, RepeatID.",
        "- Counts: N_C, N_E, N_N; all nine C/E/N transitions; N_Cdot, N_Edot, N_Ndot; N_run_NC, Sum_L_NC_obs, N_open_NC; N_E_H, N_E_F.",
        "- EndState is exactly one of C, E, N. All count fields are integers.",
        "",
        "## D. Count conservation",
        "",
        "- Per Repeat: `N_C + N_E + N_N = 200`.",
        "- Per Repeat: all nine adjacent transition counts sum to `199`.",
        "",
        "## E. Conditional denominators",
        "",
        "- `N_Cdot = N_CC + N_CE + N_CN`.",
        "- `N_Edot = N_EC + N_EE + N_EN`.",
        "- `N_Ndot = N_NC + N_NE + N_NN`.",
        "- Each origin count equals its cycle count minus the indicator that EndState is that origin.",
        "",
        "## F. NC runs",
        "",
        "- E and N jointly form maximal non-correct runs; N does not split an NC run.",
        "- Per Repeat: `Sum_L_NC_obs = N_E + N_N`.",
        "",
        "## G. Right-open",
        "",
        "- A run still E/N at cycle 200 is retained in N_run_NC and Sum_L_NC_obs and contributes one to N_open_NC.",
        "- No cycle 201 is created. `N_open_NC = I(EndState in {E,N})`.",
        "",
        "## H. Core metrics",
        "",
        "- `P_cor = sum N_C / (R * 200)`.",
        "- `P_C|C = sum N_CC / sum N_Cdot`.",
        "- `P_C|E = sum N_EC / sum N_Edot`; the numerator is E-to-C, never C-to-E.",
        "- `mean L_NC = sum Sum_L_NC_obs / sum N_run_NC`.",
        "",
        "## I. Support metrics",
        "",
        "- `P_E|C = sum N_CE / sum N_Cdot`; `P_N|C = sum N_CN / sum N_Cdot`.",
        "- `P_E|E = sum N_EE / sum N_Edot`; `P_N|E = sum N_EN / sum N_Edot`.",
        "- `P_N = sum N_N / (R * 200)`.",
        "- `P_open_NC = sum N_open_NC / sum N_run_NC`.",
        "- `P_NC_end = sum N_open_NC / R`; it is not P_open_NC.",
        "",
        "## J. Undefined metrics",
        "",
        "- Frozen reasons: C_DENOM_ZERO, E_DENOM_ZERO, NO_NC_RUN, E_COUNT_ZERO.",
        "- Undefined metrics use `defined=false`, `value=null`, and an explicit reason; they are never zero-filled or silently stored as NaN.",
        "",
        "## K. Aggregation",
        "",
        "- Every conditional metric uses a ratio of total numerator count to total denominator count across Repeats.",
        "- Per-Repeat conditional probabilities are not averaged.",
        "",
        "## L. Sources",
        "",
        "- `N_E_H + N_E_F = N_E` for every Repeat.",
        "- NI has no E; HPRF E comes only from H; IDF E comes only from F; composite E retains H/F source contributions for support interpretation.",
        "",
        "## M. H300",
        "",
        "- Integer width-level difference maps to WIDTH_STRATUM_0NS, WIDTH_STRATUM_10NS, or WIDTH_STRATUM_GE20NS.",
        "- Each Method-Repeat conserves 200 cycles and C+E equals cycle count in every stratum; H300 N is a hard error.",
        "- C-to-E uses the current cycle k width stratum for transition k to k+1.",
        "- L_NC is never stratified because a continuous NC run can cross width strata.",
        "",
        "## N. Stage boundary",
        "",
        "Stage 6 was NOT executed.",
        "Pilot R=200 was NOT executed.",
        "Formal-R decision was NOT executed.",
        "Formal was NOT executed.",
        "Bootstrap was NOT executed.",
        "No paper result was generated.",
        "No paper result figure was generated.",
        "",
        "The PILOT random namespace was used only for fixed RepeatIDs 1, 2, and 17 in a 57 Condition-Repeat implementation validation; this is not the Pilot R=200 study.",
        "",
        "## Machine checks",
        "",
    ]
    for name, passed in sorted(metric_validation.get("checks", {}).items()):
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Blocking issues", ""])
    if blocking_issues:
        lines.extend(f"- {issue}" for issue in blocking_issues)
    else:
        lines.append("None")
    (STAGE5_DIR / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_stage5() -> dict[str, object]:
    STAGE5_DIR.mkdir(parents=True, exist_ok=True)
    blocking_issues: list[str] = []

    config_path = CONFIG_DIR / "study_config.json"
    config_checksum_path = CONFIG_DIR / "study_config.sha256"
    config_hash = _sha256_file(config_path) if config_path.is_file() else ""
    recorded_config_hash = (
        config_checksum_path.read_text(encoding="utf-8").strip()
        if config_checksum_path.is_file()
        else ""
    )
    config_hash_ok = (
        config_hash == FROZEN_CONFIG_SHA256
        and recorded_config_hash == FROZEN_CONFIG_SHA256
    )
    frozen_file_checks = {
        relative: (ROOT / relative).is_file()
        and _sha256_file(ROOT / relative) == expected
        for relative, expected in FROZEN_SCIENCE_FILE_SHA256.items()
    }
    selector_hash_ok = frozen_file_checks.get(
        "src/sal_stability_stage1/selectors.py", False
    )
    private_kernel_source_hash = hashlib.sha256(
        inspect.getsource(
            sys.modules[simulate_shared_world.__module__]._simulate_shared_world
        ).encode("utf-8")
    ).hexdigest()
    private_kernel_source_ok = (
        private_kernel_source_hash == FROZEN_PRIVATE_KERNEL_SOURCE_SHA256
    )
    wrapper_signature_ok = tuple(inspect.signature(simulate_shared_world).parameters) == (
        "world",
        "study_config",
        "encoding_reference",
    )

    prior_stage_checks: dict[str, bool] = {}
    for stage_number in range(1, 5):
        status_path = ROOT / "artifacts" / f"stage{stage_number}" / "status.json"
        try:
            prior_status = json.loads(status_path.read_text(encoding="utf-8"))
            prior_pass = (
                prior_status.get("status") == "PASS"
                and prior_status.get("science_contract_changed") is False
            )
        except (OSError, json.JSONDecodeError):
            prior_pass = False
        prior_stage_checks[f"stage{stage_number}_artifact_pass"] = prior_pass

    pretest_checks = {
        "study_config_hash_unchanged": config_hash_ok,
        "selector_hash_unchanged": selector_hash_ok,
        "frozen_science_files_unchanged": all(frozen_file_checks.values()),
        "private_cycle_kernel_source_unchanged": private_kernel_source_ok,
        "public_wrapper_signature": wrapper_signature_ok,
        **prior_stage_checks,
    }
    for name, passed in pretest_checks.items():
        if not passed:
            blocking_issues.append(f"Stage 5 precondition failed: {name}")

    baseline_run = run_pytest("stage1_4_baseline", BASELINE_TEST_PATHS)
    baseline_count = baseline_run.passed + baseline_run.failed
    baseline_passed = baseline_run.returncode == 0 and baseline_count == 219
    if not baseline_passed:
        blocking_issues.append(
            f"Stage 1-4 baseline regression returned {baseline_count} tests; expected 219 passing"
        )

    hard_preconditions_passed = all(pretest_checks.values()) and baseline_passed
    if hard_preconditions_passed:
        metric_test_run = run_pytest(
            "stage5_metrics", ("tests/test_stage5_metrics.py",)
        )
        repeat_test_run = run_pytest(
            "stage5_repeat_metrics", ("tests/test_stage5_repeat_metrics.py",)
        )
        if metric_test_run.returncode != 0:
            blocking_issues.append("Stage 5 metric pytest failed")
        if repeat_test_run.returncode != 0:
            blocking_issues.append("Stage 5 RepeatMetrics pytest failed")
    else:
        metric_test_run = _skipped_test_run(
            "stage5_metrics", "Stage 1-4 hard precondition failed"
        )
        repeat_test_run = _skipped_test_run(
            "stage5_repeat_metrics", "Stage 1-4 hard precondition failed"
        )

    stage5_test_gate = (
        hard_preconditions_passed
        and metric_test_run.returncode == 0
        and repeat_test_run.returncode == 0
    )
    synthetic_validation: dict[str, object] = {
        "checks": {"synthetic_validation_completed": False},
        "details": {},
        "all_pass": False,
    }
    small_validation: dict[str, object] = {
        "checks": {"small_scale_validation_completed": False},
        "details": {},
        "all_pass": False,
    }
    repeat_rows: tuple[RepeatMetrics, ...] = ()
    h300_rows: tuple[H300WidthStratumMetrics, ...] = ()
    aggregates: tuple[AggregatedMetrics, ...] = ()
    wrapper_equivalence_passed = False
    config: Mapping[str, object] = {}
    reference: Mapping[str, np.ndarray] = {}
    if stage5_test_gate:
        try:
            config, reference = load_stage3_inputs()
            synthetic_validation = validate_synthetic_metrics()
            if not synthetic_validation["all_pass"]:
                failed_names = [
                    name
                    for name, passed in synthetic_validation["checks"].items()
                    if not passed
                ]
                blocking_issues.append(
                    "synthetic metric validation failed: " + ", ".join(failed_names)
                )
            wrapper_equivalence_passed = _validate_public_wrapper_equivalence(
                config, reference
            )
            if not wrapper_equivalence_passed:
                blocking_issues.append(
                    "public shared-world wrapper is not array-identical to the Stage 4 entry"
                )
            if synthetic_validation["all_pass"] and wrapper_equivalence_passed:
                (
                    repeat_rows,
                    h300_rows,
                    aggregates,
                    small_validation,
                ) = run_small_scale_validation(config, reference)
                if not small_validation["all_pass"]:
                    failed_names = [
                        name
                        for name, passed in small_validation["checks"].items()
                        if not passed
                    ]
                    blocking_issues.append(
                        "small-scale metric validation failed: "
                        + ", ".join(failed_names)
                    )
            else:
                blocking_issues.append(
                    "small-scale validation skipped because a mathematical hard gate failed"
                )
        except Exception as exc:
            blocking_issues.append(
                f"Stage 5 scientific validation raised {type(exc).__name__}: {exc}"
            )
    else:
        blocking_issues.append("Stage 5 scientific validation skipped after pytest gate")

    boundary_checks, boundary_details = _active_boundary_checks()
    for name, passed in boundary_checks.items():
        if not passed:
            blocking_issues.append(f"Stage boundary validation failed: {name}")

    validation_gate = (
        stage5_test_gate
        and synthetic_validation["all_pass"]
        and wrapper_equivalence_passed
        and small_validation["all_pass"]
        and all(boundary_checks.values())
    )
    if validation_gate:
        all_run = run_pytest("all", ())
        if all_run.returncode != 0:
            blocking_issues.append("complete pytest regression failed")
    else:
        all_run = _skipped_test_run(
            "all", "Stage 5 scientific validation hard gate failed"
        )

    stage5_metric_test_count = metric_test_run.passed + metric_test_run.failed
    stage5_repeat_test_count = repeat_test_run.passed + repeat_test_run.failed
    expected_total_test_count = (
        baseline_count + stage5_metric_test_count + stage5_repeat_test_count
    )
    if validation_gate and all_run.passed + all_run.failed != expected_total_test_count:
        blocking_issues.append(
            "complete pytest count does not equal 219 baseline plus Stage 5 tests"
        )

    merged_checks: dict[str, bool] = {
        **pretest_checks,
        "baseline_219_passed": baseline_passed,
        "stage5_metric_tests_passed": metric_test_run.returncode == 0,
        "stage5_repeat_metrics_tests_passed": repeat_test_run.returncode == 0,
        "public_wrapper_array_equivalence": wrapper_equivalence_passed,
        **{
            f"synthetic_{name}": bool(passed)
            for name, passed in synthetic_validation["checks"].items()
        },
        **{
            f"validation_{name}": bool(passed)
            for name, passed in small_validation["checks"].items()
        },
        **boundary_checks,
        "complete_pytest_passed": all_run.returncode == 0,
        "complete_pytest_count_correct": (
            all_run.passed + all_run.failed == expected_total_test_count
            if validation_gate
            else False
        ),
    }
    blocking_issues = list(dict.fromkeys(blocking_issues))
    science_contract_unchanged = (
        config_hash_ok
        and all(frozen_file_checks.values())
        and private_kernel_source_ok
        and wrapper_signature_ok
        and wrapper_equivalence_passed
    )
    passed = (
        not blocking_issues
        and all(merged_checks.values())
        and science_contract_unchanged
        and all_run.returncode == 0
    )

    validation_checks = small_validation.get("checks", {})
    synthetic_checks = synthetic_validation.get("checks", {})
    status: dict[str, object] = {
        "stage": 5,
        "stage_name": "METRICS_AND_REPEAT_METRICS",
        "status": "PASS" if passed else "FAIL",
        "blocking_issue_count": len(blocking_issues),
        "science_contract_changed": not science_contract_unchanged,
        "study_config_changed": not config_hash_ok,
        "selector_changed": not selector_hash_ok,
        "stage1_regression_passed": baseline_passed
        and prior_stage_checks["stage1_artifact_pass"],
        "stage2_regression_passed": baseline_passed
        and prior_stage_checks["stage2_artifact_pass"],
        "stage3_regression_passed": baseline_passed
        and prior_stage_checks["stage3_artifact_pass"],
        "stage4_regression_passed": baseline_passed
        and prior_stage_checks["stage4_artifact_pass"],
        "condition_count": len(config.get("ConditionSpecs", ())) if config else 19,
        "method_count": len(config.get("MethodSpecs", ())) if config else 5,
        "K": int(config.get("K", K)) if config else K,
        "repeat_metrics_schema_passed": bool(
            synthetic_checks.get("repeat_metrics_schema", False)
        ),
        "state_count_conservation_passed": bool(
            validation_checks.get("state_count_conservation", False)
        ),
        "full_transition_count_passed": bool(
            synthetic_checks.get("all_nine_transitions", False)
            and validation_checks.get("full_transition_count", False)
        ),
        "transition_conservation_passed": bool(
            validation_checks.get("transition_conservation", False)
        ),
        "nc_run_detection_passed": bool(
            synthetic_checks.get("n_does_not_split_nc", False)
        ),
        "nc_right_open_passed": bool(
            synthetic_checks.get("right_open_nc_retained", False)
            and validation_checks.get("nc_right_open", False)
        ),
        "nc_sum_length_conservation_passed": bool(
            validation_checks.get("nc_sum_length_conservation", False)
        ),
        "composite_source_count_passed": bool(
            validation_checks.get("composite_source_relation", False)
        ),
        "h300_width_strata_passed": bool(
            synthetic_checks.get("h300_fixed_width_strata", False)
            and validation_checks.get("h300_cycle_conservation", False)
            and validation_checks.get("h300_c_e_conservation", False)
        ),
        "h300_no_lnc_stratification_passed": bool(
            synthetic_checks.get("h300_no_lnc_stratification", False)
            and boundary_checks.get("h300_no_lnc_stratification", False)
        ),
        "metric_total_ratio_passed": bool(
            synthetic_checks.get("total_count_ratio", False)
            and validation_checks.get("aggregate_total_count_formulas", False)
        ),
        "undefined_metric_encoding_passed": bool(
            synthetic_checks.get("undefined_metric_encoding", False)
        ),
        "c_transition_probability_sum_passed": bool(
            validation_checks.get("c_transition_probability_sum", False)
        ),
        "e_transition_probability_sum_passed": bool(
            validation_checks.get("e_transition_probability_sum", False)
        ),
        "p_open_nc_passed": bool(
            synthetic_checks.get("p_open_nc_distinct_from_p_nc_end", False)
        ),
        "p_nc_end_passed": bool(
            synthetic_checks.get("p_open_nc_distinct_from_p_nc_end", False)
        ),
        "validation_repeat_count": len(repeat_rows) // 5,
        "validation_repeat_id_count": len(VALIDATION_REPEAT_IDS),
        "validation_repeat_metrics_row_count": len(repeat_rows),
        "h300_validation_row_count": len(h300_rows),
        "baseline_stage1_stage2_stage3_stage4_test_count": baseline_count,
        "stage5_metric_test_count": stage5_metric_test_count,
        "stage5_repeat_metrics_test_count": stage5_repeat_test_count,
        "stage5_test_count": stage5_metric_test_count + stage5_repeat_test_count,
        "total_test_count": all_run.passed + all_run.failed,
        "total_passed_count": all_run.passed,
        "total_failed_count": all_run.failed,
        "study_config_sha256": config_hash,
        "selectors_sha256": _sha256_file(
            ROOT / "src" / "sal_stability_stage1" / "selectors.py"
        ),
        "private_cycle_kernel_source_sha256": private_kernel_source_hash,
        "kernel_interface_only_modification": True,
        "validation_only": True,
        "validation_namespace": "PILOT",
        "validation_repeat_ids": list(VALIDATION_REPEAT_IDS),
        "stage6_executed": False,
        "pilot_executed": False,
        "r_decision_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "paper_result_figures_generated": False,
        "stop_after_stage5": True,
    }

    metric_validation = {
        "stage": 5,
        "stage_name": "METRICS_AND_REPEAT_METRICS",
        "status": status["status"],
        "checks": merged_checks,
        "all_pass": passed,
        "blocking_issues": blocking_issues,
        "details": {
            "synthetic": synthetic_validation.get("details", {}),
            "small_scale": small_validation.get("details", {}),
            "boundary": boundary_details,
            "frozen_file_checks": frozen_file_checks,
            "prior_stage_checks": prior_stage_checks,
            "baseline_test_count": baseline_count,
            "stage5_metric_test_count": stage5_metric_test_count,
            "stage5_repeat_metrics_test_count": stage5_repeat_test_count,
            "total_test_count": all_run.passed + all_run.failed,
            "private_cycle_kernel_source_sha256": private_kernel_source_hash,
            "validation_usage": "VALIDATION_ONLY",
        },
    }

    _write_repeat_metrics_schema()
    if repeat_rows:
        write_repeat_metrics_csv(
            STAGE5_DIR / "repeat_metrics_validation.csv", repeat_rows
        )
    if h300_rows:
        write_h300_width_strata_csv(
            STAGE5_DIR / "h300_width_strata_validation.csv", h300_rows
        )
    if aggregates:
        _write_json(
            STAGE5_DIR / "aggregated_metrics_validation.json",
            {
                "usage": "VALIDATION_ONLY",
                "namespace": "PILOT",
                "repeat_ids": list(VALIDATION_REPEAT_IDS),
                "paper_result": False,
                "metrics": [aggregated_metrics_to_dict(item) for item in aggregates],
            },
        )
    _write_json(STAGE5_DIR / "metric_validation.json", metric_validation)
    _write_json(STAGE5_DIR / "status.json", status)
    test_runs = (
        baseline_run,
        metric_test_run,
        repeat_test_run,
        all_run,
    )
    write_stage5_test_report(test_runs, str(status["status"]))
    write_stage5_validation_report(status, metric_validation, blocking_issues)
    write_stage5_changed_files()
    return status


__all__ = [
    "build_stage5",
    "ConditionRepeatMetricBundle",
    "H300WidthStratumMetrics",
    "H300_CONDITION_CODE",
    "H300_CONDITION_ORDINAL",
    "WIDTH_STRATA",
    "WIDTH_STRATUM_0NS",
    "WIDTH_STRATUM_10NS",
    "WIDTH_STRATUM_GE20NS",
    "compute_h300_width_strata",
    "h300_width_strata_field_names",
    "level_strata_match_physical_width_difference",
    "metric_bundle_from_world",
    "simulate_condition_repeat_with_metrics",
    "width_strata_from_levels",
    "write_h300_width_strata_csv",
    "write_repeat_metrics_csv",
]
