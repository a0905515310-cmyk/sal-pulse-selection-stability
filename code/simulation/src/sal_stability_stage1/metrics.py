from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

import numpy as np

from .kernel import ConditionRepeatTrajectory, METHOD_IDENTITIES


K = 200
EFFECTIVE_INTERVAL_COUNT = K - 1

EvaluationState = Literal["C", "E", "N"]
EVALUATION_STATES: tuple[EvaluationState, ...] = ("C", "E", "N")

C_DENOM_ZERO = "C_DENOM_ZERO"
E_DENOM_ZERO = "E_DENOM_ZERO"
NO_NC_RUN = "NO_NC_RUN"
E_COUNT_ZERO = "E_COUNT_ZERO"
UNDEFINED_REASONS = frozenset(
    (C_DENOM_ZERO, E_DENOM_ZERO, NO_NC_RUN, E_COUNT_ZERO)
)

STATE_COUNT_FIELDS = ("N_C", "N_E", "N_N")
TRANSITION_COUNT_FIELDS = (
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
ORIGIN_COUNT_FIELDS = ("N_Cdot", "N_Edot", "N_Ndot")
NC_COUNT_FIELDS = ("N_run_NC", "Sum_L_NC_obs", "N_open_NC")
ERROR_SOURCE_COUNT_FIELDS = ("N_E_H", "N_E_F")
REPEAT_COUNT_FIELDS = (
    *STATE_COUNT_FIELDS,
    *TRANSITION_COUNT_FIELDS,
    *ORIGIN_COUNT_FIELDS,
    *NC_COUNT_FIELDS,
    *ERROR_SOURCE_COUNT_FIELDS,
)


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def source_state_to_evaluation(source_state_code: object) -> EvaluationState:
    """Map the frozen Stage 4 source code to the Stage 5 evaluation state."""
    value = _strict_integer(source_state_code, "source_state_code")
    if value == 0:
        return "N"
    if value == 1:
        return "C"
    if value in (2, 3):
        return "E"
    raise ValueError("source_state_code must be one of 0, 1, 2, or 3")


def _validated_source_state_array(source_state_code_1d: object) -> np.ndarray:
    if not isinstance(source_state_code_1d, np.ndarray):
        raise TypeError("source_state_code_1d must be a numpy.ndarray")
    if source_state_code_1d.dtype != np.dtype(np.uint8):
        raise TypeError("source_state_code_1d must have dtype uint8")
    if source_state_code_1d.shape != (K,):
        raise ValueError(f"source_state_code_1d must have shape ({K},)")
    if np.any(source_state_code_1d > np.uint8(3)):
        raise ValueError("source_state_code_1d contains a value outside 0..3")
    return source_state_code_1d


def evaluation_states_from_source(source_state_code_1d: object) -> np.ndarray:
    """Return a read-only C/E/N array derived only from SourceStateCode."""
    source = _validated_source_state_array(source_state_code_1d)
    states = np.empty(K, dtype="<U1")
    states[source == np.uint8(0)] = "N"
    states[source == np.uint8(1)] = "C"
    states[(source == np.uint8(2)) | (source == np.uint8(3))] = "E"
    states.setflags(write=False)
    return states


@dataclass(frozen=True, slots=True)
class RepeatMetrics:
    ConditionOrdinal: int
    ConditionCode: str
    MethodOrdinal: int
    MethodCode: str
    RepeatID: int

    N_C: int
    N_E: int
    N_N: int

    N_CC: int
    N_CE: int
    N_CN: int
    N_EC: int
    N_EE: int
    N_EN: int
    N_NC: int
    N_NE: int
    N_NN: int

    N_Cdot: int
    N_Edot: int
    N_Ndot: int

    N_run_NC: int
    Sum_L_NC_obs: int
    N_open_NC: int

    EndState: EvaluationState

    N_E_H: int
    N_E_F: int

    def __post_init__(self) -> None:
        condition_ordinal = _strict_integer(
            self.ConditionOrdinal, "ConditionOrdinal"
        )
        method_ordinal = _strict_integer(self.MethodOrdinal, "MethodOrdinal")
        repeat_id = _strict_integer(self.RepeatID, "RepeatID")
        if not 0 <= condition_ordinal < 19:
            raise ValueError("ConditionOrdinal must lie in 0..18")
        if not 0 <= method_ordinal < len(METHOD_IDENTITIES):
            raise ValueError("MethodOrdinal must lie in 0..4")
        if repeat_id < 1:
            raise ValueError("RepeatID must be positive")
        condition_code = str(self.ConditionCode)
        method_code = str(self.MethodCode)
        if not condition_code:
            raise ValueError("ConditionCode cannot be empty")
        expected_method = METHOD_IDENTITIES[method_ordinal]
        if expected_method != (method_ordinal, method_code):
            raise ValueError("Method identity is not the frozen 5-method identity")
        object.__setattr__(self, "ConditionOrdinal", condition_ordinal)
        object.__setattr__(self, "ConditionCode", condition_code)
        object.__setattr__(self, "MethodOrdinal", method_ordinal)
        object.__setattr__(self, "MethodCode", method_code)
        object.__setattr__(self, "RepeatID", repeat_id)

        for name in REPEAT_COUNT_FIELDS:
            value = _strict_integer(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)

        end_state = str(self.EndState)
        if end_state not in EVALUATION_STATES:
            raise ValueError("EndState must be C, E, or N")
        object.__setattr__(self, "EndState", end_state)

        if self.N_C + self.N_E + self.N_N != K:
            raise ValueError("N_C + N_E + N_N must equal 200")
        if sum(getattr(self, name) for name in TRANSITION_COUNT_FIELDS) != (
            EFFECTIVE_INTERVAL_COUNT
        ):
            raise ValueError("all nine transition counts must sum to 199")
        if self.N_Cdot != self.N_CC + self.N_CE + self.N_CN:
            raise ValueError("N_Cdot does not equal the C-origin transition sum")
        if self.N_Edot != self.N_EC + self.N_EE + self.N_EN:
            raise ValueError("N_Edot does not equal the E-origin transition sum")
        if self.N_Ndot != self.N_NC + self.N_NE + self.N_NN:
            raise ValueError("N_Ndot does not equal the N-origin transition sum")
        if self.N_Cdot + self.N_Edot + self.N_Ndot != EFFECTIVE_INTERVAL_COUNT:
            raise ValueError("origin transition counts must sum to 199")
        expected_origins = {
            "C": self.N_C - int(end_state == "C"),
            "E": self.N_E - int(end_state == "E"),
            "N": self.N_N - int(end_state == "N"),
        }
        if (
            self.N_Cdot != expected_origins["C"]
            or self.N_Edot != expected_origins["E"]
            or self.N_Ndot != expected_origins["N"]
        ):
            raise ValueError("origin counts disagree with cycle counts and EndState")
        if self.Sum_L_NC_obs != self.N_E + self.N_N:
            raise ValueError("Sum_L_NC_obs must equal N_E + N_N")
        if self.N_open_NC not in (0, 1):
            raise ValueError("N_open_NC must be 0 or 1")
        if self.N_open_NC != int(end_state in ("E", "N")):
            raise ValueError("N_open_NC must identify a non-C EndState exactly")
        noncorrect_count = self.N_E + self.N_N
        if noncorrect_count == 0:
            if (self.N_run_NC, self.Sum_L_NC_obs, self.N_open_NC) != (0, 0, 0):
                raise ValueError("an all-C trajectory cannot contain an NC run")
        elif not 1 <= self.N_run_NC <= noncorrect_count:
            raise ValueError("N_run_NC is outside its scientific bounds")
        if self.N_E_H + self.N_E_F != self.N_E:
            raise ValueError("N_E_H + N_E_F must equal N_E")


@dataclass(slots=True)
class _StreamingRepeatAccumulator:
    cycle_count: int = 0
    previous_state: EvaluationState | None = None
    end_state: EvaluationState | None = None
    current_nc_len: int = 0
    n_run_nc: int = 0
    sum_l_nc_obs: int = 0
    n_open_nc: int = 0
    n_e_h: int = 0
    n_e_f: int = 0
    state_counts: dict[str, int] = field(
        default_factory=lambda: {state: 0 for state in EVALUATION_STATES}
    )
    transition_counts: dict[str, int] = field(
        default_factory=lambda: {
            f"N_{origin}{destination}": 0
            for origin in EVALUATION_STATES
            for destination in EVALUATION_STATES
        }
    )
    finalized: bool = False

    def update(self, source_state_code: object) -> None:
        if self.finalized:
            raise RuntimeError("cannot update a finalized streaming accumulator")
        if self.cycle_count >= K:
            raise ValueError("streaming accumulator cannot exceed 200 cycles")
        source = _strict_integer(source_state_code, "source_state_code")
        state = source_state_to_evaluation(source)
        self.state_counts[state] += 1
        if source == 2:
            self.n_e_h += 1
        elif source == 3:
            self.n_e_f += 1

        if self.previous_state is not None:
            self.transition_counts[f"N_{self.previous_state}{state}"] += 1
        if state in ("E", "N"):
            self.current_nc_len += 1
        elif self.current_nc_len > 0:
            self.n_run_nc += 1
            self.sum_l_nc_obs += self.current_nc_len
            self.current_nc_len = 0

        self.previous_state = state
        self.end_state = state
        self.cycle_count += 1

    def finalize(self) -> dict[str, int | EvaluationState]:
        if self.finalized:
            raise RuntimeError("streaming accumulator has already been finalized")
        if self.cycle_count != K or self.end_state is None:
            raise ValueError("streaming accumulator requires exactly 200 cycles")
        if self.current_nc_len > 0:
            self.n_run_nc += 1
            self.sum_l_nc_obs += self.current_nc_len
            self.n_open_nc += 1
            self.current_nc_len = 0
        self.finalized = True

        result: dict[str, int | EvaluationState] = {
            "N_C": self.state_counts["C"],
            "N_E": self.state_counts["E"],
            "N_N": self.state_counts["N"],
            **self.transition_counts,
            "N_Cdot": sum(
                self.transition_counts[f"N_C{destination}"]
                for destination in EVALUATION_STATES
            ),
            "N_Edot": sum(
                self.transition_counts[f"N_E{destination}"]
                for destination in EVALUATION_STATES
            ),
            "N_Ndot": sum(
                self.transition_counts[f"N_N{destination}"]
                for destination in EVALUATION_STATES
            ),
            "N_run_NC": self.n_run_nc,
            "Sum_L_NC_obs": self.sum_l_nc_obs,
            "N_open_NC": self.n_open_nc,
            "EndState": self.end_state,
            "N_E_H": self.n_e_h,
            "N_E_F": self.n_e_f,
        }
        return result


def streaming_repeat_counts(
    *, source_state_code_1d: object
) -> dict[str, int | EvaluationState]:
    """Online reconstruction of all RepeatMetrics count contributions."""
    source = _validated_source_state_array(source_state_code_1d)
    accumulator = _StreamingRepeatAccumulator()
    for source_code in source:
        accumulator.update(source_code)
    return accumulator.finalize()


def reference_repeat_counts(
    *, source_state_code_1d: object
) -> dict[str, int | EvaluationState]:
    """Independent vector/run-boundary reconstruction used as a test oracle."""
    source = _validated_source_state_array(source_state_code_1d)
    states = evaluation_states_from_source(source)

    result: dict[str, int | EvaluationState] = {
        "N_C": int(np.count_nonzero(states == "C")),
        "N_E": int(np.count_nonzero(states == "E")),
        "N_N": int(np.count_nonzero(states == "N")),
    }
    origins = states[:-1]
    destinations = states[1:]
    for origin in EVALUATION_STATES:
        for destination in EVALUATION_STATES:
            result[f"N_{origin}{destination}"] = int(
                np.count_nonzero(
                    (origins == origin) & (destinations == destination)
                )
            )
    for origin in EVALUATION_STATES:
        result[f"N_{origin}dot"] = int(np.count_nonzero(origins == origin))

    noncorrect = states != "C"
    padded = np.pad(noncorrect.astype(np.int8), (1, 1), constant_values=0)
    boundary_delta = np.diff(padded)
    starts = np.flatnonzero(boundary_delta == 1)
    ends = np.flatnonzero(boundary_delta == -1)
    lengths = ends - starts
    result.update(
        {
            "N_run_NC": int(lengths.size),
            "Sum_L_NC_obs": int(lengths.sum(dtype=np.int64)),
            "N_open_NC": int(bool(noncorrect[-1])),
            "EndState": str(states[-1]),
            "N_E_H": int(np.count_nonzero(source == np.uint8(2))),
            "N_E_F": int(np.count_nonzero(source == np.uint8(3))),
        }
    )
    return result


def compute_repeat_metrics(
    *,
    source_state_code_1d: object,
    condition_ordinal: int,
    condition_code: str,
    method_ordinal: int,
    method_code: str,
    repeat_id: int,
) -> RepeatMetrics:
    """Compute one Condition-Method-Repeat row without re-running a selector."""
    source = _validated_source_state_array(source_state_code_1d)
    streaming = streaming_repeat_counts(source_state_code_1d=source)
    reference = reference_repeat_counts(source_state_code_1d=source)
    if streaming != reference:
        differing = sorted(
            name
            for name in streaming.keys() | reference.keys()
            if streaming.get(name) != reference.get(name)
        )
        raise AssertionError(
            "streaming and reference RepeatMetrics disagree: "
            + ", ".join(differing)
        )
    return RepeatMetrics(
        ConditionOrdinal=condition_ordinal,
        ConditionCode=condition_code,
        MethodOrdinal=method_ordinal,
        MethodCode=method_code,
        RepeatID=repeat_id,
        **streaming,
    )


def repeat_metrics_from_trajectory(
    trajectory: ConditionRepeatTrajectory,
) -> tuple[RepeatMetrics, ...]:
    """Read the five frozen Stage 4 source-state rows in ordinal order."""
    if not isinstance(trajectory, ConditionRepeatTrajectory):
        raise TypeError("trajectory must be a ConditionRepeatTrajectory")
    if trajectory.method_codes != tuple(code for _, code in METHOD_IDENTITIES):
        raise ValueError("trajectory method order is not frozen FIRST/LAST/T/W/TW")
    return tuple(
        compute_repeat_metrics(
            source_state_code_1d=trajectory.source_state_code[method_ordinal],
            condition_ordinal=trajectory.condition_ordinal,
            condition_code=trajectory.condition_code,
            method_ordinal=method_ordinal,
            method_code=method_code,
            repeat_id=trajectory.repeat_id,
        )
        for method_ordinal, method_code in METHOD_IDENTITIES
    )


@dataclass(frozen=True, slots=True)
class MetricValue:
    value: float | None
    defined: bool
    undefined_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.defined, bool):
            raise TypeError("defined must be bool")
        if self.defined:
            if self.value is None:
                raise ValueError("a defined MetricValue requires a value")
            value = float(self.value)
            if not math.isfinite(value):
                raise ValueError("a defined MetricValue must be finite")
            if self.undefined_reason is not None:
                raise ValueError("a defined MetricValue cannot have an undefined reason")
            object.__setattr__(self, "value", value)
        else:
            if self.value is not None:
                raise ValueError("an undefined MetricValue must use value=None")
            reason = str(self.undefined_reason or "")
            if reason not in UNDEFINED_REASONS:
                raise ValueError("undefined_reason is not a frozen reason code")
            object.__setattr__(self, "undefined_reason", reason)


def _defined_ratio(numerator: int, denominator: int) -> MetricValue:
    if denominator <= 0:
        raise ValueError("a defined ratio requires a positive denominator")
    return MetricValue(float(numerator) / float(denominator), True, None)


def _conditional_ratio(
    numerator: int, denominator: int, zero_reason: str
) -> MetricValue:
    if denominator == 0:
        return MetricValue(None, False, zero_reason)
    return _defined_ratio(numerator, denominator)


@dataclass(frozen=True, slots=True)
class AggregatedMetrics:
    ConditionOrdinal: int
    ConditionCode: str
    MethodOrdinal: int
    MethodCode: str
    R: int
    K: int

    Total_N_C: int
    Total_N_E: int
    Total_N_N: int

    Total_N_CC: int
    Total_N_CE: int
    Total_N_CN: int
    Total_N_EC: int
    Total_N_EE: int
    Total_N_EN: int
    Total_N_NC: int
    Total_N_NE: int
    Total_N_NN: int

    Total_N_Cdot: int
    Total_N_Edot: int
    Total_N_Ndot: int

    Total_N_run_NC: int
    Total_Sum_L_NC_obs: int
    Total_N_open_NC: int

    Total_N_E_H: int
    Total_N_E_F: int

    P_cor: MetricValue
    P_C_given_C: MetricValue
    P_C_given_E: MetricValue
    Mean_L_NC: MetricValue

    P_E_given_C: MetricValue
    P_N_given_C: MetricValue
    P_E_given_E: MetricValue
    P_N_given_E: MetricValue
    P_N: MetricValue
    P_open_NC: MetricValue
    P_NC_end: MetricValue

    P_E_H: MetricValue
    P_E_F: MetricValue


def aggregate_repeat_metrics(rows: Iterable[RepeatMetrics]) -> AggregatedMetrics:
    """Aggregate RepeatMetrics by total-count ratios, never repeat-wise means."""
    materialized = tuple(rows)
    if not materialized:
        raise ValueError("aggregate_repeat_metrics requires at least one row")
    if any(not isinstance(row, RepeatMetrics) for row in materialized):
        raise TypeError("every aggregate input row must be RepeatMetrics")

    identity = (
        materialized[0].ConditionOrdinal,
        materialized[0].ConditionCode,
        materialized[0].MethodOrdinal,
        materialized[0].MethodCode,
    )
    if any(
        (
            row.ConditionOrdinal,
            row.ConditionCode,
            row.MethodOrdinal,
            row.MethodCode,
        )
        != identity
        for row in materialized[1:]
    ):
        raise ValueError("aggregate rows must belong to one Condition and one Method")
    repeat_ids = [row.RepeatID for row in materialized]
    if len(set(repeat_ids)) != len(repeat_ids):
        raise ValueError("RepeatID must be unique within an aggregate")

    totals = {
        name: sum(getattr(row, name) for row in materialized)
        for name in REPEAT_COUNT_FIELDS
    }
    r_count = len(materialized)
    c_denominator = totals["N_Cdot"]
    e_denominator = totals["N_Edot"]
    nc_run_denominator = totals["N_run_NC"]
    error_denominator = totals["N_E"]

    p_c_given_c = _conditional_ratio(
        totals["N_CC"], c_denominator, C_DENOM_ZERO
    )
    p_e_given_c = _conditional_ratio(
        totals["N_CE"], c_denominator, C_DENOM_ZERO
    )
    p_n_given_c = _conditional_ratio(
        totals["N_CN"], c_denominator, C_DENOM_ZERO
    )
    p_c_given_e = _conditional_ratio(
        totals["N_EC"], e_denominator, E_DENOM_ZERO
    )
    p_e_given_e = _conditional_ratio(
        totals["N_EE"], e_denominator, E_DENOM_ZERO
    )
    p_n_given_e = _conditional_ratio(
        totals["N_EN"], e_denominator, E_DENOM_ZERO
    )
    mean_l_nc = _conditional_ratio(
        totals["Sum_L_NC_obs"], nc_run_denominator, NO_NC_RUN
    )
    p_open_nc = _conditional_ratio(
        totals["N_open_NC"], nc_run_denominator, NO_NC_RUN
    )
    p_e_h = _conditional_ratio(totals["N_E_H"], error_denominator, E_COUNT_ZERO)
    p_e_f = _conditional_ratio(totals["N_E_F"], error_denominator, E_COUNT_ZERO)

    if c_denominator > 0:
        c_sum = sum(
            metric.value for metric in (p_c_given_c, p_e_given_c, p_n_given_c)
        )
        if not math.isclose(c_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("C-origin conditional probabilities do not sum to one")
    if e_denominator > 0:
        e_sum = sum(
            metric.value for metric in (p_c_given_e, p_e_given_e, p_n_given_e)
        )
        if not math.isclose(e_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("E-origin conditional probabilities do not sum to one")
    if error_denominator > 0:
        source_sum = p_e_h.value + p_e_f.value
        if not math.isclose(source_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("H/F error-source proportions do not sum to one")
    end_noncorrect_count = sum(
        row.EndState in ("E", "N") for row in materialized
    )
    if end_noncorrect_count != totals["N_open_NC"]:
        raise AssertionError("EndState and N_open_NC aggregate contributions disagree")

    total_kwargs = {
        f"Total_{name}": value for name, value in totals.items()
    }
    return AggregatedMetrics(
        ConditionOrdinal=identity[0],
        ConditionCode=identity[1],
        MethodOrdinal=identity[2],
        MethodCode=identity[3],
        R=r_count,
        K=K,
        **total_kwargs,
        P_cor=_defined_ratio(totals["N_C"], r_count * K),
        P_C_given_C=p_c_given_c,
        P_C_given_E=p_c_given_e,
        Mean_L_NC=mean_l_nc,
        P_E_given_C=p_e_given_c,
        P_N_given_C=p_n_given_c,
        P_E_given_E=p_e_given_e,
        P_N_given_E=p_n_given_e,
        P_N=_defined_ratio(totals["N_N"], r_count * K),
        P_open_NC=p_open_nc,
        P_NC_end=_defined_ratio(totals["N_open_NC"], r_count),
        P_E_H=p_e_h,
        P_E_F=p_e_f,
    )


def repeat_metrics_field_names() -> tuple[str, ...]:
    return tuple(field_info.name for field_info in fields(RepeatMetrics))


def metric_value_to_dict(metric: MetricValue) -> dict[str, object]:
    if not isinstance(metric, MetricValue):
        raise TypeError("metric must be MetricValue")
    return {
        "value": metric.value,
        "defined": metric.defined,
        "undefined_reason": metric.undefined_reason,
    }


def aggregated_metrics_to_dict(metrics: AggregatedMetrics) -> dict[str, object]:
    if not isinstance(metrics, AggregatedMetrics):
        raise TypeError("metrics must be AggregatedMetrics")
    return asdict(metrics)


__all__ = [
    "AggregatedMetrics",
    "C_DENOM_ZERO",
    "E_COUNT_ZERO",
    "E_DENOM_ZERO",
    "EFFECTIVE_INTERVAL_COUNT",
    "EVALUATION_STATES",
    "K",
    "MetricValue",
    "NO_NC_RUN",
    "REPEAT_COUNT_FIELDS",
    "RepeatMetrics",
    "aggregate_repeat_metrics",
    "aggregated_metrics_to_dict",
    "compute_repeat_metrics",
    "evaluation_states_from_source",
    "metric_value_to_dict",
    "reference_repeat_counts",
    "repeat_metrics_field_names",
    "repeat_metrics_from_trajectory",
    "source_state_to_evaluation",
    "streaming_repeat_counts",
]
