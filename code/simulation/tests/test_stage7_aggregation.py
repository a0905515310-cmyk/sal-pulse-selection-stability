from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sal_stability_stage1.contracts import CONDITIONS, METHODS
from sal_stability_stage1.execution import ExecutionOptions, run_condition_repeats_serial
from sal_stability_stage1.metrics import (
    E_COUNT_ZERO,
    E_DENOM_ZERO,
    NO_NC_RUN,
    RepeatMetrics,
    aggregate_repeat_metrics,
    compute_repeat_metrics,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import load_stage3_inputs
from sal_stability_stage1.stage7 import (
    aggregate_pilot_repeat_metrics,
    flatten_aggregated_metrics,
    validate_h300_width_strata,
    validate_ni_identity,
    validate_n_state_assertions,
    validate_repeat_metrics_rows,
)


def _all_c_row(condition_ordinal: int, method_ordinal: int, repeat_id: int) -> RepeatMetrics:
    return RepeatMetrics(
        ConditionOrdinal=condition_ordinal,
        ConditionCode=CONDITIONS[condition_ordinal].Code,
        MethodOrdinal=method_ordinal,
        MethodCode=METHODS[method_ordinal].Code,
        RepeatID=repeat_id,
        N_C=200,
        N_E=0,
        N_N=0,
        N_CC=199,
        N_CE=0,
        N_CN=0,
        N_EC=0,
        N_EE=0,
        N_EN=0,
        N_NC=0,
        N_NE=0,
        N_NN=0,
        N_Cdot=199,
        N_Edot=0,
        N_Ndot=0,
        N_run_NC=0,
        Sum_L_NC_obs=0,
        N_open_NC=0,
        EndState="C",
        N_E_H=0,
        N_E_F=0,
    )


@pytest.fixture(scope="module")
def full_synthetic_pilot_rows():
    return tuple(
        _all_c_row(condition.ConditionOrdinal, method.MethodOrdinal, repeat_id)
        for condition in CONDITIONS
        for method in METHODS
        for repeat_id in range(1, 201)
    )


def test_pilot_aggregation_is_ratio_of_total_counts_not_mean_of_ratios():
    first_source = np.ones(200, dtype=np.uint8)
    first_source[100:] = np.uint8(2)
    second_source = np.ones(200, dtype=np.uint8)
    second_source[1:20] = np.uint8(2)
    first = compute_repeat_metrics(
        source_state_code_1d=first_source,
        condition_ordinal=1,
        condition_code="H100",
        method_ordinal=2,
        method_code="T",
        repeat_id=1,
    )
    second = compute_repeat_metrics(
        source_state_code_1d=second_source,
        condition_ordinal=1,
        condition_code="H100",
        method_ordinal=2,
        method_code="T",
        repeat_id=2,
    )
    aggregate = aggregate_repeat_metrics((first, second))
    expected = (first.N_EC + second.N_EC) / (first.N_Edot + second.N_Edot)
    repeat_mean = (first.N_EC / first.N_Edot + second.N_EC / second.N_Edot) / 2
    assert aggregate.P_C_given_E.value == expected
    assert expected != repeat_mean


def test_zero_denominator_metrics_remain_explicitly_undefined():
    aggregate = aggregate_repeat_metrics((_all_c_row(0, 0, 1),))
    assert aggregate.P_C_given_E.value is None
    assert aggregate.P_C_given_E.defined is False
    assert aggregate.P_C_given_E.undefined_reason == E_DENOM_ZERO
    assert aggregate.Mean_L_NC.value is None
    assert aggregate.Mean_L_NC.defined is False
    assert aggregate.Mean_L_NC.undefined_reason == NO_NC_RUN
    assert aggregate.P_E_H.value is None
    assert aggregate.P_E_H.defined is False
    assert aggregate.P_E_H.undefined_reason == E_COUNT_ZERO
    flattened = flatten_aggregated_metrics(aggregate)
    assert flattened["P_C_given_E_value"] is None
    assert flattened["P_C_given_E_defined"] is False
    assert flattened["P_C_given_E_undefined_reason"] == E_DENOM_ZERO


def test_full_synthetic_pilot_conservation_and_95_aggregates(full_synthetic_pilot_rows):
    checks = validate_repeat_metrics_rows(full_synthetic_pilot_rows)
    assert checks == {
        "repeat_conservation_passed": True,
        "transition_conservation_passed": True,
    }
    aggregate_rows = aggregate_pilot_repeat_metrics(full_synthetic_pilot_rows)
    assert len(aggregate_rows) == 95
    assert all(row.R == 200 and row.K == 200 for row in aggregate_rows)
    assert all(row.P_cor.value == 1.0 for row in aggregate_rows)


def test_hprf_and_composite_no_n_assertions_accept_complete_no_n_data(
    full_synthetic_pilot_rows,
):
    assert validate_n_state_assertions(full_synthetic_pilot_rows) == (True, True)


@pytest.mark.parametrize("condition_ordinal", (1, 10))
def test_hprf_and_composite_no_n_assertions_reject_n_state(
    full_synthetic_pilot_rows, condition_ordinal
):
    source = np.ones(200, dtype=np.uint8)
    source[0] = np.uint8(0)
    violating = compute_repeat_metrics(
        source_state_code_1d=source,
        condition_ordinal=condition_ordinal,
        condition_code=CONDITIONS[condition_ordinal].Code,
        method_ordinal=0,
        method_code="FIRST",
        repeat_id=1,
    )
    rows = list(full_synthetic_pilot_rows)
    position = next(
        index
        for index, row in enumerate(rows)
        if row.ConditionOrdinal == condition_ordinal
        and row.MethodOrdinal == 0
        and row.RepeatID == 1
    )
    rows[position] = violating
    with pytest.raises(AssertionError, match="forbidden N state"):
        validate_n_state_assertions(rows)


def test_ni_scientific_identity_and_h300_three_strata_are_enforced():
    config, reference = load_stage3_inputs()
    options = ExecutionOptions(workers=1, chunk_size=1)
    ni = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_ids=(1,),
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
    )
    assert validate_ni_identity(ni) is True
    h300 = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_ids=(1,),
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
    )
    assert len(h300.h300_width_strata) == 15
    assert {row.WidthStratum for row in h300.h300_width_strata} == {
        "WIDTH_STRATUM_0NS",
        "WIDTH_STRATUM_10NS",
        "WIDTH_STRATUM_GE20NS",
    }
    assert validate_h300_width_strata(
        h300.h300_width_strata, expected_repeat_ids=(1,)
    )

