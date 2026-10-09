from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sal_stability_stage1.metrics import (
    C_DENOM_ZERO,
    E_COUNT_ZERO,
    E_DENOM_ZERO,
    K,
    NO_NC_RUN,
    MetricValue,
    aggregate_repeat_metrics,
    aggregated_metrics_to_dict,
    compute_repeat_metrics,
    evaluation_states_from_source,
    metric_value_to_dict,
    repeat_metrics_field_names,
    source_state_to_evaluation,
)


def _row(source: np.ndarray, repeat_id: int = 1, **overrides):
    arguments = {
        "source_state_code_1d": source,
        "condition_ordinal": 0,
        "condition_code": "NI",
        "method_ordinal": 0,
        "method_code": "FIRST",
        "repeat_id": repeat_id,
    }
    arguments.update(overrides)
    return compute_repeat_metrics(**arguments)


@pytest.mark.parametrize(
    ("source_code", "expected"), ((0, "N"), (1, "C"), (2, "E"), (3, "E"))
)
def test_source_state_mapping_is_exact(source_code, expected):
    assert source_state_to_evaluation(np.uint8(source_code)) == expected


@pytest.mark.parametrize("invalid", (-1, 4, 5, True, 1.0, np.nan))
def test_source_state_mapping_rejects_invalid_scalars(invalid):
    expected_error = TypeError if invalid is True or isinstance(invalid, float) else ValueError
    with pytest.raises(expected_error):
        source_state_to_evaluation(invalid)


def test_evaluation_array_maps_h_and_f_to_one_e_state_and_is_read_only():
    source = np.resize(np.array([0, 1, 2, 3], dtype=np.uint8), K)
    states = evaluation_states_from_source(source)
    assert states.tolist()[:4] == ["N", "C", "E", "E"]
    assert not states.flags.writeable


@pytest.mark.parametrize(
    "invalid",
    (
        [1] * K,
        np.ones(K, dtype=np.int64),
        np.ones(K, dtype=np.float64),
        np.ones(K - 1, dtype=np.uint8),
        np.full(K, 5, dtype=np.uint8),
    ),
)
def test_repeat_input_requires_exact_shape_dtype_and_values(invalid):
    with pytest.raises((TypeError, ValueError)):
        _row(invalid)


def test_metric_value_encodes_defined_and_undefined_without_nan_or_zero_fill():
    defined = MetricValue(0.25, True, None)
    undefined = MetricValue(None, False, E_DENOM_ZERO)
    assert metric_value_to_dict(defined) == {
        "value": 0.25,
        "defined": True,
        "undefined_reason": None,
    }
    assert metric_value_to_dict(undefined) == {
        "value": None,
        "defined": False,
        "undefined_reason": E_DENOM_ZERO,
    }
    serialized = json.dumps(metric_value_to_dict(undefined), allow_nan=False)
    assert '"value": null' in serialized


@pytest.mark.parametrize(
    "arguments",
    (
        (None, True, None),
        (np.nan, True, None),
        (0.0, False, E_DENOM_ZERO),
        (None, False, None),
        (None, False, "UNKNOWN"),
    ),
)
def test_metric_value_rejects_ambiguous_encodings(arguments):
    with pytest.raises((TypeError, ValueError)):
        MetricValue(*arguments)


def test_all_c_aggregate_has_required_defined_and_undefined_metrics():
    aggregate = aggregate_repeat_metrics([_row(np.ones(K, dtype=np.uint8))])
    assert aggregate.P_cor == MetricValue(1.0, True, None)
    assert aggregate.P_C_given_C == MetricValue(1.0, True, None)
    assert aggregate.P_E_given_C == MetricValue(0.0, True, None)
    assert aggregate.P_N_given_C == MetricValue(0.0, True, None)
    assert aggregate.P_C_given_E == MetricValue(None, False, E_DENOM_ZERO)
    assert aggregate.P_E_given_E == MetricValue(None, False, E_DENOM_ZERO)
    assert aggregate.P_N_given_E == MetricValue(None, False, E_DENOM_ZERO)
    assert aggregate.Mean_L_NC == MetricValue(None, False, NO_NC_RUN)
    assert aggregate.P_open_NC == MetricValue(None, False, NO_NC_RUN)
    assert aggregate.P_N == MetricValue(0.0, True, None)
    assert aggregate.P_NC_end == MetricValue(0.0, True, None)
    assert aggregate.P_E_H == MetricValue(None, False, E_COUNT_ZERO)
    assert aggregate.P_E_F == MetricValue(None, False, E_COUNT_ZERO)


def test_all_e_and_all_n_aggregates_preserve_right_open_runs():
    all_h = aggregate_repeat_metrics([_row(np.full(K, 2, dtype=np.uint8))])
    assert all_h.P_cor.value == 0.0
    assert all_h.P_C_given_C.undefined_reason == C_DENOM_ZERO
    assert all_h.P_C_given_E.value == 0.0
    assert all_h.P_E_given_E.value == 1.0
    assert all_h.Mean_L_NC.value == 200.0
    assert all_h.P_open_NC.value == 1.0
    assert all_h.P_NC_end.value == 1.0
    assert all_h.P_E_H.value == 1.0
    assert all_h.P_E_F.value == 0.0

    all_n = aggregate_repeat_metrics([_row(np.zeros(K, dtype=np.uint8))])
    assert all_n.P_N.value == 1.0
    assert all_n.P_C_given_E.undefined_reason == E_DENOM_ZERO
    assert all_n.Mean_L_NC.value == 200.0
    assert all_n.P_open_NC.value == 1.0
    assert all_n.P_NC_end.value == 1.0


def test_conditional_metrics_use_total_count_ratio_not_repeat_probability_mean():
    repeat_one = np.array([2, 1] + [1] * 198, dtype=np.uint8)
    repeat_two = np.array(([2] * 10 + [1]) * 10 + [1] * 90, dtype=np.uint8)
    row_one = _row(repeat_one, repeat_id=1)
    row_two = _row(repeat_two, repeat_id=2)
    assert (row_one.N_EC, row_one.N_Edot) == (1, 1)
    assert (row_two.N_EC, row_two.N_Edot) == (10, 100)
    aggregate = aggregate_repeat_metrics([row_one, row_two])
    total_ratio = 11.0 / 101.0
    repeat_probability_mean = (1.0 + 0.1) / 2.0
    assert aggregate.P_C_given_E.value == total_ratio
    assert aggregate.P_C_given_E.value != repeat_probability_mean


def test_aggregate_is_invariant_to_row_order():
    rows = [
        _row(np.resize(np.array([1, 2, 0, 3], dtype=np.uint8), K), repeat_id=1),
        _row(np.resize(np.array([2, 1, 3, 0, 1], dtype=np.uint8), K), repeat_id=2),
        _row(np.resize(np.array([3, 3, 1, 0], dtype=np.uint8), K), repeat_id=17),
    ]
    assert aggregate_repeat_metrics(rows) == aggregate_repeat_metrics(tuple(reversed(rows)))
    assert aggregate_repeat_metrics(rows) == aggregate_repeat_metrics((rows[1], rows[2], rows[0]))


def test_aggregate_rejects_duplicate_repeat_condition_and_method_mixtures():
    source = np.ones(K, dtype=np.uint8)
    base = _row(source)
    with pytest.raises(ValueError, match="RepeatID"):
        aggregate_repeat_metrics((base, base))

    other_condition = _row(
        source,
        repeat_id=2,
        condition_ordinal=3,
        condition_code="H300",
    )
    with pytest.raises(ValueError, match="Condition"):
        aggregate_repeat_metrics((base, other_condition))

    other_method = _row(
        source,
        repeat_id=2,
        method_ordinal=4,
        method_code="TW",
    )
    with pytest.raises(ValueError, match="Method"):
        aggregate_repeat_metrics((base, other_method))


def test_c_and_e_origin_probability_families_sum_to_one():
    source = np.resize(np.array([1, 1, 2, 0, 2, 3, 1, 0], dtype=np.uint8), K)
    aggregate = aggregate_repeat_metrics([_row(source)])
    assert sum(
        metric.value
        for metric in (
            aggregate.P_C_given_C,
            aggregate.P_E_given_C,
            aggregate.P_N_given_C,
        )
    ) == pytest.approx(1.0, abs=1e-15)
    assert sum(
        metric.value
        for metric in (
            aggregate.P_C_given_E,
            aggregate.P_E_given_E,
            aggregate.P_N_given_E,
        )
    ) == pytest.approx(1.0, abs=1e-15)


def test_h_f_source_proportions_use_e_count_and_sum_to_one():
    source = np.resize(np.array([1, 2, 2, 3, 1, 3], dtype=np.uint8), K)
    row = _row(source)
    aggregate = aggregate_repeat_metrics([row])
    assert row.N_E_H + row.N_E_F == row.N_E
    assert aggregate.P_E_H.value + aggregate.P_E_F.value == pytest.approx(1.0)


def test_aggregate_json_form_has_explicit_metric_value_objects():
    aggregate = aggregate_repeat_metrics([_row(np.ones(K, dtype=np.uint8))])
    payload = aggregated_metrics_to_dict(aggregate)
    assert payload["P_C_given_E"] == {
        "value": None,
        "defined": False,
        "undefined_reason": E_DENOM_ZERO,
    }
    json.dumps(payload, allow_nan=False)


def test_repeat_metrics_schema_contains_integer_contributions_not_probabilities():
    field_names = repeat_metrics_field_names()
    required = {
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
    }
    assert required.issubset(field_names)
    assert not any(name.startswith("P_") for name in field_names)
    row = _row(np.ones(K, dtype=np.uint8))
    assert all(isinstance(getattr(row, name), int) for name in required - {"EndState"})
    with pytest.raises(ValueError):
        replace(row, N_CC=row.N_CC - 1)

