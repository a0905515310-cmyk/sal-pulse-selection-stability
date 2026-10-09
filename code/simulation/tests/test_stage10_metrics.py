from __future__ import annotations

import numpy as np
import pytest

from sal_stability_stage1.bootstrap import (
    BOOTSTRAP_DRAW_COUNT,
    STATISTIC_FIELDS,
    STATISTIC_INDEX,
    compute_bootstrap_metric_block,
    formal_point_estimates,
    percentile_interval,
)
from sal_stability_stage1.stage10 import (
    load_formal_repeat_metrics,
    reconstitute_formal_point_estimates,
)


def _empty_statistics() -> np.ndarray:
    return np.zeros((5, len(STATISTIC_FIELDS), 2000), dtype=np.int64)


def test_ratio_of_resampled_sums_is_not_mean_of_repeat_ratios():
    statistics = _empty_statistics()
    statistics[:, STATISTIC_INDEX["N_CC"], 0] = 1
    statistics[:, STATISTIC_INDEX["N_Cdot"], 0] = 1
    statistics[:, STATISTIC_INDEX["N_CC"], 1] = 1
    statistics[:, STATISTIC_INDEX["N_Cdot"], 1] = 9
    repeat_ids = np.asarray([[1] * 1000 + [2] * 1000], dtype=np.uint16)
    result = compute_bootstrap_metric_block(statistics, repeat_ids)
    expected_ratio_of_sums = 2000 / 10000
    forbidden_mean_of_ratios = (1.0 + 1.0 / 9.0) / 2.0
    assert np.all(result.individual_values[:, 1, 0] == expected_ratio_of_sums)
    assert not np.any(result.individual_values[:, 1, 0] == forbidden_mean_of_ratios)


def test_p_c_given_e_uses_n_ec_and_zero_denominator_is_nan_without_redraw():
    statistics = _empty_statistics()
    statistics[:, STATISTIC_INDEX["N_EC"], :] = 3
    statistics[:, STATISTIC_INDEX["N_Edot"], :] = 10
    statistics[4, STATISTIC_INDEX["N_Edot"], :] = 0
    indices = np.tile(np.arange(1, 2001, dtype=np.uint16), (1, 1))
    result = compute_bootstrap_metric_block(statistics, indices)
    assert result.individual_values[0, 2, 0] == 0.3
    assert result.individual_defined[0, 2, 0]
    assert np.isnan(result.individual_values[4, 2, 0])
    assert not result.individual_defined[4, 2, 0]
    assert np.isnan(result.paired_values[0, 2, 0])
    assert np.isnan(result.paired_values[1, 2, 0])
    assert not result.paired_defined[0, 2, 0]
    assert not result.paired_defined[1, 2, 0]


def test_tw_minus_comparator_direction_is_not_flipped_for_mean_l_nc():
    statistics = _empty_statistics()
    statistics[:, STATISTIC_INDEX["N_run_NC"], :] = 1
    statistics[2, STATISTIC_INDEX["Sum_L_NC_obs"], :] = 2
    statistics[3, STATISTIC_INDEX["Sum_L_NC_obs"], :] = 3
    statistics[4, STATISTIC_INDEX["Sum_L_NC_obs"], :] = 4
    indices = np.arange(1, BOOTSTRAP_DRAW_COUNT + 1, dtype=np.uint16)[None, :]
    result = compute_bootstrap_metric_block(statistics, indices)
    assert result.paired_values[0, 3, 0] == 2.0
    assert result.paired_values[1, 3, 0] == 1.0


def test_percentile_ci_uses_linear_quantiles_and_insufficient_defined_rule():
    values = np.arange(2000, dtype=np.float64)
    defined = np.ones(2000, dtype=bool)
    ci_defined, reason, lower, upper = percentile_interval(values, defined)
    assert ci_defined and reason == ""
    assert lower == np.quantile(values, 0.025, method="linear")
    assert upper == np.quantile(values, 0.975, method="linear")
    sparse = np.full(4, np.nan)
    sparse[2] = 1.0
    ci_defined, reason, lower, upper = percentile_interval(
        sparse, np.asarray([False, False, True, False])
    )
    assert not ci_defined
    assert reason == "INSUFFICIENT_DEFINED_BOOTSTRAP_REPLICATES"
    assert np.isnan(lower) and np.isnan(upper)


def test_formal_point_estimates_are_reconstituted_from_stage9_mother_table():
    formal = load_formal_repeat_metrics()
    points, report = reconstitute_formal_point_estimates(formal)
    assert len(points) == 19
    assert report["AggregateRowCount"] == 95
    assert report["CoreMetricScalarCheckCount"] == 380
    assert report["Passed"] is True
    direct = formal_point_estimates(formal.statistics[1])
    assert np.array_equal(direct.defined, points[1].defined)
    assert np.array_equal(direct.sum_a, points[1].sum_a)
    assert np.array_equal(direct.sum_b, points[1].sum_b)
    assert np.allclose(direct.values, points[1].values, rtol=0, atol=0, equal_nan=True)


def test_undefined_values_cannot_be_encoded_as_zero():
    values = np.asarray([0.0, np.nan])
    with pytest.raises(ValueError, match="undefined values must be NaN"):
        percentile_interval(values, np.asarray([False, False]))
