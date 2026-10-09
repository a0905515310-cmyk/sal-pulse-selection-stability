from __future__ import annotations

import numpy as np
import pytest

from sal_stability_stage1 import execution, stage9
from sal_stability_stage1.bootstrap import (
    STATISTIC_FIELDS,
    STATISTIC_INDEX,
    bootstrap_repeat_indices,
    run_condition_bootstrap,
)
from sal_stability_stage1.stage10 import STAGE11_EXECUTED, assert_no_forbidden_execution_path


def _statistics() -> np.ndarray:
    counts = np.zeros((5, len(STATISTIC_FIELDS), 2000), dtype=np.int64)
    repeat_values = np.arange(1, 2001, dtype=np.int64)
    counts[:, STATISTIC_INDEX["N_C"], :] = repeat_values % 201
    counts[:, STATISTIC_INDEX["N_CC"], :] = repeat_values % 100
    counts[:, STATISTIC_INDEX["N_Cdot"], :] = 100
    counts[:, STATISTIC_INDEX["N_EC"], :] = repeat_values % 50
    counts[:, STATISTIC_INDEX["N_Edot"], :] = 50
    counts[:, STATISTIC_INDEX["Sum_L_NC_obs"], :] = repeat_values % 200
    counts[:, STATISTIC_INDEX["N_run_NC"], :] = 10
    return counts


def test_small_bootstrap_subset_serial_deterministic_replay_and_shared_methods():
    ids = (1, 2, 3, 4)
    first = run_condition_bootstrap(
        _statistics(),
        condition_ordinal=1,
        bootstrap_ids=ids,
        block_size=2,
        audit_ids=(1,),
    )
    second = run_condition_bootstrap(
        _statistics(),
        condition_ordinal=1,
        bootstrap_ids=ids,
        block_size=2,
        audit_ids=(1,),
    )
    assert np.array_equal(first.audit_indices, second.audit_indices)
    assert np.array_equal(first.individual_values, second.individual_values)
    assert np.array_equal(first.paired_values, second.paired_values)
    for method in range(1, 5):
        assert np.array_equal(first.individual_values[0], first.individual_values[method])


def test_vectorization_block_partition_does_not_change_any_result():
    ids = (1, 2, 3, 4, 5, 6, 7)
    by_one = run_condition_bootstrap(
        _statistics(),
        condition_ordinal=18,
        bootstrap_ids=ids,
        block_size=1,
        audit_ids=(1,),
    )
    by_four = run_condition_bootstrap(
        _statistics(),
        condition_ordinal=18,
        bootstrap_ids=ids,
        block_size=4,
        audit_ids=(1,),
    )
    assert np.array_equal(by_one.individual_values, by_four.individual_values)
    assert np.array_equal(by_one.individual_defined, by_four.individual_defined)
    assert np.array_equal(by_one.paired_values, by_four.paired_values)
    assert np.array_equal(by_one.audit_indices, by_four.audit_indices)


def test_one_index_matrix_is_generated_per_condition_bootstrap_not_per_method():
    indices = bootstrap_repeat_indices(7, [1, 2])
    assert indices.shape == (2, 2000)
    result = run_condition_bootstrap(
        _statistics(),
        condition_ordinal=7,
        bootstrap_ids=(1, 2),
        block_size=2,
        audit_ids=(1,),
    )
    assert np.array_equal(result.audit_indices[0], indices[0])
    assert np.all(result.paired_values == 0.0)


def test_ni_never_enters_inferential_engine():
    with pytest.raises(ValueError, match="NI never enters"):
        run_condition_bootstrap(
            _statistics(),
            condition_ordinal=0,
            bootstrap_ids=(1,),
            block_size=1,
            audit_ids=(1,),
        )


def test_no_physics_formal_rerun_or_stage11_call_path(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("prohibited physical/Formal path called")

    monkeypatch.setattr(execution, "run_condition_repeats_serial", prohibited)
    monkeypatch.setattr(execution, "run_condition_repeats_parallel", prohibited)
    monkeypatch.setattr(stage9, "run_formal_condition_chunked", prohibited)
    result = run_condition_bootstrap(
        _statistics(),
        condition_ordinal=3,
        bootstrap_ids=(1,),
        block_size=1,
        audit_ids=(1,),
    )
    assert result.bootstrap_ids.tolist() == [1]
    gate = assert_no_forbidden_execution_path()
    assert gate["PhysicalSimulationCallAbsent"] is True
    assert gate["FormalRerunCallAbsent"] is True
    assert gate["Stage11CallAbsent"] is True
    assert STAGE11_EXECUTED is False
