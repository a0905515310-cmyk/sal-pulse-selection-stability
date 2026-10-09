from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import sal_stability_stage1.execution as execution
from sal_stability_stage1.execution import (
    ExecutionOptions,
    assemble_condition_repeat_results,
    assert_scientific_results_equal,
    run_condition_repeat,
    run_condition_repeats_parallel,
    run_condition_repeats_serial,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import load_stage3_inputs


@pytest.fixture(scope="module")
def frozen_inputs():
    return load_stage3_inputs()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"workers": 0},
        {"workers": True},
        {"chunk_size": 0},
        {"resume": 1},
        {"diagnostic": "false"},
        {"h_cache_enabled": 0},
        {"method_execution_order": (0, 1, 2, 3, 3)},
        {"method_execution_order": (0, 1, 2, 3)},
        {"method_execution_order": (0, 1, 2, 3, 5)},
    ),
)
def test_execution_options_reject_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ExecutionOptions(**kwargs)


def test_serial_runner_sorts_repeat_ids_and_emits_canonical_axes(frozen_inputs):
    config, reference = frozen_inputs
    result = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_ids=(10002, 10001),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(),
    )
    assert result.repeat_ids == (10001, 10002)
    assert result.source_state.shape == (5, 2, 200)
    assert result.source_state.dtype == np.uint8
    assert result.ref_toa_before_s.shape == (5, 2, 200)
    assert result.selected_observed_toa_s.shape == (5, 2, 200)
    assert result.has_selection.shape == (5, 2, 200)
    assert [(row.RepeatID, row.MethodOrdinal) for row in result.repeat_metrics] == [
        (repeat_id, method_ordinal)
        for repeat_id in (10001, 10002)
        for method_ordinal in range(5)
    ]
    assert result.h300_width_strata == ()
    assert all(
        not array.flags.writeable
        for array in (
            result.source_state,
            result.ref_toa_before_s,
            result.selected_observed_toa_s,
            result.has_selection,
        )
    )


def test_condition_repeat_builds_exactly_one_shared_world(frozen_inputs, monkeypatch):
    config, reference = frozen_inputs
    original = execution.build_shared_physical_world
    calls = []

    def counted_builder(**kwargs):
        calls.append((kwargs["condition_ordinal"], kwargs["repeat_id"]))
        return original(**kwargs)

    monkeypatch.setattr(execution, "build_shared_physical_world", counted_builder)
    result = run_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_id=10001,
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(),
    )
    assert calls == [(0, 10001)]
    assert len(result.repeat_metrics) == 5


def test_collector_is_independent_of_condition_repeat_completion_order(frozen_inputs):
    config, reference = frozen_inputs
    options = ExecutionOptions()
    first = run_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_id=10001,
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
    )
    second = run_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_id=10002,
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
    )
    forward = assemble_condition_repeat_results(
        (first, second), expected_repeat_ids=(10001, 10002)
    )
    reverse = assemble_condition_repeat_results(
        (second, first), expected_repeat_ids=(10001, 10002)
    )
    assert_scientific_results_equal(forward, reverse, label="completion order")


def test_serial_parallel_workers_two_are_exact(frozen_inputs):
    config, reference = frozen_inputs
    serial_options = ExecutionOptions(workers=1)
    serial = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_ids=(10001, 10002),
        study_config=config,
        encoding_reference=reference,
        execution_options=serial_options,
    )
    parallel = run_condition_repeats_parallel(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_ids=(10001, 10002),
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(serial_options, workers=2),
    )
    assert_scientific_results_equal(serial, parallel, label="serial/parallel")


def test_duplicate_repeat_ids_are_rejected(frozen_inputs):
    config, reference = frozen_inputs
    with pytest.raises(ValueError, match="duplicates"):
        run_condition_repeats_serial(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=7,
            repeat_ids=(10001, 10001),
            study_config=config,
            encoding_reference=reference,
            execution_options=ExecutionOptions(),
        )
