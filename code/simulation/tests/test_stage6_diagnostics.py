from __future__ import annotations

import csv

import pytest

from sal_stability_stage1.diagnostics import DIAGNOSTIC_FIELD_NAMES, write_diagnostics_csv
from sal_stability_stage1.execution import (
    ExecutionOptions,
    assert_scientific_results_equal,
    run_condition_repeats_serial,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import load_stage3_inputs


@pytest.fixture(scope="module")
def diagnostic_results():
    config, reference = load_stage3_inputs()
    common = dict(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
    )
    off = run_condition_repeats_serial(
        **common, execution_options=ExecutionOptions(diagnostic=False)
    )
    on = run_condition_repeats_serial(
        **common, execution_options=ExecutionOptions(diagnostic=True)
    )
    return off, on


def test_diagnostic_on_off_is_scientifically_exact(diagnostic_results):
    off, on = diagnostic_results
    assert off.diagnostics == ()
    assert len(on.diagnostics) == 5 * 200
    assert_scientific_results_equal(off, on, label="diagnostic OFF/ON")


def test_diagnostic_source_is_read_only_after_selection_and_matches_trajectory(
    diagnostic_results,
):
    _, on = diagnostic_results
    assert [(row.Cycle, row.MethodOrdinal) for row in on.diagnostics[:5]] == [
        (1, method_ordinal) for method_ordinal in range(5)
    ]
    for record in on.diagnostics:
        expected_source = int(
            on.source_state[
                record.MethodOrdinal,
                0,
                record.Cycle - 1,
            ]
        )
        assert record.SelectedSourceAfterSelection == expected_source
        assert (record.SelectedPos is None) == (expected_source == 0)


def test_diagnostic_csv_has_stable_schema_and_row_count(
    diagnostic_results, tmp_path
):
    _, on = diagnostic_results
    path = tmp_path / "diagnostics.csv"
    write_diagnostics_csv(path, on.diagnostics)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == DIAGNOSTIC_FIELD_NAMES
    assert len(rows) == 1000
    assert rows[0]["SelectedSourceAfterSelection"] in {"1", "2", "3"}
