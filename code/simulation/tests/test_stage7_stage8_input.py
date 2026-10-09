from __future__ import annotations

import csv

import pytest

from sal_stability_stage1.contracts import CONDITIONS, METHODS
from sal_stability_stage1.metrics import RepeatMetrics
from sal_stability_stage1.stage7 import (
    STAGE8_INPUT_FIELDS,
    build_stage8_decision_rows,
    write_stage8_decision_input,
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


def test_stage8_decision_input_has_exact_scope_methods_repeats_and_order(
    full_synthetic_pilot_rows,
):
    rows = build_stage8_decision_rows(full_synthetic_pilot_rows)
    assert len(rows) == 7200
    assert tuple(rows[0]) == STAGE8_INPUT_FIELDS
    assert {row["ConditionOrdinal"] for row in rows} == set(range(1, 19))
    assert 0 not in {row["ConditionOrdinal"] for row in rows}
    assert {row["MethodOrdinal"] for row in rows} == {2, 4}
    assert {row["MethodCode"] for row in rows} == {"T", "TW"}
    for condition_ordinal in range(1, 19):
        for method_ordinal in (2, 4):
            repeats = [
                row["RepeatID"]
                for row in rows
                if row["ConditionOrdinal"] == condition_ordinal
                and row["MethodOrdinal"] == method_ordinal
            ]
            assert repeats == list(range(1, 201))
    keys = [
        (row["ConditionOrdinal"], row["MethodOrdinal"], row["RepeatID"])
        for row in rows
    ]
    assert keys == sorted(keys)
    assert len(keys) == len(set(keys))


def test_stage8_decision_input_csv_schema_and_row_count(
    full_synthetic_pilot_rows, tmp_path
):
    rows = build_stage8_decision_rows(full_synthetic_pilot_rows)
    path = tmp_path / "stage8_decision_input.csv"
    write_stage8_decision_input(path, rows)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        materialized = tuple(reader)
    assert tuple(reader.fieldnames or ()) == STAGE8_INPUT_FIELDS
    assert len(materialized) == 7200
    assert materialized[0]["ConditionOrdinal"] == "1"
    assert materialized[0]["MethodCode"] == "T"
    assert materialized[0]["RepeatID"] == "1"
    assert materialized[-1]["ConditionOrdinal"] == "18"
    assert materialized[-1]["MethodCode"] == "TW"
    assert materialized[-1]["RepeatID"] == "200"


def test_stage8_decision_input_rejects_missing_repeat(full_synthetic_pilot_rows):
    missing = tuple(
        row
        for row in full_synthetic_pilot_rows
        if not (
            row.ConditionOrdinal == 1
            and row.MethodOrdinal == 2
            and row.RepeatID == 1
        )
    )
    with pytest.raises(AssertionError, match="incomplete"):
        build_stage8_decision_rows(missing)


def test_stage8_decision_input_uses_n_ec_not_n_ce(full_synthetic_pilot_rows):
    rows = build_stage8_decision_rows(full_synthetic_pilot_rows)
    assert "N_EC" in rows[0]
    assert "N_CE" not in rows[0]

