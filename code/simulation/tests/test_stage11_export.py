from __future__ import annotations

from pathlib import Path

from sal_stability_stage1.build import ROOT
from sal_stability_stage1.results_export import (
    COMPOSITE_CONDITIONS,
    COMPOSITE_INDIVIDUAL_REL,
    COMPOSITE_PAIRED_REL,
    EVIDENCE_REGISTRY_REL,
    F_ONLY_CONDITIONS,
    F_ONLY_INDIVIDUAL_REL,
    F_ONLY_PAIRED_REL,
    H300_REL,
    HPRF_CONDITIONS,
    HPRF_INDIVIDUAL_REL,
    HPRF_PAIRED_REL,
    INDIVIDUAL_FIELDS,
    INDIVIDUAL_MASTER_REL,
    INDIVIDUAL_SOURCE_FIELDS,
    METRICS,
    NI_REL,
    PAIRED_FIELDS,
    PAIRED_MASTER_REL,
    PAIRED_SOURCE_FIELDS,
    PRECISION_FIELDS,
    PRECISION_LIMITATIONS_REL,
    PRECISION_MASTER_REL,
    PRECISION_SOURCE_FIELDS,
    STAGE11_DIR,
    build_export_tables,
    read_csv_exact,
    validate_evidence_exports,
)


def _keyed(rows, fields):
    return {tuple(row[field] for field in fields): row for row in rows}


def test_all_master_and_scenario_row_count_gates_and_unique_keys_pass():
    validation = validate_evidence_exports()
    counts = validation["RowCounts"]
    assert counts[INDIVIDUAL_MASTER_REL] == 360
    assert counts[PAIRED_MASTER_REL] == 144
    assert counts[PRECISION_MASTER_REL] == 216
    assert counts[HPRF_INDIVIDUAL_REL] == 100
    assert counts[HPRF_PAIRED_REL] == 40
    assert counts[F_ONLY_INDIVIDUAL_REL] == 80
    assert counts[F_ONLY_PAIRED_REL] == 32
    assert counts[COMPOSITE_INDIVIDUAL_REL] == 180
    assert counts[COMPOSITE_PAIRED_REL] == 72
    assert counts[NI_REL] == 5
    assert counts[H300_REL] == 15
    assert validation["EvidenceRegistryRows"] == 16


def test_individual_stage10_source_fields_are_string_exact_in_master():
    source = read_csv_exact(
        ROOT / "artifacts/stage10/bootstrap_individual_summary.csv",
        INDIVIDUAL_SOURCE_FIELDS,
        360,
    )
    master = read_csv_exact(
        STAGE11_DIR / INDIVIDUAL_MASTER_REL, INDIVIDUAL_FIELDS, 360
    )
    key_fields = ("ConditionOrdinal", "MethodOrdinal", "Metric")
    source_by_key = _keyed(source, key_fields)
    master_by_key = _keyed(master, key_fields)
    assert set(source_by_key) == set(master_by_key)
    for key, source_row in source_by_key.items():
        assert {field: master_by_key[key][field] for field in INDIVIDUAL_SOURCE_FIELDS} == (
            source_row
        )


def test_paired_stage10_source_fields_and_ci_geometry_are_exact():
    source = read_csv_exact(
        ROOT / "artifacts/stage10/bootstrap_paired_summary.csv",
        PAIRED_SOURCE_FIELDS,
        144,
    )
    master = read_csv_exact(STAGE11_DIR / PAIRED_MASTER_REL, PAIRED_FIELDS, 144)
    key_fields = ("ConditionOrdinal", "ContrastOrdinal", "Metric")
    source_by_key = _keyed(source, key_fields)
    master_by_key = _keyed(master, key_fields)
    assert set(source_by_key) == set(master_by_key)
    allowed = {"POSITIVE", "NEGATIVE", "INCLUDES_ZERO", "UNDEFINED"}
    for key, source_row in source_by_key.items():
        output = master_by_key[key]
        assert {field: output[field] for field in PAIRED_SOURCE_FIELDS} == source_row
        assert output["CIPositionRelativeToZero"] in allowed
        if source_row["CI95Defined"] == "False":
            expected = "UNDEFINED"
        elif float(source_row["CI95Lower"]) > 0:
            expected = "POSITIVE"
        elif float(source_row["CI95Upper"]) < 0:
            expected = "NEGATIVE"
        else:
            expected = "INCLUDES_ZERO"
        assert output["CIPositionRelativeToZero"] == expected


def test_precision_master_is_exact_and_limitations_are_the_complete_false_subset():
    source = read_csv_exact(
        ROOT / "artifacts/stage10/stage8_precision_target_audit.csv",
        PRECISION_SOURCE_FIELDS,
        216,
    )
    master = read_csv_exact(
        STAGE11_DIR / PRECISION_MASTER_REL, PRECISION_FIELDS, 216
    )
    assert [
        {field: row[field] for field in PRECISION_SOURCE_FIELDS} for row in master
    ] == source
    limitations = read_csv_exact(
        STAGE11_DIR / PRECISION_LIMITATIONS_REL, PRECISION_FIELDS
    )
    assert limitations == [
        row for row in master if row["MeetsStage8PrecisionTarget"] == "False"
    ]
    assert all(row["MeetsStage8PrecisionTarget"] == "False" for row in limitations)


def test_scenario_slices_cover_every_condition_without_reselection():
    expectations = (
        (HPRF_INDIVIDUAL_REL, INDIVIDUAL_FIELDS, set(HPRF_CONDITIONS)),
        (HPRF_PAIRED_REL, PAIRED_FIELDS, set(HPRF_CONDITIONS)),
        (F_ONLY_INDIVIDUAL_REL, INDIVIDUAL_FIELDS, set(F_ONLY_CONDITIONS)),
        (F_ONLY_PAIRED_REL, PAIRED_FIELDS, set(F_ONLY_CONDITIONS)),
        (
            COMPOSITE_INDIVIDUAL_REL,
            INDIVIDUAL_FIELDS,
            set(COMPOSITE_CONDITIONS),
        ),
        (COMPOSITE_PAIRED_REL, PAIRED_FIELDS, set(COMPOSITE_CONDITIONS)),
    )
    for relative, fields, expected_conditions in expectations:
        rows = read_csv_exact(STAGE11_DIR / relative, fields)
        assert {int(row["ConditionOrdinal"]) for row in rows} == expected_conditions


def test_ni_has_five_methods_and_no_invented_ci_or_zero_for_undefined_metrics():
    fields, expected = build_export_tables()[NI_REL]
    rows = read_csv_exact(STAGE11_DIR / NI_REL, fields, 5)
    assert rows == expected
    assert {int(row["MethodOrdinal"]) for row in rows} == set(range(5))
    assert all(row["P_C_given_E_defined"] == "False" for row in rows)
    assert all(row["P_C_given_E_value"] == "" for row in rows)
    assert all(row["P_C_given_E_undefined_reason"] == "E_DENOM_ZERO" for row in rows)
    assert not any("CI" in field or "p_value" in field.lower() for field in fields)


def test_h300_is_pooled_count_conserving_and_has_no_ci_pvalue_or_fabricated_rate():
    fields, expected = build_export_tables()[H300_REL]
    rows = read_csv_exact(STAGE11_DIR / H300_REL, fields, 15)
    assert rows == expected
    for row in rows:
        total = int(row["TotalCycleCount"])
        correct = int(row["TotalCorrectCount"])
        error = int(row["TotalErrorCount"])
        assert correct + error == total
        assert row["P_cor_stratum"] == repr(correct / total)
        assert row["P_error_stratum"] == repr(error / total)
        assert row["Estimator"] == "RATIO_OF_POOLED_COUNTS"
    assert "P_C_to_E_stratum" not in fields
    assert not any("CI" in field or "p_value" in field.lower() for field in fields)


def test_no_forbidden_significance_or_superiority_labels_enter_any_evidence_csv():
    forbidden = {
        "SIGNIFICANT",
        "NOT_SIGNIFICANT",
        "SUPERIOR",
        "INFERIOR",
        "WIN",
        "LOSE",
    }
    for path in STAGE11_DIR.rglob("*.csv"):
        text = path.read_text(encoding="utf-8")
        assert not any(label in text for label in forbidden), path
    assert (STAGE11_DIR / EVIDENCE_REGISTRY_REL).is_file()

