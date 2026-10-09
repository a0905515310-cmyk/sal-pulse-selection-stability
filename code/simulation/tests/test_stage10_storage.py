from __future__ import annotations

import json

import numpy as np
import pytest

from sal_stability_stage1.bootstrap import (
    BOOTSTRAP_AUDIT_IDS,
    BOOTSTRAP_B,
    CONTRAST_CODES,
    METHOD_CODES,
    METRIC_CODES,
    STATISTIC_FIELDS,
    ConditionBootstrapResult,
    formal_point_estimates,
)
from sal_stability_stage1.bootstrap_storage import (
    EXPECTED_AUDIT_SHAPE,
    EXPECTED_INDIVIDUAL_SHAPE,
    EXPECTED_PAIRED_SHAPE,
    combined_distribution_arrays,
    condition_directory,
    load_valid_condition,
    run_or_resume_condition,
    sha256_file,
    validate_combined_arrays,
    validate_condition_arrays,
    write_condition_result,
)
from sal_stability_stage1.contracts import CONDITIONS
from sal_stability_stage1.stage10 import (
    build_individual_summary_rows,
    build_paired_summary_rows,
    build_stage8_precision_audit_rows,
)


IDENTITY = {
    "FormalRepeatMetricsSHA256": "a" * 64,
    "Stage10ScienceContractSHA256": "b" * 64,
    "Stage10CodeSHA256": "c" * 64,
    "Stage9BaselineZipSHA256": "d" * 64,
}


def _result(
    condition: int,
    *,
    bootstrap_ids: tuple[int, ...] = (1, 2, 3, 4),
    audit_ids: tuple[int, ...] = (1,),
) -> ConditionBootstrapResult:
    b = len(bootstrap_ids)
    individual_values = np.zeros((5, 4, b), dtype=np.float64)
    individual_defined = np.ones((5, 4, b), dtype=bool)
    paired_values = np.zeros((2, 4, b), dtype=np.float64)
    paired_defined = np.ones((2, 4, b), dtype=bool)
    audit_indices = np.tile(
        np.arange(1, 2001, dtype=np.uint16), (len(audit_ids), 1)
    )
    return ConditionBootstrapResult(
        condition_ordinal=condition,
        bootstrap_ids=np.asarray(bootstrap_ids, dtype=np.int16),
        individual_values=individual_values,
        individual_defined=individual_defined,
        paired_values=paired_values,
        paired_defined=paired_defined,
        audit_ids=np.asarray(audit_ids, dtype=np.int16),
        audit_indices=audit_indices,
    )


def _official_result(condition: int) -> ConditionBootstrapResult:
    b_ids = np.arange(1, BOOTSTRAP_B + 1, dtype=np.int16)
    individual_values = np.zeros((5, 4, BOOTSTRAP_B), dtype=np.float64)
    individual_defined = np.ones((5, 4, BOOTSTRAP_B), dtype=bool)
    paired_values = np.zeros((2, 4, BOOTSTRAP_B), dtype=np.float64)
    paired_defined = np.ones((2, 4, BOOTSTRAP_B), dtype=bool)
    return ConditionBootstrapResult(
        condition_ordinal=condition,
        bootstrap_ids=b_ids,
        individual_values=individual_values,
        individual_defined=individual_defined,
        paired_values=paired_values,
        paired_defined=paired_defined,
        audit_ids=np.asarray(BOOTSTRAP_AUDIT_IDS, dtype=np.int16),
        audit_indices=np.tile(
            np.arange(1, 2001, dtype=np.uint16), (len(BOOTSTRAP_AUDIT_IDS), 1)
        ),
    )


def test_exactly_18_condition_manifests_and_distribution_shas(tmp_path):
    for condition in range(1, 19):
        result = _result(condition)
        distribution, manifest_path, manifest = write_condition_result(
            tmp_path,
            condition_code=CONDITIONS[condition].Code,
            result=result,
            block_size=2,
            identity_hashes=IDENTITY,
        )
        assert manifest["DistributionSHA256"] == sha256_file(distribution)
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    directories = [path for path in (tmp_path / "bootstrap_conditions").iterdir() if path.is_dir()]
    assert len(directories) == 18
    assert all((path / "distributions.npz").is_file() for path in directories)
    assert all((path / "manifest.json").is_file() for path in directories)


def test_valid_resume_skips_and_invalid_output_forces_whole_condition_recompute(tmp_path):
    calls = []

    def compute(statistics, **kwargs):
        calls.append(kwargs["condition_ordinal"])
        return _result(
            kwargs["condition_ordinal"],
            bootstrap_ids=tuple(kwargs["bootstrap_ids"]),
            audit_ids=tuple(kwargs["audit_ids"]),
        )

    statistics = np.zeros((5, len(STATISTIC_FIELDS), 2000), dtype=np.int64)
    options = dict(
        condition_ordinal=1,
        condition_code="H100",
        statistics=statistics,
        bootstrap_ids=(1, 2, 3, 4),
        audit_ids=(1,),
        block_size=2,
        identity_hashes=IDENTITY,
        compute=compute,
    )
    first, resumed, _ = run_or_resume_condition(tmp_path, **options)
    assert not resumed and calls == [1]
    second, resumed, _ = run_or_resume_condition(tmp_path, **options)
    assert resumed and calls == [1]
    assert np.array_equal(first.individual_values, second.individual_values)
    directory = condition_directory(tmp_path, 1, "H100")
    (directory / "distributions.npz").write_bytes(b"corrupt whole condition")
    third, resumed, reason = run_or_resume_condition(tmp_path, **options)
    assert not resumed and calls == [1, 1]
    assert reason.startswith("INVALID_CONDITION_OUTPUT")
    validation = load_valid_condition(
        tmp_path,
        condition_ordinal=1,
        condition_code="H100",
        bootstrap_ids=(1, 2, 3, 4),
        audit_ids=(1,),
        block_size=2,
        identity_hashes=IDENTITY,
    )
    assert validation.valid
    assert np.array_equal(third.individual_values, validation.result.individual_values)


def test_combined_shapes_nan_masks_and_360_144_216_row_counts():
    results = [_official_result(condition) for condition in range(1, 19)]
    combined = combined_distribution_arrays(results)
    validate_combined_arrays(combined)
    assert combined["individual_values"].shape == EXPECTED_INDIVIDUAL_SHAPE
    assert combined["paired_values"].shape == EXPECTED_PAIRED_SHAPE
    assert EXPECTED_AUDIT_SHAPE == (18, 3, 2000)
    statistics = np.zeros((5, len(STATISTIC_FIELDS), 2000), dtype=np.int64)
    statistics[:, 0, :] = 100
    statistics[:, 1, :] = 50
    statistics[:, 2, :] = 100
    statistics[:, 3, :] = 25
    statistics[:, 4, :] = 50
    statistics[:, 5, :] = 100
    statistics[:, 6, :] = 10
    points = tuple(formal_point_estimates(statistics) for _ in range(19))
    individual_rows = build_individual_summary_rows(combined, points)
    paired_rows = build_paired_summary_rows(combined, points)
    precision_rows = build_stage8_precision_audit_rows(individual_rows, paired_rows)
    assert len(individual_rows) == 360
    assert len(paired_rows) == 144
    assert len(precision_rows) == 216
    assert {row["ContrastCode"] for row in paired_rows} == set(CONTRAST_CODES)
    assert {row["MethodCode"] for row in individual_rows} == set(METHOD_CODES)
    assert {row["Metric"] for row in individual_rows} == set(METRIC_CODES)


def test_nan_and_defined_mask_must_be_exact_complements():
    result = _result(1)
    arrays = {
        "condition_ordinal": np.asarray(1, dtype=np.int16),
        "bootstrap_ids": result.bootstrap_ids,
        "method_ordinals": np.arange(5, dtype=np.int8),
        "metric_codes": np.asarray(METRIC_CODES),
        "contrast_codes": np.asarray(CONTRAST_CODES),
        "individual_values": result.individual_values.copy(),
        "individual_defined": result.individual_defined.copy(),
        "paired_values": result.paired_values.copy(),
        "paired_defined": result.paired_defined.copy(),
        "audit_ids": result.audit_ids,
        "audit_indices": result.audit_indices,
    }
    arrays["individual_defined"][0, 0, 0] = False
    with pytest.raises(AssertionError, match="NaN/mask"):
        validate_condition_arrays(
            arrays,
            condition_ordinal=1,
            bootstrap_ids=(1, 2, 3, 4),
            audit_ids=(1,),
        )
