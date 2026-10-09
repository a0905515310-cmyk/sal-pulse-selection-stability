from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bootstrap import (
    ACCEPTANCE_LIMIT,
    BOOTSTRAP_AUDIT_IDS,
    BOOTSTRAP_B,
    BOOTSTRAP_BLOCK_SIZE,
    BOOTSTRAP_DRAW_COUNT,
    CI_UNDEFINED_REASON,
    COMPARATOR_METHOD_ORDINALS,
    CONTRAST_CODES,
    FORMAL_REPEAT_COUNT,
    K,
    METHOD_CODES,
    METRIC_CODES,
    STATISTIC_FIELDS,
    STATISTIC_INDEX,
    TW_METHOD_ORDINAL,
    PointEstimateResult,
    bootstrap_random_address,
    bootstrap_repeat_indices,
    formal_point_estimates,
    percentile_interval,
    validate_scalar_vector_and_golden_rng,
)
from .bootstrap_storage import (
    EXPECTED_AUDIT_SHAPE,
    EXPECTED_INDIVIDUAL_SHAPE,
    EXPECTED_PAIRED_SHAPE,
    canonical_json_bytes,
    load_valid_condition,
    run_or_resume_condition,
    sha256_array,
    sha256_file,
    validate_bootstrap_audit_arrays,
    validate_combined_arrays,
    validate_combined_distributions,
    write_bootstrap_audit_indices,
    write_combined_distributions,
    write_json_atomic,
    write_text_atomic,
)
from .build import ROOT
from .contracts import CONDITIONS, METHODS


STAGE_NUMBER = 10
STAGE_NAME = "BOOTSTRAP_B2000_PAIRED_REPEAT"
STAGE10_DIR = ROOT / "artifacts" / "stage10"

STAGE9_BASELINE_ZIP_NAME = "sal_stability_stage9_20260825.zip"
STAGE9_BASELINE_ZIP_SHA256 = (
    "df1f4611d968e7911527592e32a7a24101c711e4306069f5dcaafebd745f2af5"
)
STAGE8_BASELINE_ZIP_NAME = "sal_stability_stage8_20260825.zip"
STAGE8_BASELINE_ZIP_SHA256 = (
    "a440cae9bbef5474bf30d8abc32078646bf76219910a6a37e31863065e233a3b"
)
FORMAL_REPEAT_METRICS_SHA256 = (
    "e3084733adf9fa3f2325b0407b4358aeae118df8f6a152a4c1f5984757f3f8f6"
)
FORMAL_AGGREGATED_METRICS_SHA256 = (
    "803856c4d776b53b63e7e6f8be4425ce96ddb05c49aaca97c049134b04644865"
)
FORMAL_H300_WIDTH_STRATA_SHA256 = (
    "4e70f33ba195070a75e35fc6b3d14b7d5d7851b4ca414fcac7e8f24046a108b4"
)
STAGE10_PREFORMAL_CONTRACT_SHA256 = (
    "79fdae99716ea047d44171b332b2e2fc5bdeb311fa454b8092ba1f008c6b6b7a"
)
STAGE10_SCIENCE_CONTRACT_SHA256 = (
    "fdaa84d99e3b9e8fbcebe091c7a173a90ba5a42cef054f48c79eca01179656fd"
)
STAGE10_TASKBOOK_SHA256 = (
    "c36b4df1fc3781a83d84751e3333c01c5331e5a62b7c1ac8d687e8f49f1f8070"
)

FORMAL_REPEAT_METRICS_PATH = ROOT / "artifacts/stage9/formal_repeat_metrics.csv"
FORMAL_AGGREGATED_METRICS_PATH = ROOT / "artifacts/stage9/formal_aggregated_metrics.csv"
FORMAL_H300_WIDTH_STRATA_PATH = ROOT / "artifacts/stage9/formal_h300_width_strata.csv"
STAGE10_PREFORMAL_CONTRACT_PATH = (
    ROOT / "artifacts/stage9/stage10_bootstrap_contract_preformal.json"
)
STAGE9_STATUS_PATH = ROOT / "artifacts/stage9/status.json"
SCIENCE_CONTRACT_PATH = STAGE10_DIR / "stage10_science_contract.json"
INPUT_IDENTITY_PATH = STAGE10_DIR / "stage9_input_identity.json"
INDIVIDUAL_SUMMARY_PATH = STAGE10_DIR / "bootstrap_individual_summary.csv"
PAIRED_SUMMARY_PATH = STAGE10_DIR / "bootstrap_paired_summary.csv"
PRECISION_AUDIT_PATH = STAGE10_DIR / "stage8_precision_target_audit.csv"
COMBINED_DISTRIBUTIONS_PATH = STAGE10_DIR / "bootstrap_distributions.npz"
AUDIT_INDICES_PATH = STAGE10_DIR / "bootstrap_audit_indices.npz"
NI_REPORT_PATH = STAGE10_DIR / "ni_consistency_report.json"
BOOTSTRAP_MANIFEST_PATH = STAGE10_DIR / "bootstrap_manifest.json"
STATUS_PATH = STAGE10_DIR / "status.json"
VALIDATION_REPORT_PATH = STAGE10_DIR / "validation_report.md"
TEST_REPORT_PATH = STAGE10_DIR / "test_report.txt"
CHANGED_FILES_PATH = STAGE10_DIR / "changed_files.txt"

FINAL_ZIP_NAME = "sal_stability_stage10_20260825.zip"
FINAL_ZIP_ROOT = "sal_stability_stage10_20260825"

ALLOWED_STATIC_ADDITIONS = {
    "STAGE10_TASKBOOK.md",
    "scripts/build_stage10.py",
    "src/sal_stability_stage1/bootstrap.py",
    "src/sal_stability_stage1/bootstrap_storage.py",
    "src/sal_stability_stage1/stage10.py",
    "tests/test_stage10_contract.py",
    "tests/test_stage10_rng.py",
    "tests/test_stage10_metrics.py",
    "tests/test_stage10_storage.py",
    "tests/test_stage10_execution.py",
}
ALLOWED_MODIFIED_FILES = {"README.md", "pyproject.toml"}

LEGACY_STAGE8_NODE = (
    "tests/test_stage8_decision.py::"
    "test_stage1_through_stage7_byte_identity_and_frozen_science_gate"
)
LEGACY_STAGE9_NODE = (
    "tests/test_stage9_contract.py::"
    "test_stage8_identity_and_decision_boundary_are_independently_verified"
)
LEGACY_NODES = (LEGACY_STAGE8_NODE, LEGACY_STAGE9_NODE)

PROBABILITY_TARGET = 0.020
MEAN_L_NC_TARGET = 0.25
STAGE11_EXECUTED = False

INDIVIDUAL_SUMMARY_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrdinal",
    "MethodCode",
    "Metric",
    "FormalSumA",
    "FormalSumB",
    "FormalPositiveDenominatorRepeatCount",
    "PointEstimate",
    "PointDefined",
    "PointUndefinedReason",
    "BootstrapB",
    "DefinedBootstrapCount",
    "UndefinedBootstrapCount",
    "DefinedFraction",
    "CI95Defined",
    "CI95UndefinedReason",
    "CI95Lower",
    "CI95Upper",
    "CI95Width",
    "MaxOneSidedHalfWidth",
    "Stage8PrecisionContractApplicable",
    "Stage8TargetHalfWidth",
    "MeetsStage8PrecisionTarget",
)

PAIRED_SUMMARY_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "ContrastOrdinal",
    "ContrastCode",
    "ComparatorMethod",
    "Metric",
    "PointEstimateTW",
    "PointEstimateComparator",
    "DeltaPoint",
    "DeltaPointDefined",
    "DeltaPointUndefinedReason",
    "BootstrapB",
    "DefinedBootstrapCount",
    "UndefinedBootstrapCount",
    "DefinedFraction",
    "CI95Defined",
    "CI95UndefinedReason",
    "CI95Lower",
    "CI95Upper",
    "CI95Width",
    "MaxOneSidedHalfWidth",
    "Stage8PrecisionContractApplicable",
    "Stage8TargetHalfWidth",
    "MeetsStage8PrecisionTarget",
)

PRECISION_AUDIT_FIELDS = (
    "AuditOrdinal",
    "InferenceType",
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrContrast",
    "Metric",
    "PointEstimate",
    "CI95Defined",
    "CI95Lower",
    "CI95Upper",
    "ActualMaxOneSidedHalfWidth",
    "TargetHalfWidth",
    "MeetsStage8PrecisionTarget",
    "Stage8PrecisionContractApplicable",
    "FailureToMeetTargetFailsStage10",
)


def _verify_sha(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def verify_stage9_baseline_zip(path: Path | None = None) -> str:
    target = Path(path) if path is not None else ROOT.parent / STAGE9_BASELINE_ZIP_NAME
    return _verify_sha(target, STAGE9_BASELINE_ZIP_SHA256, "Stage 9 baseline ZIP")


def verify_stage8_regression_zip(path: Path | None = None) -> str:
    target = Path(path) if path is not None else ROOT.parent / STAGE8_BASELINE_ZIP_NAME
    return _verify_sha(target, STAGE8_BASELINE_ZIP_SHA256, "Stage 8 regression ZIP")


def validate_stage9_status(path: Path = STAGE9_STATUS_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "stage": 9,
        "stage_name": "FORMAL_R2000",
        "status": "PASS",
        "blocking_issue_count": 0,
        "formal_executed": True,
        "formal_r": FORMAL_REPEAT_COUNT,
        "formal_condition_count": 19,
        "formal_method_count": 5,
        "formal_repeat_metrics_row_count": 190000,
        "projected_precision_warning": True,
        "bootstrap_executed": False,
        "condition_dropped": False,
        "parameter_changed": False,
        "p_value_generated": False,
        "outcome_direction_used": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise AssertionError(
                f"Stage 9 status field {key} is not frozen: expected {expected!r}, got {payload.get(key)!r}"
            )
    for stage in range(1, 10):
        prior = json.loads(
            (ROOT / f"artifacts/stage{stage}/status.json").read_text(encoding="utf-8")
        )
        if prior.get("status") != "PASS":
            raise AssertionError(f"Stage {stage} status is not PASS")
    return payload


def validate_science_contract(path: Path = SCIENCE_CONTRACT_PATH) -> dict[str, object]:
    _verify_sha(path, STAGE10_SCIENCE_CONTRACT_SHA256, "Stage 10 science contract")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(path).read_bytes() != canonical_json_bytes(payload):
        raise AssertionError("Stage 10 science contract is not canonical sorted/indented JSON")
    hard_expectations: dict[str, object] = {
        "BootstrapB": BOOTSTRAP_B,
        "BootstrapDrawCountPerReplicate": BOOTSTRAP_DRAW_COUNT,
        "BootstrapNamespace": "BOOTSTRAP",
        "IndependentUnit": "Repeat",
        "MetricEstimator": "RATIO_OF_RESAMPLED_SUMS",
        "NIInference": "CONSISTENCY_ONLY_NO_BOOTSTRAP",
        "StopAfterStage10": True,
        "FormalRepeatMetricsSHA256": FORMAL_REPEAT_METRICS_SHA256,
        "Stage10PreFormalContractSHA256": STAGE10_PREFORMAL_CONTRACT_SHA256,
        "Stage9BaselineZipSHA256": STAGE9_BASELINE_ZIP_SHA256,
    }
    for key, expected in hard_expectations.items():
        if payload.get(key) != expected:
            raise AssertionError(f"Stage 10 science contract field {key} drifted")
    if payload.get("Methods") != list(METHOD_CODES):
        raise AssertionError("Stage 10 method identity drifted")
    if payload.get("InterferenceConditionOrdinals") != list(range(1, 19)):
        raise AssertionError("Stage 10 inference conditions drifted")
    if [item.get("Code") for item in payload.get("PairedContrasts", [])] != list(
        CONTRAST_CODES
    ):
        raise AssertionError("Stage 10 paired contrasts drifted")
    confidence = payload.get("ConfidenceInterval", {})
    if confidence != {
        "ConfidenceLevel": 0.95,
        "DoNotUseBootstrapMeanAsPointEstimate": True,
        "LowerQuantile": 0.025,
        "PointEstimateSource": "STAGE9_FORMAL_RATIO_OF_SUMS",
        "QuantileMethod": "linear",
        "Type": "PERCENTILE_BOOTSTRAP",
        "UpperQuantile": 0.975,
    }:
        raise AssertionError("Stage 10 percentile CI contract drifted")
    if payload["BootstrapRNG"]["AcceptanceLimit"] != ACCEPTANCE_LIMIT:
        raise AssertionError("Stage 10 rejection-sampling limit drifted")
    return payload


def validate_frozen_input_hashes() -> dict[str, str]:
    hashes = {
        "Stage9BaselineZipSHA256": verify_stage9_baseline_zip(),
        "Stage8RegressionZipSHA256": verify_stage8_regression_zip(),
        "FormalRepeatMetricsSHA256": _verify_sha(
            FORMAL_REPEAT_METRICS_PATH,
            FORMAL_REPEAT_METRICS_SHA256,
            "Stage 9 Formal RepeatMetrics",
        ),
        "FormalAggregatedMetricsSHA256": _verify_sha(
            FORMAL_AGGREGATED_METRICS_PATH,
            FORMAL_AGGREGATED_METRICS_SHA256,
            "Stage 9 Formal AggregatedMetrics",
        ),
        "FormalH300WidthStrataSHA256": _verify_sha(
            FORMAL_H300_WIDTH_STRATA_PATH,
            FORMAL_H300_WIDTH_STRATA_SHA256,
            "Stage 9 Formal H300 width strata",
        ),
        "Stage10PreFormalContractSHA256": _verify_sha(
            STAGE10_PREFORMAL_CONTRACT_PATH,
            STAGE10_PREFORMAL_CONTRACT_SHA256,
            "Stage 10 preformal contract",
        ),
        "Stage10ScienceContractSHA256": _verify_sha(
            SCIENCE_CONTRACT_PATH,
            STAGE10_SCIENCE_CONTRACT_SHA256,
            "Stage 10 science contract",
        ),
        "Stage10TaskbookSHA256": _verify_sha(
            ROOT / "STAGE10_TASKBOOK.md",
            STAGE10_TASKBOOK_SHA256,
            "Stage 10 taskbook",
        ),
    }
    return hashes


def _excluded_path(path: Path) -> bool:
    return any(
        part in {"__pycache__", ".pytest_cache", ".git", ".venv"}
        or part.startswith(".venv-")
        for part in path.parts
    ) or path.suffix.lower() == ".pyc" or path.name.endswith(".tmp") or ".tmp-" in path.name


def _baseline_archive_members(path: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    roots: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                raise AssertionError("Stage 9 baseline ZIP contains a root-level file")
            roots.add(parts[0])
            relative = Path(*parts[1:])
            if _excluded_path(relative):
                continue
            key = relative.as_posix()
            if key in members:
                raise AssertionError(f"duplicate Stage 9 archive member: {key}")
            members[key] = hashlib.sha256(archive.read(info)).hexdigest()
    if len(roots) != 1:
        raise AssertionError("Stage 9 baseline ZIP must contain exactly one project root")
    return members


def _tree_fingerprint(members: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(members.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b":")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _allowed_stage10_addition(relative: str) -> bool:
    return relative in ALLOWED_STATIC_ADDITIONS or relative.startswith("artifacts/stage10/")


def validate_stage1_through_stage9_tree_unchanged(
    baseline_zip: Path | None = None,
) -> dict[str, object]:
    baseline_path = (
        Path(baseline_zip)
        if baseline_zip is not None
        else ROOT.parent / STAGE9_BASELINE_ZIP_NAME
    )
    verify_stage9_baseline_zip(baseline_path)
    baseline = _baseline_archive_members(baseline_path)
    current: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if _excluded_path(relative):
            continue
        current[relative.as_posix()] = sha256_file(path)
    added = sorted(set(current) - set(baseline))
    modified = sorted(
        relative
        for relative in set(current).intersection(baseline)
        if current[relative] != baseline[relative]
    )
    deleted = sorted(set(baseline) - set(current))
    unexpected_added = [value for value in added if not _allowed_stage10_addition(value)]
    unexpected_modified = [
        value for value in modified if value not in ALLOWED_MODIFIED_FILES
    ]
    if unexpected_added or unexpected_modified or deleted:
        raise AssertionError(
            "Stage 1-9 byte-identity boundary failed: "
            f"unexpected_added={unexpected_added}, "
            f"unexpected_modified={unexpected_modified}, deleted={deleted}"
        )
    return {
        "BaselineMemberCount": len(baseline),
        "BaselineTreeSHA256": _tree_fingerprint(baseline),
        "UnchangedBaselineMemberCount": len(baseline) - len(modified),
        "AllowedModifiedFiles": modified,
        "AddedFiles": added,
        "AddedFileCount": len(added),
        "DeletedFiles": deleted,
        "UnexpectedAddedFiles": unexpected_added,
        "UnexpectedModifiedFiles": unexpected_modified,
        "Passed": True,
    }


def validate_required_static_files() -> None:
    missing = [relative for relative in sorted(ALLOWED_STATIC_ADDITIONS) if not (ROOT / relative).is_file()]
    if missing:
        raise AssertionError(f"Stage 10 required static files are missing: {missing}")


def stage10_code_sha256() -> str:
    files = sorted(ALLOWED_STATIC_ADDITIONS | ALLOWED_MODIFIED_FILES)
    digest = hashlib.sha256()
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Stage 10 code identity file is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b":")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FormalBootstrapData:
    statistics: np.ndarray
    row_count: int
    condition_codes: tuple[str, ...]
    method_codes: tuple[str, ...]


def load_formal_repeat_metrics(
    path: Path = FORMAL_REPEAT_METRICS_PATH,
) -> FormalBootstrapData:
    _verify_sha(path, FORMAL_REPEAT_METRICS_SHA256, "Stage 9 Formal RepeatMetrics")
    shape = (len(CONDITIONS), len(METHODS), len(STATISTIC_FIELDS), FORMAL_REPEAT_COUNT)
    statistics = np.full(shape, -1, dtype=np.int64)
    filled = np.zeros((len(CONDITIONS), len(METHODS), FORMAL_REPEAT_COUNT), dtype=bool)
    expected_condition_codes = tuple(item.Code for item in CONDITIONS)
    expected_method_codes = tuple(item.Code for item in METHODS)
    row_count = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "ConditionOrdinal",
            "ConditionCode",
            "MethodOrdinal",
            "MethodCode",
            "RepeatID",
            *STATISTIC_FIELDS,
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AssertionError("Formal RepeatMetrics schema lacks a required Bootstrap field")
        for row in reader:
            row_count += 1
            condition = int(row["ConditionOrdinal"])
            method = int(row["MethodOrdinal"])
            repeat_id = int(row["RepeatID"])
            if not 0 <= condition < len(CONDITIONS):
                raise AssertionError("Formal RepeatMetrics contains an invalid ConditionOrdinal")
            if not 0 <= method < len(METHODS):
                raise AssertionError("Formal RepeatMetrics contains an invalid MethodOrdinal")
            if not 1 <= repeat_id <= FORMAL_REPEAT_COUNT:
                raise AssertionError("Formal RepeatMetrics contains an invalid RepeatID")
            if row["ConditionCode"] != expected_condition_codes[condition]:
                raise AssertionError("Formal RepeatMetrics condition identity drifted")
            if row["MethodCode"] != expected_method_codes[method]:
                raise AssertionError("Formal RepeatMetrics method identity drifted")
            target = (condition, method, repeat_id - 1)
            if filled[target]:
                raise AssertionError(f"duplicate Formal RepeatMetrics key: {target}")
            for field_index, field_name in enumerate(STATISTIC_FIELDS):
                value = int(row[field_name])
                if value < 0:
                    raise AssertionError(f"Formal RepeatMetrics {field_name} is negative")
                statistics[condition, method, field_index, repeat_id - 1] = value
            filled[target] = True
    if row_count != len(CONDITIONS) * len(METHODS) * FORMAL_REPEAT_COUNT:
        raise AssertionError("Formal RepeatMetrics row count is not 190000")
    if not np.all(filled) or np.any(statistics < 0):
        raise AssertionError("Formal RepeatMetrics key universe is incomplete")
    statistics.setflags(write=False)
    return FormalBootstrapData(
        statistics=statistics,
        row_count=row_count,
        condition_codes=expected_condition_codes,
        method_codes=expected_method_codes,
    )


def validate_ni_consistency(
    path: Path = FORMAL_REPEAT_METRICS_PATH,
) -> dict[str, object]:
    reference_by_repeat: dict[int, tuple[tuple[str, str], ...]] = {}
    seen: set[tuple[int, int]] = set()
    payload_fields: tuple[str, ...] | None = None
    row_count = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssertionError("Formal RepeatMetrics has no header")
        payload_fields = tuple(
            name
            for name in reader.fieldnames
            if name
            not in {"ConditionOrdinal", "ConditionCode", "MethodOrdinal", "MethodCode"}
        )
        for row in reader:
            condition = int(row["ConditionOrdinal"])
            if condition != 0:
                continue
            row_count += 1
            method = int(row["MethodOrdinal"])
            repeat_id = int(row["RepeatID"])
            key = (method, repeat_id)
            if key in seen:
                raise AssertionError("NI contains a duplicate Method/Repeat key")
            seen.add(key)
            if row["ConditionCode"] != "NI" or row["MethodCode"] != METHOD_CODES[method]:
                raise AssertionError("NI method or condition identity drifted")
            if (int(row["N_C"]), int(row["N_E"]), int(row["N_N"])) != (200, 0, 0):
                raise AssertionError("NI is not exactly all-C")
            payload = tuple((name, row[name]) for name in payload_fields)
            if method == 0:
                reference_by_repeat[repeat_id] = payload
            elif reference_by_repeat.get(repeat_id) != payload:
                raise AssertionError(
                    f"NI five-method RepeatMetrics identity failed at RepeatID={repeat_id}"
                )
    expected_keys = {
        (method, repeat_id)
        for method in range(len(METHOD_CODES))
        for repeat_id in range(1, FORMAL_REPEAT_COUNT + 1)
    }
    if seen != expected_keys or row_count != len(expected_keys):
        raise AssertionError("NI does not contain the exact 5 x 2000 key universe")
    if len(reference_by_repeat) != FORMAL_REPEAT_COUNT:
        raise AssertionError("NI reference Repeat universe is incomplete")
    return {
        "Stage": 10,
        "ConditionOrdinal": 0,
        "ConditionCode": "NI",
        "InferentialBootstrapExecuted": False,
        "FormalMethodCount": len(METHOD_CODES),
        "FormalRepeatCount": FORMAL_REPEAT_COUNT,
        "FormalRepeatMetricsRowCount": row_count,
        "ExactMethodRepeatKeyUniversePassed": True,
        "AllCyclesCorrectPassed": True,
        "FiveMethodRepeatMetricsIdentityPassed": True,
        "Status": "PASS",
    }


def _parse_csv_bool(value: str, field_name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AssertionError(f"{field_name} is not a canonical CSV boolean: {value!r}")


def reconstitute_formal_point_estimates(
    formal: FormalBootstrapData,
    aggregate_path: Path = FORMAL_AGGREGATED_METRICS_PATH,
) -> tuple[tuple[PointEstimateResult, ...], dict[str, object]]:
    _verify_sha(
        aggregate_path,
        FORMAL_AGGREGATED_METRICS_SHA256,
        "Stage 9 Formal AggregatedMetrics",
    )
    point_results = tuple(
        formal_point_estimates(formal.statistics[condition])
        for condition in range(len(CONDITIONS))
    )
    metric_value_fields = (
        "P_cor_value",
        "P_C_given_C_value",
        "P_C_given_E_value",
        "Mean_L_NC_value",
    )
    numerator_total_fields = (
        "Total_N_C",
        "Total_N_CC",
        "Total_N_EC",
        "Total_Sum_L_NC_obs",
    )
    denominator_total_fields = (
        None,
        "Total_N_Cdot",
        "Total_N_Edot",
        "Total_N_run_NC",
    )
    seen: set[tuple[int, int]] = set()
    scalar_check_count = 0
    maximum_absolute_difference = 0.0
    with Path(aggregate_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = int(row["ConditionOrdinal"])
            method = int(row["MethodOrdinal"])
            key = (condition, method)
            if key in seen:
                raise AssertionError("Formal AggregatedMetrics contains a duplicate key")
            seen.add(key)
            if row["ConditionCode"] != formal.condition_codes[condition]:
                raise AssertionError("Formal aggregate condition identity drifted")
            if row["MethodCode"] != formal.method_codes[method]:
                raise AssertionError("Formal aggregate method identity drifted")
            if int(row["R"]) != FORMAL_REPEAT_COUNT or int(row["K"]) != K:
                raise AssertionError("Formal aggregate R/K identity drifted")
            points = point_results[condition]
            for metric_index, value_field in enumerate(metric_value_fields):
                prefix = value_field[: -len("_value")]
                aggregate_defined = _parse_csv_bool(
                    row[f"{prefix}_defined"], f"{prefix}_defined"
                )
                if aggregate_defined != bool(points.defined[method, metric_index]):
                    raise AssertionError("Formal point defined state differs from aggregate")
                expected_reason = str(points.undefined_reason[method, metric_index])
                if row[f"{prefix}_undefined_reason"] != expected_reason:
                    raise AssertionError("Formal point undefined reason differs from aggregate")
                if int(row[numerator_total_fields[metric_index]]) != int(
                    points.sum_a[method, metric_index]
                ):
                    raise AssertionError("Formal point numerator sum differs from aggregate")
                denominator_field = denominator_total_fields[metric_index]
                if denominator_field is not None and int(row[denominator_field]) != int(
                    points.sum_b[method, metric_index]
                ):
                    raise AssertionError("Formal point denominator sum differs from aggregate")
                if aggregate_defined:
                    aggregate_value = float(row[value_field])
                    difference = abs(
                        aggregate_value - float(points.values[method, metric_index])
                    )
                    maximum_absolute_difference = max(maximum_absolute_difference, difference)
                    if not math.isclose(
                        aggregate_value,
                        float(points.values[method, metric_index]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        raise AssertionError(
                            "Formal point estimate differs scientifically from Stage 9 aggregate"
                        )
                elif row[value_field] != "":
                    raise AssertionError("undefined Formal aggregate value is not blank")
                scalar_check_count += 1
    expected_keys = {
        (condition, method)
        for condition in range(len(CONDITIONS))
        for method in range(len(METHODS))
    }
    if seen != expected_keys:
        raise AssertionError("Formal AggregatedMetrics key universe is not 19 x 5")
    return point_results, {
        "AggregateRowCount": len(seen),
        "CoreMetricScalarCheckCount": scalar_check_count,
        "MaximumAbsoluteDifference": maximum_absolute_difference,
        "IEEE754CSVSerializationTolerance": 1e-15,
        "Passed": True,
    }


def _target_half_width(metric: str) -> float:
    return MEAN_L_NC_TARGET if metric == "Mean_L_NC" else PROBABILITY_TARGET


def _ci_summary(
    values: np.ndarray,
    defined: np.ndarray,
    point_estimate: float,
    point_defined: bool,
) -> dict[str, object]:
    defined_count = int(np.count_nonzero(defined))
    ci_defined, reason, lower, upper = percentile_interval(values, defined)
    if ci_defined:
        width = upper - lower
        max_half_width = (
            max(point_estimate - lower, upper - point_estimate)
            if point_defined
            else np.nan
        )
    else:
        width = np.nan
        max_half_width = np.nan
    return {
        "DefinedBootstrapCount": defined_count,
        "UndefinedBootstrapCount": BOOTSTRAP_B - defined_count,
        "DefinedFraction": defined_count / BOOTSTRAP_B,
        "CI95Defined": ci_defined,
        "CI95UndefinedReason": reason,
        "CI95Lower": lower,
        "CI95Upper": upper,
        "CI95Width": width,
        "MaxOneSidedHalfWidth": max_half_width,
    }


def build_individual_summary_rows(
    combined: Mapping[str, np.ndarray],
    point_results: Sequence[PointEstimateResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in range(1, 19):
        points = point_results[condition]
        for method, method_code in enumerate(METHOD_CODES):
            for metric_index, metric in enumerate(METRIC_CODES):
                point_defined = bool(points.defined[method, metric_index])
                point = float(points.values[method, metric_index])
                ci = _ci_summary(
                    combined["individual_values"][condition - 1, method, metric_index],
                    combined["individual_defined"][condition - 1, method, metric_index],
                    point,
                    point_defined,
                )
                applicable = method_code in {"T", "TW"}
                target = _target_half_width(metric) if applicable else None
                meets = (
                    bool(ci["MaxOneSidedHalfWidth"] <= target)
                    if applicable and ci["CI95Defined"] and point_defined
                    else None
                )
                rows.append(
                    {
                        "ConditionOrdinal": condition,
                        "ConditionCode": CONDITIONS[condition].Code,
                        "MethodOrdinal": method,
                        "MethodCode": method_code,
                        "Metric": metric,
                        "FormalSumA": int(points.sum_a[method, metric_index]),
                        "FormalSumB": int(points.sum_b[method, metric_index]),
                        "FormalPositiveDenominatorRepeatCount": int(
                            points.positive_denominator_repeat_count[method, metric_index]
                        ),
                        "PointEstimate": point,
                        "PointDefined": point_defined,
                        "PointUndefinedReason": str(
                            points.undefined_reason[method, metric_index]
                        ),
                        "BootstrapB": BOOTSTRAP_B,
                        **ci,
                        "Stage8PrecisionContractApplicable": applicable,
                        "Stage8TargetHalfWidth": target,
                        "MeetsStage8PrecisionTarget": meets,
                    }
                )
    keys = {
        (row["ConditionOrdinal"], row["MethodOrdinal"], row["Metric"]) for row in rows
    }
    if len(rows) != 360 or len(keys) != 360:
        raise AssertionError("individual summary is not exactly 360 unique rows")
    return rows


def build_paired_summary_rows(
    combined: Mapping[str, np.ndarray],
    point_results: Sequence[PointEstimateResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in range(1, 19):
        points = point_results[condition]
        for contrast, comparator in enumerate(COMPARATOR_METHOD_ORDINALS):
            for metric_index, metric in enumerate(METRIC_CODES):
                tw_defined = bool(points.defined[TW_METHOD_ORDINAL, metric_index])
                comparator_defined = bool(points.defined[comparator, metric_index])
                delta_defined = tw_defined and comparator_defined
                tw_point = float(points.values[TW_METHOD_ORDINAL, metric_index])
                comparator_point = float(points.values[comparator, metric_index])
                delta = tw_point - comparator_point if delta_defined else np.nan
                ci = _ci_summary(
                    combined["paired_values"][condition - 1, contrast, metric_index],
                    combined["paired_defined"][condition - 1, contrast, metric_index],
                    delta,
                    delta_defined,
                )
                applicable = contrast == 0
                target = _target_half_width(metric) if applicable else None
                meets = (
                    bool(ci["MaxOneSidedHalfWidth"] <= target)
                    if applicable and ci["CI95Defined"] and delta_defined
                    else None
                )
                rows.append(
                    {
                        "ConditionOrdinal": condition,
                        "ConditionCode": CONDITIONS[condition].Code,
                        "ContrastOrdinal": contrast,
                        "ContrastCode": CONTRAST_CODES[contrast],
                        "ComparatorMethod": METHOD_CODES[comparator],
                        "Metric": metric,
                        "PointEstimateTW": tw_point,
                        "PointEstimateComparator": comparator_point,
                        "DeltaPoint": delta,
                        "DeltaPointDefined": delta_defined,
                        "DeltaPointUndefinedReason": (
                            ""
                            if delta_defined
                            else "TW_OR_COMPARATOR_FORMAL_POINT_UNDEFINED"
                        ),
                        "BootstrapB": BOOTSTRAP_B,
                        **ci,
                        "Stage8PrecisionContractApplicable": applicable,
                        "Stage8TargetHalfWidth": target,
                        "MeetsStage8PrecisionTarget": meets,
                    }
                )
    keys = {
        (row["ConditionOrdinal"], row["ContrastOrdinal"], row["Metric"])
        for row in rows
    }
    if len(rows) != 144 or len(keys) != 144:
        raise AssertionError("paired summary is not exactly 144 unique rows")
    return rows


def build_stage8_precision_audit_rows(
    individual_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in individual_rows:
        if not source["Stage8PrecisionContractApplicable"]:
            continue
        rows.append(
            {
                "AuditOrdinal": len(rows) + 1,
                "InferenceType": "INDIVIDUAL",
                "ConditionOrdinal": source["ConditionOrdinal"],
                "ConditionCode": source["ConditionCode"],
                "MethodOrContrast": source["MethodCode"],
                "Metric": source["Metric"],
                "PointEstimate": source["PointEstimate"],
                "CI95Defined": source["CI95Defined"],
                "CI95Lower": source["CI95Lower"],
                "CI95Upper": source["CI95Upper"],
                "ActualMaxOneSidedHalfWidth": source["MaxOneSidedHalfWidth"],
                "TargetHalfWidth": source["Stage8TargetHalfWidth"],
                "MeetsStage8PrecisionTarget": source["MeetsStage8PrecisionTarget"],
                "Stage8PrecisionContractApplicable": True,
                "FailureToMeetTargetFailsStage10": False,
            }
        )
    for source in paired_rows:
        if not source["Stage8PrecisionContractApplicable"]:
            continue
        rows.append(
            {
                "AuditOrdinal": len(rows) + 1,
                "InferenceType": "PAIRED",
                "ConditionOrdinal": source["ConditionOrdinal"],
                "ConditionCode": source["ConditionCode"],
                "MethodOrContrast": source["ContrastCode"],
                "Metric": source["Metric"],
                "PointEstimate": source["DeltaPoint"],
                "CI95Defined": source["CI95Defined"],
                "CI95Lower": source["CI95Lower"],
                "CI95Upper": source["CI95Upper"],
                "ActualMaxOneSidedHalfWidth": source["MaxOneSidedHalfWidth"],
                "TargetHalfWidth": source["Stage8TargetHalfWidth"],
                "MeetsStage8PrecisionTarget": source["MeetsStage8PrecisionTarget"],
                "Stage8PrecisionContractApplicable": True,
                "FailureToMeetTargetFailsStage10": False,
            }
        )
    keys = {
        (
            row["InferenceType"],
            row["ConditionOrdinal"],
            row["MethodOrContrast"],
            row["Metric"],
        )
        for row in rows
    }
    if len(rows) != 216 or len(keys) != 216:
        raise AssertionError("Stage 8 precision target audit is not exactly 216 unique rows")
    return rows


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_csv_atomic(
    path: Path,
    field_names: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(field_names),
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key)) for key in field_names})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def csv_row_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def assert_no_forbidden_execution_path() -> dict[str, object]:
    address_parameters = tuple(inspect.signature(bootstrap_random_address).parameters)
    if "method" in " ".join(address_parameters).lower():
        raise AssertionError("Method entered the Bootstrap RandomAddress interface")
    source_paths = (
        ROOT / "src/sal_stability_stage1/bootstrap.py",
        ROOT / "src/sal_stability_stage1/bootstrap_storage.py",
        ROOT / "src/sal_stability_stage1/stage10.py",
    )
    forbidden_calls = {
        "run_condition_repeats_serial",
        "run_condition_repeats_parallel",
        "generate_physical_events",
        "run_formal_condition_chunked",
        "build_stage9",
        "build_stage11",
        "default_rng",
        "choice",
    }
    offending: list[str] = []
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if isinstance(function, ast.Name):
                name = function.id
            elif isinstance(function, ast.Attribute):
                parts = [function.attr]
                parent = function.value
                while isinstance(parent, ast.Attribute):
                    parts.append(parent.attr)
                    parent = parent.value
                if isinstance(parent, ast.Name):
                    parts.append(parent.id)
                name = ".".join(reversed(parts))
            else:
                continue
            if name in forbidden_calls or name.split(".")[-1] in forbidden_calls:
                offending.append(f"{path.name}:{node.lineno}:{name}")
    if offending:
        raise AssertionError(f"forbidden execution/RNG path appears in Stage 10 code: {offending}")
    return {
        "PhysicalSimulationCallAbsent": True,
        "FormalRerunCallAbsent": True,
        "Stage11CallAbsent": True,
        "NumpyRandomCallAbsent": True,
        "MethodAbsentFromRandomAddress": True,
    }


def _pytest_result(command: Sequence[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")

    def latest(pattern: str) -> int:
        matches = re.findall(pattern, output)
        return int(matches[-1]) if matches else 0

    failed_nodes = sorted(
        {
            value.replace("\\", "/")
            for value in re.findall(r"^FAILED\s+(\S+)", output, re.MULTILINE)
        }
    )
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "passed": latest(r"([0-9]+) passed"),
        "failed": latest(r"([0-9]+) failed"),
        "errors": latest(r"([0-9]+) errors?"),
        "failed_nodes": failed_nodes,
        "wall_seconds": elapsed,
        "output": output,
    }


def _append_raw_test_snapshot(label: str, payload: Mapping[str, object]) -> None:
    existing = TEST_REPORT_PATH.read_text(encoding="utf-8") if TEST_REPORT_PATH.exists() else ""
    section = (
        f"===== {label} =====\n"
        f"COMMAND: {payload['command']}\n"
        f"RETURN CODE: {payload['returncode']}\n"
        f"PARSED: {payload['passed']} passed, {payload['failed']} failed, {payload['errors']} errors\n"
        f"WALL SECONDS: {payload['wall_seconds']:.6f}\n\n"
        f"{payload['output']}\n"
    )
    write_text_atomic(TEST_REPORT_PATH, existing + section)


def _assert_additions_only_legacy_failure(message: str, label: str) -> None:
    if (
        "unexpected_added=" not in message
        or "unexpected_modified=[]" not in message
        or "deleted=[]" not in message
    ):
        raise AssertionError(f"{label} was not an additions-only boundary failure: {message}")


def run_full_pytest_with_legacy_rules(label: str) -> dict[str, object]:
    full = _pytest_result((sys.executable, "-m", "pytest"))
    _append_raw_test_snapshot(f"{label} FULL PYTEST RAW", full)
    expected_nodes = set(LEGACY_NODES)
    if (
        full["returncode"] == 0
        or full["failed"] != 2
        or full["errors"] != 0
        or set(full["failed_nodes"]) != expected_nodes
    ):
        raise AssertionError(
            "complete pytest did not match exactly the two authorized legacy failures: "
            f"returncode={full['returncode']}, failed={full['failed']}, errors={full['errors']}, "
            f"failed_nodes={full['failed_nodes']}\n{full['output']}"
        )

    from .stage8 import validate_stage1_through_stage7_tree_unchanged
    from .stage9 import validate_stage1_through_stage8_tree_unchanged

    try:
        validate_stage1_through_stage7_tree_unchanged()
    except AssertionError as exc:
        stage8_message = str(exc)
    else:
        raise AssertionError("authorized Stage 8 legacy boundary unexpectedly passed")
    _assert_additions_only_legacy_failure(stage8_message, "Stage 8 legacy boundary")
    try:
        validate_stage1_through_stage8_tree_unchanged()
    except AssertionError as exc:
        stage9_message = str(exc)
    else:
        raise AssertionError("authorized Stage 9 legacy boundary unexpectedly passed")
    _assert_additions_only_legacy_failure(stage9_message, "Stage 9 legacy boundary")

    tree = validate_stage1_through_stage9_tree_unchanged()
    verify_stage8_regression_zip()
    verify_stage9_baseline_zip()
    validate_stage9_status()

    command = [sys.executable, "-m", "pytest"]
    for node in LEGACY_NODES:
        command.extend(("--deselect", node))
    remaining = _pytest_result(command)
    _append_raw_test_snapshot(f"{label} REMAINING PYTEST RAW", remaining)
    if (
        remaining["returncode"] != 0
        or remaining["passed"] <= 0
        or remaining["failed"] != 0
        or remaining["errors"] != 0
    ):
        raise AssertionError(
            "remaining pytest failed outside the two authorized nodes\n"
            f"{remaining['output']}"
        )
    exceptions = [
        {
            "Code": "LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE",
            "TestNode": LEGACY_STAGE8_NODE,
            "Allowed": True,
            "FailureDifferenceScope": "taskbook-authorized Stage9 and Stage10 additions only",
            "FrozenFilesModified": False,
            "DeletedFiles": [],
        },
        {
            "Code": "LEGACY_STAGE9_BOUNDARY_NOT_FORWARD_COMPATIBLE",
            "TestNode": LEGACY_STAGE9_NODE,
            "Allowed": True,
            "FailureDifferenceScope": "taskbook-authorized Stage10 additions only",
            "FrozenFilesModified": False,
            "DeletedFiles": [],
        },
    ]
    return {
        "label": label,
        "complete_pytest": full,
        "remaining_pytest": remaining,
        "legacy_boundary_exceptions": exceptions,
        "stage1_through_stage9_tree": tree,
    }


def run_stage10_unit_tests(label: str) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_stage10_contract.py",
        "tests/test_stage10_rng.py",
        "tests/test_stage10_metrics.py",
        "tests/test_stage10_storage.py",
        "tests/test_stage10_execution.py",
    ]
    result = _pytest_result(command)
    _append_raw_test_snapshot(f"{label} STAGE10 TESTS RAW", result)
    if (
        result["returncode"] != 0
        or result["passed"] <= 0
        or result["failed"] != 0
        or result["errors"] != 0
    ):
        raise AssertionError(f"Stage 10 tests failed\n{result['output']}")
    return result


def independently_regenerate_audit_indices(
    primary_results: Sequence[object],
) -> tuple[np.ndarray, dict[str, object]]:
    ordered = sorted(primary_results, key=lambda item: item.condition_ordinal)
    primary = np.stack([item.audit_indices for item in ordered], axis=0).astype(
        np.uint16, copy=False
    )
    rerun = np.stack(
        [
            bootstrap_repeat_indices(condition, BOOTSTRAP_AUDIT_IDS)
            for condition in range(1, 19)
        ],
        axis=0,
    ).astype(np.uint16, copy=False)
    if primary.shape != EXPECTED_AUDIT_SHAPE or rerun.shape != EXPECTED_AUDIT_SHAPE:
        raise AssertionError("Bootstrap audit primary/rerun shape is not 18 x 3 x 2000")
    if not np.array_equal(primary, rerun):
        mismatch = np.argwhere(primary != rerun)[0]
        raise AssertionError(
            "Bootstrap audit primary/rerun mismatch at " + str(tuple(int(x) for x in mismatch))
        )
    primary_sha = sha256_array(primary)
    rerun_sha = sha256_array(rerun)
    if primary_sha != rerun_sha:
        raise AssertionError("Bootstrap audit primary/rerun SHA differs")
    return primary, {
        "BootstrapAuditIDs": list(BOOTSTRAP_AUDIT_IDS),
        "BootstrapAuditIndexCount": int(primary.size),
        "PrimaryArraySHA256": primary_sha,
        "IndependentRerunArraySHA256": rerun_sha,
        "ExactReproducibilityPassed": True,
    }


def validate_summary_and_distribution_hard_gates(
    *,
    combined: Mapping[str, np.ndarray],
    audit_arrays: Mapping[str, np.ndarray],
    individual_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
    precision_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validate_combined_arrays(combined)
    validate_bootstrap_audit_arrays(audit_arrays)
    if len(individual_rows) != 360 or len(paired_rows) != 144 or len(precision_rows) != 216:
        raise AssertionError("Stage 10 summary row-count gate failed")
    if any(row["ContrastCode"] not in CONTRAST_CODES for row in paired_rows):
        raise AssertionError("an unregistered paired contrast entered Stage 10")
    for row in paired_rows:
        expected_delta = row["PointEstimateTW"] - row["PointEstimateComparator"]
        if row["DeltaPointDefined"] and row["DeltaPoint"] != expected_delta:
            raise AssertionError("DeltaPoint is not exactly TW point minus comparator point")
    if any(
        row["MethodOrContrast"] not in {"T", "TW", "TW-T"}
        for row in precision_rows
    ):
        raise AssertionError("Stage 8 precision audit scope expanded beyond T/TW/TW-T")
    if any(row["FailureToMeetTargetFailsStage10"] for row in precision_rows):
        raise AssertionError("precision target outcome improperly entered Stage 10 PASS")
    if (ROOT / "artifacts/stage11").exists():
        raise AssertionError("Stage 11 artifact tree exists")
    return {
        "IndividualDistributionShape": list(combined["individual_values"].shape),
        "PairedDistributionShape": list(combined["paired_values"].shape),
        "AuditIndexShape": list(audit_arrays["repeat_ids"].shape),
        "IndividualSummaryRows": len(individual_rows),
        "PairedSummaryRows": len(paired_rows),
        "Stage8PrecisionAuditRows": len(precision_rows),
        "NaNDefinedMaskIdentityPassed": True,
        "ProbabilityBoundsPassed": True,
        "MeanLNCNonnegativePassed": True,
        "DeltaDirectionPassed": True,
        "ContrastScopePassed": True,
        "Stage11Absent": True,
    }


def _compact_pytest(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "Command": payload["command"],
        "ReturnCode": payload["returncode"],
        "Passed": payload["passed"],
        "Failed": payload["failed"],
        "Errors": payload["errors"],
        "FailedNodes": payload["failed_nodes"],
        "WallSeconds": payload["wall_seconds"],
    }


def _write_final_test_report(
    pre_regression: Mapping[str, object],
    pre_unit: Mapping[str, object],
    post_regression: Mapping[str, object],
    post_unit: Mapping[str, object],
) -> None:
    sections: list[str] = ["STAGE 10 PYTEST AND LEGACY EXCEPTION REPORT\n"]
    for label, report in (("PRE-BOOTSTRAP", pre_regression), ("POST-BOOTSTRAP", post_regression)):
        sections.append(f"===== {label} FULL PYTEST =====\n")
        sections.append(json.dumps(_compact_pytest(report["complete_pytest"]), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        sections.append(str(report["complete_pytest"]["output"]) + "\n")
        sections.append(f"===== {label} AUTHORIZED LEGACY EXCEPTIONS =====\n")
        sections.append(json.dumps(report["legacy_boundary_exceptions"], ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        sections.append(f"===== {label} REMAINING PYTEST =====\n")
        sections.append(json.dumps(_compact_pytest(report["remaining_pytest"]), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        sections.append(str(report["remaining_pytest"]["output"]) + "\n")
    for label, report in (("PRE-BOOTSTRAP STAGE10 TESTS", pre_unit), ("POST-BOOTSTRAP STAGE10 TESTS", post_unit)):
        sections.append(f"===== {label} =====\n")
        sections.append(json.dumps(_compact_pytest(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        sections.append(str(report["output"]) + "\n")
    write_text_atomic(TEST_REPORT_PATH, "".join(sections))


def _build_input_identity(
    *,
    frozen_hashes: Mapping[str, str],
    tree: Mapping[str, object],
    point_crosscheck: Mapping[str, object],
) -> dict[str, object]:
    return {
        "Stage": 10,
        "ActiveBaselineZipName": STAGE9_BASELINE_ZIP_NAME,
        "ActiveBaselineZipSHA256": STAGE9_BASELINE_ZIP_SHA256,
        "ActiveBaselineZipSHA256Verified": True,
        "Stage8ZipRole": "LEGACY_REGRESSION_DEPENDENCY_ONLY_NOT_BOOTSTRAP_SOURCE",
        "Stage8RegressionZipSHA256": STAGE8_BASELINE_ZIP_SHA256,
        "Stage8RegressionZipSHA256Verified": True,
        "Stage9Status": "PASS",
        "FormalDataSource": "artifacts/stage9/formal_repeat_metrics.csv",
        "FrozenInputHashes": dict(frozen_hashes),
        "Stage1ThroughStage9Tree": dict(tree),
        "FormalPointEstimateCrosscheck": dict(point_crosscheck),
        "FormalRerunExecuted": False,
        "PhysicalSimulationExecuted": False,
    }


def _write_changed_files(tree: Mapping[str, object]) -> None:
    text = "STAGE 10 CHANGED FILES\n\nADDED STATIC FILES:\n"
    text += "".join(f"- {path}\n" for path in sorted(ALLOWED_STATIC_ADDITIONS))
    artifact_count = sum(1 for path in STAGE10_DIR.rglob("*") if path.is_file())
    text += (
        f"\nADDED ARTIFACT TREE:\n- artifacts/stage10/** | {artifact_count} files before final manifest\n"
        "\nMODIFIED FILES:\n- README.md\n- pyproject.toml\n"
        "\nFROZEN STAGE 1-9 TREE:\n"
        f"- Baseline member count: {tree['BaselineMemberCount']}\n"
        f"- Baseline tree SHA256: {tree['BaselineTreeSHA256']}\n"
        "- Unexpected modified files: NONE\n- Deleted files: NONE\n"
        "\nSCIENCE CONTRACT CHANGED: NO\n"
    )
    write_text_atomic(CHANGED_FILES_PATH, text)


def _write_validation_report(
    *,
    post_regression: Mapping[str, object],
    hard_gates: Mapping[str, object],
    audit_report: Mapping[str, object],
    resumed_count: int,
    recomputed_count: int,
) -> None:
    full = post_regression["complete_pytest"]
    remaining = post_regression["remaining_pytest"]
    text = f"""# Stage 10 validation report

## Identity and frozen boundary

- Unique Stage 9 baseline ZIP SHA256: PASS.
- Stage 9 status and Formal input table identities: PASS.
- Stage 10 canonical science-contract SHA256: PASS.
- Direct Stage 9 ZIP versus current Stage 1-9 byte-identity gate: PASS.
- No frozen Stage 1-9 file was modified or deleted outside README/pyproject version boundaries.

## Bootstrap execution integrity

- Namespace: BOOTSTRAP; independent unit: whole Repeat.
- Scale: 18 interference Conditions x 2000 Bootstrap replicates x 2000 draws.
- Workers: 1; vectorization block size: 50; persistent resume unit: Condition.
- Condition outputs reused after strict validation: {resumed_count}; wholly recomputed: {recomputed_count}.
- Five methods share one index vector for every Condition/BootstrapID.
- Method is absent from RandomAddress; vector Philox and frozen scalar Philox are bitwise exact.
- Rejection sampling acceptance limit: {ACCEPTANCE_LIMIT}.
- Audit index shape: {hard_gates['AuditIndexShape']}; primary and independent rerun exact: {audit_report['ExactReproducibilityPassed']}.
- NI was checked for 5 x 2000 completeness and identity and was not bootstrapped.

## Statistics and outputs

- Estimator: ratio of resampled sums for all four metrics.
- Undefined denominator handling: NaN plus false mask; no redraw and no zero imputation.
- Paired contrasts: TW-T and TW-W only, with TW minus comparator direction for every metric.
- CI: 95% percentile Bootstrap, NumPy linear 0.025/0.975 quantiles.
- Point estimates: independently reconstructed from the Stage 9 Formal mother table, not Bootstrap means.
- Individual distribution shape: {hard_gates['IndividualDistributionShape']} and summary rows: {hard_gates['IndividualSummaryRows']}.
- Paired distribution shape: {hard_gates['PairedDistributionShape']} and summary rows: {hard_gates['PairedSummaryRows']}.
- Stage 8 precision audit rows: {hard_gates['Stage8PrecisionAuditRows']}; failure to meet a target was not used as a Stage 10 failure condition.

## Regression boundary

- Full pytest raw result: {full['passed']} passed, {full['failed']} failed, {full['errors']} errors.
- Failed-node set equals the two authorized forward-compatibility boundary nodes exactly.
- After exact deselection: {remaining['passed']} passed, {remaining['failed']} failed, {remaining['errors']} errors.
- Raw failure output is preserved in `test_report.txt`; the report does not relabel full pytest as green.

## Prohibited actions and STOP

- No Stage 9 Formal rerun, physical simulation, p-value, paper figure, paper result conclusion, condition drop, parameter change, outcome-direction PASS rule, or post-Formal contrast addition occurred.
- ProjectedPrecisionWarning remains true.
- Stage 11 was not executed. STOP after automatic Stage 10 packaging.
"""
    write_text_atomic(VALIDATION_REPORT_PATH, text)


def _status_payload(
    *,
    post_regression: Mapping[str, object],
    audit_report: Mapping[str, object],
) -> dict[str, object]:
    return {
        "stage": STAGE_NUMBER,
        "stage_name": STAGE_NAME,
        "status": "PASS",
        "stage9_baseline_zip_sha256_verified": True,
        "formal_repeat_metrics_sha256_verified": True,
        "preformal_contract_sha256_verified": True,
        "stage10_science_contract_sha256_verified": True,
        "stage1_through_stage9_tree_unchanged": True,
        "bootstrap_executed": True,
        "bootstrap_namespace": "BOOTSTRAP",
        "bootstrap_b": BOOTSTRAP_B,
        "bootstrap_independent_unit": "Repeat",
        "bootstrap_draws_per_replicate": BOOTSTRAP_DRAW_COUNT,
        "bootstrap_condition_count": 18,
        "individual_summary_row_count": 360,
        "paired_summary_row_count": 144,
        "stage8_precision_audit_row_count": 216,
        "bootstrap_audit_ids": list(BOOTSTRAP_AUDIT_IDS),
        "bootstrap_audit_index_count": 108000,
        "bootstrap_audit_exact_reproducibility_passed": audit_report[
            "ExactReproducibilityPassed"
        ],
        "ratio_of_sums_verified": True,
        "paired_indices_shared_verified": True,
        "method_absent_from_random_address": True,
        "vectorized_philox_scalar_exact": True,
        "projected_precision_warning": True,
        "p_value_generated": False,
        "condition_dropped": False,
        "parameter_changed": False,
        "post_formal_contrast_added": False,
        "outcome_direction_used_for_pass": False,
        "physical_simulation_executed": False,
        "formal_rerun_executed": False,
        "paper_figure_generated": False,
        "paper_result_conclusion_generated": False,
        "stage11_executed": False,
        "legacy_boundary_exceptions": post_regression["legacy_boundary_exceptions"],
        "full_pytest_raw_failed": post_regression["complete_pytest"]["failed"],
        "full_pytest_raw_errors": post_regression["complete_pytest"]["errors"],
        "remaining_pytest_failed": post_regression["remaining_pytest"]["failed"],
        "remaining_pytest_errors": post_regression["remaining_pytest"]["errors"],
        "blocking_issue_count": 0,
        "stop_after_stage10": True,
    }


def _artifact_hash_map() -> dict[str, str]:
    paths: list[Path] = [
        SCIENCE_CONTRACT_PATH,
        INPUT_IDENTITY_PATH,
        NI_REPORT_PATH,
        INDIVIDUAL_SUMMARY_PATH,
        PAIRED_SUMMARY_PATH,
        PRECISION_AUDIT_PATH,
        COMBINED_DISTRIBUTIONS_PATH,
        AUDIT_INDICES_PATH,
        STATUS_PATH,
        VALIDATION_REPORT_PATH,
        TEST_REPORT_PATH,
        CHANGED_FILES_PATH,
    ]
    for condition in range(1, 19):
        directory = STAGE10_DIR / "bootstrap_conditions" / (
            f"condition_{condition:02d}_{CONDITIONS[condition].Code}"
        )
        paths.extend((directory / "distributions.npz", directory / "manifest.json"))
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())
    }


def verify_bootstrap_manifest_artifacts(
    path: Path = BOOTSTRAP_MANIFEST_PATH,
) -> dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    artifact_hashes = manifest.get("ArtifactSHA256", {})
    if not isinstance(artifact_hashes, dict):
        raise AssertionError("Bootstrap manifest ArtifactSHA256 is not an object")
    for relative, expected in artifact_hashes.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise AssertionError(
                f"Bootstrap manifest artifact SHA mismatch for {relative}: {actual}"
            )
    if len(manifest.get("ConditionOutputs", [])) != 18:
        raise AssertionError("Bootstrap manifest does not contain 18 condition outputs")
    return manifest


def _package_members() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not _excluded_path(path.relative_to(ROOT))
        ),
        key=lambda item: item.relative_to(ROOT).as_posix(),
    )


def validate_package_prerequisites() -> None:
    validate_stage1_through_stage9_tree_unchanged()
    validate_science_contract()
    validate_combined_distributions(COMBINED_DISTRIBUTIONS_PATH)
    with np.load(AUDIT_INDICES_PATH, allow_pickle=False) as archive:
        validate_bootstrap_audit_arrays({name: archive[name] for name in archive.files})
    if csv_row_count(INDIVIDUAL_SUMMARY_PATH) != 360:
        raise AssertionError("package prerequisite individual summary row count failed")
    if csv_row_count(PAIRED_SUMMARY_PATH) != 144:
        raise AssertionError("package prerequisite paired summary row count failed")
    if csv_row_count(PRECISION_AUDIT_PATH) != 216:
        raise AssertionError("package prerequisite precision audit row count failed")
    verify_bootstrap_manifest_artifacts()
    condition_root = STAGE10_DIR / "bootstrap_conditions"
    directories = sorted(path for path in condition_root.iterdir() if path.is_dir())
    if len(directories) != 18:
        raise AssertionError("package prerequisite condition directory count is not 18")
    for directory in directories:
        if not (directory / "distributions.npz").is_file() or not (
            directory / "manifest.json"
        ).is_file():
            raise AssertionError(f"incomplete condition output directory: {directory.name}")
    if (ROOT / "artifacts/stage11").exists():
        raise AssertionError("Stage 11 artifacts cannot enter the Stage 10 package")


def package_stage10(output_path: Path | None = None) -> tuple[Path, str]:
    validate_package_prerequisites()
    destination = (
        Path(output_path) if output_path is not None else ROOT.parent / FINAL_ZIP_NAME
    ).resolve()
    if destination.parent != ROOT.parent.resolve():
        raise ValueError("Stage 10 package destination must be the project parent")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    members = _package_members()
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in members:
                relative = path.relative_to(ROOT)
                archive.write(path, (Path(FINAL_ZIP_ROOT) / relative).as_posix())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    expected_names = {
        (Path(FINAL_ZIP_ROOT) / path.relative_to(ROOT)).as_posix() for path in members
    }
    manifest = verify_bootstrap_manifest_artifacts()
    with zipfile.ZipFile(destination) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        if names != expected_names:
            raise AssertionError(
                f"Stage 10 package inventory mismatch: missing={sorted(expected_names-names)}, extra={sorted(names-expected_names)}"
            )
        for relative, expected_sha in manifest["ArtifactSHA256"].items():
            member = (Path(FINAL_ZIP_ROOT) / relative).as_posix()
            actual_sha = hashlib.sha256(archive.read(member)).hexdigest()
            if actual_sha != expected_sha:
                raise AssertionError(f"post-package artifact SHA mismatch: {relative}")
        if any(
            name.startswith(f"{FINAL_ZIP_ROOT}/artifacts/stage11/") for name in names
        ):
            raise AssertionError("Stage 11 artifact entered the final ZIP")
    return destination, sha256_file(destination)


def build_stage10(progress: Callable[[str], None] = print) -> dict[str, object]:
    STAGE10_DIR.mkdir(parents=True, exist_ok=True)
    progress("[Stage10 STEP 1-5] Verify identities, contract, and Stage 1-9 tree")
    validate_required_static_files()
    frozen_hashes = validate_frozen_input_hashes()
    validate_stage9_status()
    validate_science_contract()
    tree = validate_stage1_through_stage9_tree_unchanged()
    execution_path_gate = assert_no_forbidden_execution_path()
    code_sha = stage10_code_sha256()

    progress("[Stage10 STEP 6] Run complete pytest and audit exactly two legacy failures")
    pre_regression = run_full_pytest_with_legacy_rules("PRE-BOOTSTRAP")
    progress("[Stage10 STEP 7] Run Stage 10 tests inside the remaining green suite")
    pre_unit = run_stage10_unit_tests("PRE-BOOTSTRAP")

    progress("[Stage10 STEP 8] Run scalar/vector Philox and golden-vector preflight")
    rng_preflight = validate_scalar_vector_and_golden_rng()

    progress("[Stage10 STEP 9] Load Formal mother table, reconstruct points, and verify NI")
    formal = load_formal_repeat_metrics()
    point_results, point_crosscheck = reconstitute_formal_point_estimates(formal)
    ni_report = validate_ni_consistency()
    write_json_atomic(NI_REPORT_PATH, ni_report)

    identity_hashes = {
        "FormalRepeatMetricsSHA256": FORMAL_REPEAT_METRICS_SHA256,
        "Stage10ScienceContractSHA256": STAGE10_SCIENCE_CONTRACT_SHA256,
        "Stage10CodeSHA256": code_sha,
        "Stage9BaselineZipSHA256": STAGE9_BASELINE_ZIP_SHA256,
    }
    condition_results = []
    condition_records: list[dict[str, object]] = []
    resumed_count = 0
    recomputed_count = 0
    progress("[Stage10 STEP 10] Execute 18 Conditions, B=2000, whole-Condition resume")
    for condition in range(1, 19):
        code = CONDITIONS[condition].Code
        progress(f"  [Condition {condition:02d}/18 {code}] validate resume or compute all 2000 replicates")
        result, resumed, reason = run_or_resume_condition(
            STAGE10_DIR,
            condition_ordinal=condition,
            condition_code=code,
            statistics=formal.statistics[condition],
            identity_hashes=identity_hashes,
        )
        condition_results.append(result)
        resumed_count += int(resumed)
        recomputed_count += int(not resumed)
        directory = STAGE10_DIR / "bootstrap_conditions" / f"condition_{condition:02d}_{code}"
        condition_records.append(
            {
                "ConditionOrdinal": condition,
                "ConditionCode": code,
                "ResumeUsed": resumed,
                "PreValidationReason": reason,
                "DistributionSHA256": sha256_file(directory / "distributions.npz"),
                "ManifestSHA256": sha256_file(directory / "manifest.json"),
            }
        )

    progress("[Stage10 STEP 11] Reload and strictly validate all 18 condition outputs")
    reloaded_results = []
    for condition in range(1, 19):
        validation = load_valid_condition(
            STAGE10_DIR,
            condition_ordinal=condition,
            condition_code=CONDITIONS[condition].Code,
            bootstrap_ids=tuple(range(1, BOOTSTRAP_B + 1)),
            audit_ids=BOOTSTRAP_AUDIT_IDS,
            block_size=BOOTSTRAP_BLOCK_SIZE,
            identity_hashes=identity_hashes,
        )
        if not validation.valid or validation.result is None:
            raise AssertionError(
                f"Condition {condition} failed post-execution reload: {validation.reason}"
            )
        reloaded_results.append(validation.result)

    progress("[Stage10 STEP 12] Independently regenerate audit IDs 1/1000/2000")
    primary_audit, audit_report = independently_regenerate_audit_indices(
        reloaded_results
    )
    audit_arrays = write_bootstrap_audit_indices(
        AUDIT_INDICES_PATH,
        condition_ordinals=range(1, 19),
        audit_ids=BOOTSTRAP_AUDIT_IDS,
        repeat_ids=primary_audit,
    )

    progress("[Stage10 STEP 13] Build and reload combined distributions")
    combined = write_combined_distributions(
        COMBINED_DISTRIBUTIONS_PATH, reloaded_results
    )
    progress("[Stage10 STEP 14-16] Build 360/144/216-row summaries and precision audit")
    individual_rows = build_individual_summary_rows(combined, point_results)
    paired_rows = build_paired_summary_rows(combined, point_results)
    precision_rows = build_stage8_precision_audit_rows(individual_rows, paired_rows)
    write_csv_atomic(INDIVIDUAL_SUMMARY_PATH, INDIVIDUAL_SUMMARY_FIELDS, individual_rows)
    write_csv_atomic(PAIRED_SUMMARY_PATH, PAIRED_SUMMARY_FIELDS, paired_rows)
    write_csv_atomic(PRECISION_AUDIT_PATH, PRECISION_AUDIT_FIELDS, precision_rows)

    hard_gates = validate_summary_and_distribution_hard_gates(
        combined=combined,
        audit_arrays=audit_arrays,
        individual_rows=individual_rows,
        paired_rows=paired_rows,
        precision_rows=precision_rows,
    )

    progress("[Stage10 STEP 17] Run post-Bootstrap complete regression and all hard gates")
    post_regression = run_full_pytest_with_legacy_rules("POST-BOOTSTRAP")
    post_unit = run_stage10_unit_tests("POST-BOOTSTRAP")
    final_tree = validate_stage1_through_stage9_tree_unchanged()
    if final_tree["BaselineTreeSHA256"] != tree["BaselineTreeSHA256"]:
        raise AssertionError("Stage 9 baseline tree identity changed during Stage 10")

    progress("[Stage10 STEP 18] Write identity, status, reports, and manifest")
    input_identity = _build_input_identity(
        frozen_hashes=frozen_hashes,
        tree=final_tree,
        point_crosscheck=point_crosscheck,
    )
    write_json_atomic(INPUT_IDENTITY_PATH, input_identity)
    _write_final_test_report(pre_regression, pre_unit, post_regression, post_unit)
    _write_changed_files(final_tree)
    _write_validation_report(
        post_regression=post_regression,
        hard_gates=hard_gates,
        audit_report=audit_report,
        resumed_count=resumed_count,
        recomputed_count=recomputed_count,
    )
    status = _status_payload(
        post_regression=post_regression,
        audit_report=audit_report,
    )
    write_json_atomic(STATUS_PATH, status)
    manifest = {
        "ManifestSchema": "STAGE10_BOOTSTRAP_MANIFEST_V1",
        "Stage": STAGE_NUMBER,
        "StageName": STAGE_NAME,
        "Status": "PASS",
        "IdentityHashes": dict(frozen_hashes),
        "Stage10CodeSHA256": code_sha,
        "BootstrapNamespace": "BOOTSTRAP",
        "BootstrapB": BOOTSTRAP_B,
        "BootstrapDrawCountPerReplicate": BOOTSTRAP_DRAW_COUNT,
        "IndependentUnit": "Repeat",
        "Workers": 1,
        "VectorizationBlockSize": BOOTSTRAP_BLOCK_SIZE,
        "PersistentResumeUnit": "Condition",
        "ConditionCount": 18,
        "ConditionOutputs": condition_records,
        "RNGPreflight": rng_preflight,
        "ExecutionPathGate": execution_path_gate,
        "FormalPointEstimateCrosscheck": point_crosscheck,
        "BootstrapAudit": audit_report,
        "HardGates": hard_gates,
        "FullPytestAuthorizedFailureNodes": list(LEGACY_NODES),
        "RemainingPytestFailed": 0,
        "RemainingPytestErrors": 0,
        "ProjectedPrecisionWarning": True,
        "PValueGenerated": False,
        "PhysicalSimulationExecuted": False,
        "FormalRerunExecuted": False,
        "Stage11Executed": False,
        "StopAfterStage10": True,
        "ArtifactSHA256": _artifact_hash_map(),
    }
    write_json_atomic(BOOTSTRAP_MANIFEST_PATH, manifest)
    verify_bootstrap_manifest_artifacts()
    validate_stage1_through_stage9_tree_unchanged()

    progress("[Stage10 STEP 19] Automatically package and post-validate complete Stage 10 ZIP")
    zip_path, zip_sha = package_stage10()
    progress("[Stage10 STEP 20] STOP")
    return {
        "Stage": STAGE_NUMBER,
        "Status": "PASS",
        "FinalZip": str(zip_path),
        "FinalZipSHA256": zip_sha,
        "FullPytestPassed": post_regression["complete_pytest"]["passed"],
        "FullPytestFailed": post_regression["complete_pytest"]["failed"],
        "FullPytestErrors": post_regression["complete_pytest"]["errors"],
        "RemainingPytestPassed": post_regression["remaining_pytest"]["passed"],
        "RemainingPytestFailed": post_regression["remaining_pytest"]["failed"],
        "RemainingPytestErrors": post_regression["remaining_pytest"]["errors"],
        "LegacyBoundaryExceptions": post_regression["legacy_boundary_exceptions"],
        "BootstrapConditionCount": 18,
        "BootstrapB": BOOTSTRAP_B,
        "BootstrapDrawsPerReplicate": BOOTSTRAP_DRAW_COUNT,
        "BootstrapAuditIndexCount": int(primary_audit.size),
        "BootstrapAuditExact": True,
        "IndividualSummaryRows": len(individual_rows),
        "PairedSummaryRows": len(paired_rows),
        "Stage8PrecisionAuditRows": len(precision_rows),
        "IndividualDistributionShape": list(combined["individual_values"].shape),
        "PairedDistributionShape": list(combined["paired_values"].shape),
        "StopAfterStage10": True,
    }


__all__ = [
    "ALLOWED_MODIFIED_FILES",
    "ALLOWED_STATIC_ADDITIONS",
    "BOOTSTRAP_MANIFEST_PATH",
    "FINAL_ZIP_NAME",
    "FORMAL_AGGREGATED_METRICS_SHA256",
    "FORMAL_H300_WIDTH_STRATA_SHA256",
    "FORMAL_REPEAT_METRICS_SHA256",
    "INDIVIDUAL_SUMMARY_FIELDS",
    "LEGACY_NODES",
    "PAIRED_SUMMARY_FIELDS",
    "PRECISION_AUDIT_FIELDS",
    "SCIENCE_CONTRACT_PATH",
    "STAGE10_PREFORMAL_CONTRACT_SHA256",
    "STAGE10_SCIENCE_CONTRACT_SHA256",
    "STAGE10_TASKBOOK_SHA256",
    "STAGE11_EXECUTED",
    "STAGE8_BASELINE_ZIP_SHA256",
    "STAGE9_BASELINE_ZIP_SHA256",
    "assert_no_forbidden_execution_path",
    "build_individual_summary_rows",
    "build_paired_summary_rows",
    "build_stage10",
    "build_stage8_precision_audit_rows",
    "csv_row_count",
    "load_formal_repeat_metrics",
    "package_stage10",
    "reconstitute_formal_point_estimates",
    "run_full_pytest_with_legacy_rules",
    "stage10_code_sha256",
    "validate_frozen_input_hashes",
    "validate_ni_consistency",
    "validate_required_static_files",
    "validate_science_contract",
    "validate_stage1_through_stage9_tree_unchanged",
    "validate_stage9_status",
    "verify_stage8_regression_zip",
    "verify_stage9_baseline_zip",
    "write_csv_atomic",
]
