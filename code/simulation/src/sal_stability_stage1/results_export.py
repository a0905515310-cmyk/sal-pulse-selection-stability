from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

from .build import ROOT
from .contracts import CONDITIONS, METHODS


STAGE11_DIR = ROOT / "artifacts" / "stage11"
STAGE11_CONTRACT_PATH = STAGE11_DIR / "stage11_science_contract.json"

FROZEN_SOURCE_SPECS: dict[str, tuple[str, str]] = {
    "Stage10BootstrapAuditIndicesSHA256": (
        "artifacts/stage10/bootstrap_audit_indices.npz",
        "ff21b44d5d6bf638cbde0f881e9e0bca6a8778b4cf154b125d8511e2ec873b07",
    ),
    "Stage10BootstrapDistributionsSHA256": (
        "artifacts/stage10/bootstrap_distributions.npz",
        "370e11118c828c092a0c86037488025555578e2040364d776940887a8b2c8cad",
    ),
    "Stage10BootstrapManifestSHA256": (
        "artifacts/stage10/bootstrap_manifest.json",
        "69748b6f37e3caf5d3abb561042ca190c8f0068d2217b44b14cb770faa50b71d",
    ),
    "Stage10IndividualSummarySHA256": (
        "artifacts/stage10/bootstrap_individual_summary.csv",
        "ad9bca843ff063d1a9cc26d134f8fc43b3cdeb9460f850e9e5afbda6c0d5f0b1",
    ),
    "Stage10NIConsistencySHA256": (
        "artifacts/stage10/ni_consistency_report.json",
        "3fcc11e13a40b7ae177643c71c7287878c33fb43a124d8d8604c96288f9d19bc",
    ),
    "Stage10PairedSummarySHA256": (
        "artifacts/stage10/bootstrap_paired_summary.csv",
        "2e864379ff8985fce28cfc7fd9d71f29ca5ec593fd7340dfeaf586684d7e58fe",
    ),
    "Stage10PrecisionAuditSHA256": (
        "artifacts/stage10/stage8_precision_target_audit.csv",
        "361e5b017cb078545c032a17f98104887ae46703a1a921d6cf8689e3ae9988e4",
    ),
    "Stage10ScienceContractSHA256": (
        "artifacts/stage10/stage10_science_contract.json",
        "fdaa84d99e3b9e8fbcebe091c7a173a90ba5a42cef054f48c79eca01179656fd",
    ),
    "Stage10StatusSHA256": (
        "artifacts/stage10/status.json",
        "91cb29d8c06aeabfed7b8dbfe3a46f26ee236533e12a045753b4cee24a57ecf6",
    ),
    "Stage9FormalAggregatedMetricsSHA256": (
        "artifacts/stage9/formal_aggregated_metrics.csv",
        "803856c4d776b53b63e7e6f8be4425ce96ddb05c49aaca97c049134b04644865",
    ),
    "Stage9FormalRepeatMetricsSHA256": (
        "artifacts/stage9/formal_repeat_metrics.csv",
        "e3084733adf9fa3f2325b0407b4358aeae118df8f6a152a4c1f5984757f3f8f6",
    ),
    "Stage9H300WidthStrataSHA256": (
        "artifacts/stage9/formal_h300_width_strata.csv",
        "4e70f33ba195070a75e35fc6b3d14b7d5d7851b4ca414fcac7e8f24046a108b4",
    ),
}

INDIVIDUAL_SOURCE_PATH = "artifacts/stage10/bootstrap_individual_summary.csv"
PAIRED_SOURCE_PATH = "artifacts/stage10/bootstrap_paired_summary.csv"
PRECISION_SOURCE_PATH = "artifacts/stage10/stage8_precision_target_audit.csv"
NI_REPORT_SOURCE_PATH = "artifacts/stage10/ni_consistency_report.json"
FORMAL_AGGREGATED_SOURCE_PATH = "artifacts/stage9/formal_aggregated_metrics.csv"
H300_SOURCE_PATH = "artifacts/stage9/formal_h300_width_strata.csv"
CONTRACTS_SOURCE_PATH = "src/sal_stability_stage1/contracts.py"
STAGE11_CONTRACT_ARTIFACT = "artifacts/stage11/stage11_science_contract.json"

SOURCE_SHA256_BY_PATH = {
    path: expected for path, expected in FROZEN_SOURCE_SPECS.values()
}

METRICS = (
    ("P_cor", "HIGHER"),
    ("P_C_given_C", "HIGHER"),
    ("P_C_given_E", "HIGHER"),
    ("Mean_L_NC", "LOWER"),
)
METRIC_ORDER = {code: ordinal for ordinal, (code, _) in enumerate(METRICS)}
METRIC_DIRECTION = dict(METRICS)

CONTRASTS = (
    (0, "TW-T", "T"),
    (1, "TW-W", "W"),
)
CONTRAST_BY_ORDINAL = {
    ordinal: (code, comparator) for ordinal, code, comparator in CONTRASTS
}

EXPECTED_CONDITIONS = (
    (0, "NI", "NI", None, None),
    (1, "H100", "HPRF", 100_000.0, None),
    (2, "H200", "HPRF", 200_000.0, None),
    (3, "H300", "HPRF", 300_000.0, None),
    (4, "H400", "HPRF", 400_000.0, None),
    (5, "H500", "HPRF", 500_000.0, None),
    (6, "F1", "IDF", None, 1e-6),
    (7, "F2", "IDF", None, 2e-6),
    (8, "F3", "IDF", None, 3e-6),
    (9, "F4", "IDF", None, 4e-6),
    (10, "HF200-1", "COMPOSITE", 200_000.0, 1e-6),
    (11, "HF200-2", "COMPOSITE", 200_000.0, 2e-6),
    (12, "HF200-3", "COMPOSITE", 200_000.0, 3e-6),
    (13, "HF300-1", "COMPOSITE", 300_000.0, 1e-6),
    (14, "HF300-2", "COMPOSITE", 300_000.0, 2e-6),
    (15, "HF300-3", "COMPOSITE", 300_000.0, 3e-6),
    (16, "HF500-1", "COMPOSITE", 500_000.0, 1e-6),
    (17, "HF500-2", "COMPOSITE", 500_000.0, 2e-6),
    (18, "HF500-3", "COMPOSITE", 500_000.0, 3e-6),
)
EXPECTED_METHODS = (
    (0, "FIRST", "首脉冲准则"),
    (1, "LAST", "末脉冲准则"),
    (2, "T", "最优时序准则"),
    (3, "W", "脉宽单特征准则"),
    (4, "TW", "双特征联合准则"),
)

HPRF_CONDITIONS = (1, 2, 3, 4, 5)
F_ONLY_CONDITIONS = (6, 7, 8, 9)
COMPOSITE_CONDITIONS = tuple(range(10, 19))
WIDTH_STRATUM_ORDER = (
    "WIDTH_STRATUM_0NS",
    "WIDTH_STRATUM_10NS",
    "WIDTH_STRATUM_GE20NS",
)

INDIVIDUAL_SOURCE_FIELDS = (
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
INDIVIDUAL_ADDED_FIELDS = (
    "Scene",
    "HPRF_Hz",
    "HPRF_kHz",
    "FDelay_s",
    "FDelay_us",
    "MethodNameZH",
    "MetricPreferredDirection",
    "SourceArtifact",
    "SourceArtifactSHA256",
)
INDIVIDUAL_FIELDS = INDIVIDUAL_SOURCE_FIELDS + INDIVIDUAL_ADDED_FIELDS

PAIRED_SOURCE_FIELDS = (
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
PAIRED_ADDED_FIELDS = (
    "Scene",
    "HPRF_Hz",
    "HPRF_kHz",
    "FDelay_s",
    "FDelay_us",
    "ComparatorMethodNameZH",
    "MetricPreferredDirection",
    "CIPositionRelativeToZero",
    "SourceArtifact",
    "SourceArtifactSHA256",
)
PAIRED_FIELDS = PAIRED_SOURCE_FIELDS + PAIRED_ADDED_FIELDS

PRECISION_SOURCE_FIELDS = (
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
PRECISION_FIELDS = PRECISION_SOURCE_FIELDS + (
    "SourceArtifact",
    "SourceArtifactSHA256",
)

NI_SOURCE_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrdinal",
    "MethodCode",
    "R",
    "K",
    "Total_N_C",
    "Total_N_E",
    "Total_N_N",
    "Total_N_CC",
    "Total_N_CE",
    "Total_N_CN",
    "Total_N_EC",
    "Total_N_EE",
    "Total_N_EN",
    "Total_N_NC",
    "Total_N_NE",
    "Total_N_NN",
    "Total_N_Cdot",
    "Total_N_Edot",
    "Total_N_Ndot",
    "Total_N_run_NC",
    "Total_Sum_L_NC_obs",
    "Total_N_open_NC",
    "Total_N_E_H",
    "Total_N_E_F",
    "P_cor_value",
    "P_cor_defined",
    "P_cor_undefined_reason",
    "P_C_given_C_value",
    "P_C_given_C_defined",
    "P_C_given_C_undefined_reason",
    "P_C_given_E_value",
    "P_C_given_E_defined",
    "P_C_given_E_undefined_reason",
    "Mean_L_NC_value",
    "Mean_L_NC_defined",
    "Mean_L_NC_undefined_reason",
    "P_E_given_C_value",
    "P_E_given_C_defined",
    "P_E_given_C_undefined_reason",
    "P_N_given_C_value",
    "P_N_given_C_defined",
    "P_N_given_C_undefined_reason",
    "P_E_given_E_value",
    "P_E_given_E_defined",
    "P_E_given_E_undefined_reason",
    "P_N_given_E_value",
    "P_N_given_E_defined",
    "P_N_given_E_undefined_reason",
    "P_N_value",
    "P_N_defined",
    "P_N_undefined_reason",
    "P_open_NC_value",
    "P_open_NC_defined",
    "P_open_NC_undefined_reason",
    "P_NC_end_value",
    "P_NC_end_defined",
    "P_NC_end_undefined_reason",
    "P_E_H_value",
    "P_E_H_defined",
    "P_E_H_undefined_reason",
    "P_E_F_value",
    "P_E_F_defined",
    "P_E_F_undefined_reason",
)
NI_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrdinal",
    "MethodCode",
    "MethodNameZH",
    "R",
    "K",
    "Total_N_C",
    "Total_N_E",
    "Total_N_N",
    "P_cor_value",
    "P_cor_defined",
    "P_C_given_C_value",
    "P_C_given_C_defined",
    "P_C_given_E_value",
    "P_C_given_E_defined",
    "P_C_given_E_undefined_reason",
    "Mean_L_NC_value",
    "Mean_L_NC_defined",
    "Mean_L_NC_undefined_reason",
    "SourceArtifact",
    "SourceArtifactSHA256",
    "NIIdentityArtifact",
    "NIIdentityArtifactSHA256",
)

H300_SOURCE_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrdinal",
    "MethodCode",
    "RepeatID",
    "WidthStratum",
    "StratumCycleCount",
    "StratumCorrectCount",
    "StratumErrorCount",
    "Stratum_C_to_E_Count",
)
H300_FIELDS = (
    "MethodOrdinal",
    "MethodCode",
    "WidthStratum",
    "TotalCycleCount",
    "TotalCorrectCount",
    "TotalErrorCount",
    "Total_C_to_E_Count",
    "PositiveSupportRepeatCount",
    "P_cor_stratum",
    "P_error_stratum",
    "Estimator",
    "SourceArtifact",
    "SourceArtifactSHA256",
)

CONDITION_REGISTRY_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "Scene",
    "HPRF_Hz",
    "HPRF_kHz",
    "FDelay_s",
    "FDelay_us",
    "SourceArtifact",
    "SourceArtifactSHA256",
)
METHOD_REGISTRY_FIELDS = (
    "MethodOrdinal",
    "MethodCode",
    "MethodNameZH",
    "SourceArtifact",
    "SourceArtifactSHA256",
)
METRIC_REGISTRY_FIELDS = (
    "MetricOrdinal",
    "MetricCode",
    "PreferredDirection",
    "SourceArtifact",
    "SourceArtifactSHA256",
)
CONTRAST_REGISTRY_FIELDS = (
    "ContrastOrdinal",
    "ContrastCode",
    "ComparatorMethod",
    "SourceArtifact",
    "SourceArtifactSHA256",
)
EVIDENCE_REGISTRY_FIELDS = (
    "ArtifactPath",
    "ArtifactClass",
    "RowCount",
    "SHA256",
    "ImmediateSourceArtifacts",
    "InferenceClass",
    "AllowedUse",
    "SelectiveFilteringAllowed",
)

INDIVIDUAL_MASTER_REL = "paper_results_individual_master.csv"
PAIRED_MASTER_REL = "paper_results_paired_master.csv"
PRECISION_MASTER_REL = "paper_precision_audit_master.csv"
PRECISION_LIMITATIONS_REL = "paper_statistical_precision_limitations.csv"
NI_REL = "paper_data/NI_consistency.csv"
HPRF_INDIVIDUAL_REL = "paper_data/HPRF_individual.csv"
HPRF_PAIRED_REL = "paper_data/HPRF_paired.csv"
F_ONLY_INDIVIDUAL_REL = "paper_data/F_only_individual.csv"
F_ONLY_PAIRED_REL = "paper_data/F_only_paired.csv"
COMPOSITE_INDIVIDUAL_REL = "paper_data/composite_individual.csv"
COMPOSITE_PAIRED_REL = "paper_data/composite_paired.csv"
H300_REL = "paper_data/H300_width_strata_summary.csv"
CONDITION_REGISTRY_REL = "registry/condition_registry_export.csv"
METHOD_REGISTRY_REL = "registry/method_registry_export.csv"
METRIC_REGISTRY_REL = "registry/metric_registry_export.csv"
CONTRAST_REGISTRY_REL = "registry/contrast_registry_export.csv"
EVIDENCE_REGISTRY_REL = "paper_evidence_registry.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_row_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def read_csv_exact(
    path: Path,
    expected_fields: Sequence[str] | None = None,
    expected_rows: int | None = None,
) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if expected_fields is not None and fields != tuple(expected_fields):
            raise AssertionError(
                f"CSV header drift for {path}: expected {tuple(expected_fields)}, got {fields}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise AssertionError(f"CSV has surplus unnamed values: {path}")
    if expected_rows is not None and len(rows) != expected_rows:
        raise AssertionError(
            f"CSV row count drift for {path}: expected {expected_rows}, got {len(rows)}"
        )
    return rows


def write_csv_atomic(
    path: Path,
    field_names: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
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
                writer.writerow({field: row.get(field, "") for field in field_names})
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{label} is not an integer: {value!r}") from exc


def _parse_bool(value: str, label: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AssertionError(f"{label} is not an exact frozen Boolean: {value!r}")


def _validate_frozen_registries() -> None:
    conditions = tuple(
        (
            item.ConditionOrdinal,
            item.Code,
            item.Scene,
            item.HPRF_Hz,
            item.FDelay_s,
        )
        for item in CONDITIONS
    )
    methods = tuple((item.MethodOrdinal, item.Code, item.Name) for item in METHODS)
    if conditions != EXPECTED_CONDITIONS:
        raise AssertionError("contracts.py Condition registry drifted from Stage 11 contract")
    if methods != EXPECTED_METHODS:
        raise AssertionError("contracts.py Method registry drifted from Stage 11 contract")


def _condition_metadata(ordinal: int) -> dict[str, str]:
    condition = CONDITIONS[ordinal]
    return {
        "Scene": condition.Scene,
        "HPRF_Hz": "" if condition.HPRF_Hz is None else repr(condition.HPRF_Hz),
        "HPRF_kHz": ""
        if condition.HPRF_Hz is None
        else repr(condition.HPRF_Hz / 1000.0),
        "FDelay_s": "" if condition.FDelay_s is None else repr(condition.FDelay_s),
        "FDelay_us": ""
        if condition.FDelay_s is None
        else repr(condition.FDelay_s * 1_000_000.0),
    }


def _validate_unique(
    rows: Sequence[Mapping[str, str]], fields: Sequence[str], expected_count: int
) -> None:
    keys = {tuple(row[field] for field in fields) for row in rows}
    if len(rows) != expected_count or len(keys) != expected_count:
        raise AssertionError(
            f"expected {expected_count} unique rows by {tuple(fields)}, got rows={len(rows)}, unique={len(keys)}"
        )


def _build_individual_rows() -> list[dict[str, str]]:
    source = read_csv_exact(
        ROOT / INDIVIDUAL_SOURCE_PATH, INDIVIDUAL_SOURCE_FIELDS, 360
    )
    rows: list[dict[str, str]] = []
    for original in source:
        ordinal = _parse_int(original["ConditionOrdinal"], "ConditionOrdinal")
        method_ordinal = _parse_int(original["MethodOrdinal"], "MethodOrdinal")
        if ordinal not in range(1, 19):
            raise AssertionError("individual source contains NI or an unknown Condition")
        if original["ConditionCode"] != CONDITIONS[ordinal].Code:
            raise AssertionError("individual source ConditionCode drifted")
        if method_ordinal not in range(5) or original["MethodCode"] != METHODS[
            method_ordinal
        ].Code:
            raise AssertionError("individual source Method identity drifted")
        metric = original["Metric"]
        if metric not in METRIC_ORDER:
            raise AssertionError(f"unregistered individual metric: {metric}")
        enriched = dict(original)
        enriched.update(_condition_metadata(ordinal))
        enriched.update(
            {
                "MethodNameZH": METHODS[method_ordinal].Name,
                "MetricPreferredDirection": METRIC_DIRECTION[metric],
                "SourceArtifact": INDIVIDUAL_SOURCE_PATH,
                "SourceArtifactSHA256": SOURCE_SHA256_BY_PATH[
                    INDIVIDUAL_SOURCE_PATH
                ],
            }
        )
        rows.append(enriched)
    rows.sort(
        key=lambda row: (
            int(row["ConditionOrdinal"]),
            int(row["MethodOrdinal"]),
            METRIC_ORDER[row["Metric"]],
        )
    )
    _validate_unique(
        rows, ("ConditionOrdinal", "MethodOrdinal", "Metric"), expected_count=360
    )
    expected_universe = {
        (condition, method, metric)
        for condition in range(1, 19)
        for method in range(5)
        for metric, _ in METRICS
    }
    actual_universe = {
        (int(row["ConditionOrdinal"]), int(row["MethodOrdinal"]), row["Metric"])
        for row in rows
    }
    if actual_universe != expected_universe:
        raise AssertionError("individual source key universe is incomplete")
    return rows


def ci_position_relative_to_zero(row: Mapping[str, str]) -> str:
    if not _parse_bool(row["CI95Defined"], "CI95Defined"):
        return "UNDEFINED"
    if not row["CI95Lower"] or not row["CI95Upper"]:
        raise AssertionError("defined CI is missing a frozen endpoint")
    lower = Decimal(row["CI95Lower"])
    upper = Decimal(row["CI95Upper"])
    if lower > 0:
        return "POSITIVE"
    if upper < 0:
        return "NEGATIVE"
    return "INCLUDES_ZERO"


def _build_paired_rows() -> list[dict[str, str]]:
    source = read_csv_exact(ROOT / PAIRED_SOURCE_PATH, PAIRED_SOURCE_FIELDS, 144)
    rows: list[dict[str, str]] = []
    method_by_code = {method.Code: method for method in METHODS}
    for original in source:
        ordinal = _parse_int(original["ConditionOrdinal"], "ConditionOrdinal")
        contrast_ordinal = _parse_int(
            original["ContrastOrdinal"], "ContrastOrdinal"
        )
        if ordinal not in range(1, 19):
            raise AssertionError("paired source contains NI or an unknown Condition")
        if original["ConditionCode"] != CONDITIONS[ordinal].Code:
            raise AssertionError("paired source ConditionCode drifted")
        expected_contrast = CONTRAST_BY_ORDINAL.get(contrast_ordinal)
        if expected_contrast is None or (
            original["ContrastCode"], original["ComparatorMethod"]
        ) != expected_contrast:
            raise AssertionError("paired source contrast identity drifted")
        metric = original["Metric"]
        if metric not in METRIC_ORDER:
            raise AssertionError(f"unregistered paired metric: {metric}")
        comparator = method_by_code[original["ComparatorMethod"]]
        enriched = dict(original)
        enriched.update(_condition_metadata(ordinal))
        enriched.update(
            {
                "ComparatorMethodNameZH": comparator.Name,
                "MetricPreferredDirection": METRIC_DIRECTION[metric],
                "CIPositionRelativeToZero": ci_position_relative_to_zero(original),
                "SourceArtifact": PAIRED_SOURCE_PATH,
                "SourceArtifactSHA256": SOURCE_SHA256_BY_PATH[PAIRED_SOURCE_PATH],
            }
        )
        rows.append(enriched)
    rows.sort(
        key=lambda row: (
            int(row["ConditionOrdinal"]),
            int(row["ContrastOrdinal"]),
            METRIC_ORDER[row["Metric"]],
        )
    )
    _validate_unique(
        rows, ("ConditionOrdinal", "ContrastOrdinal", "Metric"), expected_count=144
    )
    expected_universe = {
        (condition, contrast, metric)
        for condition in range(1, 19)
        for contrast in range(2)
        for metric, _ in METRICS
    }
    actual_universe = {
        (int(row["ConditionOrdinal"]), int(row["ContrastOrdinal"]), row["Metric"])
        for row in rows
    }
    if actual_universe != expected_universe:
        raise AssertionError("paired source key universe is incomplete")
    return rows


def _build_precision_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source = read_csv_exact(
        ROOT / PRECISION_SOURCE_PATH, PRECISION_SOURCE_FIELDS, 216
    )
    rows: list[dict[str, str]] = []
    for original in source:
        enriched = dict(original)
        enriched.update(
            {
                "SourceArtifact": PRECISION_SOURCE_PATH,
                "SourceArtifactSHA256": SOURCE_SHA256_BY_PATH[PRECISION_SOURCE_PATH],
            }
        )
        rows.append(enriched)
    rows.sort(key=lambda row: int(row["AuditOrdinal"]))
    _validate_unique(rows, ("AuditOrdinal",), expected_count=216)
    limitations = [
        dict(row)
        for row in rows
        if not _parse_bool(
            row["MeetsStage8PrecisionTarget"], "MeetsStage8PrecisionTarget"
        )
    ]
    return rows, limitations


def _load_ni_identity() -> dict[str, object]:
    import json

    payload = json.loads((ROOT / NI_REPORT_SOURCE_PATH).read_text(encoding="utf-8"))
    required = {
        "Status": "PASS",
        "ConditionOrdinal": 0,
        "ConditionCode": "NI",
        "AllCyclesCorrectPassed": True,
        "FiveMethodRepeatMetricsIdentityPassed": True,
        "InferentialBootstrapExecuted": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise AssertionError(
                f"Stage 10 NI identity field {field} drifted: {payload.get(field)!r}"
            )
    return payload


def _build_ni_rows() -> list[dict[str, str]]:
    _load_ni_identity()
    source = read_csv_exact(ROOT / FORMAL_AGGREGATED_SOURCE_PATH, NI_SOURCE_FIELDS, 95)
    selected = [row for row in source if row["ConditionOrdinal"] == "0"]
    if len(selected) != 5:
        raise AssertionError("NI export is not exactly the five complete Method rows")
    rows: list[dict[str, str]] = []
    for original in selected:
        method_ordinal = _parse_int(original["MethodOrdinal"], "MethodOrdinal")
        if original["ConditionCode"] != "NI" or original["MethodCode"] != METHODS[
            method_ordinal
        ].Code:
            raise AssertionError("NI method/Condition identity drifted")
        row = {field: original[field] for field in NI_FIELDS if field in original}
        row.update(
            {
                "MethodNameZH": METHODS[method_ordinal].Name,
                "SourceArtifact": FORMAL_AGGREGATED_SOURCE_PATH,
                "SourceArtifactSHA256": SOURCE_SHA256_BY_PATH[
                    FORMAL_AGGREGATED_SOURCE_PATH
                ],
                "NIIdentityArtifact": NI_REPORT_SOURCE_PATH,
                "NIIdentityArtifactSHA256": SOURCE_SHA256_BY_PATH[
                    NI_REPORT_SOURCE_PATH
                ],
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: int(row["MethodOrdinal"]))
    _validate_unique(rows, ("MethodOrdinal",), expected_count=5)
    return rows


def _build_h300_rows() -> list[dict[str, str]]:
    source = read_csv_exact(ROOT / H300_SOURCE_PATH, H300_SOURCE_FIELDS, 30_000)
    grouped: dict[tuple[int, str, str], dict[str, int]] = {}
    for row in source:
        if row["ConditionOrdinal"] != "3" or row["ConditionCode"] != "H300":
            raise AssertionError("H300 width-strata source contains another Condition")
        method_ordinal = _parse_int(row["MethodOrdinal"], "MethodOrdinal")
        if row["MethodCode"] != METHODS[method_ordinal].Code:
            raise AssertionError("H300 width-strata Method identity drifted")
        stratum = row["WidthStratum"]
        if stratum not in WIDTH_STRATUM_ORDER:
            raise AssertionError(f"unknown H300 width stratum: {stratum}")
        key = (method_ordinal, row["MethodCode"], stratum)
        bucket = grouped.setdefault(
            key,
            {
                "TotalCycleCount": 0,
                "TotalCorrectCount": 0,
                "TotalErrorCount": 0,
                "Total_C_to_E_Count": 0,
                "PositiveSupportRepeatCount": 0,
                "InputRows": 0,
            },
        )
        cycle = _parse_int(row["StratumCycleCount"], "StratumCycleCount")
        correct = _parse_int(row["StratumCorrectCount"], "StratumCorrectCount")
        error = _parse_int(row["StratumErrorCount"], "StratumErrorCount")
        c_to_e = _parse_int(row["Stratum_C_to_E_Count"], "Stratum_C_to_E_Count")
        if min(cycle, correct, error, c_to_e) < 0 or correct + error != cycle:
            raise AssertionError("H300 source row violates count conservation")
        bucket["TotalCycleCount"] += cycle
        bucket["TotalCorrectCount"] += correct
        bucket["TotalErrorCount"] += error
        bucket["Total_C_to_E_Count"] += c_to_e
        bucket["PositiveSupportRepeatCount"] += int(cycle > 0)
        bucket["InputRows"] += 1
    expected_keys = {
        (method, METHODS[method].Code, stratum)
        for method in range(5)
        for stratum in WIDTH_STRATUM_ORDER
    }
    if set(grouped) != expected_keys:
        raise AssertionError("H300 pooled-count group universe is not 5 x 3")
    rows: list[dict[str, str]] = []
    for method in range(5):
        for stratum in WIDTH_STRATUM_ORDER:
            key = (method, METHODS[method].Code, stratum)
            values = grouped[key]
            if values["InputRows"] != 2000:
                raise AssertionError("H300 group is not supported by exactly 2000 Repeat rows")
            total = values["TotalCycleCount"]
            if total <= 0:
                raise AssertionError("H300 pooled-count group has zero total support")
            if values["TotalCorrectCount"] + values["TotalErrorCount"] != total:
                raise AssertionError("H300 pooled-count conservation failed")
            rows.append(
                {
                    "MethodOrdinal": str(method),
                    "MethodCode": METHODS[method].Code,
                    "WidthStratum": stratum,
                    "TotalCycleCount": str(total),
                    "TotalCorrectCount": str(values["TotalCorrectCount"]),
                    "TotalErrorCount": str(values["TotalErrorCount"]),
                    "Total_C_to_E_Count": str(values["Total_C_to_E_Count"]),
                    "PositiveSupportRepeatCount": str(
                        values["PositiveSupportRepeatCount"]
                    ),
                    "P_cor_stratum": repr(values["TotalCorrectCount"] / total),
                    "P_error_stratum": repr(values["TotalErrorCount"] / total),
                    "Estimator": "RATIO_OF_POOLED_COUNTS",
                    "SourceArtifact": H300_SOURCE_PATH,
                    "SourceArtifactSHA256": SOURCE_SHA256_BY_PATH[H300_SOURCE_PATH],
                }
            )
    return rows


def _build_registry_rows() -> dict[str, tuple[tuple[str, ...], list[dict[str, str]]]]:
    contracts_sha = sha256_file(ROOT / CONTRACTS_SOURCE_PATH)
    contract_sha = sha256_file(STAGE11_CONTRACT_PATH)
    conditions: list[dict[str, str]] = []
    for ordinal, condition in enumerate(CONDITIONS):
        if ordinal != condition.ConditionOrdinal:
            raise AssertionError("Condition registry ordinals are not canonical")
        conditions.append(
            {
                "ConditionOrdinal": str(ordinal),
                "ConditionCode": condition.Code,
                **_condition_metadata(ordinal),
                "SourceArtifact": CONTRACTS_SOURCE_PATH,
                "SourceArtifactSHA256": contracts_sha,
            }
        )
    methods = [
        {
            "MethodOrdinal": str(method.MethodOrdinal),
            "MethodCode": method.Code,
            "MethodNameZH": method.Name,
            "SourceArtifact": CONTRACTS_SOURCE_PATH,
            "SourceArtifactSHA256": contracts_sha,
        }
        for method in METHODS
    ]
    metrics = [
        {
            "MetricOrdinal": str(ordinal),
            "MetricCode": code,
            "PreferredDirection": direction,
            "SourceArtifact": STAGE11_CONTRACT_ARTIFACT,
            "SourceArtifactSHA256": contract_sha,
        }
        for ordinal, (code, direction) in enumerate(METRICS)
    ]
    contrasts = [
        {
            "ContrastOrdinal": str(ordinal),
            "ContrastCode": code,
            "ComparatorMethod": comparator,
            "SourceArtifact": STAGE11_CONTRACT_ARTIFACT,
            "SourceArtifactSHA256": contract_sha,
        }
        for ordinal, code, comparator in CONTRASTS
    ]
    return {
        CONDITION_REGISTRY_REL: (CONDITION_REGISTRY_FIELDS, conditions),
        METHOD_REGISTRY_REL: (METHOD_REGISTRY_FIELDS, methods),
        METRIC_REGISTRY_REL: (METRIC_REGISTRY_FIELDS, metrics),
        CONTRAST_REGISTRY_REL: (CONTRAST_REGISTRY_FIELDS, contrasts),
    }


def build_export_tables() -> dict[str, tuple[tuple[str, ...], list[dict[str, str]]]]:
    _validate_frozen_registries()
    individual = _build_individual_rows()
    paired = _build_paired_rows()
    precision, limitations = _build_precision_rows()
    tables: dict[str, tuple[tuple[str, ...], list[dict[str, str]]]] = {
        INDIVIDUAL_MASTER_REL: (INDIVIDUAL_FIELDS, individual),
        PAIRED_MASTER_REL: (PAIRED_FIELDS, paired),
        PRECISION_MASTER_REL: (PRECISION_FIELDS, precision),
        PRECISION_LIMITATIONS_REL: (PRECISION_FIELDS, limitations),
        NI_REL: (NI_FIELDS, _build_ni_rows()),
        HPRF_INDIVIDUAL_REL: (
            INDIVIDUAL_FIELDS,
            [dict(row) for row in individual if int(row["ConditionOrdinal"]) in HPRF_CONDITIONS],
        ),
        HPRF_PAIRED_REL: (
            PAIRED_FIELDS,
            [dict(row) for row in paired if int(row["ConditionOrdinal"]) in HPRF_CONDITIONS],
        ),
        F_ONLY_INDIVIDUAL_REL: (
            INDIVIDUAL_FIELDS,
            [dict(row) for row in individual if int(row["ConditionOrdinal"]) in F_ONLY_CONDITIONS],
        ),
        F_ONLY_PAIRED_REL: (
            PAIRED_FIELDS,
            [dict(row) for row in paired if int(row["ConditionOrdinal"]) in F_ONLY_CONDITIONS],
        ),
        COMPOSITE_INDIVIDUAL_REL: (
            INDIVIDUAL_FIELDS,
            [dict(row) for row in individual if int(row["ConditionOrdinal"]) in COMPOSITE_CONDITIONS],
        ),
        COMPOSITE_PAIRED_REL: (
            PAIRED_FIELDS,
            [dict(row) for row in paired if int(row["ConditionOrdinal"]) in COMPOSITE_CONDITIONS],
        ),
        H300_REL: (H300_FIELDS, _build_h300_rows()),
    }
    tables.update(_build_registry_rows())
    expected_counts = {
        INDIVIDUAL_MASTER_REL: 360,
        PAIRED_MASTER_REL: 144,
        PRECISION_MASTER_REL: 216,
        NI_REL: 5,
        HPRF_INDIVIDUAL_REL: 100,
        HPRF_PAIRED_REL: 40,
        F_ONLY_INDIVIDUAL_REL: 80,
        F_ONLY_PAIRED_REL: 32,
        COMPOSITE_INDIVIDUAL_REL: 180,
        COMPOSITE_PAIRED_REL: 72,
        H300_REL: 15,
        CONDITION_REGISTRY_REL: 19,
        METHOD_REGISTRY_REL: 5,
        METRIC_REGISTRY_REL: 4,
        CONTRAST_REGISTRY_REL: 2,
    }
    for relative, expected in expected_counts.items():
        actual = len(tables[relative][1])
        if actual != expected:
            raise AssertionError(
                f"Stage 11 table {relative} has {actual} rows, expected {expected}"
            )
    return tables


EVIDENCE_METADATA: dict[str, tuple[str, str, str, str]] = {
    INDIVIDUAL_MASTER_REL: (
        "MASTER_RESULT",
        INDIVIDUAL_SOURCE_PATH,
        "STAGE10_FROZEN_INFERENCE",
        "Graph-ready complete individual estimate and frozen CI mother table",
    ),
    PAIRED_MASTER_REL: (
        "MASTER_RESULT",
        PAIRED_SOURCE_PATH,
        "STAGE10_FROZEN_INFERENCE",
        "Graph-ready complete paired contrast and frozen CI mother table",
    ),
    PRECISION_MASTER_REL: (
        "PRECISION_AUDIT",
        PRECISION_SOURCE_PATH,
        "STAGE10_FROZEN_INFERENCE",
        "Complete frozen Stage 8 precision-target audit evidence",
    ),
    PRECISION_LIMITATIONS_REL: (
        "PRECISION_LIMITATION",
        f"artifacts/stage11/{PRECISION_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Exact unfiltered false subset for statistical-precision limitations",
    ),
    NI_REL: (
        "IDENTITY_RESULT",
        f"{FORMAL_AGGREGATED_SOURCE_PATH};{NI_REPORT_SOURCE_PATH}",
        "IDENTITY_OR_METADATA",
        "NI consistency evidence without inferential intervals",
    ),
    HPRF_INDIVIDUAL_REL: (
        "SCENARIO_SLICE",
        f"artifacts/stage11/{INDIVIDUAL_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Complete HPRF individual graph-ready slice",
    ),
    HPRF_PAIRED_REL: (
        "SCENARIO_SLICE",
        f"artifacts/stage11/{PAIRED_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Complete HPRF paired graph-ready slice",
    ),
    F_ONLY_INDIVIDUAL_REL: (
        "SCENARIO_SLICE",
        f"artifacts/stage11/{INDIVIDUAL_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Complete F-only individual graph-ready slice",
    ),
    F_ONLY_PAIRED_REL: (
        "SCENARIO_SLICE",
        f"artifacts/stage11/{PAIRED_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Complete F-only paired graph-ready slice",
    ),
    COMPOSITE_INDIVIDUAL_REL: (
        "SCENARIO_SLICE",
        f"artifacts/stage11/{INDIVIDUAL_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Complete composite individual graph-ready slice",
    ),
    COMPOSITE_PAIRED_REL: (
        "SCENARIO_SLICE",
        f"artifacts/stage11/{PAIRED_MASTER_REL}",
        "STAGE10_FROZEN_INFERENCE",
        "Complete composite paired graph-ready slice",
    ),
    H300_REL: (
        "DESCRIPTIVE_AGGREGATE",
        H300_SOURCE_PATH,
        "STAGE9_FROZEN_DESCRIPTIVE",
        "Pooled-count H300 width-stratum descriptive evidence only",
    ),
    CONDITION_REGISTRY_REL: (
        "METADATA_REGISTRY",
        CONTRACTS_SOURCE_PATH,
        "IDENTITY_OR_METADATA",
        "Condition identity and physical-parameter metadata",
    ),
    METHOD_REGISTRY_REL: (
        "METADATA_REGISTRY",
        CONTRACTS_SOURCE_PATH,
        "IDENTITY_OR_METADATA",
        "Method identity and Chinese display-name metadata",
    ),
    METRIC_REGISTRY_REL: (
        "METADATA_REGISTRY",
        STAGE11_CONTRACT_ARTIFACT,
        "IDENTITY_OR_METADATA",
        "Metric identity and predeclared preferred-direction metadata",
    ),
    CONTRAST_REGISTRY_REL: (
        "METADATA_REGISTRY",
        STAGE11_CONTRACT_ARTIFACT,
        "IDENTITY_OR_METADATA",
        "Frozen paired-contrast identity metadata",
    ),
}


def _evidence_registry_rows(
    output_dir: Path,
    tables: Mapping[str, tuple[Sequence[str], Sequence[Mapping[str, str]]]],
) -> list[dict[str, str]]:
    if set(tables) != set(EVIDENCE_METADATA):
        raise AssertionError("evidence table inventory and registry metadata differ")
    rows: list[dict[str, str]] = []
    for relative in sorted(tables):
        artifact_class, sources, inference_class, allowed_use = EVIDENCE_METADATA[
            relative
        ]
        if inference_class not in {
            "STAGE10_FROZEN_INFERENCE",
            "STAGE9_FROZEN_DESCRIPTIVE",
            "IDENTITY_OR_METADATA",
        }:
            raise AssertionError("invalid evidence InferenceClass")
        rows.append(
            {
                "ArtifactPath": f"artifacts/stage11/{relative}",
                "ArtifactClass": artifact_class,
                "RowCount": str(len(tables[relative][1])),
                "SHA256": sha256_file(Path(output_dir) / relative),
                "ImmediateSourceArtifacts": sources,
                "InferenceClass": inference_class,
                "AllowedUse": allowed_use,
                "SelectiveFilteringAllowed": "false",
            }
        )
    return rows


def build_evidence_exports(
    output_dir: Path = STAGE11_DIR,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = build_export_tables()
    for relative in sorted(tables):
        fields, rows = tables[relative]
        write_csv_atomic(destination / relative, fields, rows)
    registry_rows = _evidence_registry_rows(destination, tables)
    write_csv_atomic(
        destination / EVIDENCE_REGISTRY_REL,
        EVIDENCE_REGISTRY_FIELDS,
        registry_rows,
    )
    return {
        "EvidenceArtifactCount": len(tables),
        "EvidenceRegistryRows": len(registry_rows),
        "RowCounts": {
            relative: len(rows) for relative, (_, rows) in sorted(tables.items())
        },
        "SHA256": {
            relative: sha256_file(destination / relative)
            for relative in sorted((*tables.keys(), EVIDENCE_REGISTRY_REL))
        },
    }


def validate_evidence_exports(
    output_dir: Path = STAGE11_DIR,
) -> dict[str, object]:
    destination = Path(output_dir)
    expected_tables = build_export_tables()
    for relative, (fields, expected_rows) in expected_tables.items():
        actual_rows = read_csv_exact(
            destination / relative, fields, expected_rows=len(expected_rows)
        )
        if actual_rows != expected_rows:
            raise AssertionError(f"deterministic export reconstitution failed: {relative}")
    expected_registry = _evidence_registry_rows(destination, expected_tables)
    actual_registry = read_csv_exact(
        destination / EVIDENCE_REGISTRY_REL,
        EVIDENCE_REGISTRY_FIELDS,
        expected_rows=len(expected_registry),
    )
    if actual_registry != expected_registry:
        raise AssertionError("paper evidence registry does not match evidence artifacts")
    forbidden_labels = {
        "SIGNIFICANT",
        "NOT_SIGNIFICANT",
        "SUPERIOR",
        "INFERIOR",
        "WIN",
        "LOSE",
    }
    for relative, (_, rows) in expected_tables.items():
        for row in rows:
            if forbidden_labels.intersection(row.values()):
                raise AssertionError(f"forbidden conclusion label entered {relative}")
            if any("p_value" in field.lower() or "pvalue" in field.lower() for field in row):
                raise AssertionError(f"p-value field entered {relative}")
    paired = expected_tables[PAIRED_MASTER_REL][1]
    if any(
        row["CIPositionRelativeToZero"]
        not in {"POSITIVE", "NEGATIVE", "INCLUDES_ZERO", "UNDEFINED"}
        for row in paired
    ):
        raise AssertionError("paired CI-position labels are outside the frozen geometry set")
    h300 = expected_tables[H300_REL][1]
    for row in h300:
        if int(row["TotalCorrectCount"]) + int(row["TotalErrorCount"]) != int(
            row["TotalCycleCount"]
        ):
            raise AssertionError("H300 pooled-count conservation failed")
    return {
        "EvidenceArtifactCount": len(expected_tables),
        "EvidenceRegistryRows": len(actual_registry),
        "RowCounts": {
            relative: len(rows)
            for relative, (_, rows) in sorted(expected_tables.items())
        },
        "PrecisionLimitationRows": len(
            expected_tables[PRECISION_LIMITATIONS_REL][1]
        ),
        "SourceInferenceValuesPreserved": True,
        "CIPositionLabelsVerified": True,
        "H300PooledCountConservationPassed": True,
        "EvidenceRegistryVerified": True,
        "AllScenarioConditionsExported": True,
    }


__all__ = [
    "COMPOSITE_CONDITIONS",
    "CONTRASTS",
    "EVIDENCE_METADATA",
    "EVIDENCE_REGISTRY_FIELDS",
    "EVIDENCE_REGISTRY_REL",
    "FROZEN_SOURCE_SPECS",
    "F_ONLY_CONDITIONS",
    "HPRF_CONDITIONS",
    "INDIVIDUAL_FIELDS",
    "INDIVIDUAL_MASTER_REL",
    "INDIVIDUAL_SOURCE_FIELDS",
    "METRICS",
    "PAIRED_FIELDS",
    "PAIRED_MASTER_REL",
    "PAIRED_SOURCE_FIELDS",
    "PRECISION_FIELDS",
    "PRECISION_LIMITATIONS_REL",
    "PRECISION_MASTER_REL",
    "STAGE11_CONTRACT_PATH",
    "STAGE11_DIR",
    "WIDTH_STRATUM_ORDER",
    "build_evidence_exports",
    "build_export_tables",
    "ci_position_relative_to_zero",
    "csv_row_count",
    "read_csv_exact",
    "sha256_file",
    "validate_evidence_exports",
    "write_csv_atomic",
]
