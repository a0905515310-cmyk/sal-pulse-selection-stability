from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from .build import ROOT
from .results_export import (
    COMPOSITE_INDIVIDUAL_REL,
    COMPOSITE_PAIRED_REL,
    CONDITION_REGISTRY_REL,
    CONTRAST_REGISTRY_REL,
    EVIDENCE_REGISTRY_REL,
    FROZEN_SOURCE_SPECS,
    F_ONLY_INDIVIDUAL_REL,
    F_ONLY_PAIRED_REL,
    H300_REL,
    HPRF_INDIVIDUAL_REL,
    HPRF_PAIRED_REL,
    INDIVIDUAL_MASTER_REL,
    METHOD_REGISTRY_REL,
    METRIC_REGISTRY_REL,
    NI_REL,
    PAIRED_MASTER_REL,
    PRECISION_LIMITATIONS_REL,
    PRECISION_MASTER_REL,
    STAGE11_CONTRACT_PATH,
    STAGE11_DIR,
    build_evidence_exports,
    csv_row_count,
    sha256_file,
    validate_evidence_exports,
)


STAGE_NUMBER = 11
STAGE_NAME = "RESULTS_EXPORT_AND_PAPER_EVIDENCE_FREEZE"
STAGE10_BASELINE_ZIP_NAME = "sal_stability_stage10_20260825.zip"
STAGE10_BASELINE_ZIP_SHA256 = (
    "fb3666b9a9cd74d2a52978df98b2e4f08b7ddcb11bff0134bbdaed6cc09fb389"
)
STAGE11_SCIENCE_CONTRACT_SHA256 = (
    "9fc2141695a6e73697bbee2e7590eb54678be5ca094ae06e753a36a38f3282a1"
)
STAGE11_TASKBOOK_SHA256 = (
    "645bb81479d813cc804dd2a47cb8ff48f6db0bb69187589dccddbcbbf61e8938"
)
FINAL_ZIP_NAME = "sal_stability_stage11_20260826.zip"
FINAL_ZIP_ROOT = "sal_stability_stage11_20260826"

SOURCE_IDENTITY_PATH = STAGE11_DIR / "source_identity.json"
MANIFEST_PATH = STAGE11_DIR / "stage11_manifest.json"
STATUS_PATH = STAGE11_DIR / "status.json"
VALIDATION_REPORT_PATH = STAGE11_DIR / "validation_report.md"
TEST_REPORT_PATH = STAGE11_DIR / "test_report.txt"
CHANGED_FILES_PATH = STAGE11_DIR / "changed_files.txt"
STAGE10_STATUS_PATH = ROOT / "artifacts/stage10/status.json"

ALLOWED_STATIC_ADDITIONS = {
    "STAGE11_TASKBOOK.md",
    "scripts/build_stage11.py",
    "src/sal_stability_stage1/results_export.py",
    "src/sal_stability_stage1/stage11.py",
    "tests/test_stage11_contract.py",
    "tests/test_stage11_export.py",
    "tests/test_stage11_storage.py",
    "tests/test_stage11_execution.py",
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
LEGACY_STAGE10_NODE = (
    "tests/test_stage10_contract.py::"
    "test_stage9_pass_and_stage1_through_stage9_zip_tree_identity"
)
LEGACY_NODES = (LEGACY_STAGE8_NODE, LEGACY_STAGE9_NODE, LEGACY_STAGE10_NODE)

STAGE12_EXECUTED = False

REQUIRED_EVIDENCE_ROW_COUNTS: dict[str, int] = {
    INDIVIDUAL_MASTER_REL: 360,
    PAIRED_MASTER_REL: 144,
    PRECISION_MASTER_REL: 216,
    HPRF_INDIVIDUAL_REL: 100,
    HPRF_PAIRED_REL: 40,
    F_ONLY_INDIVIDUAL_REL: 80,
    F_ONLY_PAIRED_REL: 32,
    COMPOSITE_INDIVIDUAL_REL: 180,
    COMPOSITE_PAIRED_REL: 72,
    NI_REL: 5,
    H300_REL: 15,
    CONDITION_REGISTRY_REL: 19,
    METHOD_REGISTRY_REL: 5,
    METRIC_REGISTRY_REL: 4,
    CONTRAST_REGISTRY_REL: 2,
}


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_sha(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def verify_stage10_baseline_zip(path: Path | None = None) -> str:
    target = (
        Path(path)
        if path is not None
        else ROOT.parent / STAGE10_BASELINE_ZIP_NAME
    )
    return _verify_sha(
        target, STAGE10_BASELINE_ZIP_SHA256, "Stage 10 unique active baseline ZIP"
    )


def validate_stage10_status(path: Path = STAGE10_STATUS_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "stage": 10,
        "stage_name": "BOOTSTRAP_B2000_PAIRED_REPEAT",
        "status": "PASS",
        "blocking_issue_count": 0,
        "bootstrap_executed": True,
        "stage11_executed": False,
        "condition_dropped": False,
        "parameter_changed": False,
        "formal_rerun_executed": False,
        "p_value_generated": False,
        "physical_simulation_executed": False,
        "paper_figure_generated": False,
        "paper_result_conclusion_generated": False,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise AssertionError(
                f"Stage 10 status field {field} drifted: expected {expected!r}, got {payload.get(field)!r}"
            )
    return payload


def validate_frozen_source_hashes() -> dict[str, str]:
    actual: dict[str, str] = {}
    for identity_name, (relative, expected) in FROZEN_SOURCE_SPECS.items():
        actual[identity_name] = _verify_sha(
            ROOT / relative, expected, f"frozen source {relative}"
        )
    return actual


def validate_science_contract(
    path: Path = STAGE11_CONTRACT_PATH,
) -> dict[str, object]:
    _verify_sha(
        path, STAGE11_SCIENCE_CONTRACT_SHA256, "Stage 11 frozen science contract"
    )
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    hard_values = {
        "ContractName": "STAGE11_RESULTS_EXPORT_AND_PAPER_EVIDENCE_FREEZE",
        "ContractVersion": 1,
        "FinalCodeStage": True,
        "FinalZipName": FINAL_ZIP_NAME,
        "Stage10BaselineZipName": STAGE10_BASELINE_ZIP_NAME,
        "Stage10BaselineZipSHA256": STAGE10_BASELINE_ZIP_SHA256,
        "NoFormalRerun": True,
        "NoNewBootstrap": True,
        "NoNewCI": True,
        "NoPValues": True,
        "NoNewContrasts": True,
        "NoConditionDropping": True,
        "NoParameterChange": True,
        "NoPhysicalSimulation": True,
        "NoNewRandomness": True,
        "NoPaperFigureRendering": True,
        "NoPaperConclusionWriting": True,
        "NoResultDrivenSelection": True,
        "NoStage12Execution": True,
        "StopAfterStage11": True,
    }
    for field, expected in hard_values.items():
        if contract.get(field) != expected:
            raise AssertionError(f"Stage 11 contract field {field} drifted")
    expected_source_hashes = {
        identity_name: expected
        for identity_name, (_, expected) in FROZEN_SOURCE_SPECS.items()
    }
    if contract.get("FrozenSourceSHA256") != expected_source_hashes:
        raise AssertionError("Stage 11 contract frozen-source identities drifted")
    allowed = contract.get("AllowedCodeChanges", {})
    if set(allowed.get("AddedStaticOrCode", [])) != ALLOWED_STATIC_ADDITIONS:
        raise AssertionError("Stage 11 allowed static/code additions drifted")
    if set(allowed.get("ModifiedExisting", [])) != ALLOWED_MODIFIED_FILES:
        raise AssertionError("Stage 11 allowed modified files drifted")
    if allowed.get("AddedArtifactsPrefix") != "artifacts/stage11/":
        raise AssertionError("Stage 11 artifact addition prefix drifted")
    if contract.get("Conditions") != {
        "NI": [0],
        "HPRF": [1, 2, 3, 4, 5],
        "F_ONLY": [6, 7, 8, 9],
        "COMPOSITE": list(range(10, 19)),
    }:
        raise AssertionError("Stage 11 Condition universe drifted")
    if [item.get("Code") for item in contract.get("Methods", [])] != [
        "FIRST",
        "LAST",
        "T",
        "W",
        "TW",
    ]:
        raise AssertionError("Stage 11 Method universe drifted")
    if [item.get("Code") for item in contract.get("Metrics", [])] != [
        "P_cor",
        "P_C_given_C",
        "P_C_given_E",
        "Mean_L_NC",
    ]:
        raise AssertionError("Stage 11 Metric universe drifted")
    if contract.get("Contrasts") != [
        {"Code": "TW-T", "Comparator": "T", "Ordinal": 0},
        {"Code": "TW-W", "Comparator": "W", "Ordinal": 1},
    ]:
        raise AssertionError("Stage 11 Contrast universe drifted")
    h300 = contract.get("H300WidthStrataDescriptiveAggregation", {})
    if (
        h300.get("InputRows") != 30_000
        or h300.get("OutputRows") != 15
        or h300.get("Estimator") != "RATIO_OF_POOLED_COUNTS"
        or h300.get("NoBootstrap") is not True
        or h300.get("NoCI") is not True
    ):
        raise AssertionError("Stage 11 H300 descriptive aggregation contract drifted")
    return contract


def _excluded_path(path: Path) -> bool:
    return (
        any(
            part in {"__pycache__", ".pytest_cache", ".git", ".venv"}
            or part.startswith(".venv-")
            for part in path.parts
        )
        or path.suffix.lower() == ".pyc"
        or path.name.endswith(".tmp")
        or ".tmp-" in path.name
    )


def _baseline_archive_members(path: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    roots: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                raise AssertionError("Stage 10 baseline ZIP contains a root-level file")
            roots.add(parts[0])
            relative = Path(*parts[1:])
            if _excluded_path(relative):
                continue
            key = relative.as_posix()
            if key in members:
                raise AssertionError(f"duplicate Stage 10 archive member: {key}")
            members[key] = hashlib.sha256(archive.read(info)).hexdigest()
    if len(roots) != 1:
        raise AssertionError("Stage 10 baseline ZIP must contain one project root")
    return members


def _tree_fingerprint(members: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(members.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b":")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _allowed_stage11_addition(relative: str) -> bool:
    return relative in ALLOWED_STATIC_ADDITIONS or relative.startswith(
        "artifacts/stage11/"
    )


def validate_stage1_through_stage10_tree_unchanged(
    baseline_zip: Path | None = None,
) -> dict[str, object]:
    baseline_path = (
        Path(baseline_zip)
        if baseline_zip is not None
        else ROOT.parent / STAGE10_BASELINE_ZIP_NAME
    )
    verify_stage10_baseline_zip(baseline_path)
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
    unexpected_added = [item for item in added if not _allowed_stage11_addition(item)]
    unexpected_modified = [
        item for item in modified if item not in ALLOWED_MODIFIED_FILES
    ]
    if unexpected_added or unexpected_modified or deleted:
        raise AssertionError(
            "Stage 1-10 byte-identity boundary failed: "
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
    missing = [
        relative
        for relative in sorted(ALLOWED_STATIC_ADDITIONS)
        if not (ROOT / relative).is_file()
    ]
    if missing:
        raise AssertionError(f"Stage 11 required static files are missing: {missing}")
    _verify_sha(
        ROOT / "STAGE11_TASKBOOK.md",
        STAGE11_TASKBOOK_SHA256,
        "Stage 11 frozen taskbook",
    )
    _verify_sha(
        STAGE11_CONTRACT_PATH,
        STAGE11_SCIENCE_CONTRACT_SHA256,
        "Stage 11 frozen science contract",
    )


def stage11_code_sha256() -> str:
    digest = hashlib.sha256()
    for relative in sorted(ALLOWED_STATIC_ADDITIONS | ALLOWED_MODIFIED_FILES):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Stage 11 code identity file missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b":")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_no_forbidden_execution_path() -> dict[str, object]:
    source_paths = (
        ROOT / "scripts/build_stage11.py",
        ROOT / "src/sal_stability_stage1/results_export.py",
        ROOT / "src/sal_stability_stage1/stage11.py",
    )
    forbidden_modules = {
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "plotly",
        "openpyxl",
    }
    forbidden_call_leaves = {
        "bootstrap_repeat_indices",
        "build_stage9",
        "build_stage10",
        "choice",
        "default_rng",
        "generate_physical_events",
        "percentile",
        "percentile_interval",
        "quantile",
        "rand",
        "randn",
        "random",
        "run_condition_bootstrap",
        "run_condition_repeats_parallel",
        "run_condition_repeats_serial",
        "run_formal_condition_chunked",
        "seed",
    }
    offending_imports: list[str] = []
    offending_calls: list[str] = []
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_modules:
                        offending_imports.append(f"{path.name}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = (node.module or "").split(".")[0]
                if module in forbidden_modules:
                    offending_imports.append(f"{path.name}:{node.lineno}:{node.module}")
            elif isinstance(node, ast.Call):
                function = node.func
                if isinstance(function, ast.Name):
                    leaf = function.id
                elif isinstance(function, ast.Attribute):
                    leaf = function.attr
                else:
                    continue
                if leaf in forbidden_call_leaves:
                    offending_calls.append(f"{path.name}:{node.lineno}:{leaf}")
    if offending_imports or offending_calls:
        raise AssertionError(
            "forbidden Stage 11 execution path detected: "
            f"imports={offending_imports}, calls={offending_calls}"
        )
    return {
        "BootstrapExecutionCallAbsent": True,
        "RNGCallAbsent": True,
        "NewCICallAbsent": True,
        "PValueCallAbsent": True,
        "PhysicalSimulationCallAbsent": True,
        "FormalRerunCallAbsent": True,
        "PlottingImportAbsent": True,
        "Stage12CallAbsent": True,
    }


def _pytest_result(command: Sequence[str]) -> dict[str, object]:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
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
        "skipped": latest(r"([0-9]+) skipped"),
        "failed_nodes": failed_nodes,
        "output": output,
    }


def run_stage11_tests(*, include_storage: bool = True) -> dict[str, object]:
    files = [
        "tests/test_stage11_contract.py",
        "tests/test_stage11_export.py",
        "tests/test_stage11_execution.py",
    ]
    if include_storage:
        files.append("tests/test_stage11_storage.py")
    result = _pytest_result((sys.executable, "-m", "pytest", *files))
    if (
        result["returncode"] != 0
        or result["passed"] <= 0
        or result["failed"] != 0
        or result["errors"] != 0
    ):
        raise AssertionError(f"Stage 11 tests failed\n{result['output']}")
    return result


def run_full_pytest_with_legacy_rules() -> dict[str, object]:
    result = _pytest_result((sys.executable, "-m", "pytest"))
    if (
        result["returncode"] == 0
        or result["failed"] != 3
        or result["errors"] != 0
        or set(result["failed_nodes"]) != set(LEGACY_NODES)
    ):
        raise AssertionError(
            "complete pytest did not match exactly the three authorized forward-boundary failures: "
            f"returncode={result['returncode']}, failed={result['failed']}, "
            f"errors={result['errors']}, nodes={result['failed_nodes']}\n{result['output']}"
        )
    return {
        **result,
        "legacy_boundary_exceptions": legacy_boundary_exceptions(),
        "unexpected_failure_count": 0,
    }


def legacy_boundary_exceptions() -> list[dict[str, object]]:
    return [
        {
            "Code": "LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE",
            "TestNode": LEGACY_STAGE8_NODE,
            "Allowed": True,
            "FailureDifferenceScope": "taskbook-authorized post-Stage7 additions only",
            "FrozenFilesModified": False,
            "DeletedFiles": [],
        },
        {
            "Code": "LEGACY_STAGE9_BOUNDARY_NOT_FORWARD_COMPATIBLE",
            "TestNode": LEGACY_STAGE9_NODE,
            "Allowed": True,
            "FailureDifferenceScope": "taskbook-authorized post-Stage8 additions only",
            "FrozenFilesModified": False,
            "DeletedFiles": [],
        },
        {
            "Code": "LEGACY_STAGE10_BOUNDARY_NOT_FORWARD_COMPATIBLE",
            "TestNode": LEGACY_STAGE10_NODE,
            "Allowed": True,
            "FailureDifferenceScope": "taskbook-authorized Stage11 additions only",
            "FrozenFilesModified": False,
            "DeletedFiles": [],
        },
    ]


def _source_identity_payload(
    source_hashes: Mapping[str, str], tree: Mapping[str, object]
) -> dict[str, object]:
    return {
        "IdentitySchema": "STAGE11_SOURCE_IDENTITY_V1",
        "Stage": STAGE_NUMBER,
        "Stage10BaselineZipName": STAGE10_BASELINE_ZIP_NAME,
        "Stage10BaselineZipSHA256": STAGE10_BASELINE_ZIP_SHA256,
        "Stage10BaselineZipSHA256Verified": True,
        "Stage10StatusPASSVerified": True,
        "FrozenSourceSHA256": dict(source_hashes),
        "AllFrozenSourceSHA256Verified": True,
        "Stage11ScienceContractSHA256": STAGE11_SCIENCE_CONTRACT_SHA256,
        "Stage11ScienceContractSHA256Verified": True,
        "BaselineMemberCount": tree["BaselineMemberCount"],
        "BaselineTreeSHA256": tree["BaselineTreeSHA256"],
        "Stage1ThroughStage10TreeUnchanged": True,
        "ReadBoundary": "READ_ONLY_DETERMINISTIC_EXPORT",
        "InferenceSourcesRecomputed": False,
    }


def _status_payload(
    validation: Mapping[str, object],
    stage11_tests: Mapping[str, object],
    historical_tests: Mapping[str, object],
) -> dict[str, object]:
    counts = validation["RowCounts"]
    return {
        "stage": STAGE_NUMBER,
        "stage_name": STAGE_NAME,
        "status": "PASS",
        "final_code_stage": True,
        "stage10_baseline_zip_sha256_verified": True,
        "stage10_status_pass_verified": True,
        "all_frozen_source_sha256_verified": True,
        "stage11_science_contract_sha256_verified": True,
        "stage1_through_stage10_tree_unchanged": True,
        "individual_master_rows": counts[INDIVIDUAL_MASTER_REL],
        "paired_master_rows": counts[PAIRED_MASTER_REL],
        "precision_audit_master_rows": counts[PRECISION_MASTER_REL],
        "precision_limitation_rows": validation["PrecisionLimitationRows"],
        "ni_rows": counts[NI_REL],
        "hprf_individual_rows": counts[HPRF_INDIVIDUAL_REL],
        "hprf_paired_rows": counts[HPRF_PAIRED_REL],
        "f_only_individual_rows": counts[F_ONLY_INDIVIDUAL_REL],
        "f_only_paired_rows": counts[F_ONLY_PAIRED_REL],
        "composite_individual_rows": counts[COMPOSITE_INDIVIDUAL_REL],
        "composite_paired_rows": counts[COMPOSITE_PAIRED_REL],
        "h300_width_strata_summary_rows": counts[H300_REL],
        "all_scenario_conditions_exported": True,
        "source_inference_values_preserved": True,
        "ci_position_labels_verified": True,
        "h300_pooled_count_conservation_passed": True,
        "evidence_registry_verified": True,
        "evidence_registry_rows": validation["EvidenceRegistryRows"],
        "stage11_tests_passed": stage11_tests["passed"],
        "stage11_tests_failed": 0,
        "stage11_tests_errors": 0,
        "full_pytest_raw_passed": historical_tests["passed"],
        "full_pytest_raw_failed": historical_tests["failed"],
        "full_pytest_raw_errors": historical_tests["errors"],
        "unexpected_pytest_failure_count": 0,
        "new_bootstrap_executed": False,
        "rng_executed": False,
        "new_ci_generated": False,
        "p_value_generated": False,
        "new_contrast_added": False,
        "condition_dropped": False,
        "parameter_changed": False,
        "formal_rerun_executed": False,
        "physical_simulation_executed": False,
        "paper_figure_generated": False,
        "paper_result_conclusion_generated": False,
        "result_driven_selection_used": False,
        "stage12_executed": False,
        "legacy_boundary_exceptions": legacy_boundary_exceptions(),
        "blocking_issue_count": 0,
        "stop_after_stage11": True,
    }


def _write_test_report(
    preflight: Mapping[str, object],
    stage11_tests: Mapping[str, object],
    historical_tests: Mapping[str, object],
) -> None:
    lines = [
        "STAGE 11 TEST REPORT",
        "",
        "PRE-PACKAGE STAGE11 CONTRACT/EXPORT/EXECUTION TESTS",
        f"command: {preflight['command']}",
        f"result: {preflight['passed']} passed, 0 failed, 0 errors",
        "",
        "PACKAGED STAGE11 TESTS",
        f"command: {stage11_tests['command']}",
        f"result: {stage11_tests['passed']} passed, 0 failed, 0 errors",
        "",
        "FULL HISTORICAL PYTEST RAW",
        f"command: {historical_tests['command']}",
        (
            "result: "
            f"{historical_tests['passed']} passed, {historical_tests['failed']} authorized legacy failures, "
            f"{historical_tests['errors']} errors"
        ),
        "authorized failure nodes:",
        *(f"- {node}" for node in historical_tests["failed_nodes"]),
        "unexpected failure count: 0",
        "",
        "The final ZIP is re-created after this report is frozen and is then checked by a final unrecorded Stage11-only validation run to avoid mutating the tested package.",
        "",
    ]
    write_text_atomic(TEST_REPORT_PATH, "\n".join(lines))


def _write_validation_report(
    validation: Mapping[str, object], tree: Mapping[str, object]
) -> None:
    counts = validation["RowCounts"]
    lines = [
        "# Stage 11 Validation Report",
        "",
        "Status: PASS",
        "",
        "This report records identity, completeness, and deterministic-export gates only. It does not state a paper result conclusion.",
        "",
        "## Identity gates",
        "",
        f"- Stage 10 baseline ZIP SHA256: `{STAGE10_BASELINE_ZIP_SHA256}`",
        f"- Stage 10 baseline archive members: {tree['BaselineMemberCount']}",
        f"- Stage 10 baseline tree SHA256: `{tree['BaselineTreeSHA256']}`",
        f"- Stage 11 science contract SHA256: `{STAGE11_SCIENCE_CONTRACT_SHA256}`",
        "- All twelve frozen source SHA256 gates: PASS",
        "- Stage 1-10 tree boundary: PASS",
        "",
        "## Evidence row-count gates",
        "",
        *(f"- `{relative}`: {count}" for relative, count in sorted(counts.items())),
        f"- `{PRECISION_LIMITATIONS_REL}`: {validation['PrecisionLimitationRows']}",
        f"- `{EVIDENCE_REGISTRY_REL}`: {validation['EvidenceRegistryRows']}",
        "",
        "## Prohibited-execution gates",
        "",
        "- New Bootstrap, RNG, CI, p-value, contrast, physical simulation, and Formal rerun: absent",
        "- Condition dropping, parameter change, and result-driven selection: absent",
        "- Paper figure rendering and paper result conclusion writing: absent",
        "- Stage 12: absent",
        "",
        "STOP after Stage 11.",
        "",
    ]
    write_text_atomic(VALIDATION_REPORT_PATH, "\n".join(lines))


def _write_changed_files(tree: Mapping[str, object]) -> None:
    lines = ["MODIFIED"]
    lines.extend(f"{relative}" for relative in tree["AllowedModifiedFiles"])
    lines.append("")
    lines.append("ADDED")
    lines.extend(f"{relative}" for relative in tree["AddedFiles"])
    lines.append("")
    lines.append("DELETED")
    lines.extend(f"{relative}" for relative in tree["DeletedFiles"])
    lines.append("")
    write_text_atomic(CHANGED_FILES_PATH, "\n".join(lines))


def _stage11_artifact_hashes() -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for path in STAGE11_DIR.rglob("*"):
        if not path.is_file() or path == MANIFEST_PATH:
            continue
        relative = path.relative_to(ROOT).as_posix()
        artifacts[relative] = sha256_file(path)
    return dict(sorted(artifacts.items()))


def _manifest_payload(
    source_hashes: Mapping[str, str],
    tree: Mapping[str, object],
    validation: Mapping[str, object],
    execution_gate: Mapping[str, object],
    stage11_tests: Mapping[str, object],
    historical_tests: Mapping[str, object],
) -> dict[str, object]:
    evidence_hashes = {
        relative: sha256_file(STAGE11_DIR / relative)
        for relative in sorted(
            (*validation["RowCounts"].keys(), EVIDENCE_REGISTRY_REL)
        )
    }
    return {
        "ManifestSchema": "STAGE11_EVIDENCE_FREEZE_MANIFEST_V1",
        "Stage": STAGE_NUMBER,
        "StageName": STAGE_NAME,
        "Status": "PASS",
        "FinalCodeStage": True,
        "Stage10BaselineZipName": STAGE10_BASELINE_ZIP_NAME,
        "Stage10BaselineZipSHA256": STAGE10_BASELINE_ZIP_SHA256,
        "Stage10BaselineTreeSHA256": tree["BaselineTreeSHA256"],
        "Stage10BaselineMemberCount": tree["BaselineMemberCount"],
        "Stage11ScienceContractSHA256": STAGE11_SCIENCE_CONTRACT_SHA256,
        "Stage11TaskbookSHA256": STAGE11_TASKBOOK_SHA256,
        "Stage11CodeSHA256": stage11_code_sha256(),
        "FrozenSourceSHA256": dict(source_hashes),
        "EvidenceRowCounts": dict(validation["RowCounts"]),
        "PrecisionLimitationRows": validation["PrecisionLimitationRows"],
        "EvidenceRegistryRows": validation["EvidenceRegistryRows"],
        "EvidenceArtifactSHA256": evidence_hashes,
        "PaperEvidenceRegistrySHA256": evidence_hashes[EVIDENCE_REGISTRY_REL],
        "ArtifactSHA256": _stage11_artifact_hashes(),
        "ExecutionPathGate": dict(execution_gate),
        "Stage11Tests": {
            "Passed": stage11_tests["passed"],
            "Failed": 0,
            "Errors": 0,
        },
        "FullHistoricalPytest": {
            "Passed": historical_tests["passed"],
            "AuthorizedLegacyFailures": historical_tests["failed"],
            "Errors": historical_tests["errors"],
            "UnexpectedFailures": 0,
            "FailureNodes": list(historical_tests["failed_nodes"]),
        },
        "LegacyBoundaryExceptions": legacy_boundary_exceptions(),
        "NewBootstrapExecuted": False,
        "RNGExecuted": False,
        "NewCIGenerated": False,
        "PValueGenerated": False,
        "NewContrastAdded": False,
        "ConditionDropped": False,
        "ParameterChanged": False,
        "FormalRerunExecuted": False,
        "PhysicalSimulationExecuted": False,
        "PaperFigureGenerated": False,
        "PaperResultConclusionGenerated": False,
        "ResultDrivenSelectionUsed": False,
        "Stage12Executed": False,
        "StopAfterStage11": True,
    }


def verify_stage11_manifest(path: Path = MANIFEST_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if Path(path).read_bytes() != canonical_json_bytes(payload):
        raise AssertionError("Stage 11 manifest is not canonical JSON")
    if payload.get("Stage") != 11 or payload.get("Status") != "PASS":
        raise AssertionError("Stage 11 manifest stage/status drifted")
    artifact_hashes = payload.get("ArtifactSHA256")
    if not isinstance(artifact_hashes, dict):
        raise AssertionError("Stage 11 manifest ArtifactSHA256 is not an object")
    actual_inventory = _stage11_artifact_hashes()
    if artifact_hashes != actual_inventory:
        raise AssertionError("Stage 11 manifest artifact inventory/hash drifted")
    evidence_hashes = payload.get("EvidenceArtifactSHA256")
    if not isinstance(evidence_hashes, dict):
        raise AssertionError("Stage 11 manifest evidence hashes are not an object")
    for relative, expected in evidence_hashes.items():
        if sha256_file(STAGE11_DIR / relative) != expected:
            raise AssertionError(f"Stage 11 evidence SHA mismatch: {relative}")
    if payload.get("PaperEvidenceRegistrySHA256") != sha256_file(
        STAGE11_DIR / EVIDENCE_REGISTRY_REL
    ):
        raise AssertionError("Stage 11 registry SHA is not frozen in the manifest")
    return payload


def validate_forbidden_artifact_absence(root: Path = ROOT) -> None:
    forbidden_stage11_suffixes = {".png", ".pdf", ".svg", ".xlsx", ".xls"}
    for path in (Path(root) / "artifacts/stage11").rglob("*"):
        if path.is_file() and path.suffix.lower() in forbidden_stage11_suffixes:
            raise AssertionError(f"forbidden Stage 11 rendered/figure artifact: {path}")
    for path in Path(root).rglob("*"):
        if any(part.lower() == "stage12" for part in path.relative_to(root).parts):
            raise AssertionError(f"Stage 12 artifact/code exists: {path}")


def _package_members() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not _excluded_path(path.relative_to(ROOT))
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def validate_package_prerequisites() -> dict[str, object]:
    verify_stage10_baseline_zip()
    validate_stage10_status()
    validate_frozen_source_hashes()
    validate_science_contract()
    validate_required_static_files()
    tree = validate_stage1_through_stage10_tree_unchanged()
    assert_no_forbidden_execution_path()
    validation = validate_evidence_exports()
    manifest = verify_stage11_manifest()
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("blocking_issue_count") != 0:
        raise AssertionError("Stage 11 status is not package-ready PASS")
    if manifest.get("StopAfterStage11") is not True:
        raise AssertionError("Stage 11 manifest does not stop after Stage 11")
    validate_forbidden_artifact_absence()
    return {"tree": tree, "validation": validation, "manifest": manifest}


def package_stage11(output_path: Path | None = None) -> tuple[Path, str]:
    validate_package_prerequisites()
    destination = (
        Path(output_path) if output_path is not None else ROOT.parent / FINAL_ZIP_NAME
    ).resolve()
    if destination.parent != ROOT.parent.resolve():
        raise ValueError("Stage 11 package destination must be the project parent")
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
    validate_final_zip(destination)
    return destination, sha256_file(destination)


def _zip_csv_row_count(archive: zipfile.ZipFile, member: str) -> int:
    text = archive.read(member).decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    next(reader, None)
    return sum(1 for _ in reader)


def validate_final_zip(path: Path | None = None) -> dict[str, object]:
    target = Path(path) if path is not None else ROOT.parent / FINAL_ZIP_NAME
    if not target.is_file():
        raise FileNotFoundError(f"Stage 11 final ZIP is missing: {target}")
    current_members = _package_members()
    expected_names = {
        (Path(FINAL_ZIP_ROOT) / member.relative_to(ROOT)).as_posix()
        for member in current_members
    }
    baseline = _baseline_archive_members(ROOT.parent / STAGE10_BASELINE_ZIP_NAME)
    with zipfile.ZipFile(target) as archive:
        names = {name for name in archive.namelist() if not name.endswith("/")}
        if names != expected_names:
            raise AssertionError(
                "Stage 11 ZIP inventory mismatch: "
                f"missing={sorted(expected_names - names)}, extra={sorted(names - expected_names)}"
            )
        for relative, expected_sha in baseline.items():
            if relative in ALLOWED_MODIFIED_FILES:
                continue
            member = (Path(FINAL_ZIP_ROOT) / relative).as_posix()
            actual_sha = hashlib.sha256(archive.read(member)).hexdigest()
            if actual_sha != expected_sha:
                raise AssertionError(f"Stage 10 member changed inside final ZIP: {relative}")
        contract_member = (
            Path(FINAL_ZIP_ROOT) / "artifacts/stage11/stage11_science_contract.json"
        ).as_posix()
        if hashlib.sha256(archive.read(contract_member)).hexdigest() != (
            STAGE11_SCIENCE_CONTRACT_SHA256
        ):
            raise AssertionError("Stage 11 science contract SHA drifted inside ZIP")
        manifest_member = (
            Path(FINAL_ZIP_ROOT) / "artifacts/stage11/stage11_manifest.json"
        ).as_posix()
        manifest = json.loads(archive.read(manifest_member).decode("utf-8"))
        for relative, expected_sha in manifest["ArtifactSHA256"].items():
            member = (Path(FINAL_ZIP_ROOT) / relative).as_posix()
            actual_sha = hashlib.sha256(archive.read(member)).hexdigest()
            if actual_sha != expected_sha:
                raise AssertionError(f"manifest SHA mismatch inside ZIP: {relative}")
        for relative, expected_rows in REQUIRED_EVIDENCE_ROW_COUNTS.items():
            member = (
                Path(FINAL_ZIP_ROOT) / "artifacts/stage11" / relative
            ).as_posix()
            actual_rows = _zip_csv_row_count(archive, member)
            if actual_rows != expected_rows:
                raise AssertionError(
                    f"ZIP row count mismatch for {relative}: {actual_rows}"
                )
        limitation_member = (
            Path(FINAL_ZIP_ROOT)
            / "artifacts/stage11"
            / PRECISION_LIMITATIONS_REL
        ).as_posix()
        if _zip_csv_row_count(archive, limitation_member) != manifest[
            "PrecisionLimitationRows"
        ]:
            raise AssertionError("ZIP precision-limitation row count drifted")
        registry_member = (
            Path(FINAL_ZIP_ROOT) / "artifacts/stage11" / EVIDENCE_REGISTRY_REL
        ).as_posix()
        if _zip_csv_row_count(archive, registry_member) != manifest[
            "EvidenceRegistryRows"
        ]:
            raise AssertionError("ZIP evidence-registry row count drifted")
        if any("/artifacts/stage12/" in name.lower() for name in names):
            raise AssertionError("Stage 12 artifact entered the final ZIP")
        stage11_prefix = f"{FINAL_ZIP_ROOT}/artifacts/stage11/"
        forbidden_suffixes = (".png", ".pdf", ".svg", ".xlsx", ".xls")
        if any(
            name.startswith(stage11_prefix) and name.lower().endswith(forbidden_suffixes)
            for name in names
        ):
            raise AssertionError("rendered figure/document artifact entered Stage 11 ZIP")
    return {
        "ZipMemberCount": len(expected_names),
        "Stage10BaselineMembersVerified": len(baseline) - len(ALLOWED_MODIFIED_FILES),
        "RequiredEvidenceRowCountsVerified": True,
        "ArtifactSHA256Verified": True,
        "Stage12Absent": True,
        "ForbiddenRenderedArtifactsAbsent": True,
    }


def _provisional_test_result(label: str) -> dict[str, object]:
    return {
        "command": label,
        "returncode": 0,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "failed_nodes": [],
    }


def build_stage11(progress: Callable[[str], None] = print) -> dict[str, object]:
    STAGE11_DIR.mkdir(parents=True, exist_ok=True)
    progress("[Stage11 STEP 1] Verify unique Stage 10 ZIP, PASS status, and frozen sources")
    verify_stage10_baseline_zip()
    validate_stage10_status()
    source_hashes = validate_frozen_source_hashes()
    validate_science_contract()
    validate_required_static_files()
    initial_tree = validate_stage1_through_stage10_tree_unchanged()
    execution_gate = assert_no_forbidden_execution_path()

    progress("[Stage11 STEP 2] Build deterministic complete evidence exports")
    build_evidence_exports()
    validation = validate_evidence_exports()
    write_json_atomic(
        SOURCE_IDENTITY_PATH, _source_identity_payload(source_hashes, initial_tree)
    )

    progress("[Stage11 STEP 3] Run pre-package Stage 11 contract/export/execution tests")
    preflight = run_stage11_tests(include_storage=False)

    provisional = _provisional_test_result("PROVISIONAL BEFORE PACKAGED TESTS")
    _write_test_report(preflight, provisional, provisional)
    tree = validate_stage1_through_stage10_tree_unchanged()
    _write_validation_report(validation, tree)
    _write_changed_files(tree)
    write_json_atomic(STATUS_PATH, _status_payload(validation, preflight, provisional))
    write_json_atomic(
        MANIFEST_PATH,
        _manifest_payload(
            source_hashes,
            tree,
            validation,
            execution_gate,
            preflight,
            provisional,
        ),
    )
    tree = validate_stage1_through_stage10_tree_unchanged()
    _write_validation_report(validation, tree)
    _write_changed_files(tree)
    write_json_atomic(STATUS_PATH, _status_payload(validation, preflight, provisional))
    write_json_atomic(
        MANIFEST_PATH,
        _manifest_payload(
            source_hashes,
            tree,
            validation,
            execution_gate,
            preflight,
            provisional,
        ),
    )
    verify_stage11_manifest()

    progress("[Stage11 STEP 4] Create an auditable package snapshot for storage tests")
    package_stage11()

    progress("[Stage11 STEP 5] Run full historical pytest and audit exactly three legacy boundaries")
    historical = run_full_pytest_with_legacy_rules()
    progress("[Stage11 STEP 6] Run all Stage 11 tests against the packaged snapshot")
    packaged_tests = run_stage11_tests(include_storage=True)

    progress("[Stage11 STEP 7] Freeze test/status/report/manifest artifacts")
    _write_test_report(preflight, packaged_tests, historical)
    final_tree = validate_stage1_through_stage10_tree_unchanged()
    _write_validation_report(validation, final_tree)
    _write_changed_files(final_tree)
    write_json_atomic(
        STATUS_PATH, _status_payload(validation, packaged_tests, historical)
    )
    write_json_atomic(
        MANIFEST_PATH,
        _manifest_payload(
            source_hashes,
            final_tree,
            validation,
            execution_gate,
            packaged_tests,
            historical,
        ),
    )
    verify_stage11_manifest()

    progress("[Stage11 STEP 8] Build and post-validate final Stage 11 ZIP")
    zip_path, zip_sha = package_stage11()
    progress("[Stage11 STEP 9] Run final non-mutating Stage 11 validation")
    final_tests = run_stage11_tests(include_storage=True)
    final_zip_validation = validate_final_zip(zip_path)
    validate_stage1_through_stage10_tree_unchanged()

    progress("[Stage11 STEP 10] STOP")
    return {
        "Stage": STAGE_NUMBER,
        "Status": "PASS",
        "FinalZip": str(zip_path),
        "FinalZipSHA256": zip_sha,
        "FrozenSourceCount": len(source_hashes),
        "Stage10BaselineMemberCount": final_tree["BaselineMemberCount"],
        "EvidenceRowCounts": validation["RowCounts"],
        "PrecisionLimitationRows": validation["PrecisionLimitationRows"],
        "EvidenceRegistryRows": validation["EvidenceRegistryRows"],
        "Stage11TestsPassed": final_tests["passed"],
        "HistoricalPytestPassed": historical["passed"],
        "AuthorizedLegacyFailures": historical["failed_nodes"],
        "UnexpectedHistoricalFailures": 0,
        "FinalZipValidation": final_zip_validation,
        "Stage12Executed": False,
        "StopAfterStage11": True,
    }


__all__ = [
    "ALLOWED_MODIFIED_FILES",
    "ALLOWED_STATIC_ADDITIONS",
    "FINAL_ZIP_NAME",
    "LEGACY_NODES",
    "MANIFEST_PATH",
    "REQUIRED_EVIDENCE_ROW_COUNTS",
    "STAGE10_BASELINE_ZIP_SHA256",
    "STAGE11_SCIENCE_CONTRACT_SHA256",
    "STAGE11_TASKBOOK_SHA256",
    "STAGE12_EXECUTED",
    "STATUS_PATH",
    "assert_no_forbidden_execution_path",
    "build_stage11",
    "legacy_boundary_exceptions",
    "package_stage11",
    "run_full_pytest_with_legacy_rules",
    "run_stage11_tests",
    "stage11_code_sha256",
    "validate_final_zip",
    "validate_forbidden_artifact_absence",
    "validate_frozen_source_hashes",
    "validate_package_prerequisites",
    "validate_required_static_files",
    "validate_science_contract",
    "validate_stage10_status",
    "validate_stage1_through_stage10_tree_unchanged",
    "verify_stage10_baseline_zip",
    "verify_stage11_manifest",
    "write_json_atomic",
    "write_text_atomic",
]
