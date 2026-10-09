from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .build import CONFIG_DIR, ROOT
from .contracts import CONDITIONS, METHODS
from .diagnostics import CycleDiagnostic, DIAGNOSTIC_FIELD_NAMES
from .execution import (
    ORDER_A,
    ConditionRunResult,
    ExecutionOptions,
    run_condition_repeats_parallel,
    run_condition_repeats_serial,
)
from .formal_storage import (
    AUDIT_REPEAT_IDS,
    CODE_VERSION,
    COMPOSITE_CONDITION_ORDINALS,
    FORMAL_CHUNK_SIZE,
    FORMAL_R,
    FORMAL_REPEAT_IDS,
    FORMAL_WORKERS,
    HPRF_CONDITION_ORDINALS,
    MANIFEST_FILENAME,
    STAGE8_BASELINE_ZIP_SHA256,
    STAGE8_DECISION_CONTRACT_SHA256,
    STAGE8_DECISION_INPUT_SHA256,
    STAGE8_R_DECISION_SHA256,
    STUDY_CONFIG_SHA256,
    AuditSnapshot,
    FormalChunkPaths,
    FormalIdentityHashes,
    compute_stage9_code_sha256,
    formal_chunk_paths,
    partition_formal_repeat_ids,
    reset_formal_chunk_directory,
    validate_formal_chunk,
    write_formal_chunk_atomic,
)
from .metrics import AggregatedMetrics, RepeatMetrics, aggregate_repeat_metrics
from .pilot_storage import sha256_file, write_json_atomic, write_text_atomic
from .rng import RandomNamespace
from .stage3 import load_stage3_inputs
from .stage5 import H300WidthStratumMetrics, WIDTH_STRATA
from .stage7 import (
    AGGREGATED_METRIC_NAMES,
    FROZEN_SCIENCE_HASHES,
    aggregated_metrics_field_names,
    flatten_aggregated_metrics,
    validate_frozen_science_hashes,
)
from .stage8 import (
    validate_stage1_through_stage7_tree_unchanged as validate_legacy_stage8_boundary,
)


STAGE_NUMBER = 9
STAGE_NAME = "FORMAL_R2000"
STAGE9_DIR = ROOT / "artifacts" / "stage9"
FORMAL_CHUNKS_DIR = STAGE9_DIR / "formal_chunks"
PREFLIGHT_DIR = STAGE9_DIR / "preflight"
AUDIT_DIR = STAGE9_DIR / "audit"

STAGE8_BASELINE_ZIP_NAME = "sal_stability_stage8_20260825.zip"
FINAL_ZIP_NAME = "sal_stability_stage9_20260825.zip"

STAGE10_CONTRACT_PATH = STAGE9_DIR / "stage10_bootstrap_contract_preformal.json"
MANIFEST_SEED_PATH = STAGE9_DIR / "formal_manifest_seed.json"
TREE_IDENTITY_PATH = STAGE9_DIR / "stage8_tree_identity_preformal.json"
PREFLIGHT_REPORT_PATH = PREFLIGHT_DIR / "formal_preflight_report.json"

FORMAL_REPEAT_METRICS_PATH = STAGE9_DIR / "formal_repeat_metrics.csv"
FORMAL_AGGREGATED_METRICS_PATH = STAGE9_DIR / "formal_aggregated_metrics.csv"
FORMAL_H300_STRATA_PATH = STAGE9_DIR / "formal_h300_width_strata.csv"

AUDIT_REPEAT_METRICS_PATH = AUDIT_DIR / "formal_audit_repeat_metrics.csv"
AUDIT_H300_STRATA_PATH = AUDIT_DIR / "formal_audit_h300_width_strata.csv"
AUDIT_DIAGNOSTICS_PATH = AUDIT_DIR / "formal_audit_diagnostics.csv"
AUDIT_MANIFEST_PATH = AUDIT_DIR / "audit_rerun_manifest.json"

EXPECTED_CHUNK_COUNT = 19 * (FORMAL_R // FORMAL_CHUNK_SIZE)
EXPECTED_REPEAT_METRICS_ROWS = 19 * 5 * FORMAL_R
EXPECTED_H300_STRATA_ROWS = FORMAL_R * 5 * 3
EXPECTED_AGGREGATED_ROWS = 19 * 5
EXPECTED_METHOD_CYCLES = 19 * 5 * FORMAL_R * 200
EXPECTED_AUDIT_CONDITION_REPEATS = 19 * len(AUDIT_REPEAT_IDS)
EXPECTED_AUDIT_REPEAT_METRICS_ROWS = EXPECTED_AUDIT_CONDITION_REPEATS * 5
EXPECTED_AUDIT_DIAGNOSTICS_ROWS = EXPECTED_AUDIT_CONDITION_REPEATS * 5 * 200

FORMAL_PREFLIGHT_CONDITIONS = ((3, "H300"), (7, "F2"), (14, "HF300-2"))
FORMAL_PREFLIGHT_REPEAT_IDS = tuple(range(10001, 10021))
FORMAL_PREFLIGHT_BUDGET_SECONDS = 7.0 * 3600.0

ALLOWED_STATIC_ADDITIONS = {
    "STAGE9_TASKBOOK.md",
    "src/sal_stability_stage1/formal_storage.py",
    "src/sal_stability_stage1/stage9.py",
    "scripts/build_stage9.py",
    "tests/test_stage9_contract.py",
    "tests/test_stage9_storage.py",
    "tests/test_stage9_execution.py",
}
ALLOWED_MODIFIED_FILES = {"README.md", "pyproject.toml"}

LEGACY_BOUNDARY_EXCEPTION_CODE = (
    "LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE"
)
LEGACY_BOUNDARY_EXCEPTION_NODE = (
    "tests/test_stage8_decision.py::"
    "test_stage1_through_stage7_byte_identity_and_frozen_science_gate"
)
LEGACY_BOUNDARY_EXCEPTION_REASON = (
    "Stage8 boundary test rejects taskbook-authorized Stage9 additions"
)


def stage10_bootstrap_contract() -> dict[str, object]:
    return {
        "ContractName": "STAGE10_BOOTSTRAP_SCOPE_PRE_FORMAL",
        "ContractVersion": 1,
        "LockedBeforeFormal": True,
        "BootstrapB": 2000,
        "IndependentUnit": "Repeat",
        "BootstrapNamespace": "BOOTSTRAP",
        "CoreMetrics": ["P_cor", "P_C_given_C", "P_C_given_E", "Mean_L_NC"],
        "IndividualMethodIntervals": ["FIRST", "LAST", "T", "W", "TW"],
        "PrimaryPairedContrast": "TW - T",
        "SecondaryAblationContrast": "TW - W",
        "InterferenceConditionOrdinals": list(range(1, 19)),
        "NIInference": "consistency only",
        "NoPValueRequirement": True,
        "NoOutcomeDirectionRule": True,
        "NoPostFormalContrastAddition": True,
        "NoConditionDropping": True,
        "NoParameterChange": True,
        "Stage8PrecisionContractScope": "T/TW only",
        "TWWIntervalsGuaranteedToMeetStage8HalfWidthTarget": False,
        "BootstrapExecuted": False,
    }


def lock_stage10_bootstrap_contract(path: Path = STAGE10_CONTRACT_PATH) -> str:
    path = Path(path)
    expected = stage10_bootstrap_contract()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise AssertionError("Stage 10 preformal contract changed after lock")
    else:
        write_json_atomic(path, expected)
    return sha256_file(path)


def official_formal_execution_options() -> ExecutionOptions:
    return ExecutionOptions(
        workers=FORMAL_WORKERS,
        chunk_size=FORMAL_CHUNK_SIZE,
        resume=True,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )


def validate_official_formal_execution_options(options: ExecutionOptions) -> None:
    if not isinstance(options, ExecutionOptions):
        raise TypeError("options must be ExecutionOptions")
    expected = official_formal_execution_options()
    if options != expected:
        raise ValueError("Stage 9 official execution options are frozen and do not match")


def _excluded_path(relative: Path) -> bool:
    return any(part in {"__pycache__", ".pytest_cache", ".venv"} for part in relative.parts) or relative.suffix == ".pyc" or ".tmp-" in relative.name


def _baseline_archive_members(path: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        roots: set[str] = set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                raise AssertionError("Stage 8 baseline ZIP contains a root-level file")
            roots.add(parts[0])
            relative = Path(*parts[1:])
            if _excluded_path(relative):
                continue
            key = relative.as_posix()
            if key in members:
                raise AssertionError(f"duplicate Stage 8 archive member: {key}")
            members[key] = hashlib.sha256(archive.read(info)).hexdigest()
        if len(roots) != 1:
            raise AssertionError("Stage 8 baseline ZIP must have exactly one project root")
    return members


def _stage8_tree_fingerprint(members: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(members.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b":")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_stage8_baseline_zip(path: Path | None = None) -> str:
    baseline = Path(path) if path is not None else ROOT.parent / STAGE8_BASELINE_ZIP_NAME
    actual = sha256_file(baseline)
    if actual != STAGE8_BASELINE_ZIP_SHA256:
        raise AssertionError(
            f"Stage 8 baseline ZIP SHA256 mismatch: expected {STAGE8_BASELINE_ZIP_SHA256}, got {actual}"
        )
    return actual


def _allowed_added(relative: str) -> bool:
    return relative in ALLOWED_STATIC_ADDITIONS or relative.startswith("artifacts/stage9/")


def validate_stage1_through_stage8_tree_unchanged(
    baseline_zip: Path | None = None,
) -> dict[str, object]:
    baseline_path = Path(baseline_zip) if baseline_zip is not None else ROOT.parent / STAGE8_BASELINE_ZIP_NAME
    verify_stage8_baseline_zip(baseline_path)
    baseline = _baseline_archive_members(baseline_path)
    current: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(ROOT)
        if _excluded_path(relative_path):
            continue
        current[relative_path.as_posix()] = sha256_file(path)
    added = sorted(set(current) - set(baseline))
    modified = sorted(
        relative
        for relative in set(current).intersection(baseline)
        if current[relative] != baseline[relative]
    )
    deleted = sorted(set(baseline) - set(current))
    unexpected_added = tuple(relative for relative in added if not _allowed_added(relative))
    unexpected_modified = tuple(relative for relative in modified if relative not in ALLOWED_MODIFIED_FILES)
    if unexpected_added or unexpected_modified or deleted:
        raise AssertionError(
            "Stage 1-8 byte-identity boundary failed: "
            f"unexpected_added={list(unexpected_added)}, "
            f"unexpected_modified={list(unexpected_modified)}, deleted={deleted}"
        )
    return {
        "BaselineMemberCount": len(baseline),
        "BaselineTreeSHA256": _stage8_tree_fingerprint(baseline),
        "UnchangedBaselineMemberCount": len(baseline) - len(modified),
        "AllowedModifiedFiles": modified,
        "AddedFileCount": len(added),
        "DeletedFiles": deleted,
        "Passed": True,
    }


def _validate_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise AssertionError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def validate_stage8_inputs(baseline_zip: Path | None = None) -> dict[str, object]:
    baseline_sha = verify_stage8_baseline_zip(baseline_zip)
    decision_sha = _validate_sha(
        ROOT / "artifacts/stage8/stage8_r_decision.json",
        STAGE8_R_DECISION_SHA256,
        "Stage 8 R decision",
    )
    config_sha = _validate_sha(
        CONFIG_DIR / "study_config.json", STUDY_CONFIG_SHA256, "StudyConfig"
    )
    contract_sha = _validate_sha(
        ROOT / "artifacts/stage7/stage8_r_decision_contract.json",
        STAGE8_DECISION_CONTRACT_SHA256,
        "Stage 8 decision contract",
    )
    input_sha = _validate_sha(
        ROOT / "artifacts/stage7/stage8_decision_input.csv",
        STAGE8_DECISION_INPUT_SHA256,
        "Stage 8 decision input",
    )
    decision = json.loads(
        (ROOT / "artifacts/stage8/stage8_r_decision.json").read_text(encoding="utf-8")
    )
    expected_decision = {
        "Rformal": FORMAL_R,
        "FormalRSource": "STAGE8_FROZEN_ALL_RULE",
        "ProjectedPrecisionWarning": True,
        "FormalExecuted": False,
        "BootstrapExecuted": False,
        "ConditionDropped": False,
        "ParameterChanged": False,
        "OutcomeDirectionUsed": False,
        "ManualOverrideApplied": False,
    }
    for field_name, expected in expected_decision.items():
        if decision.get(field_name) != expected:
            raise AssertionError(f"Stage 8 decision field {field_name} is not frozen")
    status = json.loads((ROOT / "artifacts/stage8/status.json").read_text(encoding="utf-8"))
    if status.get("status") != "PASS" or status.get("blocking_issue_count") != 0:
        raise AssertionError("Stage 8 status is not an unblocked PASS")
    for field_name in (
        "formal_executed",
        "bootstrap_executed",
        "condition_dropped",
        "parameter_changed",
        "outcome_direction_used",
        "manual_override_applied",
        "paper_result_generated",
    ):
        if status.get(field_name) is not False:
            raise AssertionError(f"Stage 8 status field {field_name} must be false")
    if status.get("projected_precision_warning") is not True or status.get("formal_r") != FORMAL_R:
        raise AssertionError("Stage 8 status does not carry the frozen R/warning")
    for stage_number in range(1, 9):
        prior = json.loads(
            (ROOT / f"artifacts/stage{stage_number}/status.json").read_text(encoding="utf-8")
        )
        if prior.get("status") != "PASS":
            raise AssertionError(f"Stage {stage_number} status is not PASS")
    frozen_science = validate_frozen_science_hashes()
    if frozen_science != FROZEN_SCIENCE_HASHES:
        raise AssertionError("Stage 1-8 frozen science hashes are not exact")
    tree = validate_stage1_through_stage8_tree_unchanged(baseline_zip)
    return {
        "Stage8BaselineZIPSHA256": baseline_sha,
        "Stage8RDecisionSHA256": decision_sha,
        "StudyConfigSHA256": config_sha,
        "Stage8DecisionContractSHA256": contract_sha,
        "Stage8DecisionInputSHA256": input_sha,
        "Stage8StatusPassed": True,
        "FrozenScienceHashesVerified": True,
        "FrozenScienceFileCount": len(frozen_science),
        "Stage1ThroughStage8Tree": tree,
    }


def _lock_tree_identity_artifact(tree: Mapping[str, object]) -> str:
    payload = {
        "Stage8BaselineZIPSHA256": STAGE8_BASELINE_ZIP_SHA256,
        "BaselineMemberCount": tree["BaselineMemberCount"],
        "BaselineTreeSHA256": tree["BaselineTreeSHA256"],
        "Stage1ThroughStage8TreeUnchanged": True,
        "AllowedModifiedFiles": ["README.md", "pyproject.toml"],
        "LockedBeforeFormal": True,
    }
    if TREE_IDENTITY_PATH.exists():
        current = json.loads(TREE_IDENTITY_PATH.read_text(encoding="utf-8"))
        if current != payload:
            raise AssertionError("preformal Stage 8 tree identity artifact changed")
    else:
        write_json_atomic(TREE_IDENTITY_PATH, payload)
    return sha256_file(TREE_IDENTITY_PATH)


def lock_formal_manifest_seed(
    *,
    identity: FormalIdentityHashes,
    tree_identity_sha256: str,
) -> str:
    payload = {
        "StageNumber": STAGE_NUMBER,
        "StageName": STAGE_NAME,
        "LockedBeforeFormal": True,
        "FormalStartedAtLock": False,
        "Stage8BaselineZIPSHA256": identity.stage8_baseline_zip_sha256,
        "Stage8RDecisionSHA256": identity.stage8_r_decision_sha256,
        "Stage8DecisionContractSHA256": identity.stage8_decision_contract_sha256,
        "Stage8DecisionInputSHA256": identity.stage8_decision_input_sha256,
        "StudyConfigSHA256": identity.study_config_sha256,
        "Stage10PreformalContractSHA256": identity.stage10_preformal_contract_sha256,
        "Stage8TreeIdentitySHA256": tree_identity_sha256,
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": identity.code_sha256,
        "Namespace": "FORMAL",
        "FormalR": FORMAL_R,
        "RepeatRange": [1, FORMAL_R],
        "ConditionCount": 19,
        "MethodCount": 5,
        "Workers": FORMAL_WORKERS,
        "ChunkSize": FORMAL_CHUNK_SIZE,
        "Resume": True,
        "Diagnostic": False,
        "HCacheEnabled": True,
        "MethodExecutionOrder": list(ORDER_A),
        "AuditRepeatIDs": list(AUDIT_REPEAT_IDS),
        "ProjectedPrecisionWarning": True,
        "BootstrapExecuted": False,
        "OutcomeDirectionUsed": False,
        "LegacyBoundaryExceptionApproval": {
            "Code": LEGACY_BOUNDARY_EXCEPTION_CODE,
            "TestNode": LEGACY_BOUNDARY_EXCEPTION_NODE,
            "Allowed": True,
            "Scope": "only taskbook-authorized Stage9 additions",
        },
    }
    if MANIFEST_SEED_PATH.exists():
        current = json.loads(MANIFEST_SEED_PATH.read_text(encoding="utf-8"))
        if current != payload:
            raise AssertionError("Formal manifest seed changed after preformal lock")
    else:
        write_json_atomic(MANIFEST_SEED_PATH, payload)
    return sha256_file(MANIFEST_SEED_PATH)


def _pytest_result(command: Sequence[str]) -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command), cwd=ROOT, text=True, capture_output=True, check=False
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    matches = re.findall(r"([0-9]+) passed", output)
    passed = int(matches[-1]) if matches else 0
    failed_matches = re.findall(r"([0-9]+) failed", output)
    failed = int(failed_matches[-1]) if failed_matches else 0
    error_matches = re.findall(r"([0-9]+) errors?", output)
    errors = int(error_matches[-1]) if error_matches else 0
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "wall_seconds": elapsed,
        "output": output,
    }


def run_stage9_unit_tests() -> dict[str, object]:
    command = (
        sys.executable,
        "-m",
        "pytest",
        "tests/test_stage9_contract.py",
        "tests/test_stage9_storage.py",
        "tests/test_stage9_execution.py",
    )
    result = _pytest_result(command)
    if result["returncode"] != 0 or result["passed"] <= 0 or result["failed"] or result["errors"]:
        raise AssertionError(f"Stage 9 unit tests failed\n{result['output']}")
    return result


def _normalize_pytest_node_id(value: str) -> str:
    return str(value).replace("\\", "/")


def _legacy_boundary_exception_record(
    *,
    full: Mapping[str, object],
    remaining: Mapping[str, object],
    tree: Mapping[str, object],
) -> dict[str, object]:
    return {
        "Code": LEGACY_BOUNDARY_EXCEPTION_CODE,
        "TestNode": LEGACY_BOUNDARY_EXCEPTION_NODE,
        "Allowed": True,
        "Reason": LEGACY_BOUNDARY_EXCEPTION_REASON,
        "FailureDifferenceScope": "taskbook-authorized Stage9 additions only",
        "AllowedStage9AddedFileCount": int(tree["AddedFileCount"]),
        "FrozenStage8Modified": False,
        "Stage8ZipIdentityVerified": True,
        "FrozenScienceHashesVerified": True,
        "FullPytestRawPassed": int(full["passed"]),
        "FullPytestRawFailed": int(full["failed"]),
        "FullPytestRawErrors": int(full["errors"]),
        "RemainingTestsPassed": int(remaining["passed"]),
        "OtherTestFailures": int(remaining["failed"]) + int(remaining["errors"]),
    }


def run_full_pytest_with_legacy_archive_rule() -> dict[str, object]:
    full = _pytest_result((sys.executable, "-m", "pytest"))
    failed_nodes = {
        _normalize_pytest_node_id(value)
        for value in re.findall(r"^FAILED\s+(\S+)", str(full["output"]), re.MULTILINE)
    }
    if (
        full["returncode"] == 0
        or full["failed"] != 1
        or full["errors"] != 0
        or failed_nodes != {LEGACY_BOUNDARY_EXCEPTION_NODE}
    ):
        raise AssertionError(
            "complete pytest did not match the one authorized legacy boundary failure: "
            f"returncode={full['returncode']}, failed={full['failed']}, "
            f"errors={full['errors']}, failed_nodes={sorted(failed_nodes)}\n"
            f"{full['output']}"
        )

    try:
        validate_legacy_stage8_boundary()
    except AssertionError as exc:
        legacy_message = str(exc)
    else:
        raise AssertionError("the authorized legacy Stage 8 boundary test unexpectedly passed")
    if (
        "unexpected_added=" not in legacy_message
        or "unexpected_modified=[]" not in legacy_message
        or "deleted=[]" not in legacy_message
    ):
        raise AssertionError(
            "legacy Stage 8 boundary failure was not additions-only: " + legacy_message
        )

    tree = validate_stage1_through_stage8_tree_unchanged()
    if validate_frozen_science_hashes() != FROZEN_SCIENCE_HASHES:
        raise AssertionError("frozen science hashes failed during legacy exception audit")
    verify_stage8_baseline_zip()

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--deselect",
        LEGACY_BOUNDARY_EXCEPTION_NODE,
    ]
    remaining = _pytest_result(command)
    if remaining["returncode"] != 0 or remaining["passed"] <= 0 or remaining["failed"] or remaining["errors"]:
        raise AssertionError(
            "pytest still failed after the one authorized legacy boundary exclusion\n"
            f"complete output:\n{full['output']}\nremaining output:\n{remaining['output']}"
        )
    exception = _legacy_boundary_exception_record(
        full=full,
        remaining=remaining,
        tree=tree,
    )
    if exception["OtherTestFailures"] != 0:
        raise AssertionError("another regression failure exists outside the authorized node")
    return {
        "complete_pytest": full,
        "legacy_boundary_exception_count": 1,
        "legacy_boundary_exception": exception,
        "legacy_external_archive_exclusion_count": 0,
        "legacy_external_archive_exclusion_node_ids": [],
        "legacy_external_archive_exclusion_reason": None,
        "remaining_regression_suite": remaining,
    }


def run_formal_preflight(
    *,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    identity: FormalIdentityHashes,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    options = official_formal_execution_options()
    validate_official_formal_execution_options(options)
    details: list[dict[str, object]] = []
    total_started = time.perf_counter()
    for condition_ordinal, condition_code in FORMAL_PREFLIGHT_CONDITIONS:
        started = time.perf_counter()
        result = run_condition_repeats_parallel(
            namespace=RandomNamespace.FORMAL,
            condition_ordinal=condition_ordinal,
            repeat_ids=FORMAL_PREFLIGHT_REPEAT_IDS,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=options,
        )
        wall = time.perf_counter() - started
        if result.namespace is not RandomNamespace.FORMAL or len(result.repeat_metrics) != 100:
            raise AssertionError("FORMAL preflight returned an incomplete representative chunk")
        if condition_code == "H300" and len(result.h300_width_strata) != 300:
            raise AssertionError("FORMAL H300 preflight strata are incomplete")
        if condition_ordinal in HPRF_CONDITION_ORDINALS | COMPOSITE_CONDITION_ORDINALS and any(
            row.N_N != 0 for row in result.repeat_metrics
        ):
            raise AssertionError("FORMAL preflight violated the no-N scenario gate")
        details.append(
            {
                "ConditionOrdinal": condition_ordinal,
                "ConditionCode": condition_code,
                "RepeatIDs": list(FORMAL_PREFLIGHT_REPEAT_IDS),
                "RepeatMetricsRows": len(result.repeat_metrics),
                "H300WidthStrataRows": len(result.h300_width_strata),
                "WallSeconds": wall,
                "Passed": True,
            }
        )
        progress(f"[Stage9 preflight] {condition_code} FORMAL representative chunk PASS ({wall:.2f}s)")
    total_wall = time.perf_counter() - total_started
    condition_repeats = len(FORMAL_PREFLIGHT_CONDITIONS) * len(FORMAL_PREFLIGHT_REPEAT_IDS)
    projected = total_wall / condition_repeats * (19 * FORMAL_R)
    if projected > FORMAL_PREFLIGHT_BUDGET_SECONDS:
        raise AssertionError(
            f"FORMAL representative preflight projects {projected:.1f}s, exceeding the frozen 7h budget"
        )
    report = {
        "Passed": True,
        "RunKind": "VALIDATION",
        "Namespace": "FORMAL",
        "StatisticalDataIncluded": False,
        "Conditions": details,
        "RepeatIDs": list(FORMAL_PREFLIGHT_REPEAT_IDS),
        "Workers": FORMAL_WORKERS,
        "ChunkSize": FORMAL_CHUNK_SIZE,
        "Resume": True,
        "Diagnostic": False,
        "HCacheEnabled": True,
        "MethodExecutionOrder": list(ORDER_A),
        "Stage10PreformalContractSHA256": identity.stage10_preformal_contract_sha256,
        "CodeSHA256": identity.code_sha256,
        "WallSeconds": total_wall,
        "ProjectedFormalWallSeconds": projected,
        "BudgetSeconds": FORMAL_PREFLIGHT_BUDGET_SECONDS,
        "NoOutcomeDirectionUsed": True,
    }
    write_json_atomic(PREFLIGHT_REPORT_PATH, report)
    return report


@dataclass(frozen=True, slots=True)
class FormalConditionExecutionOutcome:
    condition_ordinal: int
    condition_code: str
    completed: bool
    computed_chunks: int
    skipped_chunks: int
    recomputed_invalid_chunks: int
    invalidated_chunks: tuple[str, ...]
    chunk_wall_seconds: tuple[float, ...]


def run_formal_condition_chunked(
    *,
    condition_ordinal: int,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    execution_options: ExecutionOptions,
    identity: FormalIdentityHashes,
    output_root: Path = FORMAL_CHUNKS_DIR,
    stop_after_chunks: int | None = None,
    progress: Callable[[str], None] | None = print,
) -> FormalConditionExecutionOutcome:
    validate_official_formal_execution_options(execution_options)
    if not isinstance(identity, FormalIdentityHashes):
        raise TypeError("identity must be FormalIdentityHashes")
    try:
        condition = CONDITIONS[int(condition_ordinal)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("condition_ordinal must lie in 0..18") from exc
    if condition.ConditionOrdinal != condition_ordinal:
        raise AssertionError("frozen condition order is not canonical")
    if stop_after_chunks is not None and (
        isinstance(stop_after_chunks, bool)
        or not isinstance(stop_after_chunks, int)
        or stop_after_chunks < 1
    ):
        raise ValueError("stop_after_chunks must be a positive integer or None")

    computed = 0
    skipped = 0
    recomputed_invalid = 0
    invalidated: list[str] = []
    timings: list[float] = []
    chunks = partition_formal_repeat_ids()
    processed = 0
    for chunk_index, repeat_ids in enumerate(chunks):
        if stop_after_chunks is not None and chunk_index >= stop_after_chunks:
            break
        started = time.perf_counter()
        paths = formal_chunk_paths(
            output_root,
            condition_ordinal=condition_ordinal,
            condition_code=condition.Code,
            repeat_ids=repeat_ids,
        )
        validation = validate_formal_chunk(
            paths,
            identity=identity,
            condition_ordinal=condition_ordinal,
            condition_code=condition.Code,
            repeat_ids=repeat_ids,
        )
        if validation.valid:
            skipped += 1
            action = "skipped-valid"
        else:
            had_artifact = paths.directory.exists()
            invalidated.append(f"{repeat_ids[0]}-{repeat_ids[-1]}:{validation.reason}")
            if had_artifact:
                recomputed_invalid += 1
            reset_formal_chunk_directory(paths, output_root=output_root)
            result = run_condition_repeats_parallel(
                namespace=RandomNamespace.FORMAL,
                condition_ordinal=condition_ordinal,
                repeat_ids=repeat_ids,
                study_config=study_config,
                encoding_reference=encoding_reference,
                execution_options=execution_options,
            )
            write_formal_chunk_atomic(output_root, result=result, identity=identity)
            committed = validate_formal_chunk(
                paths,
                identity=identity,
                condition_ordinal=condition_ordinal,
                condition_code=condition.Code,
                repeat_ids=repeat_ids,
            )
            if not committed.valid:
                raise AssertionError(
                    f"fresh Formal chunk failed commit validation: {condition.Code} {repeat_ids[0]}-{repeat_ids[-1]} {committed.reason}"
                )
            computed += 1
            action = (
                f"recomputed-invalid:{validation.reason}" if had_artifact else "computed"
            )
        wall = time.perf_counter() - started
        timings.append(wall)
        processed += 1
        if progress is not None:
            progress(
                f"[Stage9 FORMAL] {condition.Code} chunk {repeat_ids[0]}-{repeat_ids[-1]} "
                f"{action} ({wall:.2f}s; condition {processed}/100)"
            )
    return FormalConditionExecutionOutcome(
        condition_ordinal=condition_ordinal,
        condition_code=condition.Code,
        completed=processed == len(chunks),
        computed_chunks=computed,
        skipped_chunks=skipped,
        recomputed_invalid_chunks=recomputed_invalid,
        invalidated_chunks=tuple(invalidated),
        chunk_wall_seconds=tuple(timings),
    )


@dataclass(frozen=True, slots=True)
class FormalDataset:
    repeat_metrics: tuple[RepeatMetrics, ...]
    h300_width_strata: tuple[H300WidthStratumMetrics, ...]
    audit_snapshots: dict[tuple[int, int], AuditSnapshot]
    chunk_manifests: tuple[dict[str, object], ...]
    chunk_paths: dict[tuple[int, int], FormalChunkPaths]


def validate_and_load_all_formal_chunks(
    *,
    identity: FormalIdentityHashes,
    output_root: Path = FORMAL_CHUNKS_DIR,
) -> FormalDataset:
    repeat_chunks = partition_formal_repeat_ids()
    expected_manifest_paths: set[Path] = set()
    metrics: list[RepeatMetrics] = []
    strata: list[H300WidthStratumMetrics] = []
    snapshots: dict[tuple[int, int], AuditSnapshot] = {}
    manifests: list[dict[str, object]] = []
    chunk_path_map: dict[tuple[int, int], FormalChunkPaths] = {}
    for condition in CONDITIONS:
        for repeat_ids in repeat_chunks:
            paths = formal_chunk_paths(
                output_root,
                condition_ordinal=condition.ConditionOrdinal,
                condition_code=condition.Code,
                repeat_ids=repeat_ids,
            )
            expected_manifest_paths.add(paths.manifest.resolve())
            validation = validate_formal_chunk(
                paths,
                identity=identity,
                condition_ordinal=condition.ConditionOrdinal,
                condition_code=condition.Code,
                repeat_ids=repeat_ids,
            )
            if not validation.valid or validation.data is None:
                raise AssertionError(
                    f"invalid Formal chunk {condition.Code} {repeat_ids[0]}-{repeat_ids[-1]}: {validation.reason}"
                )
            data = validation.data
            metrics.extend(data.repeat_metrics)
            strata.extend(data.h300_width_strata)
            manifests.append(data.manifest)
            if data.audit_snapshot is not None:
                key = (condition.ConditionOrdinal, data.audit_snapshot.repeat_id)
                if key in snapshots:
                    raise AssertionError("duplicate primary audit snapshot")
                snapshots[key] = data.audit_snapshot
                chunk_path_map[key] = paths
    actual_manifest_paths = {
        path.resolve() for path in Path(output_root).rglob(MANIFEST_FILENAME) if path.is_file()
    }
    if actual_manifest_paths != expected_manifest_paths:
        raise AssertionError("Formal chunk tree contains a missing or unexpected manifest")
    if len(manifests) != EXPECTED_CHUNK_COUNT:
        raise AssertionError("Formal chunk validation did not cover exactly 1900 chunks")
    if len(metrics) != EXPECTED_REPEAT_METRICS_ROWS:
        raise AssertionError("Formal chunk merge did not load exactly 190000 RepeatMetrics")
    if len(strata) != EXPECTED_H300_STRATA_ROWS:
        raise AssertionError("Formal chunk merge did not load exactly 30000 H300 strata")
    if set(snapshots) != {
        (condition.ConditionOrdinal, repeat_id)
        for condition in CONDITIONS
        for repeat_id in AUDIT_REPEAT_IDS
    }:
        raise AssertionError("primary audit snapshot key set is not the preregistered 57 units")
    rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
    return FormalDataset(
        repeat_metrics=tuple(
            sorted(metrics, key=lambda row: (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID))
        ),
        h300_width_strata=tuple(
            sorted(
                strata,
                key=lambda row: (
                    row.ConditionOrdinal,
                    row.MethodOrdinal,
                    row.RepeatID,
                    rank[row.WidthStratum],
                ),
            )
        ),
        audit_snapshots=snapshots,
        chunk_manifests=tuple(manifests),
        chunk_paths=chunk_path_map,
    )


def validate_formal_repeat_metrics(rows: Sequence[RepeatMetrics]) -> dict[str, bool]:
    if len(rows) != EXPECTED_REPEAT_METRICS_ROWS or any(
        not isinstance(row, RepeatMetrics) for row in rows
    ):
        raise AssertionError("Formal RepeatMetrics row count/type is invalid")
    expected_keys = {
        (condition.ConditionOrdinal, method.MethodOrdinal, repeat_id)
        for condition in CONDITIONS
        for method in METHODS
        for repeat_id in FORMAL_REPEAT_IDS
    }
    actual_keys = [(row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise AssertionError("Formal RepeatMetrics key universe is incomplete or duplicated")
    condition_codes = {item.ConditionOrdinal: item.Code for item in CONDITIONS}
    method_codes = {item.MethodOrdinal: item.Code for item in METHODS}
    for row in rows:
        if row.ConditionCode != condition_codes[row.ConditionOrdinal]:
            raise AssertionError("Formal RepeatMetrics condition identity drifted")
        if row.MethodCode != method_codes[row.MethodOrdinal]:
            raise AssertionError("Formal RepeatMetrics method identity drifted")
        if row.N_C + row.N_E + row.N_N != 200:
            raise AssertionError("Formal RepeatMetrics state conservation failed")
        if (
            row.N_CC + row.N_CE + row.N_CN + row.N_EC + row.N_EE + row.N_EN + row.N_NC + row.N_NE + row.N_NN
        ) != 199:
            raise AssertionError("Formal RepeatMetrics transition total failed")
        if row.N_Cdot != row.N_CC + row.N_CE + row.N_CN:
            raise AssertionError("Formal C-origin conservation failed")
        if row.N_Edot != row.N_EC + row.N_EE + row.N_EN:
            raise AssertionError("Formal E-origin conservation failed")
        if row.N_Ndot != row.N_NC + row.N_NE + row.N_NN:
            raise AssertionError("Formal N-origin conservation failed")
        if row.N_Cdot + row.N_Edot + row.N_Ndot != 199:
            raise AssertionError("Formal origin denominator conservation failed")
        if row.Sum_L_NC_obs != row.N_E + row.N_N:
            raise AssertionError("Formal non-correct run-length conservation failed")
        if row.N_E_H + row.N_E_F != row.N_E:
            raise AssertionError("Formal error-source conservation failed")
        expected_origins = {
            "C": row.N_C - int(row.EndState == "C"),
            "E": row.N_E - int(row.EndState == "E"),
            "N": row.N_N - int(row.EndState == "N"),
        }
        if (row.N_Cdot, row.N_Edot, row.N_Ndot) != (
            expected_origins["C"], expected_origins["E"], expected_origins["N"]
        ):
            raise AssertionError("Formal EndState/origin identity failed")
    ni_rows = tuple(row for row in rows if row.ConditionOrdinal == 0)
    if len(ni_rows) != 5 * FORMAL_R or any(
        (row.N_C, row.N_E, row.N_N) != (200, 0, 0) for row in ni_rows
    ):
        raise AssertionError("NI Formal counts are not exactly all-C")
    hprf_rows = tuple(row for row in rows if row.ConditionOrdinal in HPRF_CONDITION_ORDINALS)
    composite_rows = tuple(
        row for row in rows if row.ConditionOrdinal in COMPOSITE_CONDITION_ORDINALS
    )
    if len(hprf_rows) != len(HPRF_CONDITION_ORDINALS) * 5 * FORMAL_R or any(
        row.N_N != 0 for row in hprf_rows
    ):
        raise AssertionError("HPRF Formal no-N gate failed")
    if len(composite_rows) != len(COMPOSITE_CONDITION_ORDINALS) * 5 * FORMAL_R or any(
        row.N_N != 0 for row in composite_rows
    ):
        raise AssertionError("Composite Formal no-N gate failed")
    return {
        "repeat_conservation_passed": True,
        "transition_conservation_passed": True,
        "ni_counts_passed": True,
        "hprf_no_n_passed": True,
        "composite_no_n_passed": True,
    }


def validate_formal_h300_width_strata(
    rows: Sequence[H300WidthStratumMetrics],
) -> bool:
    if len(rows) != EXPECTED_H300_STRATA_ROWS:
        raise AssertionError("Formal H300 width-stratum row count is not 30000")
    expected_keys = {
        (method.MethodOrdinal, repeat_id, stratum)
        for method in METHODS
        for repeat_id in FORMAL_REPEAT_IDS
        for stratum in WIDTH_STRATA
    }
    actual_keys = [(row.MethodOrdinal, row.RepeatID, row.WidthStratum) for row in rows]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise AssertionError("Formal H300 width-stratum key universe is invalid")
    for row in rows:
        if row.ConditionOrdinal != 3 or row.ConditionCode != "H300":
            raise AssertionError("Formal H300 width-stratum condition identity failed")
    grouped: dict[tuple[int, int], int] = {}
    for row in rows:
        key = (row.MethodOrdinal, row.RepeatID)
        grouped[key] = grouped.get(key, 0) + row.StratumCycleCount
    if len(grouped) != 5 * FORMAL_R or any(value != 200 for value in grouped.values()):
        raise AssertionError("Formal H300 strata do not conserve 200 cycles per Method-Repeat")
    return True


def aggregate_formal_repeat_metrics(
    rows: Sequence[RepeatMetrics],
) -> tuple[AggregatedMetrics, ...]:
    groups: dict[tuple[int, int], list[RepeatMetrics]] = {
        (condition.ConditionOrdinal, method.MethodOrdinal): []
        for condition in CONDITIONS
        for method in METHODS
    }
    for row in rows:
        groups[(row.ConditionOrdinal, row.MethodOrdinal)].append(row)
    result: list[AggregatedMetrics] = []
    for condition in CONDITIONS:
        for method in METHODS:
            group = groups[(condition.ConditionOrdinal, method.MethodOrdinal)]
            if len(group) != FORMAL_R or {row.RepeatID for row in group} != set(FORMAL_REPEAT_IDS):
                raise AssertionError(
                    f"Formal aggregate input incomplete for {condition.Code}/{method.Code}"
                )
            aggregate = aggregate_repeat_metrics(group)
            if aggregate.R != FORMAL_R:
                raise AssertionError("Formal aggregate R is not 2000")
            result.append(aggregate)
    if len(result) != EXPECTED_AGGREGATED_ROWS:
        raise AssertionError("Formal aggregation did not produce exactly 95 rows")
    return tuple(result)


def _atomic_csv_stream(
    path: Path,
    *,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    names = tuple(field_names)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(names), lineterminator="\n", extrasaction="raise"
            )
            writer.writeheader()
            for row in rows:
                payload = dict(row)
                if set(payload) != set(names):
                    raise ValueError(f"CSV row schema disagrees with {path.name}")
                writer.writerow(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_row_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {path}") from exc
        return sum(1 for _ in reader)


def write_formal_global_outputs(
    *,
    repeat_metrics: Sequence[RepeatMetrics],
    aggregated_metrics: Sequence[AggregatedMetrics],
    h300_width_strata: Sequence[H300WidthStratumMetrics],
) -> dict[str, object]:
    _atomic_csv_stream(
        FORMAL_REPEAT_METRICS_PATH,
        field_names=tuple(field.name for field in fields(RepeatMetrics)),
        rows=(asdict(row) for row in repeat_metrics),
    )
    _atomic_csv_stream(
        FORMAL_AGGREGATED_METRICS_PATH,
        field_names=aggregated_metrics_field_names(),
        rows=(flatten_aggregated_metrics(row) for row in aggregated_metrics),
    )
    _atomic_csv_stream(
        FORMAL_H300_STRATA_PATH,
        field_names=tuple(field.name for field in fields(H300WidthStratumMetrics)),
        rows=(asdict(row) for row in h300_width_strata),
    )
    counts = {
        "FormalRepeatMetricsRowCount": _csv_row_count(FORMAL_REPEAT_METRICS_PATH),
        "FormalAggregatedMetricsRowCount": _csv_row_count(FORMAL_AGGREGATED_METRICS_PATH),
        "FormalH300WidthStrataRowCount": _csv_row_count(FORMAL_H300_STRATA_PATH),
    }
    if counts != {
        "FormalRepeatMetricsRowCount": EXPECTED_REPEAT_METRICS_ROWS,
        "FormalAggregatedMetricsRowCount": EXPECTED_AGGREGATED_ROWS,
        "FormalH300WidthStrataRowCount": EXPECTED_H300_STRATA_ROWS,
    }:
        raise AssertionError(f"global Formal disk row counts failed: {counts}")
    return {
        **counts,
        "FormalRepeatMetricsSHA256": sha256_file(FORMAL_REPEAT_METRICS_PATH),
        "FormalAggregatedMetricsSHA256": sha256_file(FORMAL_AGGREGATED_METRICS_PATH),
        "FormalH300WidthStrataSHA256": sha256_file(FORMAL_H300_STRATA_PATH),
    }


@dataclass(frozen=True, slots=True)
class AuditResult:
    repeat_metrics: tuple[RepeatMetrics, ...]
    h300_width_strata: tuple[H300WidthStratumMetrics, ...]
    diagnostics: tuple[CycleDiagnostic, ...]
    manifest: dict[str, object]


def run_independent_audit_reruns(
    *,
    dataset: FormalDataset,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    progress: Callable[[str], None] = print,
) -> AuditResult:
    metric_lookup = {
        (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID): row
        for row in dataset.repeat_metrics
    }
    primary_h300 = {
        (row.MethodOrdinal, row.RepeatID, row.WidthStratum): row
        for row in dataset.h300_width_strata
    }
    options = ExecutionOptions(
        workers=1,
        chunk_size=1,
        resume=False,
        diagnostic=True,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    audit_metrics: list[RepeatMetrics] = []
    audit_strata: list[H300WidthStratumMetrics] = []
    diagnostics: list[CycleDiagnostic] = []
    unit_details: list[dict[str, object]] = []
    for condition in CONDITIONS:
        for repeat_id in AUDIT_REPEAT_IDS:
            result = run_condition_repeats_serial(
                namespace=RandomNamespace.FORMAL,
                condition_ordinal=condition.ConditionOrdinal,
                repeat_ids=(repeat_id,),
                study_config=study_config,
                encoding_reference=encoding_reference,
                execution_options=options,
            )
            expected_metrics = tuple(
                metric_lookup[(condition.ConditionOrdinal, method.MethodOrdinal, repeat_id)]
                for method in METHODS
            )
            if result.repeat_metrics != expected_metrics:
                raise AssertionError(
                    f"audit RepeatMetrics mismatch: {condition.Code} RepeatID={repeat_id}"
                )
            snapshot = dataset.audit_snapshots[(condition.ConditionOrdinal, repeat_id)]
            source_equal = np.array_equal(result.source_state[:, 0, :], snapshot.source_state_code)
            reference_equal = np.array_equal(result.ref_toa_before_s[:, 0, :], snapshot.ref_toa_before_s)
            selected_equal = np.array_equal(
                result.selected_observed_toa_s[:, 0, :],
                snapshot.selected_observed_toa_s,
                equal_nan=True,
            )
            selection_equal = np.array_equal(result.has_selection[:, 0, :], snapshot.has_selection)
            if not all((source_equal, reference_equal, selected_equal, selection_equal)):
                raise AssertionError(
                    f"audit primary/rerun trajectory mismatch: {condition.Code} RepeatID={repeat_id}"
                )
            h300_equal = True
            if condition.Code == "H300":
                expected_strata = tuple(
                    primary_h300[(method.MethodOrdinal, repeat_id, stratum)]
                    for method in METHODS
                    for stratum in WIDTH_STRATA
                )
                h300_equal = result.h300_width_strata == expected_strata
                if not h300_equal:
                    raise AssertionError(f"audit H300 strata mismatch: RepeatID={repeat_id}")
                audit_strata.extend(result.h300_width_strata)
            if len(result.diagnostics) != 1000:
                raise AssertionError("audit rerun did not generate exactly 1000 diagnostics")
            audit_metrics.extend(result.repeat_metrics)
            diagnostics.extend(result.diagnostics)
            primary_paths = dataset.chunk_paths[(condition.ConditionOrdinal, repeat_id)]
            primary_manifest = json.loads(primary_paths.manifest.read_text(encoding="utf-8"))
            unit_details.append(
                {
                    "ConditionOrdinal": condition.ConditionOrdinal,
                    "ConditionCode": condition.Code,
                    "RepeatID": repeat_id,
                    "RepeatMetricsExact": True,
                    "SourceStateExact": source_equal,
                    "RefTOABeforeExact": reference_equal,
                    "SelectedObservedTOAEqualNaNExact": selected_equal,
                    "HasSelectionExact": selection_equal,
                    "H300WidthStrataExact": h300_equal,
                    "DiagnosticRowCount": len(result.diagnostics),
                    "PrimaryAuditSnapshotSHA256": primary_manifest["AuditSnapshotSHA256"],
                    "PrimaryChunkDataSHA256": primary_manifest["DataSHA256"],
                    "PrimaryH300WidthStrataSHA256": primary_manifest.get("H300WidthStrataSHA256"),
                }
            )
            progress(f"[Stage9 audit] {condition.Code} RepeatID={repeat_id} exact PASS")
    if len(audit_metrics) != EXPECTED_AUDIT_REPEAT_METRICS_ROWS:
        raise AssertionError("audit RepeatMetrics row count is not 285")
    if len(diagnostics) != EXPECTED_AUDIT_DIAGNOSTICS_ROWS:
        raise AssertionError("audit diagnostic row count is not 57000")
    rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
    audit_metrics_tuple = tuple(
        sorted(audit_metrics, key=lambda row: (row.ConditionOrdinal, row.RepeatID, row.MethodOrdinal))
    )
    audit_strata_tuple = tuple(
        sorted(
            audit_strata,
            key=lambda row: (row.RepeatID, row.MethodOrdinal, rank[row.WidthStratum]),
        )
    )
    diagnostics_tuple = tuple(
        sorted(
            diagnostics,
            key=lambda row: (
                row.ConditionOrdinal, row.RepeatID, row.Cycle, row.MethodOrdinal
            ),
        )
    )
    _atomic_csv_stream(
        AUDIT_REPEAT_METRICS_PATH,
        field_names=tuple(field.name for field in fields(RepeatMetrics)),
        rows=(asdict(row) for row in audit_metrics_tuple),
    )
    _atomic_csv_stream(
        AUDIT_H300_STRATA_PATH,
        field_names=tuple(field.name for field in fields(H300WidthStratumMetrics)),
        rows=(asdict(row) for row in audit_strata_tuple),
    )
    _atomic_csv_stream(
        AUDIT_DIAGNOSTICS_PATH,
        field_names=DIAGNOSTIC_FIELD_NAMES,
        rows=(asdict(row) for row in diagnostics_tuple),
    )
    if _csv_row_count(AUDIT_REPEAT_METRICS_PATH) != EXPECTED_AUDIT_REPEAT_METRICS_ROWS:
        raise AssertionError("audit RepeatMetrics disk row count failed")
    if _csv_row_count(AUDIT_DIAGNOSTICS_PATH) != EXPECTED_AUDIT_DIAGNOSTICS_ROWS:
        raise AssertionError("audit diagnostics disk row count failed")
    manifest = {
        "AuditRepeatIDs": list(AUDIT_REPEAT_IDS),
        "ConditionRepeatCount": EXPECTED_AUDIT_CONDITION_REPEATS,
        "RepeatMetricsRowCount": EXPECTED_AUDIT_REPEAT_METRICS_ROWS,
        "H300WidthStrataRowCount": len(audit_strata_tuple),
        "DiagnosticsRowCount": EXPECTED_AUDIT_DIAGNOSTICS_ROWS,
        "RepeatMetricsSHA256": sha256_file(AUDIT_REPEAT_METRICS_PATH),
        "H300WidthStrataSHA256": sha256_file(AUDIT_H300_STRATA_PATH),
        "DiagnosticsSHA256": sha256_file(AUDIT_DIAGNOSTICS_PATH),
        "ExactReproducibilityPassed": True,
        "DiagnosticDataIncludedInFormalStatistics": False,
        "Units": unit_details,
    }
    write_json_atomic(AUDIT_MANIFEST_PATH, manifest)
    manifest["ManifestSHA256"] = sha256_file(AUDIT_MANIFEST_PATH)
    return AuditResult(
        repeat_metrics=audit_metrics_tuple,
        h300_width_strata=audit_strata_tuple,
        diagnostics=diagnostics_tuple,
        manifest=manifest,
    )


def _chunk_data_set_sha256(manifests: Sequence[Mapping[str, object]]) -> str:
    records = sorted(
        (
            int(item["ConditionOrdinal"]),
            int(item["RepeatStart"]),
            str(item["DataSHA256"]),
        )
        for item in manifests
    )
    digest = hashlib.sha256()
    for condition_ordinal, repeat_start, data_sha in records:
        digest.update(f"{condition_ordinal:02d}:{repeat_start:04d}:{data_sha}\n".encode("ascii"))
    return digest.hexdigest()


def _scan_formal_chunk_storage() -> dict[str, bool]:
    forbidden_names = {"chunk_state.npy", "chunk_trajectory.npz"}
    names = {path.name for path in FORMAL_CHUNKS_DIR.rglob("*") if path.is_file()}
    if names.intersection(forbidden_names):
        raise AssertionError("Stage 9 Formal storage contains a forbidden full-chunk state/trajectory file")
    if any(".tmp-" in name for name in names):
        raise AssertionError("Stage 9 Formal storage contains an uncommitted temporary file")
    for manifest_path in FORMAL_CHUNKS_DIR.rglob(MANIFEST_FILENAME):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        forbidden = {
            "GlobalRNGState",
            "ThreadRNGState",
            "LoopCounterState",
            "RandomState",
            "MutableRNGState",
            "BootstrapIndexSHA256",
        }.intersection(manifest)
        if forbidden:
            raise AssertionError("Stage 9 Formal manifest contains mutable RNG/Bootstrap state")
    return {"NoMutableRNGStateStored": True, "NoFullChunkTrajectoryStored": True}


def build_formal_manifest(
    *,
    identity: FormalIdentityHashes,
    manifest_seed_sha256: str,
    tree_identity_sha256: str,
    dataset: FormalDataset,
    global_outputs: Mapping[str, object],
    audit: AuditResult,
    regression: Mapping[str, object],
) -> dict[str, object]:
    if len(dataset.chunk_manifests) != EXPECTED_CHUNK_COUNT:
        raise AssertionError("formal manifest cannot be built from an incomplete chunk set")
    return {
        "StageNumber": STAGE_NUMBER,
        "StageName": STAGE_NAME,
        "Stage8BaselineZIPSHA256": identity.stage8_baseline_zip_sha256,
        "Stage8RDecisionSHA256": identity.stage8_r_decision_sha256,
        "Stage8DecisionContractSHA256": identity.stage8_decision_contract_sha256,
        "Stage8DecisionInputSHA256": identity.stage8_decision_input_sha256,
        "Stage10PreformalContractSHA256": identity.stage10_preformal_contract_sha256,
        "FormalManifestSeedSHA256": manifest_seed_sha256,
        "StudyConfigSHA256": identity.study_config_sha256,
        "Stage8TreeIdentitySHA256": tree_identity_sha256,
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": identity.code_sha256,
        "FormalR": FORMAL_R,
        "Namespace": "FORMAL",
        "Workers": FORMAL_WORKERS,
        "ChunkSize": FORMAL_CHUNK_SIZE,
        "Resume": True,
        "Diagnostic": False,
        "HCacheEnabled": True,
        "MethodExecutionOrder": list(ORDER_A),
        "ConditionCount": 19,
        "MethodCount": 5,
        "RepeatStart": 1,
        "RepeatEnd": FORMAL_R,
        "ExpectedChunkCount": EXPECTED_CHUNK_COUNT,
        "CompletedChunkCount": EXPECTED_CHUNK_COUNT,
        "AllChunksValid": True,
        "FormalChunksDataSetSHA256": _chunk_data_set_sha256(dataset.chunk_manifests),
        **dict(global_outputs),
        "AuditRepeatIDs": list(AUDIT_REPEAT_IDS),
        "PrimaryAuditSnapshotCount": len(dataset.audit_snapshots),
        "AuditConditionRepeatCount": EXPECTED_AUDIT_CONDITION_REPEATS,
        "AuditRepeatMetricsRowCount": len(audit.repeat_metrics),
        "AuditDiagnosticsRowCount": len(audit.diagnostics),
        "AuditRerunManifestSHA256": audit.manifest["ManifestSHA256"],
        "AuditExactReproducibilityPassed": True,
        "ProjectedPrecisionWarning": True,
        "BootstrapExecuted": False,
        "ConfidenceIntervalGenerated": False,
        "PValueGenerated": False,
        "PaperResultGenerated": False,
        "ConditionDropped": False,
        "ParameterChanged": False,
        "OutcomeDirectionUsed": False,
        "LegacyBoundaryException": dict(regression["legacy_boundary_exception"]),
        "StopAfterStage9": True,
    }


def _status_payload(regression: Mapping[str, object]) -> dict[str, object]:
    return {
        "stage": 9,
        "stage_name": STAGE_NAME,
        "status": "PASS",
        "baseline_stage8_zip_sha256_verified": True,
        "stage8_r_decision_sha256_verified": True,
        "stage8_status_passed": True,
        "stage1_through_stage8_tree_unchanged": True,
        "science_contract_changed": False,
        "formal_executed": True,
        "formal_namespace": "FORMAL",
        "formal_r": FORMAL_R,
        "formal_repeat_range": [1, FORMAL_R],
        "formal_condition_count": 19,
        "formal_method_count": 5,
        "formal_expected_chunk_count": EXPECTED_CHUNK_COUNT,
        "formal_completed_chunk_count": EXPECTED_CHUNK_COUNT,
        "formal_all_chunks_valid": True,
        "formal_repeat_metrics_row_count": EXPECTED_REPEAT_METRICS_ROWS,
        "formal_aggregated_metrics_row_count": EXPECTED_AGGREGATED_ROWS,
        "formal_h300_strata_row_count": EXPECTED_H300_STRATA_ROWS,
        "audit_repeat_ids": list(AUDIT_REPEAT_IDS),
        "audit_condition_repeat_count": EXPECTED_AUDIT_CONDITION_REPEATS,
        "audit_repeat_metrics_row_count": EXPECTED_AUDIT_REPEAT_METRICS_ROWS,
        "audit_diagnostics_row_count": EXPECTED_AUDIT_DIAGNOSTICS_ROWS,
        "audit_exact_reproducibility_passed": True,
        "repeat_conservation_passed": True,
        "transition_conservation_passed": True,
        "ni_identity_passed": True,
        "hprf_no_n_passed": True,
        "composite_no_n_passed": True,
        "stage10_preformal_contract_locked": True,
        "projected_precision_warning": True,
        "bootstrap_executed": False,
        "confidence_interval_generated": False,
        "p_value_generated": False,
        "paper_result_generated": False,
        "condition_dropped": False,
        "parameter_changed": False,
        "outcome_direction_used": False,
        "LegacyBoundaryException": dict(regression["legacy_boundary_exception"]),
        "blocking_issue_count": 0,
        "stop_after_stage9": True,
    }


def _format_pytest_section(label: str, payload: Mapping[str, object]) -> str:
    return (
        f"=== {label} ===\n"
        f"Command: {payload['command']}\n"
        f"Return code: {payload['returncode']}\n"
        f"Passed: {payload['passed']}\n"
        f"Failed: {payload['failed']}\n"
        f"Errors: {payload['errors']}\n"
        f"Wall seconds: {float(payload['wall_seconds']):.6f}\n\n"
        f"{payload['output']}\n"
    )


def _write_test_report(
    *,
    unit_tests: Mapping[str, object],
    preformal_regression: Mapping[str, object],
    postformal_regression: Mapping[str, object],
) -> None:
    sections = [_format_pytest_section("STAGE9 UNIT TESTS", unit_tests)]
    for label, report in (
        ("PREFORMAL COMPLETE PYTEST RAW RESULT", preformal_regression),
        ("POSTFORMAL COMPLETE PYTEST RAW RESULT", postformal_regression),
    ):
        sections.append(_format_pytest_section(label, report["complete_pytest"]))
        exception = report["legacy_boundary_exception"]
        sections.append(
            "AUTHORIZED LEGACY BOUNDARY EXCEPTION (RAW FAILURE RETAINED)\n"
            f"Code: {exception['Code']}\n"
            f"Test node: {exception['TestNode']}\n"
            f"Allowed: {exception['Allowed']}\n"
            f"Reason: {exception['Reason']}\n"
            f"Failure difference scope: {exception['FailureDifferenceScope']}\n"
            f"Frozen Stage 8 modified: {exception['FrozenStage8Modified']}\n"
            f"Stage 8 ZIP identity verified: {exception['Stage8ZipIdentityVerified']}\n"
            f"Frozen science hashes verified: {exception['FrozenScienceHashesVerified']}\n"
            f"Other test failures: {exception['OtherTestFailures']}\n"
            "Legacy external archive exclusions: 0\n\n"
            + _format_pytest_section(
                f"{label} REMAINING SUITE AFTER ONE AUTHORIZED DESELECT",
                report["remaining_regression_suite"],
            )
        )
    write_text_atomic(STAGE9_DIR / "test_report.txt", "\n".join(sections))


def _write_validation_report(
    *,
    postformal_regression: Mapping[str, object],
    formal_wall_seconds: float,
    audit_wall_seconds: float,
) -> None:
    complete = postformal_regression["complete_pytest"]
    remaining = postformal_regression["remaining_regression_suite"]
    exception = postformal_regression["legacy_boundary_exception"]
    text = f"""# Stage 9 Validation Report

## Identity and frozen-boundary gates

- Stage 8 baseline ZIP SHA256 `{STAGE8_BASELINE_ZIP_SHA256}`: PASS.
- Stage 8 R decision, StudyConfig, decision contract, and decision input frozen SHA256 gates: PASS.
- All {len(FROZEN_SCIENCE_HASHES)} independently frozen science/configuration hashes: PASS.
- Stage 1-8 baseline tree byte identity, excluding only the permitted README/pyproject version boundary: PASS.
- Frozen scientific implementation changed: no.

## Preformal lock and execution identity

- Stage 10 Bootstrap scope contract was locked before the first official FORMAL Repeat.
- Namespace=FORMAL, FormalR=2000, RepeatID=1..2000, workers=2, ChunkSize=20, resume=true, diagnostic=false, H cache=true, ORDER_A: PASS.
- The FORMAL representative preflight used validation-only RepeatIDs and was excluded from the statistical dataset.

## Chunk and global data gates

- 1900/1900 chunk manifests passed strict inventory, identity, SHA256, DataSHA256, row-key, and full-load validation.
- `formal_repeat_metrics.csv`: 190000 rows with the exact 19 x 5 x 2000 key universe.
- `formal_aggregated_metrics.csv`: 95 point-estimate/support rows; no Bootstrap interval or p-value fields.
- `formal_h300_width_strata.csv`: 30000 rows; each Method-Repeat conserves 200 cycles across the three frozen strata.
- Repeat, transition, origin, EndState, run-length, and error-source conservation: PASS.
- NI full five-method identity-at-computation and all-C counts: PASS.
- HPRF and Composite no-N gates: PASS.
- Mutable RNG state and full Formal chunk trajectory persistence: absent.

## Preregistered audit

- Primary snapshots: 57/57 for RepeatIDs 1, 1000, and 2000 across all 19 Conditions.
- Independent diagnostic reruns: 57/57; 285 RepeatMetrics and 57000 CycleDiagnostic rows.
- Primary/rerun RepeatMetrics, source state, reference TOA, selected observed TOA (equal_nan), selection mask, and H300 strata: exact PASS.
- Audit reruns were excluded from the formal point-estimate population.

## Regression handling

- Complete pytest raw return code: {complete['returncode']}.
- Complete pytest parsed result: {complete['passed']} passed, {complete['failed']} failed, {complete['errors']} errors.
- Authorized exception code: `{exception['Code']}`.
- Authorized exception node: `{exception['TestNode']}`.
- Reason: {exception['Reason']}.
- Failure difference scope: {exception['FailureDifferenceScope']}.
- Frozen Stage 8 modified: {exception['FrozenStage8Modified']}.
- Stage 8 ZIP identity independently verified: {exception['Stage8ZipIdentityVerified']}.
- Frozen science hashes independently verified: {exception['FrozenScienceHashesVerified']}.
- Other test failures: {exception['OtherTestFailures']}.
- Legacy external archive exclusions: 0.
- Remaining regression suite: {remaining['passed']} passed, {remaining['failed']} failed, {remaining['errors']} errors.
- Raw complete pytest output, including the one failure, is preserved in `test_report.txt`; this report does not claim full-suite PASS.

## Timing and STOP boundary

- Official Formal invocation wall time: {formal_wall_seconds:.6f} seconds.
- Independent audit wall time: {audit_wall_seconds:.6f} seconds.
- ProjectedPrecisionWarning remains true; Stage 9 does not claim that it disappeared.
- Bootstrap, confidence intervals, p-values, paper figures, outcome-direction rules, condition dropping, and parameter changes: not executed.
- STOP after Stage 9: yes.
"""
    write_text_atomic(STAGE9_DIR / "validation_report.md", text)


def _write_changed_files(tree: Mapping[str, object]) -> None:
    artifact_files = tuple(
        path for path in STAGE9_DIR.rglob("*") if path.is_file() and not _excluded_path(path.relative_to(ROOT))
    )
    text = f"""STAGE 9 CHANGED FILES

ADDED STATIC FILES:
- STAGE9_TASKBOOK.md
- src/sal_stability_stage1/formal_storage.py
- src/sal_stability_stage1/stage9.py
- scripts/build_stage9.py
- tests/test_stage9_contract.py
- tests/test_stage9_storage.py
- tests/test_stage9_execution.py

ADDED ARTIFACT TREE:
- artifacts/stage9/** | {len(artifact_files)} files at report generation

MODIFIED FILES:
- README.md | Stage 9 execution and STOP boundary documentation
- pyproject.toml | Version 0.9.0; dependency set unchanged

FROZEN STAGE 1-8 TREE:
- Baseline member count: {tree['BaselineMemberCount']}
- Baseline tree SHA256: {tree['BaselineTreeSHA256']}
- Unexpected modified files: NONE
- Deleted files: NONE

SCIENCE CONTRACT CHANGED: NO
"""
    write_text_atomic(STAGE9_DIR / "changed_files.txt", text)


def _package_members() -> list[Path]:
    return sorted(
        (
            path
            for path in ROOT.rglob("*")
            if path.is_file() and not _excluded_path(path.relative_to(ROOT))
        ),
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )


def package_stage9(output_path: Path | None = None) -> tuple[Path, str]:
    destination = (
        Path(output_path) if output_path is not None else ROOT.parent / FINAL_ZIP_NAME
    ).resolve()
    if destination.parent != ROOT.parent.resolve():
        raise ValueError("Stage 9 package destination must be the workspace parent")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in _package_members():
                relative = path.relative_to(ROOT)
                archive.write(path, (Path(ROOT.name) / relative).as_posix())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    with zipfile.ZipFile(destination) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        relative_names = {
            Path(*Path(name).parts[1:]).as_posix()
            for name in names
            if len(Path(name).parts) >= 2
        }
        required = {
            "STAGE9_TASKBOOK.md",
            "src/sal_stability_stage1/formal_storage.py",
            "src/sal_stability_stage1/stage9.py",
            "scripts/build_stage9.py",
            "artifacts/stage9/formal_repeat_metrics.csv",
            "artifacts/stage9/formal_aggregated_metrics.csv",
            "artifacts/stage9/formal_h300_width_strata.csv",
            "artifacts/stage9/formal_manifest.json",
            "artifacts/stage9/status.json",
            "artifacts/stage9/validation_report.md",
            "artifacts/stage9/test_report.txt",
            "artifacts/stage9/audit/formal_audit_diagnostics.csv",
        }
        missing = sorted(required - relative_names)
        formal_manifests = [
            name
            for name in relative_names
            if name.startswith("artifacts/stage9/formal_chunks/")
            and name.endswith("/chunk_manifest.json")
        ]
        forbidden = [
            name
            for name in relative_names
            if name.startswith("artifacts/stage9/formal_chunks/")
            and Path(name).name in {"chunk_state.npy", "chunk_trajectory.npz"}
        ]
        if missing or len(formal_manifests) != EXPECTED_CHUNK_COUNT or forbidden:
            raise AssertionError(
                f"Stage 9 ZIP validation failed: missing={missing}, "
                f"formal_manifests={len(formal_manifests)}, forbidden={forbidden}"
            )
    return destination, sha256_file(destination)


def build_stage9(progress: Callable[[str], None] = print) -> dict[str, object]:
    baseline_zip = ROOT.parent / STAGE8_BASELINE_ZIP_NAME
    build_started = time.perf_counter()
    progress("[Stage9 STEP 1] Verify Stage 8 baseline ZIP and all frozen inputs")
    identity_report = validate_stage8_inputs(baseline_zip)

    progress("[Stage9 STEP 2] Freeze Stage 10 scope and preformal identity seed")
    STAGE9_DIR.mkdir(parents=True, exist_ok=True)
    stage10_sha = lock_stage10_bootstrap_contract()
    tree_identity_sha = _lock_tree_identity_artifact(
        identity_report["Stage1ThroughStage8Tree"]
    )
    code_sha = compute_stage9_code_sha256()
    identity = FormalIdentityHashes(
        study_config_sha256=STUDY_CONFIG_SHA256,
        stage8_baseline_zip_sha256=STAGE8_BASELINE_ZIP_SHA256,
        stage8_r_decision_sha256=STAGE8_R_DECISION_SHA256,
        stage8_decision_contract_sha256=STAGE8_DECISION_CONTRACT_SHA256,
        stage8_decision_input_sha256=STAGE8_DECISION_INPUT_SHA256,
        stage10_preformal_contract_sha256=stage10_sha,
        code_sha256=code_sha,
    )
    manifest_seed_sha = lock_formal_manifest_seed(
        identity=identity, tree_identity_sha256=tree_identity_sha
    )

    progress("[Stage9 STEP 3] Run Stage 9 unit tests and complete preformal pytest")
    unit_tests = run_stage9_unit_tests()
    preformal_regression = run_full_pytest_with_legacy_archive_rule()

    progress("[Stage9 STEP 4] Run representative FORMAL namespace preflight")
    study_config, encoding_reference = load_stage3_inputs()
    preflight = run_formal_preflight(
        study_config=study_config,
        encoding_reference=encoding_reference,
        identity=identity,
        progress=progress,
    )

    progress("[Stage9 STEP 5] Execute 19 Conditions x R=2000 with official options")
    options = official_formal_execution_options()
    formal_started = time.perf_counter()
    outcomes: list[FormalConditionExecutionOutcome] = []
    for condition in CONDITIONS:
        progress(
            f"[Stage9 FORMAL CONDITION] {condition.ConditionOrdinal:02d} {condition.Code} start"
        )
        outcome = run_formal_condition_chunked(
            condition_ordinal=condition.ConditionOrdinal,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=options,
            identity=identity,
            output_root=FORMAL_CHUNKS_DIR,
            progress=progress,
        )
        if not outcome.completed:
            raise AssertionError(f"Formal condition did not complete: {condition.Code}")
        outcomes.append(outcome)
        progress(
            f"[Stage9 FORMAL CONDITION] {condition.Code} complete; "
            f"computed={outcome.computed_chunks}, skipped={outcome.skipped_chunks}"
        )
    formal_wall = time.perf_counter() - formal_started

    progress("[Stage9 STEP 6] Reload and validate all 1900 chunks strictly from disk")
    dataset = validate_and_load_all_formal_chunks(identity=identity)
    gates = validate_formal_repeat_metrics(dataset.repeat_metrics)
    validate_formal_h300_width_strata(dataset.h300_width_strata)
    if any(
        manifest.get("NIIdentityPassed") is not True
        for manifest in dataset.chunk_manifests
        if manifest["ConditionOrdinal"] == 0
    ):
        raise AssertionError("an NI Resume manifest lacks NIIdentityPassed=true")
    storage_gates = _scan_formal_chunk_storage()

    progress("[Stage9 STEP 7] Write global point estimates and support tables")
    aggregated = aggregate_formal_repeat_metrics(dataset.repeat_metrics)
    global_outputs = write_formal_global_outputs(
        repeat_metrics=dataset.repeat_metrics,
        aggregated_metrics=aggregated,
        h300_width_strata=dataset.h300_width_strata,
    )

    progress("[Stage9 STEP 8] Run 57 independent diagnostic audit reruns")
    audit_started = time.perf_counter()
    audit = run_independent_audit_reruns(
        dataset=dataset,
        study_config=study_config,
        encoding_reference=encoding_reference,
        progress=progress,
    )
    audit_wall = time.perf_counter() - audit_started

    progress("[Stage9 STEP 9] Run post-artifact complete pytest regression")
    postformal_regression = run_full_pytest_with_legacy_archive_rule()
    final_identity_report = validate_stage8_inputs(baseline_zip)
    if final_identity_report["Stage1ThroughStage8Tree"]["BaselineTreeSHA256"] != identity_report["Stage1ThroughStage8Tree"]["BaselineTreeSHA256"]:
        raise AssertionError("Stage 8 baseline tree identity changed during Stage 9")

    progress("[Stage9 STEP 10] Write performance, manifest, validation, and PASS status")
    performance = {
        "Environment": platform.platform(),
        "PythonVersion": platform.python_version(),
        "NumPyVersion": np.__version__,
        "CPUCount": os.cpu_count(),
        "FormalInvocationWallSeconds": formal_wall,
        "AuditWallSeconds": audit_wall,
        "TotalBuildWallSecondsBeforePackaging": time.perf_counter() - build_started,
        "FormalMethodCycles": EXPECTED_METHOD_CYCLES,
        "ComputedChunksThisInvocation": sum(item.computed_chunks for item in outcomes),
        "SkippedValidChunksThisInvocation": sum(item.skipped_chunks for item in outcomes),
        "RecomputedInvalidChunksThisInvocation": sum(
            item.recomputed_invalid_chunks for item in outcomes
        ),
        "ConditionTimings": [
            {
                "ConditionOrdinal": item.condition_ordinal,
                "ConditionCode": item.condition_code,
                "ChunkWallSecondsSum": sum(item.chunk_wall_seconds),
                "ComputedChunks": item.computed_chunks,
                "SkippedChunks": item.skipped_chunks,
                "RecomputedInvalidChunks": item.recomputed_invalid_chunks,
            }
            for item in outcomes
        ],
        "PreflightWallSeconds": preflight["WallSeconds"],
        "BootstrapExecuted": False,
    }
    write_json_atomic(STAGE9_DIR / "performance.json", performance)
    _write_test_report(
        unit_tests=unit_tests,
        preformal_regression=preformal_regression,
        postformal_regression=postformal_regression,
    )
    _write_validation_report(
        postformal_regression=postformal_regression,
        formal_wall_seconds=formal_wall,
        audit_wall_seconds=audit_wall,
    )
    _write_changed_files(final_identity_report["Stage1ThroughStage8Tree"])
    formal_manifest = build_formal_manifest(
        identity=identity,
        manifest_seed_sha256=manifest_seed_sha,
        tree_identity_sha256=tree_identity_sha,
        dataset=dataset,
        global_outputs=global_outputs,
        audit=audit,
        regression=postformal_regression,
    )
    formal_manifest.update(gates)
    formal_manifest.update(storage_gates)
    write_json_atomic(STAGE9_DIR / "formal_manifest.json", formal_manifest)
    status = _status_payload(postformal_regression)
    write_json_atomic(STAGE9_DIR / "status.json", status)
    validate_stage1_through_stage8_tree_unchanged(baseline_zip)

    progress("[Stage9 STEP 11] Package complete Stage 9 ZIP and STOP")
    zip_path, zip_sha = package_stage9()
    return {
        "Stage9Status": status["status"],
        "BaselineStage8ZipSHA256": STAGE8_BASELINE_ZIP_SHA256,
        "CodeSHA256": code_sha,
        "Stage10PreformalContractSHA256": stage10_sha,
        "FormalR": FORMAL_R,
        "Namespace": "FORMAL",
        "CompletedChunks": EXPECTED_CHUNK_COUNT,
        "FormalRepeatMetricsRows": EXPECTED_REPEAT_METRICS_ROWS,
        "FormalAggregatedMetricsRows": EXPECTED_AGGREGATED_ROWS,
        "FormalH300WidthStrataRows": EXPECTED_H300_STRATA_ROWS,
        "AuditConditionRepeats": EXPECTED_AUDIT_CONDITION_REPEATS,
        "AuditDiagnosticsRows": EXPECTED_AUDIT_DIAGNOSTICS_ROWS,
        "ProjectedPrecisionWarning": True,
        "BootstrapExecuted": False,
        "ConfidenceIntervalGenerated": False,
        "PValueGenerated": False,
        "OutcomeDirectionUsed": False,
        "FinalZip": str(zip_path),
        "FinalZipSHA256": zip_sha,
        "StopAfterStage9": True,
    }


__all__ = [
    "AUDIT_DIAGNOSTICS_PATH",
    "AUDIT_DIR",
    "AUDIT_H300_STRATA_PATH",
    "AUDIT_MANIFEST_PATH",
    "AUDIT_REPEAT_METRICS_PATH",
    "EXPECTED_AGGREGATED_ROWS",
    "EXPECTED_AUDIT_CONDITION_REPEATS",
    "EXPECTED_AUDIT_DIAGNOSTICS_ROWS",
    "EXPECTED_AUDIT_REPEAT_METRICS_ROWS",
    "EXPECTED_CHUNK_COUNT",
    "EXPECTED_H300_STRATA_ROWS",
    "EXPECTED_REPEAT_METRICS_ROWS",
    "FINAL_ZIP_NAME",
    "FORMAL_AGGREGATED_METRICS_PATH",
    "FORMAL_CHUNKS_DIR",
    "FORMAL_H300_STRATA_PATH",
    "FORMAL_REPEAT_METRICS_PATH",
    "LEGACY_BOUNDARY_EXCEPTION_CODE",
    "LEGACY_BOUNDARY_EXCEPTION_NODE",
    "FormalConditionExecutionOutcome",
    "FormalDataset",
    "aggregate_formal_repeat_metrics",
    "build_stage9",
    "lock_formal_manifest_seed",
    "lock_stage10_bootstrap_contract",
    "official_formal_execution_options",
    "package_stage9",
    "run_formal_condition_chunked",
    "run_formal_preflight",
    "run_full_pytest_with_legacy_archive_rule",
    "run_independent_audit_reruns",
    "stage10_bootstrap_contract",
    "validate_and_load_all_formal_chunks",
    "validate_formal_h300_width_strata",
    "validate_formal_repeat_metrics",
    "validate_official_formal_execution_options",
    "validate_stage1_through_stage8_tree_unchanged",
    "validate_stage8_inputs",
    "verify_stage8_baseline_zip",
    "write_formal_global_outputs",
]
