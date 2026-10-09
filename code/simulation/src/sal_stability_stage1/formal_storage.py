from __future__ import annotations

import csv
import hashlib
import json
import operator
import os
import re
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TypeVar

import numpy as np

from .build import ROOT
from .contracts import METHODS
from .execution import ConditionRunResult, ORDER_A
from .metrics import RepeatMetrics
from .pilot_storage import (
    FORBIDDEN_RNG_STATE_FIELDS,
    compute_data_sha256,
    sha256_file,
    write_csv_dicts_atomic,
    write_json_atomic,
)
from .rng import RandomNamespace
from .stage5 import H300WidthStratumMetrics, WIDTH_STRATA


SCHEMA_VERSION = 1
CODE_VERSION = "0.9.0"
RUN_KIND = "FORMAL"
FORMAL_R = 2000
FORMAL_WORKERS = 2
FORMAL_CHUNK_SIZE = 20
FORMAL_REPEAT_IDS = tuple(range(1, FORMAL_R + 1))
AUDIT_REPEAT_IDS = (1, 1000, 2000)

STAGE8_BASELINE_ZIP_SHA256 = (
    "a440cae9bbef5474bf30d8abc32078646bf76219910a6a37e31863065e233a3b"
)
STUDY_CONFIG_SHA256 = (
    "4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb"
)
STAGE8_R_DECISION_SHA256 = (
    "13a8c9ec8a7debb0c0d3c64ac790efb12fa4f444320ec77ff5812c5e7a48c70c"
)
STAGE8_DECISION_CONTRACT_SHA256 = (
    "1991d8cec0a91438ccd95a5e96ed02cfd4d27285e9e2fa4eba047f9938e285e6"
)
STAGE8_DECISION_INPUT_SHA256 = (
    "655ffd9f03a7a7824662c9a7a1a2281ed620c73019239236e386fb15eac0eb7b"
)

REPEAT_METRICS_FILENAME = "chunk_repeat_metrics.csv"
H300_STRATA_FILENAME = "chunk_h300_width_strata.csv"
MANIFEST_FILENAME = "chunk_manifest.json"
AUDIT_SNAPSHOT_TEMPLATE = "audit_repeat_{repeat_id:04d}_snapshot.npz"

HPRF_CONDITION_ORDINALS = frozenset(range(1, 6))
COMPOSITE_CONDITION_ORDINALS = frozenset(range(10, 19))

CODE_FINGERPRINT_FILES = (
    "src/sal_stability_stage1/contracts.py",
    "src/sal_stability_stage1/encoding.py",
    "src/sal_stability_stage1/rng.py",
    "src/sal_stability_stage1/events.py",
    "src/sal_stability_stage1/hprf.py",
    "src/sal_stability_stage1/selectors.py",
    "src/sal_stability_stage1/kernel.py",
    "src/sal_stability_stage1/metrics.py",
    "src/sal_stability_stage1/diagnostics.py",
    "src/sal_stability_stage1/execution.py",
    "src/sal_stability_stage1/storage.py",
    "src/sal_stability_stage1/pilot_storage.py",
    "src/sal_stability_stage1/stage3.py",
    "src/sal_stability_stage1/stage4.py",
    "src/sal_stability_stage1/stage5.py",
    "src/sal_stability_stage1/stage6.py",
    "src/sal_stability_stage1/stage7.py",
    "src/sal_stability_stage1/stage8.py",
    "src/sal_stability_stage1/formal_storage.py",
    "src/sal_stability_stage1/stage9.py",
    "scripts/build_stage9.py",
)


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _require_lower_sha256(value: object, field_name: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA256")
    return result


def canonical_formal_repeat_ids(values: Iterable[int]) -> tuple[int, ...]:
    repeat_ids = tuple(_strict_integer(value, "RepeatID") for value in values)
    if len(repeat_ids) != FORMAL_CHUNK_SIZE:
        raise ValueError("a Stage 9 Formal chunk must contain exactly 20 RepeatIDs")
    if len(repeat_ids) != len(set(repeat_ids)):
        raise ValueError("Formal chunk RepeatIDs cannot contain duplicates")
    canonical = tuple(sorted(repeat_ids))
    if canonical[0] < 1 or canonical[-1] > FORMAL_R:
        raise ValueError("Formal chunk RepeatIDs must lie in 1..2000")
    if any(right != left + 1 for left, right in zip(canonical, canonical[1:])):
        raise ValueError("Formal chunk RepeatIDs must be contiguous")
    if (canonical[0] - 1) % FORMAL_CHUNK_SIZE != 0:
        raise ValueError("Formal chunk must align to the frozen 20-Repeat partition")
    return canonical


def partition_formal_repeat_ids() -> tuple[tuple[int, ...], ...]:
    return tuple(
        FORMAL_REPEAT_IDS[offset : offset + FORMAL_CHUNK_SIZE]
        for offset in range(0, FORMAL_R, FORMAL_CHUNK_SIZE)
    )


def _condition_slug(condition_code: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(condition_code).lower()).strip("-")
    if not slug:
        raise ValueError("condition_code cannot produce an empty path component")
    return slug


@dataclass(frozen=True, slots=True)
class FormalIdentityHashes:
    study_config_sha256: str
    stage8_baseline_zip_sha256: str
    stage8_r_decision_sha256: str
    stage8_decision_contract_sha256: str
    stage8_decision_input_sha256: str
    stage10_preformal_contract_sha256: str
    code_sha256: str

    def __post_init__(self) -> None:
        for field_info in fields(type(self)):
            object.__setattr__(
                self,
                field_info.name,
                _require_lower_sha256(getattr(self, field_info.name), field_info.name),
            )


@dataclass(frozen=True, slots=True)
class FormalChunkPaths:
    directory: Path
    repeat_metrics: Path
    h300_width_strata: Path
    audit_snapshot: Path | None
    manifest: Path


def formal_chunk_paths(
    root: Path,
    *,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
) -> FormalChunkPaths:
    canonical = canonical_formal_repeat_ids(repeat_ids)
    ordinal = _strict_integer(condition_ordinal, "condition_ordinal")
    if not 0 <= ordinal < 19:
        raise ValueError("condition_ordinal must lie in 0..18")
    audit_ids = tuple(repeat_id for repeat_id in canonical if repeat_id in AUDIT_REPEAT_IDS)
    if len(audit_ids) > 1:
        raise AssertionError("a frozen Stage 9 chunk cannot contain multiple audit repeats")
    directory = (
        Path(root)
        / f"condition_{ordinal:02d}_{_condition_slug(condition_code)}"
        / f"repeats_{canonical[0]}_{canonical[-1]}"
    )
    audit_snapshot = (
        directory / AUDIT_SNAPSHOT_TEMPLATE.format(repeat_id=audit_ids[0])
        if audit_ids
        else None
    )
    return FormalChunkPaths(
        directory=directory,
        repeat_metrics=directory / REPEAT_METRICS_FILENAME,
        h300_width_strata=directory / H300_STRATA_FILENAME,
        audit_snapshot=audit_snapshot,
        manifest=directory / MANIFEST_FILENAME,
    )


def compute_stage9_code_sha256(root: Path = ROOT) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for relative in CODE_FINGERPRINT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Stage 9 code fingerprint input is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_binary_write(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_dataclass_csv(path: Path, rows: Sequence[object], row_type: type) -> None:
    materialized = tuple(rows)
    if any(not isinstance(row, row_type) for row in materialized):
        raise TypeError(f"CSV rows must all be {row_type.__name__}")
    write_csv_dicts_atomic(
        path,
        field_names=tuple(field.name for field in fields(row_type)),
        rows=(asdict(row) for row in materialized),
    )


def _write_audit_snapshot(
    path: Path,
    *,
    result: ConditionRunResult,
    repeat_id: int,
) -> None:
    position = result.repeat_ids.index(repeat_id)

    def writer(handle) -> None:
        np.savez(
            handle,
            Namespace=np.array(RandomNamespace.FORMAL.name),
            ConditionOrdinal=np.array(result.condition_ordinal, dtype=np.int16),
            ConditionCode=np.array(result.condition_code),
            RepeatID=np.array(repeat_id, dtype=np.int32),
            MethodOrdinals=np.arange(5, dtype=np.int8),
            source_state_code=result.source_state[:, position, :],
            ref_toa_before_s=result.ref_toa_before_s[:, position, :],
            selected_observed_toa_s=result.selected_observed_toa_s[:, position, :],
            has_selection=result.has_selection[:, position, :],
        )

    _atomic_binary_write(path, writer)


def _ni_metrics_identity_passed(rows: Sequence[RepeatMetrics]) -> bool:
    for repeat_id in sorted({row.RepeatID for row in rows}):
        subset = tuple(row for row in rows if row.RepeatID == repeat_id)
        if len(subset) != 5 or any((row.N_C, row.N_E, row.N_N) != (200, 0, 0) for row in subset):
            return False
        payloads = []
        for row in subset:
            payload = asdict(row)
            payload.pop("MethodOrdinal")
            payload.pop("MethodCode")
            payloads.append(payload)
        if any(payload != payloads[0] for payload in payloads[1:]):
            return False
    return True


def _ni_full_trajectory_identity_passed(result: ConditionRunResult) -> bool:
    if result.condition_ordinal != 0 or result.condition_code != "NI":
        return False
    for position in range(len(result.repeat_ids)):
        for method_ordinal in range(1, 5):
            if not np.array_equal(result.source_state[0, position], result.source_state[method_ordinal, position]):
                return False
            if not np.array_equal(result.ref_toa_before_s[0, position], result.ref_toa_before_s[method_ordinal, position]):
                return False
            if not np.array_equal(
                result.selected_observed_toa_s[0, position],
                result.selected_observed_toa_s[method_ordinal, position],
                equal_nan=True,
            ):
                return False
            if not np.array_equal(result.has_selection[0, position], result.has_selection[method_ordinal, position]):
                return False
    return _ni_metrics_identity_passed(result.repeat_metrics)


def write_formal_chunk_atomic(
    root: Path,
    *,
    result: ConditionRunResult,
    identity: FormalIdentityHashes,
) -> FormalChunkPaths:
    if not isinstance(result, ConditionRunResult):
        raise TypeError("result must be ConditionRunResult")
    if not isinstance(identity, FormalIdentityHashes):
        raise TypeError("identity must be FormalIdentityHashes")
    if result.namespace is not RandomNamespace.FORMAL:
        raise ValueError("Stage 9 chunks require Namespace=FORMAL")
    if result.diagnostics:
        raise ValueError("Stage 9 primary Formal chunks cannot contain diagnostics")
    repeat_ids = canonical_formal_repeat_ids(result.repeat_ids)
    if len(result.repeat_metrics) != FORMAL_CHUNK_SIZE * 5:
        raise ValueError("a Formal chunk must contain exactly 100 RepeatMetrics rows")
    if result.condition_code == "H300":
        if len(result.h300_width_strata) != FORMAL_CHUNK_SIZE * 5 * 3:
            raise ValueError("an H300 Formal chunk must contain exactly 300 strata rows")
    elif result.h300_width_strata:
        raise ValueError("only H300 may contain width-stratum rows")

    paths = formal_chunk_paths(
        root,
        condition_ordinal=result.condition_ordinal,
        condition_code=result.condition_code,
        repeat_ids=repeat_ids,
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    _write_dataclass_csv(paths.repeat_metrics, result.repeat_metrics, RepeatMetrics)
    ordered_hashes = [(REPEAT_METRICS_FILENAME, sha256_file(paths.repeat_metrics))]

    if result.condition_code == "H300":
        _write_dataclass_csv(
            paths.h300_width_strata,
            result.h300_width_strata,
            H300WidthStratumMetrics,
        )
        ordered_hashes.append(
            (H300_STRATA_FILENAME, sha256_file(paths.h300_width_strata))
        )

    audit_ids = tuple(repeat_id for repeat_id in repeat_ids if repeat_id in AUDIT_REPEAT_IDS)
    if audit_ids:
        if paths.audit_snapshot is None:
            raise AssertionError("audit snapshot path was not constructed")
        _write_audit_snapshot(paths.audit_snapshot, result=result, repeat_id=audit_ids[0])
        ordered_hashes.append((paths.audit_snapshot.name, sha256_file(paths.audit_snapshot)))

    manifest: dict[str, object] = {
        "SchemaVersion": SCHEMA_VERSION,
        "StudyConfigSHA256": identity.study_config_sha256,
        "Stage8BaselineZIPSHA256": identity.stage8_baseline_zip_sha256,
        "Stage8RDecisionSHA256": identity.stage8_r_decision_sha256,
        "Stage8DecisionContractSHA256": identity.stage8_decision_contract_sha256,
        "Stage8DecisionInputSHA256": identity.stage8_decision_input_sha256,
        "Stage10PreformalContractSHA256": identity.stage10_preformal_contract_sha256,
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": identity.code_sha256,
        "RunKind": RUN_KIND,
        "Namespace": RandomNamespace.FORMAL.name,
        "FormalR": FORMAL_R,
        "Workers": FORMAL_WORKERS,
        "ChunkSize": FORMAL_CHUNK_SIZE,
        "Resume": True,
        "Diagnostic": False,
        "HCacheEnabled": True,
        "MethodExecutionOrder": list(ORDER_A),
        "ConditionOrdinal": result.condition_ordinal,
        "ConditionCode": result.condition_code,
        "RepeatStart": repeat_ids[0],
        "RepeatEnd": repeat_ids[-1],
        "RepeatCount": len(repeat_ids),
        "Completed": True,
        "RepeatMetricsSHA256": ordered_hashes[0][1],
        "AuditRepeatIDs": list(audit_ids),
    }
    if result.condition_code == "H300":
        manifest["H300WidthStrataSHA256"] = dict(ordered_hashes)[H300_STRATA_FILENAME]
    if audit_ids:
        manifest["AuditSnapshotSHA256"] = dict(ordered_hashes)[paths.audit_snapshot.name]
    if result.condition_ordinal == 0:
        if not _ni_full_trajectory_identity_passed(result):
            raise AssertionError("NI five-method full trajectory identity failed")
        manifest["NIIdentityPassed"] = True
    if result.condition_ordinal in HPRF_CONDITION_ORDINALS | COMPOSITE_CONDITION_ORDINALS:
        if any(row.N_N != 0 for row in result.repeat_metrics):
            raise AssertionError("an HPRF/composite Formal row contains forbidden N state")
        manifest["NoNPassed"] = True
    manifest["DataSHA256"] = compute_data_sha256(ordered_hashes)
    write_json_atomic(paths.manifest, manifest)
    return paths


@dataclass(frozen=True, slots=True)
class AuditSnapshot:
    namespace: str
    condition_ordinal: int
    condition_code: str
    repeat_id: int
    method_ordinals: np.ndarray
    source_state_code: np.ndarray
    ref_toa_before_s: np.ndarray
    selected_observed_toa_s: np.ndarray
    has_selection: np.ndarray

    def __post_init__(self) -> None:
        if self.namespace != "FORMAL":
            raise ValueError("audit snapshot Namespace must be FORMAL")
        if not 0 <= self.condition_ordinal < 19:
            raise ValueError("audit snapshot ConditionOrdinal must lie in 0..18")
        if self.repeat_id not in AUDIT_REPEAT_IDS:
            raise ValueError("audit snapshot RepeatID is not preregistered")
        method_ordinals = np.array(self.method_ordinals, dtype=np.int8, copy=True)
        source = np.array(self.source_state_code, dtype=np.uint8, copy=True)
        reference = np.array(self.ref_toa_before_s, dtype=np.float64, copy=True)
        selected = np.array(self.selected_observed_toa_s, dtype=np.float64, copy=True)
        selection = np.array(self.has_selection, dtype=np.bool_, copy=True)
        if not np.array_equal(method_ordinals, np.arange(5, dtype=np.int8)):
            raise ValueError("audit snapshot method identity is not 0..4")
        if any(values.shape != (5, 200) for values in (source, reference, selected, selection)):
            raise ValueError("audit snapshot trajectory arrays must have shape (5, 200)")
        if not np.array_equal(selection, source != np.uint8(0)):
            raise ValueError("audit snapshot selection mask disagrees with source state")
        if np.any(~np.isfinite(reference)):
            raise ValueError("audit reference TOAs must be finite")
        if np.any(~np.isfinite(selected[selection])) or np.any(~np.isnan(selected[~selection])):
            raise ValueError("audit selected TOAs violate finite/NaN identity")
        for values in (method_ordinals, source, reference, selected, selection):
            values.setflags(write=False)
        object.__setattr__(self, "method_ordinals", method_ordinals)
        object.__setattr__(self, "source_state_code", source)
        object.__setattr__(self, "ref_toa_before_s", reference)
        object.__setattr__(self, "selected_observed_toa_s", selected)
        object.__setattr__(self, "has_selection", selection)


def _load_audit_snapshot(path: Path) -> AuditSnapshot:
    required = {
        "Namespace",
        "ConditionOrdinal",
        "ConditionCode",
        "RepeatID",
        "MethodOrdinals",
        "source_state_code",
        "ref_toa_before_s",
        "selected_observed_toa_s",
        "has_selection",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("audit snapshot archive has an invalid schema")
        return AuditSnapshot(
            namespace=str(archive["Namespace"].item()),
            condition_ordinal=int(archive["ConditionOrdinal"].item()),
            condition_code=str(archive["ConditionCode"].item()),
            repeat_id=int(archive["RepeatID"].item()),
            method_ordinals=archive["MethodOrdinals"],
            source_state_code=archive["source_state_code"],
            ref_toa_before_s=archive["ref_toa_before_s"],
            selected_observed_toa_s=archive["selected_observed_toa_s"],
            has_selection=archive["has_selection"],
        )


T = TypeVar("T")


def _parse_dataclass_csv(
    path: Path,
    row_type: type[T],
    *,
    string_fields: frozenset[str],
) -> tuple[T, ...]:
    names = tuple(field.name for field in fields(row_type))
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != names:
            raise ValueError(f"{Path(path).name} has an invalid schema")
        result: list[T] = []
        for raw in reader:
            if None in raw or set(raw) != set(names):
                raise ValueError(f"{Path(path).name} contains an invalid row schema")
            values = {
                name: raw[name] if name in string_fields else int(raw[name])
                for name in names
            }
            result.append(row_type(**values))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FormalChunkData:
    manifest: dict[str, object]
    repeat_metrics: tuple[RepeatMetrics, ...]
    h300_width_strata: tuple[H300WidthStratumMetrics, ...]
    audit_snapshot: AuditSnapshot | None


@dataclass(frozen=True, slots=True)
class FormalChunkValidation:
    valid: bool
    reason: str
    manifest: dict[str, object] | None = None
    data: FormalChunkData | None = None


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Formal chunk manifest must be a JSON object")
    return payload


def _expected_manifest_identity(
    *,
    identity: FormalIdentityHashes,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: tuple[int, ...],
) -> dict[str, object]:
    return {
        "SchemaVersion": SCHEMA_VERSION,
        "StudyConfigSHA256": identity.study_config_sha256,
        "Stage8BaselineZIPSHA256": identity.stage8_baseline_zip_sha256,
        "Stage8RDecisionSHA256": identity.stage8_r_decision_sha256,
        "Stage8DecisionContractSHA256": identity.stage8_decision_contract_sha256,
        "Stage8DecisionInputSHA256": identity.stage8_decision_input_sha256,
        "Stage10PreformalContractSHA256": identity.stage10_preformal_contract_sha256,
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": identity.code_sha256,
        "RunKind": RUN_KIND,
        "Namespace": "FORMAL",
        "FormalR": FORMAL_R,
        "Workers": FORMAL_WORKERS,
        "ChunkSize": FORMAL_CHUNK_SIZE,
        "Resume": True,
        "Diagnostic": False,
        "HCacheEnabled": True,
        "MethodExecutionOrder": list(ORDER_A),
        "ConditionOrdinal": condition_ordinal,
        "ConditionCode": condition_code,
        "RepeatStart": repeat_ids[0],
        "RepeatEnd": repeat_ids[-1],
        "RepeatCount": FORMAL_CHUNK_SIZE,
        "Completed": True,
        "AuditRepeatIDs": [value for value in repeat_ids if value in AUDIT_REPEAT_IDS],
    }


def _validate_row_identity(
    rows: Sequence[RepeatMetrics],
    *,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: tuple[int, ...],
) -> None:
    expected = {
        (repeat_id, method.MethodOrdinal)
        for repeat_id in repeat_ids
        for method in METHODS
    }
    actual = [(row.RepeatID, row.MethodOrdinal) for row in rows]
    if len(rows) != 100 or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Formal chunk RepeatMetrics keys are incomplete or duplicated")
    if any(
        row.ConditionOrdinal != condition_ordinal or row.ConditionCode != condition_code
        for row in rows
    ):
        raise ValueError("Formal chunk RepeatMetrics condition identity is invalid")


def _validate_h300_rows(
    rows: Sequence[H300WidthStratumMetrics], repeat_ids: tuple[int, ...]
) -> None:
    rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
    expected = {
        (repeat_id, method.MethodOrdinal, stratum)
        for repeat_id in repeat_ids
        for method in METHODS
        for stratum in WIDTH_STRATA
    }
    actual = [(row.RepeatID, row.MethodOrdinal, row.WidthStratum) for row in rows]
    if len(rows) != 300 or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("H300 chunk width-stratum keys are incomplete or duplicated")
    for repeat_id in repeat_ids:
        for method in METHODS:
            subset = tuple(
                row
                for row in rows
                if row.RepeatID == repeat_id and row.MethodOrdinal == method.MethodOrdinal
            )
            if len(subset) != 3 or sum(row.StratumCycleCount for row in subset) != 200:
                raise ValueError("H300 chunk width strata do not conserve 200 cycles")
            if tuple(sorted((rank[row.WidthStratum] for row in subset))) != (0, 1, 2):
                raise ValueError("H300 chunk width strata are not canonical")


def validate_formal_chunk(
    paths: FormalChunkPaths,
    *,
    identity: FormalIdentityHashes,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
) -> FormalChunkValidation:
    canonical = canonical_formal_repeat_ids(repeat_ids)
    expected_identity = _expected_manifest_identity(
        identity=identity,
        condition_ordinal=_strict_integer(condition_ordinal, "condition_ordinal"),
        condition_code=str(condition_code),
        repeat_ids=canonical,
    )
    try:
        manifest = _read_manifest(paths.manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return FormalChunkValidation(False, f"manifest_unreadable:{type(exc).__name__}")
    forbidden = FORBIDDEN_RNG_STATE_FIELDS.intersection(manifest)
    forbidden |= {"StateSHA256", "TrajectorySHA256", "BootstrapIndexSHA256"}.intersection(manifest)
    if forbidden:
        return FormalChunkValidation(False, f"forbidden_manifest_field:{sorted(forbidden)[0]}", manifest)
    for field_name, expected_value in expected_identity.items():
        if manifest.get(field_name) != expected_value:
            return FormalChunkValidation(False, f"manifest_mismatch:{field_name}", manifest)

    audit_ids = tuple(expected_identity["AuditRepeatIDs"])
    expected_files = {REPEAT_METRICS_FILENAME, MANIFEST_FILENAME}
    if condition_code == "H300":
        expected_files.add(H300_STRATA_FILENAME)
    if audit_ids:
        if paths.audit_snapshot is None:
            return FormalChunkValidation(False, "audit_path_missing", manifest)
        expected_files.add(paths.audit_snapshot.name)
    try:
        actual_files = {path.name for path in paths.directory.iterdir() if path.is_file()}
        actual_directories = tuple(path for path in paths.directory.iterdir() if path.is_dir())
    except OSError as exc:
        return FormalChunkValidation(False, f"chunk_directory_unreadable:{type(exc).__name__}", manifest)
    if actual_directories or actual_files != expected_files:
        return FormalChunkValidation(False, "unexpected_or_missing_chunk_artifact", manifest)

    try:
        metrics_sha = sha256_file(paths.repeat_metrics)
        if manifest.get("RepeatMetricsSHA256") != metrics_sha:
            return FormalChunkValidation(False, "hash_mismatch:repeat_metrics", manifest)
        ordered_hashes = [(REPEAT_METRICS_FILENAME, metrics_sha)]
        strata: tuple[H300WidthStratumMetrics, ...] = ()
        if condition_code == "H300":
            strata_sha = sha256_file(paths.h300_width_strata)
            if manifest.get("H300WidthStrataSHA256") != strata_sha:
                return FormalChunkValidation(False, "hash_mismatch:h300_strata", manifest)
            ordered_hashes.append((H300_STRATA_FILENAME, strata_sha))
        elif "H300WidthStrataSHA256" in manifest:
            return FormalChunkValidation(False, "unexpected_h300_manifest_field", manifest)

        snapshot: AuditSnapshot | None = None
        if audit_ids:
            snapshot_sha = sha256_file(paths.audit_snapshot)
            if manifest.get("AuditSnapshotSHA256") != snapshot_sha:
                return FormalChunkValidation(False, "hash_mismatch:audit_snapshot", manifest)
            ordered_hashes.append((paths.audit_snapshot.name, snapshot_sha))
        elif "AuditSnapshotSHA256" in manifest:
            return FormalChunkValidation(False, "unexpected_audit_manifest_field", manifest)
        if manifest.get("DataSHA256") != compute_data_sha256(ordered_hashes):
            return FormalChunkValidation(False, "hash_mismatch:data", manifest)

        metrics = _parse_dataclass_csv(
            paths.repeat_metrics,
            RepeatMetrics,
            string_fields=frozenset(("ConditionCode", "MethodCode", "EndState")),
        )
        _validate_row_identity(
            metrics,
            condition_ordinal=condition_ordinal,
            condition_code=condition_code,
            repeat_ids=canonical,
        )
        if condition_code == "H300":
            strata = _parse_dataclass_csv(
                paths.h300_width_strata,
                H300WidthStratumMetrics,
                string_fields=frozenset(("ConditionCode", "MethodCode", "WidthStratum")),
            )
            _validate_h300_rows(strata, canonical)
        if condition_ordinal == 0:
            if manifest.get("NIIdentityPassed") is not True or not _ni_metrics_identity_passed(metrics):
                return FormalChunkValidation(False, "ni_identity_failed", manifest)
        elif "NIIdentityPassed" in manifest:
            return FormalChunkValidation(False, "unexpected_ni_manifest_field", manifest)
        if condition_ordinal in HPRF_CONDITION_ORDINALS | COMPOSITE_CONDITION_ORDINALS:
            if manifest.get("NoNPassed") is not True or any(row.N_N != 0 for row in metrics):
                return FormalChunkValidation(False, "no_n_gate_failed", manifest)
        elif "NoNPassed" in manifest:
            return FormalChunkValidation(False, "unexpected_no_n_manifest_field", manifest)
        if audit_ids:
            snapshot = _load_audit_snapshot(paths.audit_snapshot)
            if (
                snapshot.condition_ordinal != condition_ordinal
                or snapshot.condition_code != condition_code
                or snapshot.repeat_id != audit_ids[0]
            ):
                return FormalChunkValidation(False, "audit_snapshot_identity_failed", manifest)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        return FormalChunkValidation(False, f"data_invalid:{type(exc).__name__}", manifest)

    data = FormalChunkData(
        manifest=dict(manifest),
        repeat_metrics=metrics,
        h300_width_strata=strata,
        audit_snapshot=snapshot,
    )
    return FormalChunkValidation(True, "valid", manifest, data)


def load_validated_formal_chunk(
    paths: FormalChunkPaths,
    *,
    identity: FormalIdentityHashes,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
) -> FormalChunkData:
    validation = validate_formal_chunk(
        paths,
        identity=identity,
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        repeat_ids=repeat_ids,
    )
    if not validation.valid or validation.data is None:
        raise ValueError(f"Formal chunk is not resume-eligible: {validation.reason}")
    return validation.data


def reset_formal_chunk_directory(paths: FormalChunkPaths, *, output_root: Path) -> None:
    root = Path(output_root).resolve()
    directory = paths.directory.resolve()
    if root == Path(root.anchor) or root not in directory.parents:
        raise ValueError("refusing to reset a Formal chunk outside its output root")
    relative = directory.relative_to(root)
    if len(relative.parts) != 2 or not relative.parts[0].startswith("condition_") or not relative.parts[1].startswith("repeats_"):
        raise ValueError("refusing to reset a path that is not a canonical Formal chunk")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "AUDIT_REPEAT_IDS",
    "AUDIT_SNAPSHOT_TEMPLATE",
    "CODE_FINGERPRINT_FILES",
    "CODE_VERSION",
    "COMPOSITE_CONDITION_ORDINALS",
    "FORMAL_CHUNK_SIZE",
    "FORMAL_R",
    "FORMAL_REPEAT_IDS",
    "FORMAL_WORKERS",
    "H300_STRATA_FILENAME",
    "HPRF_CONDITION_ORDINALS",
    "MANIFEST_FILENAME",
    "REPEAT_METRICS_FILENAME",
    "RUN_KIND",
    "SCHEMA_VERSION",
    "STAGE8_BASELINE_ZIP_SHA256",
    "STAGE8_DECISION_CONTRACT_SHA256",
    "STAGE8_DECISION_INPUT_SHA256",
    "STAGE8_R_DECISION_SHA256",
    "STUDY_CONFIG_SHA256",
    "AuditSnapshot",
    "FormalChunkData",
    "FormalChunkPaths",
    "FormalChunkValidation",
    "FormalIdentityHashes",
    "canonical_formal_repeat_ids",
    "compute_stage9_code_sha256",
    "formal_chunk_paths",
    "load_validated_formal_chunk",
    "partition_formal_repeat_ids",
    "reset_formal_chunk_directory",
    "validate_formal_chunk",
    "write_formal_chunk_atomic",
]
