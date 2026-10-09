from __future__ import annotations

import csv
import hashlib
import json
import os
import operator
import re
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from .build import ROOT
from .diagnostics import canonical_diagnostics
from .execution import ConditionRunResult
from .metrics import RepeatMetrics
from .rng import RandomNamespace
from .stage5 import H300_CONDITION_CODE, H300WidthStratumMetrics, WIDTH_STRATA


CODE_VERSION = "0.6.0"
RUN_KIND = "VALIDATION"
STATE_FILENAME = "chunk_state.npy"
TRAJECTORY_FILENAME = "chunk_trajectory.npz"
REPEAT_METRICS_FILENAME = "chunk_repeat_metrics.csv"
H300_STRATA_FILENAME = "chunk_h300_width_strata.csv"
MANIFEST_FILENAME = "chunk_manifest.json"
CODE_FINGERPRINT_FILES = (
    "src/sal_stability_stage1/kernel.py",
    "src/sal_stability_stage1/diagnostics.py",
    "src/sal_stability_stage1/execution.py",
    "src/sal_stability_stage1/storage.py",
    "src/sal_stability_stage1/stage6.py",
)


def sha256_file(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_code_sha256(root: Path = ROOT) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for relative in CODE_FINGERPRINT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"code fingerprint input is missing: {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def compute_data_sha256(file_hashes: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for filename, file_hash in file_hashes:
        digest.update(filename.encode("utf-8"))
        digest.update(b":")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _canonical_repeat_ids(values: Iterable[int]) -> tuple[int, ...]:
    repeat_ids = tuple(_strict_integer(value, "RepeatID") for value in values)
    if not repeat_ids:
        raise ValueError("repeat_ids cannot be empty")
    if any(value < 1 for value in repeat_ids):
        raise ValueError("RepeatID must be positive")
    if len(repeat_ids) != len(set(repeat_ids)):
        raise ValueError("repeat_ids cannot contain duplicates")
    result = tuple(sorted(repeat_ids))
    if any(right != left + 1 for left, right in zip(result, result[1:])):
        raise ValueError("RepeatChunk RepeatID values must be contiguous")
    return result


def partition_repeat_ids(
    repeat_ids: Iterable[int], chunk_size: int
) -> tuple[tuple[int, ...], ...]:
    canonical = _canonical_repeat_ids(repeat_ids)
    size = _strict_integer(chunk_size, "chunk_size")
    if size < 1:
        raise ValueError("chunk_size must be at least one")
    return tuple(
        canonical[offset : offset + size]
        for offset in range(0, len(canonical), size)
    )


def _condition_slug(condition_code: str) -> str:
    code = str(condition_code)
    slug = re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")
    if not slug:
        raise ValueError("condition_code cannot produce an empty path component")
    return slug


@dataclass(frozen=True, slots=True)
class ChunkPaths:
    directory: Path
    state: Path
    trajectory: Path
    repeat_metrics: Path
    h300_width_strata: Path
    manifest: Path


def chunk_paths(
    root: Path,
    *,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
) -> ChunkPaths:
    canonical = _canonical_repeat_ids(repeat_ids)
    ordinal = _strict_integer(condition_ordinal, "condition_ordinal")
    if not 0 <= ordinal < 19:
        raise ValueError("condition_ordinal must lie in 0..18")
    directory = (
        Path(root)
        / f"condition_{ordinal:02d}_{_condition_slug(condition_code)}"
        / f"repeats_{canonical[0]}_{canonical[-1]}"
    )
    return ChunkPaths(
        directory=directory,
        state=directory / STATE_FILENAME,
        trajectory=directory / TRAJECTORY_FILENAME,
        repeat_metrics=directory / REPEAT_METRICS_FILENAME,
        h300_width_strata=directory / H300_STRATA_FILENAME,
        manifest=directory / MANIFEST_FILENAME,
    )


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_npy(path: Path, values: np.ndarray) -> None:
    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(path, writer)


def _write_trajectory(path: Path, result: ConditionRunResult) -> None:
    def writer(temporary: Path) -> None:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                ref_toa_before_s=result.ref_toa_before_s,
                selected_observed_toa_s=result.selected_observed_toa_s,
                has_selection=result.has_selection,
            )
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(path, writer)


T = TypeVar("T")


def _write_dataclass_csv(
    path: Path,
    rows: Sequence[T],
    row_type: type[T],
) -> None:
    field_names = tuple(field.name for field in fields(row_type))

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=list(field_names),
                lineterminator="\n",
            )
            csv_writer.writeheader()
            for row in rows:
                if not isinstance(row, row_type):
                    raise TypeError(f"CSV row must be {row_type.__name__}")
                csv_writer.writerow(
                    {field_name: getattr(row, field_name) for field_name in field_names}
                )
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(path, writer)


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(path, writer)


def write_chunk_atomic(
    root: Path,
    *,
    result: ConditionRunResult,
    study_config_sha256: str,
    code_sha256: str,
    validation_repeat_count: int,
) -> ChunkPaths:
    if not isinstance(result, ConditionRunResult):
        raise TypeError("result must be ConditionRunResult")
    repeat_ids = _canonical_repeat_ids(result.repeat_ids)
    validation_count = _strict_integer(
        validation_repeat_count, "validation_repeat_count"
    )
    if validation_count < len(repeat_ids):
        raise ValueError("validation_repeat_count cannot be smaller than the chunk")
    if not re.fullmatch(r"[0-9a-f]{64}", str(study_config_sha256)):
        raise ValueError("study_config_sha256 must be a lowercase SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(code_sha256)):
        raise ValueError("code_sha256 must be a lowercase SHA256")
    paths = chunk_paths(
        root,
        condition_ordinal=result.condition_ordinal,
        condition_code=result.condition_code,
        repeat_ids=repeat_ids,
    )
    paths.directory.mkdir(parents=True, exist_ok=True)

    _write_npy(paths.state, result.source_state)
    _write_trajectory(paths.trajectory, result)
    _write_dataclass_csv(paths.repeat_metrics, result.repeat_metrics, RepeatMetrics)
    if result.condition_code == H300_CONDITION_CODE:
        _write_dataclass_csv(
            paths.h300_width_strata,
            result.h300_width_strata,
            H300WidthStratumMetrics,
        )

    state_sha = sha256_file(paths.state)
    trajectory_sha = sha256_file(paths.trajectory)
    metrics_sha = sha256_file(paths.repeat_metrics)
    ordered_hashes = [
        (STATE_FILENAME, state_sha),
        (TRAJECTORY_FILENAME, trajectory_sha),
        (REPEAT_METRICS_FILENAME, metrics_sha),
    ]
    manifest: dict[str, object] = {
        "StudyConfigSHA256": str(study_config_sha256),
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": str(code_sha256),
        "Namespace": result.namespace.name,
        "ConditionOrdinal": result.condition_ordinal,
        "ConditionCode": result.condition_code,
        "RepeatStart": repeat_ids[0],
        "RepeatEnd": repeat_ids[-1],
        "RepeatCount": len(repeat_ids),
        "FormalR_or_PilotR": None,
        "RunKind": RUN_KIND,
        "ValidationRepeatCount": validation_count,
        "Completed": True,
        "StateSHA256": state_sha,
        "TrajectorySHA256": trajectory_sha,
        "RepeatMetricsSHA256": metrics_sha,
    }
    if result.condition_code == H300_CONDITION_CODE:
        strata_sha = sha256_file(paths.h300_width_strata)
        ordered_hashes.append((H300_STRATA_FILENAME, strata_sha))
        manifest["H300WidthStrataSHA256"] = strata_sha
    manifest["DataSHA256"] = compute_data_sha256(ordered_hashes)

    _write_json(paths.manifest, manifest)
    return paths


@dataclass(frozen=True, slots=True)
class ChunkValidation:
    valid: bool
    reason: str
    manifest: dict[str, object] | None = None


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("chunk manifest must be a JSON object")
    return payload


def _parse_csv_rows(
    path: Path,
    row_type: type[T],
    string_fields: frozenset[str],
) -> tuple[T, ...]:
    field_names = tuple(field.name for field in fields(row_type))
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != field_names:
            raise ValueError(f"{Path(path).name} has an invalid schema")
        rows: list[T] = []
        for raw in reader:
            if None in raw:
                raise ValueError(f"{Path(path).name} contains extra columns")
            values: dict[str, object] = {}
            for name in field_names:
                value = raw[name]
                values[name] = value if name in string_fields else int(value)
            rows.append(row_type(**values))
    return tuple(rows)


def _load_chunk_data(paths: ChunkPaths, manifest: dict[str, object]) -> ConditionRunResult:
    repeat_start = int(manifest["RepeatStart"])
    repeat_end = int(manifest["RepeatEnd"])
    repeat_ids = tuple(range(repeat_start, repeat_end + 1))
    with paths.state.open("rb") as handle:
        source_state = np.load(handle, allow_pickle=False)
    with np.load(paths.trajectory, allow_pickle=False) as archive:
        if set(archive.files) != {
            "ref_toa_before_s",
            "selected_observed_toa_s",
            "has_selection",
        }:
            raise ValueError("chunk trajectory archive has an invalid schema")
        reference = np.array(archive["ref_toa_before_s"], copy=True)
        selected = np.array(archive["selected_observed_toa_s"], copy=True)
        has_selection = np.array(archive["has_selection"], copy=True)
    metrics = _parse_csv_rows(
        paths.repeat_metrics,
        RepeatMetrics,
        frozenset(("ConditionCode", "MethodCode", "EndState")),
    )
    strata: tuple[H300WidthStratumMetrics, ...] = ()
    if str(manifest["ConditionCode"]) == H300_CONDITION_CODE:
        strata = _parse_csv_rows(
            paths.h300_width_strata,
            H300WidthStratumMetrics,
            frozenset(("ConditionCode", "MethodCode", "WidthStratum")),
        )
    namespace = RandomNamespace[str(manifest["Namespace"])]
    return ConditionRunResult(
        namespace=namespace,
        condition_ordinal=int(manifest["ConditionOrdinal"]),
        condition_code=str(manifest["ConditionCode"]),
        repeat_ids=repeat_ids,
        source_state=source_state,
        ref_toa_before_s=reference,
        selected_observed_toa_s=selected,
        has_selection=has_selection,
        repeat_metrics=metrics,
        h300_width_strata=strata,
    )


def validate_chunk(
    paths: ChunkPaths,
    *,
    study_config_sha256: str,
    code_sha256: str,
    namespace: RandomNamespace,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
    validation_repeat_count: int,
) -> ChunkValidation:
    canonical = _canonical_repeat_ids(repeat_ids)
    expected = {
        "StudyConfigSHA256": str(study_config_sha256),
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": str(code_sha256),
        "Namespace": RandomNamespace(namespace).name,
        "ConditionOrdinal": _strict_integer(condition_ordinal, "condition_ordinal"),
        "ConditionCode": str(condition_code),
        "RepeatStart": canonical[0],
        "RepeatEnd": canonical[-1],
        "RepeatCount": len(canonical),
        "FormalR_or_PilotR": None,
        "RunKind": RUN_KIND,
        "ValidationRepeatCount": _strict_integer(
            validation_repeat_count, "validation_repeat_count"
        ),
        "Completed": True,
    }
    try:
        manifest = _read_manifest(paths.manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return ChunkValidation(False, f"manifest_unreadable:{type(exc).__name__}")
    for field_name, expected_value in expected.items():
        if manifest.get(field_name) != expected_value:
            return ChunkValidation(False, f"manifest_mismatch:{field_name}", manifest)

    required_files = (paths.state, paths.trajectory, paths.repeat_metrics)
    if condition_code == H300_CONDITION_CODE:
        required_files += (paths.h300_width_strata,)
    if any(not path.is_file() for path in required_files):
        return ChunkValidation(False, "data_file_missing", manifest)
    try:
        state_sha = sha256_file(paths.state)
        trajectory_sha = sha256_file(paths.trajectory)
        metrics_sha = sha256_file(paths.repeat_metrics)
        if manifest.get("StateSHA256") != state_sha:
            return ChunkValidation(False, "hash_mismatch:state", manifest)
        if manifest.get("TrajectorySHA256") != trajectory_sha:
            return ChunkValidation(False, "hash_mismatch:trajectory", manifest)
        if manifest.get("RepeatMetricsSHA256") != metrics_sha:
            return ChunkValidation(False, "hash_mismatch:repeat_metrics", manifest)
        ordered_hashes = [
            (STATE_FILENAME, state_sha),
            (TRAJECTORY_FILENAME, trajectory_sha),
            (REPEAT_METRICS_FILENAME, metrics_sha),
        ]
        if condition_code == H300_CONDITION_CODE:
            strata_sha = sha256_file(paths.h300_width_strata)
            if manifest.get("H300WidthStrataSHA256") != strata_sha:
                return ChunkValidation(False, "hash_mismatch:h300_strata", manifest)
            ordered_hashes.append((H300_STRATA_FILENAME, strata_sha))
        if manifest.get("DataSHA256") != compute_data_sha256(ordered_hashes):
            return ChunkValidation(False, "hash_mismatch:data", manifest)
        _load_chunk_data(paths, manifest)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        return ChunkValidation(False, f"data_invalid:{type(exc).__name__}", manifest)
    return ChunkValidation(True, "valid", manifest)


def load_validated_chunk(
    paths: ChunkPaths,
    *,
    study_config_sha256: str,
    code_sha256: str,
    namespace: RandomNamespace,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
    validation_repeat_count: int,
) -> ConditionRunResult:
    validation = validate_chunk(
        paths,
        study_config_sha256=study_config_sha256,
        code_sha256=code_sha256,
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        repeat_ids=repeat_ids,
        validation_repeat_count=validation_repeat_count,
    )
    if not validation.valid or validation.manifest is None:
        raise ValueError(f"chunk is not resume-eligible: {validation.reason}")
    return _load_chunk_data(paths, validation.manifest)


def assemble_condition_chunks(
    chunks: Iterable[ConditionRunResult],
    *,
    expected_repeat_ids: Iterable[int],
) -> ConditionRunResult:
    materialized = tuple(chunks)
    if not materialized:
        raise ValueError("at least one chunk is required")
    if any(not isinstance(chunk, ConditionRunResult) for chunk in materialized):
        raise TypeError("all chunks must be ConditionRunResult instances")
    namespace_values = {chunk.namespace for chunk in materialized}
    condition_ordinals = {chunk.condition_ordinal for chunk in materialized}
    condition_codes = {chunk.condition_code for chunk in materialized}
    if (
        len(namespace_values) != 1
        or len(condition_ordinals) != 1
        or len(condition_codes) != 1
    ):
        raise ValueError("chunks do not share one namespace and condition")
    all_repeat_ids = tuple(
        repeat_id for chunk in materialized for repeat_id in chunk.repeat_ids
    )
    if len(all_repeat_ids) != len(set(all_repeat_ids)):
        raise ValueError("chunk RepeatID ranges overlap")
    expected = _canonical_repeat_ids(expected_repeat_ids)
    if tuple(sorted(all_repeat_ids)) != expected:
        raise ValueError("chunk assembly has a RepeatID gap or unexpected RepeatID")
    ordered = tuple(sorted(materialized, key=lambda chunk: chunk.repeat_ids[0]))
    source = np.concatenate(tuple(chunk.source_state for chunk in ordered), axis=1)
    reference = np.concatenate(
        tuple(chunk.ref_toa_before_s for chunk in ordered), axis=1
    )
    selected = np.concatenate(
        tuple(chunk.selected_observed_toa_s for chunk in ordered), axis=1
    )
    has_selection = np.concatenate(
        tuple(chunk.has_selection for chunk in ordered), axis=1
    )
    metrics = tuple(
        sorted(
            (row for chunk in ordered for row in chunk.repeat_metrics),
            key=lambda row: (row.RepeatID, row.MethodOrdinal),
        )
    )
    stratum_rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
    strata = tuple(
        sorted(
            (row for chunk in ordered for row in chunk.h300_width_strata),
            key=lambda row: (
                row.RepeatID,
                row.MethodOrdinal,
                stratum_rank[row.WidthStratum],
            ),
        )
    )
    diagnostics = canonical_diagnostics(
        record for chunk in ordered for record in chunk.diagnostics
    )
    first = ordered[0]
    return ConditionRunResult(
        namespace=first.namespace,
        condition_ordinal=first.condition_ordinal,
        condition_code=first.condition_code,
        repeat_ids=expected,
        source_state=source,
        ref_toa_before_s=reference,
        selected_observed_toa_s=selected,
        has_selection=has_selection,
        repeat_metrics=metrics,
        h300_width_strata=strata,
        diagnostics=diagnostics,
    )


__all__ = [
    "CODE_FINGERPRINT_FILES",
    "CODE_VERSION",
    "ChunkPaths",
    "ChunkValidation",
    "H300_STRATA_FILENAME",
    "MANIFEST_FILENAME",
    "REPEAT_METRICS_FILENAME",
    "RUN_KIND",
    "STATE_FILENAME",
    "TRAJECTORY_FILENAME",
    "assemble_condition_chunks",
    "chunk_paths",
    "compute_code_sha256",
    "compute_data_sha256",
    "load_validated_chunk",
    "partition_repeat_ids",
    "sha256_file",
    "validate_chunk",
    "write_chunk_atomic",
]
