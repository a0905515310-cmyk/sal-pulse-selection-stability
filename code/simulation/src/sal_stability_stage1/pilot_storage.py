from __future__ import annotations

import csv
import hashlib
import json
import os
import operator
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

from .build import ROOT
from .execution import ConditionRunResult
from .metrics import RepeatMetrics
from .rng import RandomNamespace
from .stage5 import H300_CONDITION_CODE, H300WidthStratumMetrics
from .storage import assemble_condition_chunks


SCHEMA_VERSION = 1
CODE_VERSION = "0.7.0"
RUN_KIND = "PILOT"
PILOT_R = 200

STATE_FILENAME = "chunk_state.npy"
TRAJECTORY_FILENAME = "chunk_trajectory.npz"
REPEAT_METRICS_FILENAME = "chunk_repeat_metrics.csv"
H300_STRATA_FILENAME = "chunk_h300_width_strata.csv"
MANIFEST_FILENAME = "chunk_manifest.json"

CODE_FINGERPRINT_FILES = (
    "src/sal_stability_stage1/contracts.py",
    "src/sal_stability_stage1/encoding.py",
    "src/sal_stability_stage1/rng.py",
    "src/sal_stability_stage1/events.py",
    "src/sal_stability_stage1/hprf.py",
    "src/sal_stability_stage1/selectors.py",
    "src/sal_stability_stage1/metrics.py",
    "src/sal_stability_stage1/kernel.py",
    "src/sal_stability_stage1/diagnostics.py",
    "src/sal_stability_stage1/execution.py",
    "src/sal_stability_stage1/storage.py",
    "src/sal_stability_stage1/stage3.py",
    "src/sal_stability_stage1/stage4.py",
    "src/sal_stability_stage1/stage5.py",
    "src/sal_stability_stage1/stage6.py",
    "src/sal_stability_stage1/pilot_storage.py",
    "src/sal_stability_stage1/stage7.py",
    "scripts/build_stage7.py",
)

FORBIDDEN_RNG_STATE_FIELDS = frozenset(
    (
        "GlobalRNGState",
        "ThreadRNGState",
        "LoopCounterState",
        "RandomState",
        "MutableRNGState",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_stage7_code_sha256(root: Path = ROOT) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for relative in CODE_FINGERPRINT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Stage 7 code fingerprint input is missing: {relative}")
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


def canonical_repeat_ids(values: Iterable[int]) -> tuple[int, ...]:
    repeat_ids = tuple(_strict_integer(value, "RepeatID") for value in values)
    if not repeat_ids:
        raise ValueError("repeat_ids cannot be empty")
    if any(value < 1 for value in repeat_ids):
        raise ValueError("RepeatID must be positive")
    if len(repeat_ids) != len(set(repeat_ids)):
        raise ValueError("repeat_ids cannot contain duplicates")
    result = tuple(sorted(repeat_ids))
    if any(right != left + 1 for left, right in zip(result, result[1:])):
        raise ValueError("Pilot chunk RepeatID values must be contiguous")
    return result


def partition_repeat_ids(
    repeat_ids: Iterable[int], chunk_size: int
) -> tuple[tuple[int, ...], ...]:
    canonical = canonical_repeat_ids(repeat_ids)
    size = _strict_integer(chunk_size, "chunk_size")
    if size < 1:
        raise ValueError("chunk_size must be at least one")
    return tuple(
        canonical[offset : offset + size]
        for offset in range(0, len(canonical), size)
    )


def _condition_slug(condition_code: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(condition_code).lower()).strip("-")
    if not slug:
        raise ValueError("condition_code cannot produce an empty path component")
    return slug


@dataclass(frozen=True, slots=True)
class PilotChunkPaths:
    directory: Path
    state: Path
    trajectory: Path
    repeat_metrics: Path
    h300_width_strata: Path
    manifest: Path


def pilot_chunk_paths(
    root: Path,
    *,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
) -> PilotChunkPaths:
    canonical = canonical_repeat_ids(repeat_ids)
    ordinal = _strict_integer(condition_ordinal, "condition_ordinal")
    if not 0 <= ordinal < 19:
        raise ValueError("condition_ordinal must lie in 0..18")
    directory = (
        Path(root)
        / f"condition_{ordinal:02d}_{_condition_slug(condition_code)}"
        / f"repeats_{canonical[0]}_{canonical[-1]}"
    )
    return PilotChunkPaths(
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


def write_text_atomic(path: Path, text: str) -> None:
    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(Path(path), writer)


def write_json_atomic(path: Path, payload: object) -> None:
    write_text_atomic(
        Path(path),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def write_csv_dicts_atomic(
    path: Path,
    *,
    field_names: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    names = tuple(str(name) for name in field_names)
    materialized = tuple(dict(row) for row in rows)

    def writer(temporary: Path) -> None:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=list(names),
                lineterminator="\n",
                extrasaction="raise",
            )
            csv_writer.writeheader()
            for row in materialized:
                if set(row) != set(names):
                    raise ValueError(f"CSV row schema disagrees with {Path(path).name}")
                csv_writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    _atomic_write(Path(path), writer)


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


def _write_dataclass_csv(path: Path, rows: Sequence[T], row_type: type[T]) -> None:
    names = tuple(field.name for field in fields(row_type))
    materialized = tuple(rows)
    if any(not isinstance(row, row_type) for row in materialized):
        raise TypeError(f"CSV rows must all be {row_type.__name__}")
    write_csv_dicts_atomic(
        path,
        field_names=names,
        rows=(asdict(row) for row in materialized),
    )


def _require_lower_sha256(value: object, field_name: str) -> str:
    result = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA256")
    return result


def write_pilot_chunk_atomic(
    root: Path,
    *,
    result: ConditionRunResult,
    study_config_sha256: str,
    code_sha256: str,
    pilot_r: int = PILOT_R,
) -> PilotChunkPaths:
    if not isinstance(result, ConditionRunResult):
        raise TypeError("result must be ConditionRunResult")
    if result.namespace is not RandomNamespace.PILOT:
        raise ValueError("formal Pilot chunks require Namespace=PILOT")
    if result.diagnostics:
        raise ValueError("formal Pilot chunks cannot contain diagnostics")
    pilot_count = _strict_integer(pilot_r, "pilot_r")
    if pilot_count != PILOT_R:
        raise ValueError("Stage 7 PilotR is frozen at 200")
    repeat_ids = canonical_repeat_ids(result.repeat_ids)
    if repeat_ids[0] < 1 or repeat_ids[-1] > pilot_count:
        raise ValueError("formal Pilot RepeatID must lie in 1..200")
    config_sha = _require_lower_sha256(study_config_sha256, "study_config_sha256")
    stage7_code_sha = _require_lower_sha256(code_sha256, "code_sha256")

    paths = pilot_chunk_paths(
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
        "SchemaVersion": SCHEMA_VERSION,
        "StudyConfigSHA256": config_sha,
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": stage7_code_sha,
        "RunKind": RUN_KIND,
        "Namespace": RandomNamespace.PILOT.name,
        "PilotR": pilot_count,
        "ConditionOrdinal": result.condition_ordinal,
        "ConditionCode": result.condition_code,
        "RepeatStart": repeat_ids[0],
        "RepeatEnd": repeat_ids[-1],
        "RepeatCount": len(repeat_ids),
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
    write_json_atomic(paths.manifest, manifest)
    return paths


@dataclass(frozen=True, slots=True)
class PilotChunkValidation:
    valid: bool
    reason: str
    manifest: dict[str, object] | None = None


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pilot chunk manifest must be a JSON object")
    return payload


def _parse_csv_rows(
    path: Path,
    row_type: type[T],
    string_fields: frozenset[str],
) -> tuple[T, ...]:
    names = tuple(field.name for field in fields(row_type))
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != names:
            raise ValueError(f"{Path(path).name} has an invalid schema")
        rows: list[T] = []
        for raw in reader:
            if None in raw:
                raise ValueError(f"{Path(path).name} contains extra columns")
            values: dict[str, object] = {}
            for name in names:
                value = raw[name]
                values[name] = value if name in string_fields else int(value)
            rows.append(row_type(**values))
    return tuple(rows)


def _load_pilot_chunk_data(
    paths: PilotChunkPaths, manifest: Mapping[str, object]
) -> ConditionRunResult:
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
            raise ValueError("Pilot chunk trajectory archive has an invalid schema")
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
    return ConditionRunResult(
        namespace=RandomNamespace[str(manifest["Namespace"])],
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


def validate_pilot_chunk(
    paths: PilotChunkPaths,
    *,
    study_config_sha256: str,
    code_sha256: str,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
    pilot_r: int = PILOT_R,
) -> PilotChunkValidation:
    canonical = canonical_repeat_ids(repeat_ids)
    pilot_count = _strict_integer(pilot_r, "pilot_r")
    expected = {
        "SchemaVersion": SCHEMA_VERSION,
        "StudyConfigSHA256": str(study_config_sha256),
        "CodeVersion": CODE_VERSION,
        "CodeSHA256": str(code_sha256),
        "RunKind": RUN_KIND,
        "Namespace": RandomNamespace.PILOT.name,
        "PilotR": pilot_count,
        "ConditionOrdinal": _strict_integer(condition_ordinal, "condition_ordinal"),
        "ConditionCode": str(condition_code),
        "RepeatStart": canonical[0],
        "RepeatEnd": canonical[-1],
        "RepeatCount": len(canonical),
        "Completed": True,
    }
    try:
        manifest = _read_manifest(paths.manifest)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return PilotChunkValidation(False, f"manifest_unreadable:{type(exc).__name__}")
    forbidden = FORBIDDEN_RNG_STATE_FIELDS.intersection(manifest)
    if forbidden:
        return PilotChunkValidation(
            False, f"forbidden_rng_state:{sorted(forbidden)[0]}", manifest
        )
    if "FormalR_or_PilotR" in manifest or "ValidationRepeatCount" in manifest:
        return PilotChunkValidation(False, "validation_identity_leak", manifest)
    for field_name, expected_value in expected.items():
        if manifest.get(field_name) != expected_value:
            return PilotChunkValidation(
                False, f"manifest_mismatch:{field_name}", manifest
            )

    required_files = (paths.state, paths.trajectory, paths.repeat_metrics)
    if condition_code == H300_CONDITION_CODE:
        required_files += (paths.h300_width_strata,)
    if any(not path.is_file() for path in required_files):
        return PilotChunkValidation(False, "data_file_missing", manifest)
    if condition_code != H300_CONDITION_CODE and (
        "H300WidthStrataSHA256" in manifest or paths.h300_width_strata.exists()
    ):
        return PilotChunkValidation(False, "unexpected_h300_strata", manifest)

    try:
        state_sha = sha256_file(paths.state)
        trajectory_sha = sha256_file(paths.trajectory)
        metrics_sha = sha256_file(paths.repeat_metrics)
        if manifest.get("StateSHA256") != state_sha:
            return PilotChunkValidation(False, "hash_mismatch:state", manifest)
        if manifest.get("TrajectorySHA256") != trajectory_sha:
            return PilotChunkValidation(False, "hash_mismatch:trajectory", manifest)
        if manifest.get("RepeatMetricsSHA256") != metrics_sha:
            return PilotChunkValidation(False, "hash_mismatch:repeat_metrics", manifest)
        ordered_hashes = [
            (STATE_FILENAME, state_sha),
            (TRAJECTORY_FILENAME, trajectory_sha),
            (REPEAT_METRICS_FILENAME, metrics_sha),
        ]
        if condition_code == H300_CONDITION_CODE:
            strata_sha = sha256_file(paths.h300_width_strata)
            if manifest.get("H300WidthStrataSHA256") != strata_sha:
                return PilotChunkValidation(False, "hash_mismatch:h300_strata", manifest)
            ordered_hashes.append((H300_STRATA_FILENAME, strata_sha))
        if manifest.get("DataSHA256") != compute_data_sha256(ordered_hashes):
            return PilotChunkValidation(False, "hash_mismatch:data", manifest)
        _load_pilot_chunk_data(paths, manifest)
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        return PilotChunkValidation(
            False, f"data_invalid:{type(exc).__name__}", manifest
        )
    return PilotChunkValidation(True, "valid", manifest)


def load_validated_pilot_chunk(
    paths: PilotChunkPaths,
    *,
    study_config_sha256: str,
    code_sha256: str,
    condition_ordinal: int,
    condition_code: str,
    repeat_ids: Iterable[int],
    pilot_r: int = PILOT_R,
) -> ConditionRunResult:
    validation = validate_pilot_chunk(
        paths,
        study_config_sha256=study_config_sha256,
        code_sha256=code_sha256,
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        repeat_ids=repeat_ids,
        pilot_r=pilot_r,
    )
    if not validation.valid or validation.manifest is None:
        raise ValueError(f"Pilot chunk is not resume-eligible: {validation.reason}")
    return _load_pilot_chunk_data(paths, validation.manifest)


def assemble_pilot_condition_chunks(
    chunks: Iterable[ConditionRunResult],
    *,
    expected_repeat_ids: Iterable[int],
) -> ConditionRunResult:
    return assemble_condition_chunks(
        chunks,
        expected_repeat_ids=expected_repeat_ids,
    )


__all__ = [
    "CODE_FINGERPRINT_FILES",
    "CODE_VERSION",
    "FORBIDDEN_RNG_STATE_FIELDS",
    "H300_STRATA_FILENAME",
    "MANIFEST_FILENAME",
    "PILOT_R",
    "REPEAT_METRICS_FILENAME",
    "RUN_KIND",
    "SCHEMA_VERSION",
    "STATE_FILENAME",
    "TRAJECTORY_FILENAME",
    "PilotChunkPaths",
    "PilotChunkValidation",
    "assemble_pilot_condition_chunks",
    "canonical_repeat_ids",
    "compute_data_sha256",
    "compute_stage7_code_sha256",
    "load_validated_pilot_chunk",
    "partition_repeat_ids",
    "pilot_chunk_paths",
    "sha256_file",
    "validate_pilot_chunk",
    "write_csv_dicts_atomic",
    "write_json_atomic",
    "write_pilot_chunk_atomic",
    "write_text_atomic",
]
