from __future__ import annotations

import hashlib
import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .bootstrap import (
    BOOTSTRAP_AUDIT_IDS,
    BOOTSTRAP_B,
    BOOTSTRAP_BLOCK_SIZE,
    BOOTSTRAP_DRAW_COUNT,
    CONTRAST_CODES,
    FORMAL_REPEAT_COUNT,
    METHOD_CODES,
    METHOD_ORDINALS,
    METRIC_CODES,
    ConditionBootstrapResult,
    run_condition_bootstrap,
)


CONDITION_COUNT = 18
EXPECTED_INDIVIDUAL_SHAPE = (
    CONDITION_COUNT,
    len(METHOD_CODES),
    len(METRIC_CODES),
    BOOTSTRAP_B,
)
EXPECTED_PAIRED_SHAPE = (
    CONDITION_COUNT,
    len(CONTRAST_CODES),
    len(METRIC_CODES),
    BOOTSTRAP_B,
)
EXPECTED_AUDIT_SHAPE = (
    CONDITION_COUNT,
    len(BOOTSTRAP_AUDIT_IDS),
    BOOTSTRAP_DRAW_COUNT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_atomic(path: Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def condition_directory(stage10_dir: Path, condition_ordinal: int, condition_code: str) -> Path:
    return Path(stage10_dir) / "bootstrap_conditions" / (
        f"condition_{int(condition_ordinal):02d}_{condition_code}"
    )


def condition_result_arrays(result: ConditionBootstrapResult) -> dict[str, np.ndarray]:
    return {
        "condition_ordinal": np.asarray(result.condition_ordinal, dtype=np.int16),
        "bootstrap_ids": np.asarray(result.bootstrap_ids, dtype=np.int16),
        "method_ordinals": np.asarray(METHOD_ORDINALS, dtype=np.int8),
        "metric_codes": np.asarray(METRIC_CODES),
        "contrast_codes": np.asarray(CONTRAST_CODES),
        "individual_values": np.asarray(result.individual_values, dtype=np.float64),
        "individual_defined": np.asarray(result.individual_defined, dtype=bool),
        "paired_values": np.asarray(result.paired_values, dtype=np.float64),
        "paired_defined": np.asarray(result.paired_defined, dtype=bool),
        "audit_ids": np.asarray(result.audit_ids, dtype=np.int16),
        "audit_indices": np.asarray(result.audit_indices, dtype=np.uint16),
    }


def _require_exact_keys(arrays: Mapping[str, np.ndarray], expected: set[str], label: str) -> None:
    actual = set(arrays)
    if actual != expected:
        raise AssertionError(
            f"{label} NPZ keys differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )


def validate_condition_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    condition_ordinal: int,
    bootstrap_ids: Sequence[int],
    audit_ids: Sequence[int],
) -> None:
    expected_keys = {
        "condition_ordinal",
        "bootstrap_ids",
        "method_ordinals",
        "metric_codes",
        "contrast_codes",
        "individual_values",
        "individual_defined",
        "paired_values",
        "paired_defined",
        "audit_ids",
        "audit_indices",
    }
    _require_exact_keys(arrays, expected_keys, "condition distribution")
    ids = np.asarray(tuple(bootstrap_ids), dtype=np.int16)
    audits = np.asarray(tuple(audit_ids), dtype=np.int16)
    bootstrap_count = ids.size
    if np.asarray(arrays["condition_ordinal"]).shape != () or int(
        arrays["condition_ordinal"]
    ) != int(condition_ordinal):
        raise AssertionError("condition distribution ordinal is not exact")
    if not np.array_equal(arrays["bootstrap_ids"], ids):
        raise AssertionError("condition distribution BootstrapID sequence is not exact")
    if not np.array_equal(
        arrays["method_ordinals"], np.asarray(METHOD_ORDINALS, dtype=np.int8)
    ):
        raise AssertionError("condition distribution method ordinals are not exact")
    if tuple(str(value) for value in arrays["metric_codes"]) != METRIC_CODES:
        raise AssertionError("condition distribution metric codes are not exact")
    if tuple(str(value) for value in arrays["contrast_codes"]) != CONTRAST_CODES:
        raise AssertionError("condition distribution contrast codes are not exact")
    if not np.array_equal(arrays["audit_ids"], audits):
        raise AssertionError("condition distribution audit IDs are not exact")

    individual_shape = (len(METHOD_CODES), len(METRIC_CODES), bootstrap_count)
    paired_shape = (len(CONTRAST_CODES), len(METRIC_CODES), bootstrap_count)
    audit_shape = (audits.size, BOOTSTRAP_DRAW_COUNT)
    if arrays["individual_values"].shape != individual_shape:
        raise AssertionError("condition individual distribution shape is invalid")
    if arrays["individual_values"].dtype != np.dtype(np.float64):
        raise AssertionError("condition individual values must be float64")
    if arrays["individual_defined"].shape != individual_shape or arrays[
        "individual_defined"
    ].dtype != np.dtype(bool):
        raise AssertionError("condition individual defined mask is invalid")
    if arrays["paired_values"].shape != paired_shape:
        raise AssertionError("condition paired distribution shape is invalid")
    if arrays["paired_values"].dtype != np.dtype(np.float64):
        raise AssertionError("condition paired values must be float64")
    if arrays["paired_defined"].shape != paired_shape or arrays[
        "paired_defined"
    ].dtype != np.dtype(bool):
        raise AssertionError("condition paired defined mask is invalid")
    if arrays["audit_indices"].shape != audit_shape or arrays[
        "audit_indices"
    ].dtype != np.dtype(np.uint16):
        raise AssertionError("condition audit index array is invalid")

    individual_values = arrays["individual_values"]
    individual_defined = arrays["individual_defined"]
    paired_values = arrays["paired_values"]
    paired_defined = arrays["paired_defined"]
    if not np.array_equal(np.isnan(individual_values), ~individual_defined):
        raise AssertionError("condition individual NaN/mask identity failed")
    if not np.array_equal(np.isnan(paired_values), ~paired_defined):
        raise AssertionError("condition paired NaN/mask identity failed")
    probability_individual = individual_values[:, :3, :][
        individual_defined[:, :3, :]
    ]
    if np.any(probability_individual < 0.0) or np.any(probability_individual > 1.0):
        raise AssertionError("individual probability distribution left [0,1]")
    mean_individual = individual_values[:, 3, :][individual_defined[:, 3, :]]
    if np.any(mean_individual < 0.0):
        raise AssertionError("Mean_L_NC distribution contains a negative value")
    probability_paired = paired_values[:, :3, :][paired_defined[:, :3, :]]
    if np.any(probability_paired < -1.0) or np.any(probability_paired > 1.0):
        raise AssertionError("paired probability difference left [-1,1]")
    audit_indices = arrays["audit_indices"]
    if np.any(audit_indices < 1) or np.any(audit_indices > FORMAL_REPEAT_COUNT):
        raise AssertionError("condition audit indices left 1..2000")


def _manifest_identity(
    *,
    condition_ordinal: int,
    condition_code: str,
    bootstrap_ids: Sequence[int],
    audit_ids: Sequence[int],
    block_size: int,
    identity_hashes: Mapping[str, str],
    distribution_sha256: str,
    audit_indices_sha256: str,
) -> dict[str, object]:
    return {
        "ManifestSchema": "STAGE10_CONDITION_BOOTSTRAP_MANIFEST_V1",
        "Stage": 10,
        "ConditionOrdinal": int(condition_ordinal),
        "ConditionCode": str(condition_code),
        "BootstrapNamespace": "BOOTSTRAP",
        "BootstrapB": len(tuple(bootstrap_ids)),
        "BootstrapIDRange": [int(min(bootstrap_ids)), int(max(bootstrap_ids))],
        "BootstrapDrawCountPerReplicate": BOOTSTRAP_DRAW_COUNT,
        "IndependentUnit": "Repeat",
        "FormalRepeatCount": FORMAL_REPEAT_COUNT,
        "Workers": 1,
        "VectorizationBlockSize": int(block_size),
        "PersistentResumeUnit": "Condition",
        "MethodCodes": list(METHOD_CODES),
        "MetricCodes": list(METRIC_CODES),
        "ContrastCodes": list(CONTRAST_CODES),
        "AuditIDs": [int(value) for value in audit_ids],
        "DistributionSHA256": distribution_sha256,
        "AuditIndicesSHA256": audit_indices_sha256,
        "IdentityHashes": dict(sorted(identity_hashes.items())),
        "Completed": True,
    }


@dataclass(frozen=True, slots=True)
class ConditionValidation:
    valid: bool
    reason: str
    result: ConditionBootstrapResult | None
    manifest: dict[str, object] | None


def write_condition_result(
    stage10_dir: Path,
    *,
    condition_code: str,
    result: ConditionBootstrapResult,
    block_size: int,
    identity_hashes: Mapping[str, str],
) -> tuple[Path, Path, dict[str, object]]:
    directory = condition_directory(
        stage10_dir, result.condition_ordinal, condition_code
    )
    directory.mkdir(parents=True, exist_ok=True)
    distribution_path = directory / "distributions.npz"
    manifest_path = directory / "manifest.json"
    arrays = condition_result_arrays(result)
    validate_condition_arrays(
        arrays,
        condition_ordinal=result.condition_ordinal,
        bootstrap_ids=result.bootstrap_ids,
        audit_ids=result.audit_ids,
    )
    write_npz_atomic(distribution_path, arrays)
    distribution_sha = sha256_file(distribution_path)
    audit_sha = sha256_array(arrays["audit_indices"])
    manifest = _manifest_identity(
        condition_ordinal=result.condition_ordinal,
        condition_code=condition_code,
        bootstrap_ids=result.bootstrap_ids,
        audit_ids=result.audit_ids,
        block_size=block_size,
        identity_hashes=identity_hashes,
        distribution_sha256=distribution_sha,
        audit_indices_sha256=audit_sha,
    )
    write_json_atomic(manifest_path, manifest)
    return distribution_path, manifest_path, manifest


def load_valid_condition(
    stage10_dir: Path,
    *,
    condition_ordinal: int,
    condition_code: str,
    bootstrap_ids: Sequence[int],
    audit_ids: Sequence[int],
    block_size: int,
    identity_hashes: Mapping[str, str],
) -> ConditionValidation:
    directory = condition_directory(stage10_dir, condition_ordinal, condition_code)
    distribution_path = directory / "distributions.npz"
    manifest_path = directory / "manifest.json"
    if not distribution_path.is_file() or not manifest_path.is_file():
        return ConditionValidation(False, "MISSING_CONDITION_OUTPUT", None, None)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise AssertionError("manifest is not a JSON object")
        distribution_sha = sha256_file(distribution_path)
        arrays = load_npz(distribution_path)
        validate_condition_arrays(
            arrays,
            condition_ordinal=condition_ordinal,
            bootstrap_ids=bootstrap_ids,
            audit_ids=audit_ids,
        )
        expected_manifest = _manifest_identity(
            condition_ordinal=condition_ordinal,
            condition_code=condition_code,
            bootstrap_ids=bootstrap_ids,
            audit_ids=audit_ids,
            block_size=block_size,
            identity_hashes=identity_hashes,
            distribution_sha256=distribution_sha,
            audit_indices_sha256=sha256_array(arrays["audit_indices"]),
        )
        if manifest != expected_manifest:
            raise AssertionError("condition manifest identity is not exact")
        result = ConditionBootstrapResult(
            condition_ordinal=int(arrays["condition_ordinal"]),
            bootstrap_ids=arrays["bootstrap_ids"],
            individual_values=arrays["individual_values"],
            individual_defined=arrays["individual_defined"],
            paired_values=arrays["paired_values"],
            paired_defined=arrays["paired_defined"],
            audit_ids=arrays["audit_ids"],
            audit_indices=arrays["audit_indices"],
        )
        return ConditionValidation(True, "VALID_CONDITION_RESUME", result, manifest)
    except (
        AssertionError,
        EOFError,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        return ConditionValidation(False, f"INVALID_CONDITION_OUTPUT: {exc}", None, None)


def run_or_resume_condition(
    stage10_dir: Path,
    *,
    condition_ordinal: int,
    condition_code: str,
    statistics: np.ndarray,
    bootstrap_ids: Sequence[int] = tuple(range(1, BOOTSTRAP_B + 1)),
    audit_ids: Sequence[int] = BOOTSTRAP_AUDIT_IDS,
    block_size: int = BOOTSTRAP_BLOCK_SIZE,
    identity_hashes: Mapping[str, str],
    compute: Callable[..., ConditionBootstrapResult] = run_condition_bootstrap,
) -> tuple[ConditionBootstrapResult, bool, str]:
    validation = load_valid_condition(
        stage10_dir,
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        bootstrap_ids=bootstrap_ids,
        audit_ids=audit_ids,
        block_size=block_size,
        identity_hashes=identity_hashes,
    )
    if validation.valid:
        assert validation.result is not None
        return validation.result, True, validation.reason

    result = compute(
        statistics,
        condition_ordinal=condition_ordinal,
        bootstrap_ids=bootstrap_ids,
        block_size=block_size,
        audit_ids=audit_ids,
    )
    write_condition_result(
        stage10_dir,
        condition_code=condition_code,
        result=result,
        block_size=block_size,
        identity_hashes=identity_hashes,
    )
    final_validation = load_valid_condition(
        stage10_dir,
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        bootstrap_ids=bootstrap_ids,
        audit_ids=audit_ids,
        block_size=block_size,
        identity_hashes=identity_hashes,
    )
    if not final_validation.valid or final_validation.result is None:
        raise AssertionError(
            "whole-condition recompute did not produce a valid condition output: "
            + final_validation.reason
        )
    return final_validation.result, False, validation.reason


def combined_distribution_arrays(
    results: Sequence[ConditionBootstrapResult],
) -> dict[str, np.ndarray]:
    ordered = sorted(results, key=lambda item: item.condition_ordinal)
    if [item.condition_ordinal for item in ordered] != list(range(1, 19)):
        raise AssertionError("combined distributions require exactly Conditions 1..18")
    bootstrap_ids = np.arange(1, BOOTSTRAP_B + 1, dtype=np.int16)
    for item in ordered:
        if not np.array_equal(item.bootstrap_ids, bootstrap_ids):
            raise AssertionError("combined condition BootstrapID identity is not 1..2000")
    return {
        "condition_ordinals": np.arange(1, 19, dtype=np.int16),
        "method_ordinals": np.asarray(METHOD_ORDINALS, dtype=np.int8),
        "metric_codes": np.asarray(METRIC_CODES),
        "contrast_codes": np.asarray(CONTRAST_CODES),
        "individual_values": np.stack(
            [item.individual_values for item in ordered], axis=0
        ).astype(np.float64, copy=False),
        "individual_defined": np.stack(
            [item.individual_defined for item in ordered], axis=0
        ).astype(bool, copy=False),
        "paired_values": np.stack(
            [item.paired_values for item in ordered], axis=0
        ).astype(np.float64, copy=False),
        "paired_defined": np.stack(
            [item.paired_defined for item in ordered], axis=0
        ).astype(bool, copy=False),
    }


def validate_combined_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected_keys = {
        "condition_ordinals",
        "method_ordinals",
        "metric_codes",
        "contrast_codes",
        "individual_values",
        "individual_defined",
        "paired_values",
        "paired_defined",
    }
    _require_exact_keys(arrays, expected_keys, "combined distribution")
    if not np.array_equal(
        arrays["condition_ordinals"], np.arange(1, 19, dtype=np.int16)
    ):
        raise AssertionError("combined condition ordinals are not 1..18")
    if not np.array_equal(
        arrays["method_ordinals"], np.asarray(METHOD_ORDINALS, dtype=np.int8)
    ):
        raise AssertionError("combined method ordinals are not exact")
    if tuple(str(value) for value in arrays["metric_codes"]) != METRIC_CODES:
        raise AssertionError("combined metric codes are not exact")
    if tuple(str(value) for value in arrays["contrast_codes"]) != CONTRAST_CODES:
        raise AssertionError("combined contrast codes are not exact")
    if arrays["individual_values"].shape != EXPECTED_INDIVIDUAL_SHAPE:
        raise AssertionError("combined individual distribution shape is invalid")
    if arrays["individual_values"].dtype != np.dtype(np.float64):
        raise AssertionError("combined individual values must be float64")
    if arrays["individual_defined"].shape != EXPECTED_INDIVIDUAL_SHAPE or arrays[
        "individual_defined"
    ].dtype != np.dtype(bool):
        raise AssertionError("combined individual defined mask is invalid")
    if arrays["paired_values"].shape != EXPECTED_PAIRED_SHAPE:
        raise AssertionError("combined paired distribution shape is invalid")
    if arrays["paired_values"].dtype != np.dtype(np.float64):
        raise AssertionError("combined paired values must be float64")
    if arrays["paired_defined"].shape != EXPECTED_PAIRED_SHAPE or arrays[
        "paired_defined"
    ].dtype != np.dtype(bool):
        raise AssertionError("combined paired defined mask is invalid")
    if not np.array_equal(
        np.isnan(arrays["individual_values"]), ~arrays["individual_defined"]
    ):
        raise AssertionError("combined individual NaN/mask identity failed")
    if not np.array_equal(
        np.isnan(arrays["paired_values"]), ~arrays["paired_defined"]
    ):
        raise AssertionError("combined paired NaN/mask identity failed")
    probability_individual = arrays["individual_values"][:, :, :3, :][
        arrays["individual_defined"][:, :, :3, :]
    ]
    if np.any(probability_individual < 0.0) or np.any(probability_individual > 1.0):
        raise AssertionError("combined individual probability values left [0,1]")
    mean_values = arrays["individual_values"][:, :, 3, :][
        arrays["individual_defined"][:, :, 3, :]
    ]
    if np.any(mean_values < 0.0):
        raise AssertionError("combined Mean_L_NC values contain a negative value")
    probability_paired = arrays["paired_values"][:, :, :3, :][
        arrays["paired_defined"][:, :, :3, :]
    ]
    if np.any(probability_paired < -1.0) or np.any(probability_paired > 1.0):
        raise AssertionError("combined paired probability differences left [-1,1]")


def write_combined_distributions(
    path: Path,
    results: Sequence[ConditionBootstrapResult],
) -> dict[str, np.ndarray]:
    arrays = combined_distribution_arrays(results)
    validate_combined_arrays(arrays)
    write_npz_atomic(path, arrays)
    reloaded = load_npz(path)
    validate_combined_arrays(reloaded)
    return reloaded


def validate_combined_distributions(path: Path) -> dict[str, np.ndarray]:
    arrays = load_npz(path)
    validate_combined_arrays(arrays)
    return arrays


def write_bootstrap_audit_indices(
    path: Path,
    *,
    condition_ordinals: Sequence[int],
    audit_ids: Sequence[int],
    repeat_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {
        "condition_ordinals": np.asarray(condition_ordinals, dtype=np.int16),
        "bootstrap_audit_ids": np.asarray(audit_ids, dtype=np.int16),
        "repeat_ids": np.asarray(repeat_ids, dtype=np.uint16),
    }
    validate_bootstrap_audit_arrays(arrays)
    write_npz_atomic(path, arrays)
    reloaded = load_npz(path)
    validate_bootstrap_audit_arrays(reloaded)
    return reloaded


def validate_bootstrap_audit_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected_keys = {"condition_ordinals", "bootstrap_audit_ids", "repeat_ids"}
    _require_exact_keys(arrays, expected_keys, "Bootstrap audit")
    if not np.array_equal(
        arrays["condition_ordinals"], np.arange(1, 19, dtype=np.int16)
    ):
        raise AssertionError("Bootstrap audit condition ordinals are not 1..18")
    if not np.array_equal(
        arrays["bootstrap_audit_ids"],
        np.asarray(BOOTSTRAP_AUDIT_IDS, dtype=np.int16),
    ):
        raise AssertionError("Bootstrap audit IDs are not 1/1000/2000")
    repeat_ids = arrays["repeat_ids"]
    if repeat_ids.shape != EXPECTED_AUDIT_SHAPE or repeat_ids.dtype != np.dtype(
        np.uint16
    ):
        raise AssertionError("Bootstrap audit index shape or dtype is invalid")
    if np.any(repeat_ids < 1) or np.any(repeat_ids > FORMAL_REPEAT_COUNT):
        raise AssertionError("Bootstrap audit index left 1..2000")


__all__ = [
    "CONDITION_COUNT",
    "ConditionValidation",
    "EXPECTED_AUDIT_SHAPE",
    "EXPECTED_INDIVIDUAL_SHAPE",
    "EXPECTED_PAIRED_SHAPE",
    "canonical_json_bytes",
    "combined_distribution_arrays",
    "condition_directory",
    "condition_result_arrays",
    "load_npz",
    "load_valid_condition",
    "run_or_resume_condition",
    "sha256_array",
    "sha256_file",
    "validate_bootstrap_audit_arrays",
    "validate_combined_arrays",
    "validate_combined_distributions",
    "validate_condition_arrays",
    "write_bootstrap_audit_indices",
    "write_combined_distributions",
    "write_condition_result",
    "write_json_atomic",
    "write_npz_atomic",
    "write_text_atomic",
]
