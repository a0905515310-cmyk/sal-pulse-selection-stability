from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from sal_stability_stage1.execution import (
    ExecutionOptions,
    assert_scientific_results_equal,
)
from sal_stability_stage1.pilot_storage import (
    CODE_VERSION,
    RUN_KIND,
    SCHEMA_VERSION,
)
from sal_stability_stage1.stage3 import load_stage3_inputs
from sal_stability_stage1.stage7 import run_pilot_condition_chunked


@pytest.fixture(scope="module")
def frozen_inputs():
    return load_stage3_inputs()


@pytest.fixture(scope="module")
def clean_f2_pilot_chunk(tmp_path_factory, frozen_inputs):
    config, reference = frozen_inputs
    root = tmp_path_factory.mktemp("stage7-clean-f2-pilot")
    outcome = run_pilot_condition_chunked(
        condition_ordinal=7,
        repeat_ids=(1,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(workers=1, chunk_size=1),
        output_root=root,
    )
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    return root, outcome.result


@pytest.fixture(scope="module")
def clean_h300_pilot_chunk(tmp_path_factory, frozen_inputs):
    config, reference = frozen_inputs
    root = tmp_path_factory.mktemp("stage7-clean-h300-pilot")
    outcome = run_pilot_condition_chunked(
        condition_ordinal=3,
        repeat_ids=(1,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(workers=1, chunk_size=1),
        output_root=root,
    )
    assert outcome.completed and outcome.result is not None
    return root, outcome.result


def _copy_root(source_root: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "copy"
    shutil.copytree(source_root, destination)
    return destination


def _resume(root: Path, frozen_inputs, *, condition_ordinal: int = 7):
    config, reference = frozen_inputs
    return run_pilot_condition_chunked(
        condition_ordinal=condition_ordinal,
        repeat_ids=(1,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(workers=1, chunk_size=1, resume=True),
        output_root=root,
    )


def _mutate_manifest(root: Path, field_name: str, value: object) -> None:
    path = next(root.rglob("chunk_manifest.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field_name] = value
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _corrupt_last_byte(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes((original[0] ^ 1,)))


def test_pilot_manifest_and_atomic_files_have_isolated_identity(clean_f2_pilot_chunk):
    root, _ = clean_f2_pilot_chunk
    path = next(root.rglob("chunk_manifest.json"))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["SchemaVersion"] == SCHEMA_VERSION == 1
    assert payload["CodeVersion"] == CODE_VERSION == "0.7.0"
    assert payload["RunKind"] == RUN_KIND == "PILOT"
    assert payload["Namespace"] == "PILOT"
    assert payload["PilotR"] == 200
    assert payload["ConditionOrdinal"] == 7
    assert payload["ConditionCode"] == "F2"
    assert payload["RepeatStart"] == payload["RepeatEnd"] == 1
    assert payload["RepeatCount"] == 1
    assert payload["Completed"] is True
    assert payload["StateSHA256"]
    assert payload["TrajectorySHA256"]
    assert payload["RepeatMetricsSHA256"]
    assert payload["DataSHA256"]
    assert "FormalR_or_PilotR" not in payload
    assert "ValidationRepeatCount" not in payload
    assert not any("RNGState" in key or key == "RandomState" for key in payload)
    directory = path.parent
    assert (directory / "chunk_state.npy").is_file()
    assert (directory / "chunk_trajectory.npz").is_file()
    assert (directory / "chunk_repeat_metrics.csv").is_file()
    assert not (directory / "chunk_h300_width_strata.csv").exists()
    assert not any(".tmp-" in item.name for item in root.rglob("*"))


def test_valid_pilot_chunk_resume_is_really_skipped(
    clean_f2_pilot_chunk, frozen_inputs, tmp_path
):
    root = _copy_root(clean_f2_pilot_chunk[0], tmp_path)
    outcome = _resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 0
    assert outcome.skipped_chunks == 1
    assert outcome.recomputed_invalid_chunks == 0
    assert outcome.invalidated_chunks == ()
    assert_scientific_results_equal(
        clean_f2_pilot_chunk[1], outcome.result, label="valid Pilot skip"
    )


def test_missing_pilot_chunk_is_computed(frozen_inputs, tmp_path, clean_f2_pilot_chunk):
    root = tmp_path / "missing"
    outcome = _resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    assert outcome.skipped_chunks == 0
    assert outcome.recomputed_invalid_chunks == 0
    assert any("manifest_unreadable" in item for item in outcome.invalidated_chunks)
    assert_scientific_results_equal(
        clean_f2_pilot_chunk[1], outcome.result, label="missing Pilot recompute"
    )


@pytest.mark.parametrize(
    ("field_name", "value", "reason"),
    (
        ("SchemaVersion", 2, "manifest_mismatch:SchemaVersion"),
        ("Completed", False, "manifest_mismatch:Completed"),
        ("CodeVersion", "0.6.0", "manifest_mismatch:CodeVersion"),
        ("CodeSHA256", "0" * 64, "manifest_mismatch:CodeSHA256"),
        ("StudyConfigSHA256", "0" * 64, "manifest_mismatch:StudyConfigSHA256"),
        ("PilotR", 1000, "manifest_mismatch:PilotR"),
        ("RunKind", "VALIDATION", "manifest_mismatch:RunKind"),
        ("Namespace", "FORMAL", "manifest_mismatch:Namespace"),
        ("RepeatStart", 2, "manifest_mismatch:RepeatStart"),
        ("RepeatEnd", 2, "manifest_mismatch:RepeatEnd"),
        ("RepeatCount", 2, "manifest_mismatch:RepeatCount"),
        ("DataSHA256", "0" * 64, "hash_mismatch:data"),
    ),
)
def test_pilot_manifest_mismatch_recomputes_whole_chunk(
    clean_f2_pilot_chunk,
    frozen_inputs,
    tmp_path,
    field_name,
    value,
    reason,
):
    root = _copy_root(clean_f2_pilot_chunk[0], tmp_path)
    _mutate_manifest(root, field_name, value)
    outcome = _resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    assert outcome.skipped_chunks == 0
    assert outcome.recomputed_invalid_chunks == 1
    assert any(reason in item for item in outcome.invalidated_chunks)
    assert_scientific_results_equal(
        clean_f2_pilot_chunk[1], outcome.result, label=reason
    )


@pytest.mark.parametrize(
    ("filename", "reason"),
    (
        ("chunk_state.npy", "hash_mismatch:state"),
        ("chunk_trajectory.npz", "hash_mismatch:trajectory"),
        ("chunk_repeat_metrics.csv", "hash_mismatch:repeat_metrics"),
    ),
)
def test_corrupt_pilot_data_file_recomputes_whole_chunk(
    clean_f2_pilot_chunk,
    frozen_inputs,
    tmp_path,
    filename,
    reason,
):
    root = _copy_root(clean_f2_pilot_chunk[0], tmp_path)
    _corrupt_last_byte(next(root.rglob(filename)))
    outcome = _resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    assert outcome.skipped_chunks == 0
    assert outcome.recomputed_invalid_chunks == 1
    assert any(reason in item for item in outcome.invalidated_chunks)
    assert_scientific_results_equal(
        clean_f2_pilot_chunk[1], outcome.result, label=reason
    )


def test_corrupt_h300_strata_recomputes_whole_chunk(
    clean_h300_pilot_chunk, frozen_inputs, tmp_path
):
    root = _copy_root(clean_h300_pilot_chunk[0], tmp_path)
    _corrupt_last_byte(next(root.rglob("chunk_h300_width_strata.csv")))
    outcome = _resume(root, frozen_inputs, condition_ordinal=3)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    assert outcome.skipped_chunks == 0
    assert outcome.recomputed_invalid_chunks == 1
    assert any("hash_mismatch:h300_strata" in item for item in outcome.invalidated_chunks)
    assert len(outcome.result.h300_width_strata) == 15
    assert_scientific_results_equal(
        clean_h300_pilot_chunk[1], outcome.result, label="H300 strata corruption"
    )

