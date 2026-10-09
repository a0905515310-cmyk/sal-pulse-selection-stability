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
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import load_stage3_inputs
from sal_stability_stage1.stage6 import run_condition_repeats_chunked
from sal_stability_stage1.storage import (
    CODE_VERSION,
    assemble_condition_chunks,
    validate_chunk,
)


@pytest.fixture(scope="module")
def frozen_inputs():
    return load_stage3_inputs()


@pytest.fixture(scope="module")
def clean_f2_chunk(tmp_path_factory, frozen_inputs):
    config, reference = frozen_inputs
    root = tmp_path_factory.mktemp("stage6-clean-chunk")
    outcome = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(chunk_size=1),
        output_root=root,
    )
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    return root, outcome.result


def _copy_clean_chunk(clean_f2_chunk, tmp_path: Path) -> Path:
    source_root, _ = clean_f2_chunk
    destination = tmp_path / "copied"
    shutil.copytree(source_root, destination)
    return destination


def _run_resume(root: Path, frozen_inputs):
    config, reference = frozen_inputs
    return run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(chunk_size=1, resume=True),
        output_root=root,
    )


def test_chunk_manifest_and_atomic_files_are_complete(clean_f2_chunk):
    root, _ = clean_f2_chunk
    manifests = list(root.rglob("chunk_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["StudyConfigSHA256"]
    assert manifest["CodeVersion"] == CODE_VERSION
    assert manifest["CodeSHA256"]
    assert manifest["Namespace"] == "PILOT"
    assert manifest["ConditionOrdinal"] == 7
    assert manifest["ConditionCode"] == "F2"
    assert manifest["RepeatStart"] == manifest["RepeatEnd"] == 10001
    assert manifest["RepeatCount"] == 1
    assert manifest["FormalR_or_PilotR"] is None
    assert manifest["RunKind"] == "VALIDATION"
    assert manifest["ValidationRepeatCount"] == 1
    assert manifest["Completed"] is True
    assert manifest["StateSHA256"]
    assert manifest["TrajectorySHA256"]
    assert manifest["RepeatMetricsSHA256"]
    assert manifest["DataSHA256"]
    assert not any("RNGState" in key for key in manifest)
    directory = manifests[0].parent
    assert (directory / "chunk_state.npy").is_file()
    assert (directory / "chunk_trajectory.npz").is_file()
    assert (directory / "chunk_repeat_metrics.csv").is_file()
    assert not (directory / "chunk_h300_width_strata.csv").exists()
    assert not any(".tmp-" in path.name for path in root.rglob("*"))


def test_valid_completed_chunk_is_really_skipped(
    clean_f2_chunk, frozen_inputs, tmp_path
):
    root = _copy_clean_chunk(clean_f2_chunk, tmp_path)
    outcome = _run_resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.skipped_chunks == 1
    assert outcome.computed_chunks == 0
    assert outcome.invalidated_chunks == ()
    assert_scientific_results_equal(
        clean_f2_chunk[1], outcome.result, label="valid chunk skip"
    )


def test_missing_chunk_is_computed(clean_f2_chunk, frozen_inputs, tmp_path):
    root = tmp_path / "missing"
    outcome = _run_resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    assert outcome.skipped_chunks == 0
    assert any("manifest_unreadable" in item for item in outcome.invalidated_chunks)
    assert_scientific_results_equal(
        clean_f2_chunk[1], outcome.result, label="missing chunk recompute"
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        (("manifest", "Completed", False), "manifest_mismatch:Completed"),
        (("manifest", "CodeVersion", "0.5.0"), "manifest_mismatch:CodeVersion"),
        (
            ("manifest", "StudyConfigSHA256", "0" * 64),
            "manifest_mismatch:StudyConfigSHA256",
        ),
        (("state", None, None), "hash_mismatch:state"),
    ),
)
def test_invalid_chunk_is_recomputed_as_a_whole(
    clean_f2_chunk,
    frozen_inputs,
    tmp_path,
    mutation,
    reason,
):
    root = _copy_clean_chunk(clean_f2_chunk, tmp_path)
    manifest_path = next(root.rglob("chunk_manifest.json"))
    if mutation[0] == "manifest":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload[mutation[1]] = mutation[2]
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        state_path = manifest_path.parent / "chunk_state.npy"
        with state_path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            value = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes((value[0] ^ 1,)))
    outcome = _run_resume(root, frozen_inputs)
    assert outcome.completed and outcome.result is not None
    assert outcome.computed_chunks == 1
    assert outcome.skipped_chunks == 0
    assert any(reason in item for item in outcome.invalidated_chunks)
    assert_scientific_results_equal(
        clean_f2_chunk[1], outcome.result, label=reason
    )


def test_chunk_size_one_and_two_are_exact(frozen_inputs, tmp_path):
    config, reference = frozen_inputs
    common = dict(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_ids=(10001, 10002),
        study_config=config,
        encoding_reference=reference,
    )
    size_one = run_condition_repeats_chunked(
        **common,
        execution_options=ExecutionOptions(chunk_size=1),
        output_root=tmp_path / "size1",
    )
    size_two = run_condition_repeats_chunked(
        **common,
        execution_options=ExecutionOptions(chunk_size=2),
        output_root=tmp_path / "size2",
    )
    assert size_one.result is not None and size_two.result is not None
    assert_scientific_results_equal(
        size_one.result, size_two.result, label="ChunkSize 1/2"
    )


def test_h300_chunk_roundtrip_contains_exact_width_strata(frozen_inputs, tmp_path):
    config, reference = frozen_inputs
    clean = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(chunk_size=1),
        output_root=tmp_path / "h300",
    )
    assert clean.result is not None
    assert len(clean.result.h300_width_strata) == 15
    assert next((tmp_path / "h300").rglob("chunk_h300_width_strata.csv")).is_file()
    resumed = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(chunk_size=1, resume=True),
        output_root=tmp_path / "h300",
    )
    assert resumed.result is not None and resumed.skipped_chunks == 1
    assert_scientific_results_equal(
        clean.result, resumed.result, label="H300 storage roundtrip"
    )


def test_overlapping_chunks_are_rejected(clean_f2_chunk):
    _, result = clean_f2_chunk
    with pytest.raises(ValueError, match="overlap"):
        assemble_condition_chunks(
            (result, result), expected_repeat_ids=(10001,)
        )
