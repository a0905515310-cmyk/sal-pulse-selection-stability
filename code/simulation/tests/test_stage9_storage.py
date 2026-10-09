from __future__ import annotations

import json

import pytest

from sal_stability_stage1.execution import ExecutionOptions, ORDER_A, run_condition_repeats_serial
from sal_stability_stage1.formal_storage import (
    AUDIT_REPEAT_IDS,
    FORMAL_REPEAT_IDS,
    STAGE8_BASELINE_ZIP_SHA256,
    STAGE8_DECISION_CONTRACT_SHA256,
    STAGE8_DECISION_INPUT_SHA256,
    STAGE8_R_DECISION_SHA256,
    STUDY_CONFIG_SHA256,
    FormalIdentityHashes,
    compute_stage9_code_sha256,
    formal_chunk_paths,
    load_validated_formal_chunk,
    validate_formal_chunk,
    write_formal_chunk_atomic,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import load_stage3_inputs


@pytest.fixture(scope="module")
def frozen_inputs():
    return load_stage3_inputs()


@pytest.fixture(scope="module")
def identity():
    return FormalIdentityHashes(
        study_config_sha256=STUDY_CONFIG_SHA256,
        stage8_baseline_zip_sha256=STAGE8_BASELINE_ZIP_SHA256,
        stage8_r_decision_sha256=STAGE8_R_DECISION_SHA256,
        stage8_decision_contract_sha256=STAGE8_DECISION_CONTRACT_SHA256,
        stage8_decision_input_sha256=STAGE8_DECISION_INPUT_SHA256,
        stage10_preformal_contract_sha256="1" * 64,
        code_sha256=compute_stage9_code_sha256(),
    )


@pytest.fixture(scope="module")
def ni_result(frozen_inputs):
    config, reference = frozen_inputs
    return run_condition_repeats_serial(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=0,
        repeat_ids=FORMAL_REPEAT_IDS[:20],
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(
            workers=1,
            chunk_size=20,
            resume=False,
            diagnostic=False,
            h_cache_enabled=True,
            method_execution_order=ORDER_A,
        ),
    )


def test_formal_chunk_persists_only_metrics_and_preregistered_snapshot(
    tmp_path, identity, ni_result
):
    paths = write_formal_chunk_atomic(tmp_path, result=ni_result, identity=identity)
    assert {path.name for path in paths.directory.iterdir()} == {
        "chunk_repeat_metrics.csv",
        "audit_repeat_0001_snapshot.npz",
        "chunk_manifest.json",
    }
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["RunKind"] == "FORMAL"
    assert manifest["Namespace"] == "FORMAL"
    assert manifest["FormalR"] == 2000
    assert manifest["Workers"] == 2
    assert manifest["ChunkSize"] == 20
    assert manifest["AuditRepeatIDs"] == [AUDIT_REPEAT_IDS[0]]
    assert manifest["NIIdentityPassed"] is True
    assert "StateSHA256" not in manifest
    assert "TrajectorySHA256" not in manifest
    validation = validate_formal_chunk(
        paths,
        identity=identity,
        condition_ordinal=0,
        condition_code="NI",
        repeat_ids=FORMAL_REPEAT_IDS[:20],
    )
    assert validation.valid, validation.reason
    loaded = load_validated_formal_chunk(
        paths,
        identity=identity,
        condition_ordinal=0,
        condition_code="NI",
        repeat_ids=FORMAL_REPEAT_IDS[:20],
    )
    assert len(loaded.repeat_metrics) == 100
    assert loaded.audit_snapshot is not None
    assert loaded.audit_snapshot.source_state_code.shape == (5, 200)


def test_any_corruption_disqualifies_resume(tmp_path, identity, ni_result):
    paths = write_formal_chunk_atomic(tmp_path, result=ni_result, identity=identity)
    payload = bytearray(paths.repeat_metrics.read_bytes())
    payload[-2] ^= 1
    paths.repeat_metrics.write_bytes(payload)
    validation = validate_formal_chunk(
        paths,
        identity=identity,
        condition_ordinal=0,
        condition_code="NI",
        repeat_ids=FORMAL_REPEAT_IDS[:20],
    )
    assert not validation.valid
    assert validation.reason == "hash_mismatch:repeat_metrics"


def test_unexpected_full_trajectory_file_disqualifies_resume(
    tmp_path, identity, ni_result
):
    paths = write_formal_chunk_atomic(tmp_path, result=ni_result, identity=identity)
    (paths.directory / "chunk_trajectory.npz").write_bytes(b"forbidden")
    validation = validate_formal_chunk(
        paths,
        identity=identity,
        condition_ordinal=0,
        condition_code="NI",
        repeat_ids=FORMAL_REPEAT_IDS[:20],
    )
    assert not validation.valid
    assert validation.reason == "unexpected_or_missing_chunk_artifact"
