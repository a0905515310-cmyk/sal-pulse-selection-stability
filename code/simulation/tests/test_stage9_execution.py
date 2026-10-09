from __future__ import annotations

import json

from sal_stability_stage1.formal_storage import (
    STAGE8_BASELINE_ZIP_SHA256,
    STAGE8_DECISION_CONTRACT_SHA256,
    STAGE8_DECISION_INPUT_SHA256,
    STAGE8_R_DECISION_SHA256,
    STUDY_CONFIG_SHA256,
    FormalIdentityHashes,
    compute_stage9_code_sha256,
    formal_chunk_paths,
)
from sal_stability_stage1.stage3 import load_stage3_inputs
from sal_stability_stage1.stage9 import (
    official_formal_execution_options,
    run_formal_condition_chunked,
)


def _identity() -> FormalIdentityHashes:
    return FormalIdentityHashes(
        study_config_sha256=STUDY_CONFIG_SHA256,
        stage8_baseline_zip_sha256=STAGE8_BASELINE_ZIP_SHA256,
        stage8_r_decision_sha256=STAGE8_R_DECISION_SHA256,
        stage8_decision_contract_sha256=STAGE8_DECISION_CONTRACT_SHA256,
        stage8_decision_input_sha256=STAGE8_DECISION_INPUT_SHA256,
        stage10_preformal_contract_sha256="2" * 64,
        code_sha256=compute_stage9_code_sha256(),
    )


def test_formal_chunk_execution_resume_and_whole_chunk_recompute(tmp_path):
    config, reference = load_stage3_inputs()
    identity = _identity()
    options = official_formal_execution_options()
    first = run_formal_condition_chunked(
        condition_ordinal=3,
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
        identity=identity,
        output_root=tmp_path,
        stop_after_chunks=1,
        progress=None,
    )
    assert first.completed is False
    assert first.computed_chunks == 1
    assert first.skipped_chunks == 0
    second = run_formal_condition_chunked(
        condition_ordinal=3,
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
        identity=identity,
        output_root=tmp_path,
        stop_after_chunks=1,
        progress=None,
    )
    assert second.computed_chunks == 0
    assert second.skipped_chunks == 1

    paths = formal_chunk_paths(
        tmp_path,
        condition_ordinal=3,
        condition_code="H300",
        repeat_ids=tuple(range(1, 21)),
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["Completed"] = False
    paths.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    third = run_formal_condition_chunked(
        condition_ordinal=3,
        study_config=config,
        encoding_reference=reference,
        execution_options=options,
        identity=identity,
        output_root=tmp_path,
        stop_after_chunks=1,
        progress=None,
    )
    assert third.computed_chunks == 1
    assert third.skipped_chunks == 0
    assert third.recomputed_invalid_chunks == 1
    assert any("manifest_mismatch:Completed" in item for item in third.invalidated_chunks)
