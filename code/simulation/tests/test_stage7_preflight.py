from __future__ import annotations

import json

import pytest

from sal_stability_stage1.stage3 import load_stage3_inputs
from sal_stability_stage1.stage7 import (
    PREFLIGHT_CONDITIONS,
    PREFLIGHT_REPEAT_IDS,
    run_combined_parallel_chunk_resume_preflight,
)


@pytest.fixture(scope="module")
def preflight_result(tmp_path_factory):
    config, reference = load_stage3_inputs()
    root = tmp_path_factory.mktemp("stage7-combined-preflight")
    report = run_combined_parallel_chunk_resume_preflight(
        study_config=config,
        encoding_reference=reference,
        output_root=root,
        progress=lambda _: None,
    )
    return root, report


def test_combined_parallel_chunk_resume_preflight_is_exact(preflight_result):
    _, report = preflight_result
    assert report["Passed"] is True
    assert report["ExactEquality"] is True
    assert report["EqualNaNOnlyForSelectedObservedTOA"] is True
    assert report["RepeatIDs"] == list(PREFLIGHT_REPEAT_IDS)
    assert report["Workers"] == 2
    assert report["ChunkSize"] == 2
    assert [
        (row["ConditionOrdinal"], row["ConditionCode"])
        for row in report["Conditions"]
    ] == list(PREFLIGHT_CONDITIONS)
    assert all(
        row["SerialEqualsFreshParallelChunk"]
        and row["SerialEqualsInterruptedThenParallelChunkResume"]
        and row["InterruptedCompleted"] is False
        and row["ResumeSkippedChunks"] == 1
        and row["ResumeComputedChunks"] == 2
        for row in report["Conditions"]
    )


def test_preflight_storage_is_validation_identity_and_separate(preflight_result):
    root, _ = preflight_result
    manifests = tuple(root.rglob("chunk_manifest.json"))
    assert len(manifests) == 18
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["RunKind"] == "VALIDATION"
        assert payload["Namespace"] == "PILOT"
        assert payload["FormalR_or_PilotR"] is None
        assert "PilotR" not in payload
        assert "pilot_chunks" not in path.parts
    report = json.loads((root / "preflight_report.json").read_text(encoding="utf-8"))
    assert report["Passed"] is True

