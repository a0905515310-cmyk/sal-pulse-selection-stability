from __future__ import annotations

import json

from sal_stability_stage1.results_export import (
    EVIDENCE_REGISTRY_FIELDS,
    EVIDENCE_REGISTRY_REL,
    STAGE11_DIR,
    read_csv_exact,
    sha256_file,
    validate_evidence_exports,
)
from sal_stability_stage1.stage11 import (
    CHANGED_FILES_PATH,
    FINAL_ZIP_NAME,
    MANIFEST_PATH,
    REQUIRED_EVIDENCE_ROW_COUNTS,
    STATUS_PATH,
    validate_final_zip,
    validate_forbidden_artifact_absence,
    validate_stage1_through_stage10_tree_unchanged,
    verify_stage11_manifest,
)
from sal_stability_stage1.build import ROOT


def test_evidence_files_registry_rows_hashes_and_lineage_are_complete():
    validation = validate_evidence_exports()
    registry = read_csv_exact(
        STAGE11_DIR / EVIDENCE_REGISTRY_REL,
        EVIDENCE_REGISTRY_FIELDS,
        validation["EvidenceRegistryRows"],
    )
    assert len(registry) == 16
    assert len({row["ArtifactPath"] for row in registry}) == 16
    assert all(row["SelectiveFilteringAllowed"] == "false" for row in registry)
    assert all(
        row["InferenceClass"]
        in {
            "STAGE10_FROZEN_INFERENCE",
            "STAGE9_FROZEN_DESCRIPTIVE",
            "IDENTITY_OR_METADATA",
        }
        for row in registry
    )
    assert not any(row["ArtifactPath"].endswith(EVIDENCE_REGISTRY_REL) for row in registry)
    for row in registry:
        relative = row["ArtifactPath"].removeprefix("artifacts/stage11/")
        path = STAGE11_DIR / relative
        assert path.is_file()
        assert int(row["RowCount"]) == validation["RowCounts"][relative]
        assert row["SHA256"] == sha256_file(path)
        assert row["ImmediateSourceArtifacts"]
        assert row["AllowedUse"]


def test_manifest_status_and_changed_file_scope_are_consistent():
    manifest = verify_stage11_manifest()
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    assert status["status"] == "PASS"
    assert status["blocking_issue_count"] == 0
    assert status["final_code_stage"] is True
    assert status["stop_after_stage11"] is True
    assert status["stage12_executed"] is False
    assert manifest["StopAfterStage11"] is True
    assert manifest["Stage12Executed"] is False
    assert manifest["PaperEvidenceRegistrySHA256"] == sha256_file(
        STAGE11_DIR / EVIDENCE_REGISTRY_REL
    )
    tree = validate_stage1_through_stage10_tree_unchanged()
    changed = CHANGED_FILES_PATH.read_text(encoding="utf-8")
    for relative in tree["AllowedModifiedFiles"] + tree["AddedFiles"]:
        assert relative in changed
    assert tree["DeletedFiles"] == []


def test_required_row_counts_are_frozen_in_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for relative, expected in REQUIRED_EVIDENCE_ROW_COUNTS.items():
        assert manifest["EvidenceRowCounts"][relative] == expected
    assert manifest["EvidenceRegistryRows"] == 16


def test_final_zip_members_tree_hashes_counts_and_forbidden_absence_pass():
    result = validate_final_zip(ROOT.parent / FINAL_ZIP_NAME)
    assert result["RequiredEvidenceRowCountsVerified"] is True
    assert result["ArtifactSHA256Verified"] is True
    assert result["Stage12Absent"] is True
    assert result["ForbiddenRenderedArtifactsAbsent"] is True
    validate_forbidden_artifact_absence()

