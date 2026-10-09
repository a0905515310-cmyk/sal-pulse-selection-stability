from __future__ import annotations

import json

from sal_stability_stage1.results_export import FROZEN_SOURCE_SPECS
from sal_stability_stage1.stage11 import (
    ALLOWED_MODIFIED_FILES,
    ALLOWED_STATIC_ADDITIONS,
    FINAL_ZIP_NAME,
    LEGACY_NODES,
    STAGE10_BASELINE_ZIP_SHA256,
    STAGE11_SCIENCE_CONTRACT_SHA256,
    STAGE11_TASKBOOK_SHA256,
    validate_frozen_source_hashes,
    validate_required_static_files,
    validate_science_contract,
    validate_stage10_status,
    validate_stage1_through_stage10_tree_unchanged,
    verify_stage10_baseline_zip,
)


def test_unique_stage10_baseline_status_and_all_frozen_source_hashes_are_exact():
    assert verify_stage10_baseline_zip() == STAGE10_BASELINE_ZIP_SHA256
    status = validate_stage10_status()
    assert status["status"] == "PASS"
    assert status["blocking_issue_count"] == 0
    assert status["bootstrap_executed"] is True
    assert status["stage11_executed"] is False
    hashes = validate_frozen_source_hashes()
    assert len(hashes) == 12
    assert hashes == {
        name: expected for name, (_, expected) in FROZEN_SOURCE_SPECS.items()
    }


def test_frozen_science_contract_closes_every_stage11_scope_boundary():
    contract = validate_science_contract()
    assert contract["FinalZipName"] == FINAL_ZIP_NAME
    assert contract["FinalCodeStage"] is True
    assert contract["StopAfterStage11"] is True
    assert contract["NoStage12Execution"] is True
    assert contract["NoFormalRerun"] is True
    assert contract["NoNewBootstrap"] is True
    assert contract["NoNewRandomness"] is True
    assert contract["NoNewCI"] is True
    assert contract["NoPValues"] is True
    assert contract["NoNewContrasts"] is True
    assert contract["NoConditionDropping"] is True
    assert contract["NoParameterChange"] is True
    assert contract["NoPaperFigureRendering"] is True
    assert contract["NoPaperConclusionWriting"] is True
    assert contract["NoResultDrivenSelection"] is True
    assert contract["Conditions"] == {
        "NI": [0],
        "HPRF": [1, 2, 3, 4, 5],
        "F_ONLY": [6, 7, 8, 9],
        "COMPOSITE": list(range(10, 19)),
    }
    assert [item["Code"] for item in contract["Contrasts"]] == ["TW-T", "TW-W"]


def test_static_inputs_and_allowed_change_scope_are_exact():
    validate_required_static_files()
    contract = validate_science_contract()
    assert set(contract["AllowedCodeChanges"]["AddedStaticOrCode"]) == (
        ALLOWED_STATIC_ADDITIONS
    )
    assert set(contract["AllowedCodeChanges"]["ModifiedExisting"]) == (
        ALLOWED_MODIFIED_FILES
    )
    assert contract["FrozenSourceSHA256"] == {
        name: expected for name, (_, expected) in FROZEN_SOURCE_SPECS.items()
    }
    assert len(STAGE11_SCIENCE_CONTRACT_SHA256) == 64
    assert len(STAGE11_TASKBOOK_SHA256) == 64


def test_stage1_through_stage10_zip_tree_is_authoritative_and_unchanged():
    tree = validate_stage1_through_stage10_tree_unchanged()
    assert tree["Passed"] is True
    assert tree["DeletedFiles"] == []
    assert tree["UnexpectedAddedFiles"] == []
    assert tree["UnexpectedModifiedFiles"] == []
    assert set(tree["AllowedModifiedFiles"]) == ALLOWED_MODIFIED_FILES
    assert all(
        relative in ALLOWED_STATIC_ADDITIONS
        or relative.startswith("artifacts/stage11/")
        for relative in tree["AddedFiles"]
    )


def test_historical_forward_boundary_exception_nodes_are_exactly_three():
    assert LEGACY_NODES == (
        "tests/test_stage8_decision.py::test_stage1_through_stage7_byte_identity_and_frozen_science_gate",
        "tests/test_stage9_contract.py::test_stage8_identity_and_decision_boundary_are_independently_verified",
        "tests/test_stage10_contract.py::test_stage9_pass_and_stage1_through_stage9_zip_tree_identity",
    )

