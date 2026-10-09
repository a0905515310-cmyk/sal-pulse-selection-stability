from __future__ import annotations

import json

import pytest

from sal_stability_stage1.execution import ExecutionOptions, ORDER_A
from sal_stability_stage1.pilot_storage import CODE_VERSION, RUN_KIND
from sal_stability_stage1.stage7 import (
    BASELINE_ZIP_SHA256,
    FROZEN_SCIENCE_HASHES,
    build_pilot_manifest,
    lock_stage8_r_decision_contract,
    stage7_boundary_flags,
    stage8_r_decision_contract,
    validate_frozen_science_hashes,
    validate_official_pilot_execution_options,
    validate_stage1_through_stage6_statuses,
    verify_baseline_zip,
)
from sal_stability_stage1.storage import (
    CODE_VERSION as STAGE6_CODE_VERSION,
    RUN_KIND as STAGE6_RUN_KIND,
)


def test_stage8_decision_contract_all_frozen_constants():
    contract = stage8_r_decision_contract()
    assert contract["ContractName"] == "STAGE8_FORMAL_R_AUTOMATIC_DECISION"
    assert contract["ContractVersion"] == 1.0
    assert contract["LockedBeforePilot"] is True
    assert contract["Rpre"] == 200
    assert contract["InterferenceConditionOrdinals"] == list(range(1, 19))
    assert contract["MethodOrdinals"] == [2, 4]
    assert contract["MethodCodes"] == ["T", "TW"]
    assert contract["Metrics"] == [
        "P_cor",
        "P_C_given_C",
        "P_C_given_E",
        "Mean_L_NC",
    ]
    assert contract["RatioDefinitions"]["P_C_given_E"] == {
        "A_r": "N_EC",
        "B_r": "N_Edot",
    }
    assert contract["IndependentUnit"] == "Repeat"
    assert contract["VarianceUCBMultiplier"] == 1.1890464657347386
    assert contract["Z975"] == 1.96
    assert contract["ProbabilityHalfWidthTarget"] == 0.020
    assert contract["MeanLncHalfWidthTarget"] == 0.25
    assert contract["MinimumTotalDenominator"] == 100
    assert contract["MinimumPositiveDenominatorRepeats"] == 30
    assert contract["IndividualPrecisionCellCount"] == 144
    assert contract["PairedPrecisionCellCount"] == 72
    assert contract["TotalPrecisionCellCount"] == 216
    assert contract["RformalCandidates"] == [1000, 2000]
    assert contract["NoOutcomeDirectionRule"] is True
    assert contract["NoManualOverride"] is True
    assert contract["NoConditionDropping"] is True
    assert contract["NoParameterChange"] is True
    assert contract["FormalUsesSingleRForAllConditionsAndMethods"] is True


def test_contract_lock_is_idempotent_and_rejects_post_lock_change(tmp_path):
    path = tmp_path / "stage8_r_decision_contract.json"
    first = lock_stage8_r_decision_contract(path)
    original = path.read_bytes()
    second = lock_stage8_r_decision_contract(path)
    assert first == second
    assert path.read_bytes() == original
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["ProbabilityHalfWidthTarget"] = 0.021
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="not the frozen contract"):
        lock_stage8_r_decision_contract(path)


def test_stage7_boundary_prohibits_later_stage_execution():
    assert stage7_boundary_flags() == {
        "r_decision_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "stop_after_stage7": True,
    }


def test_pilot_manifest_has_frozen_pilot_identity():
    sha = "1" * 64
    manifest = build_pilot_manifest(
        frozen_hashes=FROZEN_SCIENCE_HASHES,
        stage7_code_sha256=sha,
        stage8_contract_sha256=sha,
        stage8_input_sha256=sha,
        repeat_metrics_sha256=sha,
        aggregated_metrics_sha256=sha,
        h300_strata_sha256=sha,
    )
    assert manifest["StageNumber"] == 7
    assert manifest["StageName"] == "PILOT_R200"
    assert manifest["RunKind"] == "PILOT" == RUN_KIND
    assert manifest["Namespace"] == "PILOT"
    assert manifest["PilotR"] == 200
    assert manifest["RepeatStart"] == 1
    assert manifest["RepeatEnd"] == 200
    assert manifest["ConditionCount"] == 19
    assert manifest["MethodCount"] == 5
    assert manifest["ExpectedMethodCycles"] == 3_800_000
    assert manifest["ChunkSize"] == 20
    assert manifest["ExpectedChunkCount"] == 190
    assert manifest["CompletedChunkCount"] == 190
    assert manifest["PilotCompleted"] is True
    assert manifest["FormalExecuted"] is False
    assert manifest["BootstrapExecuted"] is False
    assert manifest["RDecisionExecuted"] is False


def test_stage6_validation_storage_identity_is_unchanged():
    assert STAGE6_CODE_VERSION == "0.6.0"
    assert STAGE6_RUN_KIND == "VALIDATION"
    assert CODE_VERSION == "0.7.0"
    assert RUN_KIND == "PILOT"


def test_official_options_are_exactly_workers_two_chunk20_resume_on():
    valid = ExecutionOptions(
        workers=2,
        chunk_size=20,
        resume=True,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    validate_official_pilot_execution_options(valid)
    for candidate in (
        ExecutionOptions(workers=1, chunk_size=20, resume=True),
        ExecutionOptions(workers=2, chunk_size=19, resume=True),
        ExecutionOptions(workers=2, chunk_size=20, resume=False),
        ExecutionOptions(workers=2, chunk_size=20, resume=True, diagnostic=True),
        ExecutionOptions(workers=2, chunk_size=20, resume=True, h_cache_enabled=False),
    ):
        with pytest.raises(ValueError):
            validate_official_pilot_execution_options(candidate)


def test_baseline_zip_frozen_hashes_and_prior_statuses_pass():
    assert verify_baseline_zip() == BASELINE_ZIP_SHA256
    assert validate_frozen_science_hashes() == FROZEN_SCIENCE_HASHES
    statuses = validate_stage1_through_stage6_statuses()
    assert list(statuses) == [1, 2, 3, 4, 5, 6]
    assert all(status["status"] == "PASS" for status in statuses.values())

