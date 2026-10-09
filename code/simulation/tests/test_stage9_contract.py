from __future__ import annotations

import json

import pytest

from sal_stability_stage1.execution import ExecutionOptions, ORDER_A
from sal_stability_stage1.formal_storage import (
    AUDIT_REPEAT_IDS,
    CODE_VERSION,
    FORMAL_CHUNK_SIZE,
    FORMAL_R,
    FORMAL_WORKERS,
    STAGE8_BASELINE_ZIP_SHA256,
    STAGE8_DECISION_CONTRACT_SHA256,
    STAGE8_DECISION_INPUT_SHA256,
    STAGE8_R_DECISION_SHA256,
    STUDY_CONFIG_SHA256,
    compute_stage9_code_sha256,
    partition_formal_repeat_ids,
)
from sal_stability_stage1.stage9 import (
    EXPECTED_AUDIT_CONDITION_REPEATS,
    EXPECTED_AUDIT_DIAGNOSTICS_ROWS,
    EXPECTED_CHUNK_COUNT,
    EXPECTED_H300_STRATA_ROWS,
    EXPECTED_REPEAT_METRICS_ROWS,
    LEGACY_BOUNDARY_EXCEPTION_CODE,
    LEGACY_BOUNDARY_EXCEPTION_NODE,
    lock_stage10_bootstrap_contract,
    official_formal_execution_options,
    stage10_bootstrap_contract,
    validate_official_formal_execution_options,
    validate_stage1_through_stage8_tree_unchanged,
    validate_stage8_inputs,
    verify_stage8_baseline_zip,
)


def test_formal_scale_and_partition_are_exactly_frozen():
    assert CODE_VERSION == "0.9.0"
    assert FORMAL_R == 2000
    assert FORMAL_WORKERS == 2
    assert FORMAL_CHUNK_SIZE == 20
    assert AUDIT_REPEAT_IDS == (1, 1000, 2000)
    chunks = partition_formal_repeat_ids()
    assert len(chunks) == 100
    assert chunks[0] == tuple(range(1, 21))
    assert chunks[-1] == tuple(range(1981, 2001))
    assert EXPECTED_CHUNK_COUNT == 1900
    assert EXPECTED_REPEAT_METRICS_ROWS == 190000
    assert EXPECTED_H300_STRATA_ROWS == 30000
    assert EXPECTED_AUDIT_CONDITION_REPEATS == 57
    assert EXPECTED_AUDIT_DIAGNOSTICS_ROWS == 57000
    assert LEGACY_BOUNDARY_EXCEPTION_CODE == "LEGACY_STAGE_BOUNDARY_NOT_FORWARD_COMPATIBLE"
    assert LEGACY_BOUNDARY_EXCEPTION_NODE == (
        "tests/test_stage8_decision.py::"
        "test_stage1_through_stage7_byte_identity_and_frozen_science_gate"
    )


def test_stage10_scope_is_preformal_and_contains_no_execution():
    contract = stage10_bootstrap_contract()
    assert contract["ContractName"] == "STAGE10_BOOTSTRAP_SCOPE_PRE_FORMAL"
    assert contract["LockedBeforeFormal"] is True
    assert contract["BootstrapB"] == 2000
    assert contract["IndependentUnit"] == "Repeat"
    assert contract["BootstrapNamespace"] == "BOOTSTRAP"
    assert contract["CoreMetrics"] == [
        "P_cor",
        "P_C_given_C",
        "P_C_given_E",
        "Mean_L_NC",
    ]
    assert contract["IndividualMethodIntervals"] == ["FIRST", "LAST", "T", "W", "TW"]
    assert contract["PrimaryPairedContrast"] == "TW - T"
    assert contract["SecondaryAblationContrast"] == "TW - W"
    assert contract["InterferenceConditionOrdinals"] == list(range(1, 19))
    assert contract["NoPValueRequirement"] is True
    assert contract["NoOutcomeDirectionRule"] is True
    assert contract["NoPostFormalContrastAddition"] is True
    assert contract["NoConditionDropping"] is True
    assert contract["NoParameterChange"] is True
    assert contract["TWWIntervalsGuaranteedToMeetStage8HalfWidthTarget"] is False
    assert contract["BootstrapExecuted"] is False


def test_stage10_contract_lock_is_idempotent_and_rejects_drift(tmp_path):
    path = tmp_path / "stage10_bootstrap_contract_preformal.json"
    first = lock_stage10_bootstrap_contract(path)
    original = path.read_bytes()
    second = lock_stage10_bootstrap_contract(path)
    assert first == second
    assert path.read_bytes() == original
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["BootstrapB"] = 1999
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="changed after lock"):
        lock_stage10_bootstrap_contract(path)


def test_official_formal_options_are_exact_and_no_faster_variant_is_accepted():
    expected = ExecutionOptions(
        workers=2,
        chunk_size=20,
        resume=True,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    assert official_formal_execution_options() == expected
    validate_official_formal_execution_options(expected)
    for candidate in (
        ExecutionOptions(workers=1, chunk_size=20, resume=True),
        ExecutionOptions(workers=4, chunk_size=20, resume=True),
        ExecutionOptions(workers=2, chunk_size=50, resume=True),
        ExecutionOptions(workers=2, chunk_size=20, resume=False),
        ExecutionOptions(workers=2, chunk_size=20, resume=True, diagnostic=True),
        ExecutionOptions(workers=2, chunk_size=20, resume=True, h_cache_enabled=False),
    ):
        with pytest.raises(ValueError, match="frozen"):
            validate_official_formal_execution_options(candidate)


def test_stage8_identity_and_decision_boundary_are_independently_verified():
    assert verify_stage8_baseline_zip() == STAGE8_BASELINE_ZIP_SHA256
    report = validate_stage8_inputs()
    assert report["Stage8BaselineZIPSHA256"] == STAGE8_BASELINE_ZIP_SHA256
    assert report["Stage8RDecisionSHA256"] == STAGE8_R_DECISION_SHA256
    assert report["StudyConfigSHA256"] == STUDY_CONFIG_SHA256
    assert report["Stage8DecisionContractSHA256"] == STAGE8_DECISION_CONTRACT_SHA256
    assert report["Stage8DecisionInputSHA256"] == STAGE8_DECISION_INPUT_SHA256
    assert report["Stage8StatusPassed"] is True
    assert report["FrozenScienceHashesVerified"] is True
    tree = validate_stage1_through_stage8_tree_unchanged()
    assert tree["Passed"] is True
    assert tree["DeletedFiles"] == []
    assert compute_stage9_code_sha256()
