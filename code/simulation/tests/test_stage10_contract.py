from __future__ import annotations

import inspect
import json

from sal_stability_stage1.bootstrap import (
    BOOTSTRAP_AUDIT_IDS,
    BOOTSTRAP_B,
    BOOTSTRAP_BLOCK_SIZE,
    BOOTSTRAP_DRAW_COUNT,
    CONTRAST_CODES,
    METHOD_CODES,
    METRIC_CODES,
    bootstrap_random_address,
)
from sal_stability_stage1.stage10 import (
    FORMAL_AGGREGATED_METRICS_SHA256,
    FORMAL_H300_WIDTH_STRATA_SHA256,
    FORMAL_REPEAT_METRICS_SHA256,
    LEGACY_NODES,
    SCIENCE_CONTRACT_PATH,
    STAGE10_PREFORMAL_CONTRACT_SHA256,
    STAGE10_SCIENCE_CONTRACT_SHA256,
    STAGE8_BASELINE_ZIP_SHA256,
    STAGE9_BASELINE_ZIP_SHA256,
    assert_no_forbidden_execution_path,
    validate_frozen_input_hashes,
    validate_science_contract,
    validate_stage1_through_stage9_tree_unchanged,
    validate_stage9_status,
)


def test_unique_stage9_baseline_and_all_frozen_input_hashes_are_exact():
    hashes = validate_frozen_input_hashes()
    assert hashes["Stage9BaselineZipSHA256"] == STAGE9_BASELINE_ZIP_SHA256
    assert hashes["Stage8RegressionZipSHA256"] == STAGE8_BASELINE_ZIP_SHA256
    assert hashes["FormalRepeatMetricsSHA256"] == FORMAL_REPEAT_METRICS_SHA256
    assert hashes["FormalAggregatedMetricsSHA256"] == FORMAL_AGGREGATED_METRICS_SHA256
    assert hashes["FormalH300WidthStrataSHA256"] == FORMAL_H300_WIDTH_STRATA_SHA256
    assert hashes["Stage10PreFormalContractSHA256"] == STAGE10_PREFORMAL_CONTRACT_SHA256
    assert hashes["Stage10ScienceContractSHA256"] == STAGE10_SCIENCE_CONTRACT_SHA256


def test_stage9_pass_and_stage1_through_stage9_zip_tree_identity():
    status = validate_stage9_status()
    assert status["status"] == "PASS"
    assert status["projected_precision_warning"] is True
    assert status["formal_r"] == 2000
    assert status["bootstrap_executed"] is False
    tree = validate_stage1_through_stage9_tree_unchanged()
    assert tree["Passed"] is True
    assert tree["DeletedFiles"] == []
    assert tree["UnexpectedAddedFiles"] == []
    assert tree["UnexpectedModifiedFiles"] == []


def test_canonical_science_contract_freezes_every_stage10_scope_boundary():
    contract = validate_science_contract()
    assert json.loads(SCIENCE_CONTRACT_PATH.read_text(encoding="utf-8")) == contract
    assert contract["BootstrapB"] == BOOTSTRAP_B == 2000
    assert contract["BootstrapDrawCountPerReplicate"] == BOOTSTRAP_DRAW_COUNT == 2000
    assert contract["IndependentUnit"] == "Repeat"
    assert contract["InterferenceConditionOrdinals"] == list(range(1, 19))
    assert contract["NIInference"] == "CONSISTENCY_ONLY_NO_BOOTSTRAP"
    assert contract["Methods"] == list(METHOD_CODES)
    assert contract["CoreMetrics"] == {
        "Mean_L_NC": {"Denominator": "N_run_NC", "Numerator": "Sum_L_NC_obs"},
        "P_C_given_C": {"Denominator": "N_Cdot", "Numerator": "N_CC"},
        "P_C_given_E": {"Denominator": "N_Edot", "Numerator": "N_EC"},
        "P_cor": {"Denominator": "200 per resampled Repeat", "Numerator": "N_C"},
    }
    assert [item["Code"] for item in contract["PairedContrasts"]] == list(CONTRAST_CODES)
    assert contract["MetricEstimator"] == "RATIO_OF_RESAMPLED_SUMS"
    assert contract["ConfidenceInterval"]["Type"] == "PERCENTILE_BOOTSTRAP"
    assert contract["ConfidenceInterval"]["QuantileMethod"] == "linear"
    assert contract["Execution"] == {
        "BootstrapMayRerunFormalWorld": False,
        "PersistentResumeUnit": "Condition",
        "PhysicalSimulationMayBeCalled": False,
        "VectorizationBlockSize": BOOTSTRAP_BLOCK_SIZE,
        "VectorizationBlockSizeIsNotStatisticalUnit": True,
        "Workers": 1,
    }


def test_no_p_value_physics_stage11_or_precision_outcome_pass_rule():
    contract = validate_science_contract()
    forbidden = contract["Forbidden"]
    assert forbidden["PValues"] is True
    assert forbidden["PhysicalSimulation"] is True
    assert forbidden["Stage11Execution"] is True
    assert forbidden["OutcomeDirectionPassRule"] is True
    precision = contract["Stage8PrecisionContract"]
    assert precision["ApplicableIndividualMethods"] == ["T", "TW"]
    assert precision["ApplicablePairedContrast"] == "TW-T"
    assert precision["TW-WNotGuaranteedByStage8PrecisionContract"] is True
    assert precision["FailureToMeetTargetDoesNotFailStage10"] is True
    assert assert_no_forbidden_execution_path()["PhysicalSimulationCallAbsent"] is True


def test_method_is_absent_from_random_address_and_legacy_scope_is_exact():
    parameters = inspect.signature(bootstrap_random_address).parameters
    assert "method" not in " ".join(parameters).lower()
    assert BOOTSTRAP_AUDIT_IDS == (1, 1000, 2000)
    assert METRIC_CODES == ("P_cor", "P_C_given_C", "P_C_given_E", "Mean_L_NC")
    assert LEGACY_NODES == (
        "tests/test_stage8_decision.py::test_stage1_through_stage7_byte_identity_and_frozen_science_gate",
        "tests/test_stage9_contract.py::test_stage8_identity_and_decision_boundary_are_independently_verified",
    )
