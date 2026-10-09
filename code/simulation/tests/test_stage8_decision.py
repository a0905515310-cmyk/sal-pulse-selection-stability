from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from sal_stability_stage1.stage7 import (
    FROZEN_SCIENCE_HASHES,
    validate_frozen_science_hashes,
)
from sal_stability_stage1.stage8 import (
    BASELINE_STAGE7_ZIP_SHA256,
    DECISION_CONTRACT_SHA256,
    DECISION_INPUT_FIELDS,
    DECISION_INPUT_SHA256,
    INDIVIDUAL_PRECISION_FIELDS,
    METRICS,
    ORACLE,
    PAIRED_PRECISION_FIELDS,
    VARIANCE_UCB_MULTIPLIER,
    build_precision_warning,
    build_r_decision,
    compute_individual_precision,
    compute_paired_precision,
    denominator_support_passes,
    load_and_validate_decision_contract,
    load_and_validate_decision_input,
    ratio_influence,
    validate_frozen_oracle,
    validate_stage1_through_stage7_tree_unchanged,
    validate_stage7_status,
    verify_stage7_baseline_zip,
)


@pytest.fixture(scope="module")
def official():
    status = validate_stage7_status()
    contract, contract_sha = load_and_validate_decision_contract()
    decision_input = load_and_validate_decision_input()
    individual, influences = compute_individual_precision(decision_input)
    paired = compute_paired_precision(influences)
    decision = build_r_decision(
        stage7_status=status,
        decision_input=decision_input,
        contract=contract,
        contract_sha256=contract_sha,
        individual_rows=individual,
        paired_rows=paired,
    )
    warning = build_precision_warning(individual, paired)
    validate_frozen_oracle(decision, individual, paired)
    return {
        "status": status,
        "contract": contract,
        "contract_sha": contract_sha,
        "input": decision_input,
        "individual": individual,
        "influences": influences,
        "paired": paired,
        "decision": decision,
        "warning": warning,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(rows, **identity):
    return next(
        row
        for row in rows
        if all(row[field_name] == value for field_name, value in identity.items())
    )


def test_stage7_baseline_status_and_prohibited_execution_gate(official):
    assert verify_stage7_baseline_zip() == BASELINE_STAGE7_ZIP_SHA256
    status = official["status"]
    assert status["status"] == "PASS"
    assert status["pilot_executed"] is True
    assert status["pilot_r"] == 200
    assert status["pilot_namespace"] == "PILOT"
    assert status["r_decision_executed"] is False
    assert status["formal_executed"] is False
    assert status["bootstrap_executed"] is False
    assert status["paper_result_generated"] is False


def test_stage1_through_stage7_byte_identity_and_frozen_science_gate():
    diff = validate_stage1_through_stage7_tree_unchanged()
    assert diff["deleted"] == []
    assert validate_frozen_science_hashes() == FROZEN_SCIENCE_HASHES


def test_decision_contract_sha_and_every_frozen_rule(official):
    contract = official["contract"]
    assert official["contract_sha"] == DECISION_CONTRACT_SHA256
    assert contract["ContractName"] == "STAGE8_FORMAL_R_AUTOMATIC_DECISION"
    assert contract["ContractVersion"] == 1.0
    assert contract["LockedBeforePilot"] is True
    assert contract["Rpre"] == 200
    assert contract["IndependentUnit"] == "Repeat"
    assert contract["InterferenceConditionOrdinals"] == list(range(1, 19))
    assert contract["MethodOrdinals"] == [2, 4]
    assert contract["MethodCodes"] == ["T", "TW"]
    assert contract["Metrics"] == list(METRICS)
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
    assert contract["R1000Rule"]["Operator"] == "ALL"
    assert contract["NoOutcomeDirectionRule"] is True
    assert contract["NoManualOverride"] is True
    assert contract["NoConditionDropping"] is True
    assert contract["NoParameterChange"] is True
    assert contract["FormalUsesSingleRForAllConditionsAndMethods"] is True


def test_contract_drift_is_rejected_even_when_file_hash_is_self_consistent(tmp_path):
    contract, _ = load_and_validate_decision_contract()
    drifted = dict(contract)
    drifted["NoManualOverride"] = False
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(drifted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="constants drifted"):
        load_and_validate_decision_contract(path, expected_sha256=_sha256(path))


def test_decision_input_sha_schema_rows_and_exact_key_universe(official):
    decision_input = official["input"]
    assert decision_input.sha256 == DECISION_INPUT_SHA256
    assert decision_input.row_count == 7200
    assert decision_input.condition_ordinals == tuple(range(1, 19))
    assert decision_input.method_ordinals == (2, 4)
    assert decision_input.repeat_ids == tuple(range(1, 201))
    keys = {
        (row["ConditionOrdinal"], row["MethodOrdinal"], row["RepeatID"])
        for row in decision_input.rows
    }
    assert len(keys) == 7200
    assert {row["ConditionCode"] for row in decision_input.rows}.isdisjoint({"NI"})
    assert {row["MethodCode"] for row in decision_input.rows} == {"T", "TW"}


def test_decision_input_duplicate_key_is_rejected(tmp_path, official):
    path = tmp_path / "decision_input.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECISION_INPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in official["input"].rows:
            writer.writerow(row)
        writer.writerow(official["input"].rows[0])
    with pytest.raises(AssertionError, match="exactly 7200 rows"):
        load_and_validate_decision_input(path, expected_sha256=_sha256(path))


def test_individual_and_paired_cell_counts_schemas_definition_and_support(official):
    individual = official["individual"]
    paired = official["paired"]
    assert len(individual) == 144
    assert len(paired) == 72
    assert all(set(row) == set(INDIVIDUAL_PRECISION_FIELDS) for row in individual)
    assert all(set(row) == set(PAIRED_PRECISION_FIELDS) for row in paired)
    assert all(row["Defined"] is True for row in individual)
    assert all(row["Defined"] is True for row in paired)
    assert all(row["SupportPass"] is True for row in individual)


def test_p_c_given_e_uses_n_ec_ratio_of_sums_not_per_repeat_means(official):
    cell = _cell(
        official["individual"],
        ConditionCode="H100",
        MethodCode="T",
        Metric="P_C_given_E",
    )
    source = [
        row
        for row in official["input"].rows
        if row["ConditionCode"] == "H100" and row["MethodCode"] == "T"
    ]
    expected = sum(row["N_EC"] for row in source) / sum(row["N_Edot"] for row in source)
    positive_repeat_ratios = [
        row["N_EC"] / row["N_Edot"] for row in source if row["N_Edot"] > 0
    ]
    assert cell["NumeratorField"] == "N_EC"
    assert cell["DenominatorField"] == "N_Edot"
    assert cell["ThetaHat"] == pytest.approx(expected, rel=0, abs=1e-15)
    assert abs(cell["ThetaHat"] - sum(positive_repeat_ratios) / len(positive_repeat_ratios)) > 1e-6


def test_repeat_is_the_independent_unit_for_individual_variance(official):
    cell = _cell(
        official["individual"],
        ConditionCode="H100",
        MethodCode="T",
        Metric="P_C_given_E",
    )
    source = [
        row
        for row in official["input"].rows
        if row["ConditionCode"] == "H100" and row["MethodCode"] == "T"
    ]
    a_values = [row["N_EC"] for row in source]
    b_values = [row["N_Edot"] for row in source]
    theta, b_bar, psi, psi_mean, variance = ratio_influence(a_values, b_values)
    assert len(psi) == 200
    assert cell["ThetaHat"] == theta
    assert cell["BBar"] == b_bar
    assert cell["PsiMean"] == psi_mean
    assert cell["SPsiSq"] == variance
    assert cell["SPsiUpperSq"] == pytest.approx(
        VARIANCE_UCB_MULTIPLIER * variance, rel=0, abs=1e-15
    )
    assert abs(psi_mean) < 1e-13


def test_denominator_support_gate_has_both_frozen_thresholds():
    assert denominator_support_passes("P_cor", 0, 0) is True
    assert denominator_support_passes("P_C_given_C", 99, 200) is False
    assert denominator_support_passes("P_C_given_C", 100, 29) is False
    assert denominator_support_passes("P_C_given_C", 100, 30) is True
    assert denominator_support_passes("Mean_L_NC", 100, 30) is True


def test_ratio_influence_rejects_cycles_as_units_and_zero_total_denominator():
    with pytest.raises(AssertionError, match="200 Repeat"):
        ratio_influence([1] * 199, [1] * 199)
    with pytest.raises(ZeroDivisionError, match=r"sum\(B_r\)=0"):
        ratio_influence([0] * 200, [0] * 200)


def test_paired_variance_is_strict_repeatwise_tw_minus_t(official):
    paired = _cell(
        official["paired"], ConditionCode="H100", Metric="P_C_given_E"
    )
    t = official["influences"][(1, 2, "P_C_given_E")].psi_by_repeat
    tw = official["influences"][(1, 4, "P_C_given_E")].psi_by_repeat
    assert set(t) == set(tw) == set(range(1, 201))
    delta = [tw[repeat_id] - t[repeat_id] for repeat_id in range(1, 201)]
    delta_mean = math.fsum(delta) / 200
    variance = math.fsum((value - delta_mean) ** 2 for value in delta) / 199
    assert paired["SPsiDeltaSq"] == variance
    assert paired["SPsiDeltaUpperSq"] == pytest.approx(
        VARIANCE_UCB_MULTIPLIER * variance, rel=0, abs=1e-15
    )
    assert paired["DeltaHat"] == pytest.approx(
        paired["ThetaTW"] - paired["ThetaT"], rel=0, abs=1e-15
    )


def test_frozen_oracle_counts_maxima_and_automatic_formal_r(official):
    decision = official["decision"]
    assert decision["IndividualPrecisionCellCount"] == ORACLE["IndividualPrecisionCellCount"]
    assert decision["PairedPrecisionCellCount"] == ORACLE["PairedPrecisionCellCount"]
    assert decision["TotalPrecisionCellCount"] == ORACLE["TotalPrecisionCellCount"]
    assert decision["IndividualFailR1000Count"] == 11
    assert decision["PairedFailR1000Count"] == 9
    assert decision["IndividualExceedR2000Count"] == 4
    assert decision["PairedExceedR2000Count"] == 3
    assert decision["MaxIndividualRequiredR"] == 41967
    assert decision["MaxPairedRequiredR"] == 42840
    assert decision["R1000Eligible"] is False
    assert decision["Rformal"] == 2000
    assert decision["ProjectedPrecisionWarning"] is True


def test_two_critical_h100_oracle_cells(official):
    individual = _cell(
        official["individual"],
        ConditionCode="H100",
        MethodCode="TW",
        Metric="P_C_given_E",
    )
    paired = _cell(
        official["paired"], ConditionCode="H100", Metric="P_C_given_E"
    )
    assert individual["RequiredR"] == 41967
    assert paired["RequiredR"] == 42840


def test_r1000_all_rule_fails_only_precision_checks(official):
    checks = official["decision"]["R1000Checks"]
    assert checks == {
        "Stage7PilotDataComplete": True,
        "R0Equals200": True,
        "Conditions1Through18Complete": True,
        "MethodsTAndTWComplete": True,
        "RepeatIDs1Through200UniqueAndComplete": True,
        "AllFourMetricsDefined": True,
        "AllNonfixedTotalDenominatorsAtLeast100": True,
        "AllNonfixedPositiveDenominatorRepeatCountsAtLeast30": True,
        "All144IndividualRequiredRValuesAtMost1000": False,
        "All72PairedRequiredRValuesAtMost1000": False,
    }
    assert all(checks.values()) is False


def test_precision_warning_contains_exact_exceedance_sets(official):
    warning = official["warning"]
    assert warning["ProjectedPrecisionWarning"] is True
    assert warning["IndividualExceedR2000Count"] == 4
    assert warning["PairedExceedR2000Count"] == 3
    assert len(warning["IndividualCells"]) == 4
    assert len(warning["PairedCells"]) == 3
    assert any(
        cell["ConditionCode"] == "H100"
        and cell["MethodCode"] == "TW"
        and cell["Metric"] == "P_C_given_E"
        and cell["RequiredR"] == 41967
        for cell in warning["IndividualCells"]
    )


def test_outcome_direction_never_enters_decision_and_no_prohibited_action(official):
    decision = official["decision"]
    assert "DeltaHat" not in decision["R1000Checks"]
    assert decision["FormalRSource"] == "STAGE8_FROZEN_ALL_RULE"
    assert decision["ManualOverrideApplied"] is False
    assert decision["OutcomeDirectionUsed"] is False
    assert decision["ConditionDropped"] is False
    assert decision["ParameterChanged"] is False
    assert decision["FormalUsesSingleRForAllConditionsAndMethods"] is True
    assert decision["FormalExecuted"] is False
    assert decision["BootstrapExecuted"] is False
    assert decision["PaperResultGenerated"] is False
