from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .build import ROOT
from .contracts import CONDITIONS, METHODS
from .pilot_storage import (
    sha256_file,
    write_csv_dicts_atomic,
    write_json_atomic,
    write_text_atomic,
)
from .stage7 import (
    FROZEN_SCIENCE_HASHES,
    stage8_r_decision_contract,
    validate_frozen_science_hashes,
    validate_stage1_through_stage6_statuses,
)


STAGE_NUMBER = 8
STAGE_NAME = "FORMAL_R_AUTOMATIC_DECISION"
STAGE8_DIR = ROOT / "artifacts" / "stage8"
STAGE7_STATUS_PATH = ROOT / "artifacts" / "stage7" / "status.json"
DECISION_CONTRACT_PATH = (
    ROOT / "artifacts" / "stage7" / "stage8_r_decision_contract.json"
)
DECISION_INPUT_PATH = ROOT / "artifacts" / "stage7" / "stage8_decision_input.csv"

BASELINE_STAGE7_ZIP_NAME = "sal_stability_stage7_20260824.zip"
BASELINE_STAGE7_ZIP_SHA256 = (
    "d6412152a199a820ef40c1485689846ee80d9227c51f9e3b6f5ff4dbac55e145"
)
DECISION_CONTRACT_SHA256 = (
    "1991d8cec0a91438ccd95a5e96ed02cfd4d27285e9e2fa4eba047f9938e285e6"
)
DECISION_INPUT_SHA256 = (
    "655ffd9f03a7a7824662c9a7a1a2281ed620c73019239236e386fb15eac0eb7b"
)
FINAL_ZIP_NAME = "sal_stability_stage8_20260825.zip"

R0 = 200
INTERFERENCE_CONDITION_ORDINALS = tuple(range(1, 19))
METHOD_ORDINALS = (2, 4)
METHOD_CODES = {2: "T", 4: "TW"}
METRICS = ("P_cor", "P_C_given_C", "P_C_given_E", "Mean_L_NC")
VARIANCE_UCB_MULTIPLIER = 1.1890464657347386
Z975 = 1.96
PROBABILITY_HALF_WIDTH_TARGET = 0.020
MEAN_L_NC_HALF_WIDTH_TARGET = 0.25
MINIMUM_TOTAL_DENOMINATOR = 100
MINIMUM_POSITIVE_DENOMINATOR_REPEATS = 30

DECISION_INPUT_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrdinal",
    "MethodCode",
    "RepeatID",
    "N_C",
    "N_CC",
    "N_Cdot",
    "N_EC",
    "N_Edot",
    "Sum_L_NC_obs",
    "N_run_NC",
)
INTEGER_INPUT_FIELDS = (
    "ConditionOrdinal",
    "MethodOrdinal",
    "RepeatID",
    "N_C",
    "N_CC",
    "N_Cdot",
    "N_EC",
    "N_Edot",
    "Sum_L_NC_obs",
    "N_run_NC",
)

METRIC_DEFINITIONS: dict[str, tuple[str, str | int, float]] = {
    "P_cor": ("N_C", 200, PROBABILITY_HALF_WIDTH_TARGET),
    "P_C_given_C": ("N_CC", "N_Cdot", PROBABILITY_HALF_WIDTH_TARGET),
    "P_C_given_E": ("N_EC", "N_Edot", PROBABILITY_HALF_WIDTH_TARGET),
    "Mean_L_NC": ("Sum_L_NC_obs", "N_run_NC", MEAN_L_NC_HALF_WIDTH_TARGET),
}

INDIVIDUAL_PRECISION_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "MethodOrdinal",
    "MethodCode",
    "Metric",
    "NumeratorField",
    "DenominatorField",
    "TargetHalfWidth",
    "Defined",
    "UndefinedReason",
    "SumA",
    "SumB",
    "PositiveDenominatorRepeatCount",
    "SupportPass",
    "ThetaHat",
    "BBar",
    "PsiMean",
    "SPsiSq",
    "SPsiUpperSq",
    "RequiredR",
    "ProjectedHalfWidthR1000",
    "ProjectedHalfWidthR2000",
    "R1000Pass",
    "R2000TargetPass",
    "ExceedsR2000",
)

PAIRED_PRECISION_FIELDS = (
    "ConditionOrdinal",
    "ConditionCode",
    "Metric",
    "TargetHalfWidth",
    "Defined",
    "UndefinedReason",
    "ThetaT",
    "ThetaTW",
    "DeltaHat",
    "SPsiDeltaSq",
    "SPsiDeltaUpperSq",
    "RequiredR",
    "ProjectedHalfWidthR1000",
    "ProjectedHalfWidthR2000",
    "R1000Pass",
    "R2000TargetPass",
    "ExceedsR2000",
)

ORACLE = {
    "IndividualPrecisionCellCount": 144,
    "PairedPrecisionCellCount": 72,
    "TotalPrecisionCellCount": 216,
    "IndividualFailR1000Count": 11,
    "PairedFailR1000Count": 9,
    "IndividualExceedR2000Count": 4,
    "PairedExceedR2000Count": 3,
    "MaxIndividualRequiredR": 41967,
    "MaxPairedRequiredR": 42840,
    "R1000Eligible": False,
    "FormalR": 2000,
    "ProjectedPrecisionWarning": True,
}


@dataclass(frozen=True)
class ValidatedDecisionInput:
    rows: tuple[dict[str, object], ...]
    sha256: str
    row_count: int
    condition_ordinals: tuple[int, ...]
    method_ordinals: tuple[int, ...]
    repeat_ids: tuple[int, ...]


@dataclass(frozen=True)
class InfluenceCell:
    output: dict[str, object]
    psi_by_repeat: dict[int, float]


def verify_stage7_baseline_zip(path: Path | None = None) -> str:
    baseline = Path(path) if path is not None else ROOT.parent / BASELINE_STAGE7_ZIP_NAME
    if not baseline.is_file():
        raise FileNotFoundError(f"unique Stage 7 baseline ZIP is missing: {baseline}")
    actual = sha256_file(baseline)
    if actual != BASELINE_STAGE7_ZIP_SHA256:
        raise AssertionError(
            "Stage 7 baseline ZIP SHA256 mismatch: "
            f"expected {BASELINE_STAGE7_ZIP_SHA256}, got {actual}"
        )
    return actual


def validate_stage7_status(path: Path = STAGE7_STATUS_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Stage 7 status must be a JSON object")
    required: dict[str, object] = {
        "stage": 7,
        "stage_name": "PILOT_R200",
        "status": "PASS",
        "pilot_executed": True,
        "pilot_r": 200,
        "pilot_namespace": "PILOT",
        "pilot_condition_count": 19,
        "pilot_method_count": 5,
        "pilot_repeat_metrics_row_count": 19000,
        "pilot_aggregated_metrics_row_count": 95,
        "pilot_h300_strata_row_count": 3000,
        "stage8_decision_input_row_count": 7200,
        "science_contract_changed": False,
        "frozen_hashes_passed": True,
        "r_decision_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "blocking_issue_count": 0,
    }
    for field_name, expected in required.items():
        actual = payload.get(field_name)
        if actual != expected:
            raise AssertionError(
                f"Stage 7 status field {field_name} must be {expected!r}, got {actual!r}"
            )
    if payload.get("pilot_repeat_range") != [1, 200]:
        raise AssertionError("Stage 7 Pilot RepeatID range must be exactly 1..200")
    return payload


def load_and_validate_decision_contract(
    path: Path = DECISION_CONTRACT_PATH,
    *,
    expected_sha256: str = DECISION_CONTRACT_SHA256,
) -> tuple[dict[str, object], str]:
    contract_path = Path(path)
    actual_sha = sha256_file(contract_path)
    if actual_sha != expected_sha256:
        raise AssertionError(
            f"Stage 8 decision contract SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha}"
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = stage8_r_decision_contract()
    if payload != expected:
        raise AssertionError("Stage 8 decision contract constants drifted")
    if payload["RatioDefinitions"]["P_C_given_E"] != {
        "A_r": "N_EC",
        "B_r": "N_Edot",
    }:
        raise AssertionError("P_C_given_E must use N_EC/N_Edot")
    if payload["R1000Rule"]["Operator"] != "ALL":
        raise AssertionError("Stage 8 R=1000 decision operator must be ALL")
    return payload, actual_sha


def _parse_nonnegative_integer(value: str | None, field_name: str) -> int:
    if value is None or not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise AssertionError(f"{field_name} must be a canonical nonnegative integer")
    return int(value)


def load_and_validate_decision_input(
    path: Path = DECISION_INPUT_PATH,
    *,
    expected_sha256: str = DECISION_INPUT_SHA256,
) -> ValidatedDecisionInput:
    input_path = Path(path)
    actual_sha = sha256_file(input_path)
    if actual_sha != expected_sha256:
        raise AssertionError(
            f"Stage 8 decision input SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha}"
        )

    condition_codes = {
        condition.ConditionOrdinal: condition.Code
        for condition in CONDITIONS
        if condition.ConditionOrdinal in INTERFERENCE_CONDITION_ORDINALS
    }
    rows: list[dict[str, object]] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DECISION_INPUT_FIELDS:
            raise AssertionError("Stage 8 decision input schema or field order drifted")
        for raw in reader:
            row: dict[str, object] = {
                name: _parse_nonnegative_integer(raw.get(name), name)
                for name in INTEGER_INPUT_FIELDS
            }
            row["ConditionCode"] = raw.get("ConditionCode", "")
            row["MethodCode"] = raw.get("MethodCode", "")
            condition_ordinal = int(row["ConditionOrdinal"])
            method_ordinal = int(row["MethodOrdinal"])
            if condition_ordinal not in condition_codes:
                raise AssertionError("decision input contains NI or an invalid Condition")
            if row["ConditionCode"] != condition_codes[condition_ordinal]:
                raise AssertionError("Condition ordinal/code identity mismatch")
            if method_ordinal not in METHOD_ORDINALS:
                raise AssertionError("decision input contains a method other than T/TW")
            if row["MethodCode"] != METHOD_CODES[method_ordinal]:
                raise AssertionError("Method ordinal/code identity mismatch")
            if not 1 <= int(row["RepeatID"]) <= R0:
                raise AssertionError("decision input RepeatID must lie in 1..200")
            if int(row["N_C"]) > 200:
                raise AssertionError("N_C cannot exceed the frozen 200 cycles")
            rows.append(row)

    if len(rows) != 18 * 2 * R0:
        raise AssertionError(f"decision input must contain exactly 7200 rows, got {len(rows)}")
    keys = [
        (int(row["ConditionOrdinal"]), int(row["MethodOrdinal"]), int(row["RepeatID"]))
        for row in rows
    ]
    if len(set(keys)) != len(keys):
        raise AssertionError("decision input contains duplicate Condition-Method-Repeat keys")
    expected_keys = {
        (condition, method, repeat_id)
        for condition in INTERFERENCE_CONDITION_ORDINALS
        for method in METHOD_ORDINALS
        for repeat_id in range(1, R0 + 1)
    }
    if set(keys) != expected_keys:
        missing = len(expected_keys - set(keys))
        extra = len(set(keys) - expected_keys)
        raise AssertionError(
            f"decision input key universe is incomplete: missing={missing}, extra={extra}"
        )
    rows.sort(
        key=lambda row: (
            int(row["ConditionOrdinal"]),
            int(row["MethodOrdinal"]),
            int(row["RepeatID"]),
        )
    )
    return ValidatedDecisionInput(
        rows=tuple(rows),
        sha256=actual_sha,
        row_count=len(rows),
        condition_ordinals=INTERFERENCE_CONDITION_ORDINALS,
        method_ordinals=METHOD_ORDINALS,
        repeat_ids=tuple(range(1, R0 + 1)),
    )


def denominator_support_passes(
    metric: str,
    total_denominator: int,
    positive_denominator_repeat_count: int,
) -> bool:
    if metric == "P_cor":
        return True
    if metric not in METRICS:
        raise ValueError(f"unknown Stage 8 metric: {metric}")
    return (
        total_denominator >= MINIMUM_TOTAL_DENOMINATOR
        and positive_denominator_repeat_count
        >= MINIMUM_POSITIVE_DENOMINATOR_REPEATS
    )


def ratio_influence(
    a_values: Sequence[int],
    b_values: Sequence[int],
) -> tuple[float, float, tuple[float, ...], float, float]:
    if len(a_values) != R0 or len(b_values) != R0:
        raise AssertionError("each precision cell must contain exactly 200 Repeat units")
    sum_a = sum(int(value) for value in a_values)
    sum_b = sum(int(value) for value in b_values)
    if sum_b <= 0:
        raise ZeroDivisionError("ratio-of-sums is undefined because sum(B_r)=0")
    theta_hat = sum_a / sum_b
    b_bar = sum_b / R0
    psi = tuple(
        (int(a_value) - theta_hat * int(b_value)) / b_bar
        for a_value, b_value in zip(a_values, b_values, strict=True)
    )
    psi_mean = math.fsum(psi) / R0
    s_psi_sq = math.fsum((value - psi_mean) ** 2 for value in psi) / (R0 - 1)
    return theta_hat, b_bar, psi, psi_mean, s_psi_sq


def _projected_half_width(upper_variance: float, repeat_count: int) -> float:
    return Z975 * math.sqrt(upper_variance / repeat_count)


def _required_r(upper_variance: float, target_half_width: float) -> int:
    return math.ceil(Z975**2 * upper_variance / target_half_width**2)


def compute_individual_precision(
    decision_input: ValidatedDecisionInput,
) -> tuple[list[dict[str, object]], dict[tuple[int, int, str], InfluenceCell]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in decision_input.rows:
        grouped[(int(row["ConditionOrdinal"]), int(row["MethodOrdinal"]))].append(row)

    output_rows: list[dict[str, object]] = []
    influences: dict[tuple[int, int, str], InfluenceCell] = {}
    for condition_ordinal in INTERFERENCE_CONDITION_ORDINALS:
        for method_ordinal in METHOD_ORDINALS:
            cell_rows = sorted(
                grouped[(condition_ordinal, method_ordinal)],
                key=lambda row: int(row["RepeatID"]),
            )
            if len(cell_rows) != R0:
                raise AssertionError("individual cell does not contain 200 Repeat units")
            condition_code = str(cell_rows[0]["ConditionCode"])
            method_code = str(cell_rows[0]["MethodCode"])
            for metric in METRICS:
                numerator_field, denominator_definition, target = METRIC_DEFINITIONS[metric]
                a_values = [int(row[numerator_field]) for row in cell_rows]
                if isinstance(denominator_definition, int):
                    b_values = [denominator_definition] * R0
                    denominator_field = str(denominator_definition)
                else:
                    b_values = [int(row[denominator_definition]) for row in cell_rows]
                    denominator_field = denominator_definition
                sum_a = sum(a_values)
                sum_b = sum(b_values)
                positive_count = sum(value > 0 for value in b_values)
                support_pass = denominator_support_passes(metric, sum_b, positive_count)
                defined = sum_b > 0
                undefined_reason = "" if defined else "ZERO_TOTAL_DENOMINATOR"
                theta_hat: float | None = None
                b_bar: float | None = None
                psi_mean: float | None = None
                s_psi_sq: float | None = None
                upper_variance: float | None = None
                required_r: int | None = None
                projected_1000: float | None = None
                projected_2000: float | None = None
                psi_by_repeat: dict[int, float] = {}
                if defined:
                    theta_hat, b_bar, psi, psi_mean, s_psi_sq = ratio_influence(
                        a_values, b_values
                    )
                    upper_variance = VARIANCE_UCB_MULTIPLIER * s_psi_sq
                    required_r = _required_r(upper_variance, target)
                    projected_1000 = _projected_half_width(upper_variance, 1000)
                    projected_2000 = _projected_half_width(upper_variance, 2000)
                    psi_by_repeat = {
                        int(row["RepeatID"]): value
                        for row, value in zip(cell_rows, psi, strict=True)
                    }
                r1000_pass = bool(
                    defined
                    and support_pass
                    and required_r is not None
                    and required_r <= 1000
                )
                r2000_pass = bool(
                    defined
                    and support_pass
                    and required_r is not None
                    and required_r <= 2000
                )
                output = {
                    "ConditionOrdinal": condition_ordinal,
                    "ConditionCode": condition_code,
                    "MethodOrdinal": method_ordinal,
                    "MethodCode": method_code,
                    "Metric": metric,
                    "NumeratorField": numerator_field,
                    "DenominatorField": denominator_field,
                    "TargetHalfWidth": target,
                    "Defined": defined,
                    "UndefinedReason": undefined_reason,
                    "SumA": sum_a,
                    "SumB": sum_b,
                    "PositiveDenominatorRepeatCount": positive_count,
                    "SupportPass": support_pass,
                    "ThetaHat": theta_hat,
                    "BBar": b_bar,
                    "PsiMean": psi_mean,
                    "SPsiSq": s_psi_sq,
                    "SPsiUpperSq": upper_variance,
                    "RequiredR": required_r,
                    "ProjectedHalfWidthR1000": projected_1000,
                    "ProjectedHalfWidthR2000": projected_2000,
                    "R1000Pass": r1000_pass,
                    "R2000TargetPass": r2000_pass,
                    "ExceedsR2000": bool(required_r is not None and required_r > 2000),
                }
                output_rows.append(output)
                influences[(condition_ordinal, method_ordinal, metric)] = InfluenceCell(
                    output=output,
                    psi_by_repeat=psi_by_repeat,
                )
    if len(output_rows) != 144:
        raise AssertionError(f"expected 144 individual cells, got {len(output_rows)}")
    return output_rows, influences


def compute_paired_precision(
    influences: Mapping[tuple[int, int, str], InfluenceCell],
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    condition_codes = {
        condition.ConditionOrdinal: condition.Code for condition in CONDITIONS
    }
    for condition_ordinal in INTERFERENCE_CONDITION_ORDINALS:
        for metric in METRICS:
            target = METRIC_DEFINITIONS[metric][2]
            t_cell = influences[(condition_ordinal, 2, metric)]
            tw_cell = influences[(condition_ordinal, 4, metric)]
            t_output = t_cell.output
            tw_output = tw_cell.output
            defined = bool(t_output["Defined"] and tw_output["Defined"])
            reasons: list[str] = []
            if not t_output["Defined"]:
                reasons.append("T_UNDEFINED")
            if not tw_output["Defined"]:
                reasons.append("TW_UNDEFINED")
            theta_t = t_output["ThetaHat"] if defined else None
            theta_tw = tw_output["ThetaHat"] if defined else None
            delta_hat = (
                float(theta_tw) - float(theta_t) if defined else None
            )
            s_delta_sq: float | None = None
            upper_variance: float | None = None
            required_r: int | None = None
            projected_1000: float | None = None
            projected_2000: float | None = None
            if defined:
                if set(t_cell.psi_by_repeat) != set(range(1, R0 + 1)):
                    raise AssertionError("T paired RepeatID universe is incomplete")
                if set(tw_cell.psi_by_repeat) != set(range(1, R0 + 1)):
                    raise AssertionError("TW paired RepeatID universe is incomplete")
                psi_delta = tuple(
                    tw_cell.psi_by_repeat[repeat_id] - t_cell.psi_by_repeat[repeat_id]
                    for repeat_id in range(1, R0 + 1)
                )
                delta_mean = math.fsum(psi_delta) / R0
                s_delta_sq = (
                    math.fsum((value - delta_mean) ** 2 for value in psi_delta)
                    / (R0 - 1)
                )
                upper_variance = VARIANCE_UCB_MULTIPLIER * s_delta_sq
                required_r = _required_r(upper_variance, target)
                projected_1000 = _projected_half_width(upper_variance, 1000)
                projected_2000 = _projected_half_width(upper_variance, 2000)
            support_pass = bool(t_output["SupportPass"] and tw_output["SupportPass"])
            output_rows.append(
                {
                    "ConditionOrdinal": condition_ordinal,
                    "ConditionCode": condition_codes[condition_ordinal],
                    "Metric": metric,
                    "TargetHalfWidth": target,
                    "Defined": defined,
                    "UndefinedReason": ";".join(reasons),
                    "ThetaT": theta_t,
                    "ThetaTW": theta_tw,
                    "DeltaHat": delta_hat,
                    "SPsiDeltaSq": s_delta_sq,
                    "SPsiDeltaUpperSq": upper_variance,
                    "RequiredR": required_r,
                    "ProjectedHalfWidthR1000": projected_1000,
                    "ProjectedHalfWidthR2000": projected_2000,
                    "R1000Pass": bool(
                        defined
                        and support_pass
                        and required_r is not None
                        and required_r <= 1000
                    ),
                    "R2000TargetPass": bool(
                        defined
                        and support_pass
                        and required_r is not None
                        and required_r <= 2000
                    ),
                    "ExceedsR2000": bool(
                        required_r is not None and required_r > 2000
                    ),
                }
            )
    if len(output_rows) != 72:
        raise AssertionError(f"expected 72 paired cells, got {len(output_rows)}")
    return output_rows


def build_r_decision(
    *,
    stage7_status: Mapping[str, object],
    decision_input: ValidatedDecisionInput,
    contract: Mapping[str, object],
    contract_sha256: str,
    individual_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    nonfixed = [row for row in individual_rows if row["Metric"] != "P_cor"]
    all_defined = all(bool(row["Defined"]) for row in individual_rows) and all(
        bool(row["Defined"]) for row in paired_rows
    )
    checks = {
        "Stage7PilotDataComplete": bool(
            stage7_status.get("status") == "PASS"
            and stage7_status.get("pilot_executed") is True
            and stage7_status.get("pilot_r") == R0
            and decision_input.row_count == 7200
        ),
        "R0Equals200": bool(contract.get("Rpre") == R0),
        "Conditions1Through18Complete": decision_input.condition_ordinals
        == INTERFERENCE_CONDITION_ORDINALS,
        "MethodsTAndTWComplete": decision_input.method_ordinals == METHOD_ORDINALS,
        "RepeatIDs1Through200UniqueAndComplete": decision_input.repeat_ids
        == tuple(range(1, R0 + 1)),
        "AllFourMetricsDefined": all_defined,
        "AllNonfixedTotalDenominatorsAtLeast100": all(
            int(row["SumB"]) >= MINIMUM_TOTAL_DENOMINATOR for row in nonfixed
        ),
        "AllNonfixedPositiveDenominatorRepeatCountsAtLeast30": all(
            int(row["PositiveDenominatorRepeatCount"])
            >= MINIMUM_POSITIVE_DENOMINATOR_REPEATS
            for row in nonfixed
        ),
        "All144IndividualRequiredRValuesAtMost1000": len(individual_rows) == 144
        and all(
            row["RequiredR"] is not None and int(row["RequiredR"]) <= 1000
            for row in individual_rows
        ),
        "All72PairedRequiredRValuesAtMost1000": len(paired_rows) == 72
        and all(
            row["RequiredR"] is not None and int(row["RequiredR"]) <= 1000
            for row in paired_rows
        ),
    }
    if tuple(checks) != tuple(contract["R1000Rule"]["RequiredChecks"]):
        raise AssertionError("implemented R1000 check order drifted from frozen contract")
    eligible = all(checks.values())
    formal_r = 1000 if eligible else 2000
    individual_fail = sum(not bool(row["R1000Pass"]) for row in individual_rows)
    paired_fail = sum(not bool(row["R1000Pass"]) for row in paired_rows)
    individual_exceed = sum(bool(row["ExceedsR2000"]) for row in individual_rows)
    paired_exceed = sum(bool(row["ExceedsR2000"]) for row in paired_rows)
    max_individual = max(int(row["RequiredR"]) for row in individual_rows)
    max_paired = max(int(row["RequiredR"]) for row in paired_rows)
    warning = bool(individual_exceed or paired_exceed)
    return {
        "StageNumber": STAGE_NUMBER,
        "StageName": STAGE_NAME,
        "DecisionContractSHA256": contract_sha256,
        "DecisionInputSHA256": decision_input.sha256,
        "Rpre": R0,
        "IndependentUnit": "Repeat",
        "IndividualPrecisionCellCount": len(individual_rows),
        "PairedPrecisionCellCount": len(paired_rows),
        "TotalPrecisionCellCount": len(individual_rows) + len(paired_rows),
        "R1000Checks": checks,
        "R1000Eligible": eligible,
        "Rformal": formal_r,
        "FormalRSource": "STAGE8_FROZEN_ALL_RULE",
        "IndividualFailR1000Count": individual_fail,
        "PairedFailR1000Count": paired_fail,
        "IndividualExceedR2000Count": individual_exceed,
        "PairedExceedR2000Count": paired_exceed,
        "MaxIndividualRequiredR": max_individual,
        "MaxPairedRequiredR": max_paired,
        "ProjectedPrecisionWarning": warning,
        "ManualOverrideApplied": False,
        "OutcomeDirectionUsed": False,
        "ConditionDropped": False,
        "ParameterChanged": False,
        "FormalUsesSingleRForAllConditionsAndMethods": True,
        "FormalExecuted": False,
        "BootstrapExecuted": False,
        "PaperResultGenerated": False,
    }


def build_precision_warning(
    individual_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    individual_cells = [
        {
            "ConditionOrdinal": row["ConditionOrdinal"],
            "ConditionCode": row["ConditionCode"],
            "MethodOrdinal": row["MethodOrdinal"],
            "MethodCode": row["MethodCode"],
            "Metric": row["Metric"],
            "RequiredR": row["RequiredR"],
            "ProjectedHalfWidthR2000": row["ProjectedHalfWidthR2000"],
        }
        for row in individual_rows
        if bool(row["ExceedsR2000"])
    ]
    paired_cells = [
        {
            "ConditionOrdinal": row["ConditionOrdinal"],
            "ConditionCode": row["ConditionCode"],
            "MethodPair": "T-TW",
            "Metric": row["Metric"],
            "RequiredR": row["RequiredR"],
            "ProjectedHalfWidthR2000": row["ProjectedHalfWidthR2000"],
        }
        for row in paired_rows
        if bool(row["ExceedsR2000"])
    ]
    return {
        "ProjectedPrecisionWarning": bool(individual_cells or paired_cells),
        "IndividualExceedR2000Count": len(individual_cells),
        "PairedExceedR2000Count": len(paired_cells),
        "IndividualCells": individual_cells,
        "PairedCells": paired_cells,
    }


def validate_frozen_oracle(
    decision: Mapping[str, object],
    individual_rows: Sequence[Mapping[str, object]],
    paired_rows: Sequence[Mapping[str, object]],
) -> None:
    actual = {
        "IndividualPrecisionCellCount": decision["IndividualPrecisionCellCount"],
        "PairedPrecisionCellCount": decision["PairedPrecisionCellCount"],
        "TotalPrecisionCellCount": decision["TotalPrecisionCellCount"],
        "IndividualFailR1000Count": decision["IndividualFailR1000Count"],
        "PairedFailR1000Count": decision["PairedFailR1000Count"],
        "IndividualExceedR2000Count": decision["IndividualExceedR2000Count"],
        "PairedExceedR2000Count": decision["PairedExceedR2000Count"],
        "MaxIndividualRequiredR": decision["MaxIndividualRequiredR"],
        "MaxPairedRequiredR": decision["MaxPairedRequiredR"],
        "R1000Eligible": decision["R1000Eligible"],
        "FormalR": decision["Rformal"],
        "ProjectedPrecisionWarning": decision["ProjectedPrecisionWarning"],
    }
    if actual != ORACLE:
        raise AssertionError(f"Stage 8 independent oracle mismatch: {actual!r}")
    critical_individual = next(
        row
        for row in individual_rows
        if row["ConditionCode"] == "H100"
        and row["MethodCode"] == "TW"
        and row["Metric"] == "P_C_given_E"
    )
    if critical_individual["RequiredR"] != 41967:
        raise AssertionError("H100/TW/P_C_given_E RequiredR oracle mismatch")
    critical_paired = next(
        row
        for row in paired_rows
        if row["ConditionCode"] == "H100" and row["Metric"] == "P_C_given_E"
    )
    if critical_paired["RequiredR"] != 42840:
        raise AssertionError("H100 paired P_C_given_E RequiredR oracle mismatch")


def _excluded_path(path: Path) -> bool:
    if any(
        part in {"__pycache__", ".pytest_cache", ".git"}
        or part == ".venv"
        or part.startswith(".venv-")
        for part in path.parts
    ):
        return True
    return (
        path.suffix.lower() == ".pyc"
        or path.name.endswith(".tmp")
        or ".tmp-" in path.name
    )


def _baseline_archive_members(path: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                continue
            relative = Path(*parts[1:])
            if _excluded_path(relative):
                continue
            members[relative.as_posix()] = hashlib.sha256(archive.read(info)).hexdigest()
    return members


ALLOWED_ADDED_FILES = {
    "STAGE8_TASKBOOK.md",
    "src/sal_stability_stage1/stage8.py",
    "scripts/build_stage8.py",
    "tests/test_stage8_decision.py",
    "artifacts/stage8/stage8_individual_precision.csv",
    "artifacts/stage8/stage8_paired_precision.csv",
    "artifacts/stage8/stage8_r_decision.json",
    "artifacts/stage8/stage8_precision_warning.json",
    "artifacts/stage8/test_report.txt",
    "artifacts/stage8/changed_files.txt",
    "artifacts/stage8/validation_report.md",
    "artifacts/stage8/status.json",
}
ALLOWED_MODIFIED_FILES = {"README.md", "pyproject.toml"}


def validate_stage1_through_stage7_tree_unchanged(
    baseline_zip: Path | None = None,
) -> dict[str, list[str]]:
    baseline_path = (
        Path(baseline_zip)
        if baseline_zip is not None
        else ROOT.parent / BASELINE_STAGE7_ZIP_NAME
    )
    verify_stage7_baseline_zip(baseline_path)
    baseline = _baseline_archive_members(baseline_path)
    current: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if _excluded_path(relative):
            continue
        current[relative.as_posix()] = sha256_file(path)
    added = sorted(set(current) - set(baseline))
    modified = sorted(
        relative
        for relative in set(current).intersection(baseline)
        if current[relative] != baseline[relative]
    )
    deleted = sorted(set(baseline) - set(current))
    unexpected_added = sorted(set(added) - ALLOWED_ADDED_FILES)
    unexpected_modified = sorted(set(modified) - ALLOWED_MODIFIED_FILES)
    if unexpected_added or unexpected_modified or deleted:
        raise AssertionError(
            "Stage 1-7 byte-identity boundary failed: "
            f"unexpected_added={unexpected_added}, "
            f"unexpected_modified={unexpected_modified}, deleted={deleted}"
        )
    return {"added": added, "modified": modified, "deleted": deleted}


def _run_full_pytest() -> dict[str, object]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = (completed.stdout or "") + (completed.stderr or "")
    matches = re.findall(r"([0-9]+) passed", output)
    passed_count = int(matches[-1]) if matches else 0
    if completed.returncode != 0 or passed_count <= 0 or re.search(r"[0-9]+ failed", output):
        raise AssertionError(f"full pytest failed\n{output}")
    return {
        "command": f"{sys.executable} -m pytest",
        "returncode": completed.returncode,
        "passed": passed_count,
        "failed": 0,
        "wall_seconds": elapsed,
        "output": output,
    }


def _stage8_status(decision: Mapping[str, object]) -> dict[str, object]:
    checks = decision["R1000Checks"]
    return {
        "stage": STAGE_NUMBER,
        "stage_name": STAGE_NAME,
        "status": "PASS",
        "baseline_stage7_zip_sha256_verified": True,
        "stage7_status_passed": True,
        "science_contract_changed": False,
        "frozen_hashes_passed": True,
        "stage8_contract_sha256_verified": True,
        "stage8_decision_input_sha256_verified": True,
        "stage8_decision_input_row_count": 7200,
        "individual_precision_cell_count": decision["IndividualPrecisionCellCount"],
        "paired_precision_cell_count": decision["PairedPrecisionCellCount"],
        "total_precision_cell_count": decision["TotalPrecisionCellCount"],
        "individual_fail_r1000_count": decision["IndividualFailR1000Count"],
        "paired_fail_r1000_count": decision["PairedFailR1000Count"],
        "individual_exceed_r2000_count": decision["IndividualExceedR2000Count"],
        "paired_exceed_r2000_count": decision["PairedExceedR2000Count"],
        "max_individual_required_r": decision["MaxIndividualRequiredR"],
        "max_paired_required_r": decision["MaxPairedRequiredR"],
        "all_metrics_defined": checks["AllFourMetricsDefined"],
        "all_denominator_support_passed": bool(
            checks["AllNonfixedTotalDenominatorsAtLeast100"]
            and checks["AllNonfixedPositiveDenominatorRepeatCountsAtLeast30"]
        ),
        "r1000_eligible": decision["R1000Eligible"],
        "r_decision_executed": True,
        "formal_r": decision["Rformal"],
        "projected_precision_warning": decision["ProjectedPrecisionWarning"],
        "manual_override_applied": False,
        "outcome_direction_used": False,
        "condition_dropped": False,
        "parameter_changed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "blocking_issue_count": 0,
        "stop_after_stage8": True,
    }


def _validation_report(decision: Mapping[str, object], test_count: int) -> str:
    return f"""# Stage 8 Validation Report

## Identity gates

- Unique Stage 7 baseline ZIP SHA256: PASS.
- Stage 7 PASS status and STOP boundary: PASS.
- All 19 frozen science/configuration SHA256 values: PASS at Stage 8 start and end.
- Full Stage 1-7 tree changed only at the explicitly allowed README/pyproject boundary.
- Stage 8 decision contract and decision input SHA256 values: PASS.

## Decision input

- Rows: 7200 (18 Conditions x 2 Methods x 200 Repeats).
- Unique key universe: Condition 1..18, Method T/TW, RepeatID 1..200.
- Independent statistical unit: Repeat.
- Estimator: ratio of sums; P_C_given_E uses N_EC/N_Edot.

## Precision cells and frozen oracle

- Individual precision cells: {decision['IndividualPrecisionCellCount']}.
- T-TW paired precision cells: {decision['PairedPrecisionCellCount']}.
- Individual RequiredR > 1000: {decision['IndividualFailR1000Count']}.
- Paired RequiredR > 1000: {decision['PairedFailR1000Count']}.
- Individual RequiredR > 2000: {decision['IndividualExceedR2000Count']}.
- Paired RequiredR > 2000: {decision['PairedExceedR2000Count']}.
- Maximum individual RequiredR: {decision['MaxIndividualRequiredR']}.
- Maximum paired RequiredR: {decision['MaxPairedRequiredR']}.
- H100/TW/P_C_given_E and H100 paired/P_C_given_E critical oracles: PASS.

## Automatic decision and prohibited actions

- Frozen ALL-rule R1000 eligibility: {str(decision['R1000Eligible']).lower()}.
- Formal R selected for later stages: {decision['Rformal']}.
- ProjectedPrecisionWarning: {str(decision['ProjectedPrecisionWarning']).lower()}.
- Outcome direction, manual override, Condition dropping, and parameter changes: none.
- FORMAL execution, Bootstrap, confidence intervals, p-values, and paper results: not run.
- Full pytest: {test_count} passed, 0 failed.
- Stop after Stage 8: yes.
"""


def _changed_files_report(
    diff: Mapping[str, Sequence[str]],
) -> str:
    added = sorted(set(diff["added"]) | {"artifacts/stage8/changed_files.txt"})
    modified = list(diff["modified"])

    def reason(relative: str) -> str:
        if relative.startswith("artifacts/stage8/"):
            return "Stage 8 automatic decision artifact"
        if relative.startswith("tests/test_stage8_"):
            return "Stage 8 decision and boundary regression coverage"
        if relative == "src/sal_stability_stage1/stage8.py":
            return "Stage 8 isolated automatic-decision implementation"
        if relative == "scripts/build_stage8.py":
            return "Stage 8 fixed-boundary build entry point"
        if relative == "STAGE8_TASKBOOK.md":
            return "Frozen Stage 8 execution taskbook"
        if relative == "pyproject.toml":
            return "Project version and Stage 8 description only"
        if relative == "README.md":
            return "Stage 8 status and STOP boundary documentation"
        return "Stage 8 required project artifact"

    lines = ["STAGE 8 CHANGED FILES", "", "ADDED FILES:"]
    lines.extend(f"- {relative} | {reason(relative)}" for relative in added)
    lines.extend(("", "MODIFIED FILES:"))
    lines.extend(f"- {relative} | {reason(relative)}" for relative in modified)
    lines.extend(("", "FROZEN SCIENCE FILES:"))
    lines.extend(
        f"- {relative} | UNCHANGED | {expected}"
        for relative, expected in FROZEN_SCIENCE_HASHES.items()
    )
    lines.extend(("", "DELETED FILES:", "- NONE", ""))
    return "\n".join(lines)


def _package_members() -> list[Path]:
    members: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not _excluded_path(path.relative_to(ROOT)):
            members.append(path)
    return sorted(members, key=lambda path: path.relative_to(ROOT).as_posix())


def package_stage8(output_path: Path | None = None) -> tuple[Path, str]:
    destination = (
        Path(output_path) if output_path is not None else ROOT.parent / FINAL_ZIP_NAME
    )
    destination = destination.resolve()
    expected_parent = ROOT.parent.resolve()
    if destination.parent != expected_parent:
        raise ValueError("Stage 8 package destination must be the workspace parent")
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in _package_members():
                relative = path.relative_to(ROOT)
                archive.write(path, (Path(ROOT.name) / relative).as_posix())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    with zipfile.ZipFile(destination) as archive:
        file_names = [name for name in archive.namelist() if not name.endswith("/")]
        relative_names = {
            Path(*Path(name).parts[1:]).as_posix()
            for name in file_names
            if len(Path(name).parts) >= 2
        }
        required = {
            "STAGE8_TASKBOOK.md",
            "README.md",
            "pyproject.toml",
            "src/sal_stability_stage1/stage8.py",
            "scripts/build_stage8.py",
            "tests/test_stage8_decision.py",
            *{
                f"artifacts/stage8/{name}"
                for name in (
                    "stage8_individual_precision.csv",
                    "stage8_paired_precision.csv",
                    "stage8_r_decision.json",
                    "stage8_precision_warning.json",
                    "test_report.txt",
                    "changed_files.txt",
                    "validation_report.md",
                    "status.json",
                )
            },
        }
        missing = sorted(required - relative_names)
        forbidden = [
            name
            for name in file_names
            if _excluded_path(Path(name))
            or name.lower().endswith((".pyc", ".tmp"))
            or Path(name).name in {
                "sal_stability_stage6_20260823.zip",
                BASELINE_STAGE7_ZIP_NAME,
            }
        ]
        if missing or forbidden:
            raise AssertionError(
                f"final ZIP validation failed: missing={missing}, forbidden={forbidden}"
            )
    return destination, sha256_file(destination)


def build_stage8(progress: Callable[[str], None] = print) -> dict[str, object]:
    baseline_zip = ROOT.parent / BASELINE_STAGE7_ZIP_NAME
    progress("[Stage8 STEP 1] Verify Stage 7 baseline ZIP and allowed file boundary")
    baseline_sha = verify_stage7_baseline_zip(baseline_zip)
    initial_diff = validate_stage1_through_stage7_tree_unchanged(baseline_zip)

    progress("[Stage8 STEP 2] Verify frozen science hashes and Stage 1-7 PASS gates")
    frozen_start = validate_frozen_science_hashes()
    validate_stage1_through_stage6_statuses()
    stage7_status = validate_stage7_status()

    progress("[Stage8 STEP 3] Verify frozen decision contract and 7200-row input")
    contract, contract_sha = load_and_validate_decision_contract()
    decision_input = load_and_validate_decision_input()

    progress("[Stage8 STEP 4] Compute 144 individual repeat-level precision cells")
    individual_rows, influences = compute_individual_precision(decision_input)
    progress("[Stage8 STEP 5] Compute 72 strictly paired T-TW precision cells")
    paired_rows = compute_paired_precision(influences)

    progress("[Stage8 STEP 6] Apply frozen ALL-rule and verify independent oracle")
    decision = build_r_decision(
        stage7_status=stage7_status,
        decision_input=decision_input,
        contract=contract,
        contract_sha256=contract_sha,
        individual_rows=individual_rows,
        paired_rows=paired_rows,
    )
    warning = build_precision_warning(individual_rows, paired_rows)
    validate_frozen_oracle(decision, individual_rows, paired_rows)
    if warning["ProjectedPrecisionWarning"] != decision["ProjectedPrecisionWarning"]:
        raise AssertionError("warning artifact disagrees with R decision")

    progress("[Stage8 STEP 7] Run complete pytest regression suite")
    pytest_result = _run_full_pytest()

    progress("[Stage8 STEP 8] Write Stage 8 decision artifacts and PASS status")
    STAGE8_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_dicts_atomic(
        STAGE8_DIR / "stage8_individual_precision.csv",
        field_names=INDIVIDUAL_PRECISION_FIELDS,
        rows=individual_rows,
    )
    write_csv_dicts_atomic(
        STAGE8_DIR / "stage8_paired_precision.csv",
        field_names=PAIRED_PRECISION_FIELDS,
        rows=paired_rows,
    )
    write_json_atomic(STAGE8_DIR / "stage8_r_decision.json", decision)
    write_json_atomic(STAGE8_DIR / "stage8_precision_warning.json", warning)
    test_report = (
        f"Command: {pytest_result['command']}\n"
        f"Return code: {pytest_result['returncode']}\n"
        f"Result: {pytest_result['passed']} passed, 0 failed\n"
        f"Wall seconds: {pytest_result['wall_seconds']:.6f}\n\n"
        f"{pytest_result['output']}"
    )
    write_text_atomic(STAGE8_DIR / "test_report.txt", test_report)
    write_text_atomic(
        STAGE8_DIR / "validation_report.md",
        _validation_report(decision, int(pytest_result["passed"])),
    )
    status = _stage8_status(decision)
    write_json_atomic(STAGE8_DIR / "status.json", status)
    final_diff = validate_stage1_through_stage7_tree_unchanged(baseline_zip)
    write_text_atomic(STAGE8_DIR / "changed_files.txt", _changed_files_report(final_diff))

    progress("[Stage8 STEP 9] Re-verify frozen hashes and package complete Stage 1-8 tree")
    frozen_end = validate_frozen_science_hashes()
    if frozen_end != frozen_start:
        raise AssertionError("frozen science hashes changed during Stage 8")
    validate_stage1_through_stage7_tree_unchanged(baseline_zip)
    zip_path, zip_sha = package_stage8()

    progress("[Stage8 STEP 10] STOP after Stage 8")
    return {
        "Stage8Status": status["status"],
        "BaselineStage7ZipSHA256": baseline_sha,
        "FrozenScienceHashesPassed": True,
        "DecisionContractSHA256": contract_sha,
        "DecisionInputSHA256": decision_input.sha256,
        "DecisionInputRows": decision_input.row_count,
        "IndividualPrecisionCells": decision["IndividualPrecisionCellCount"],
        "PairedPrecisionCells": decision["PairedPrecisionCellCount"],
        "IndividualFailR1000": decision["IndividualFailR1000Count"],
        "PairedFailR1000": decision["PairedFailR1000Count"],
        "IndividualExceedR2000": decision["IndividualExceedR2000Count"],
        "PairedExceedR2000": decision["PairedExceedR2000Count"],
        "MaxIndividualRequiredR": decision["MaxIndividualRequiredR"],
        "MaxPairedRequiredR": decision["MaxPairedRequiredR"],
        "R1000Eligible": decision["R1000Eligible"],
        "FormalR": decision["Rformal"],
        "ProjectedPrecisionWarning": decision["ProjectedPrecisionWarning"],
        "OutcomeDirectionUsed": False,
        "ManualOverrideApplied": False,
        "FormalExecuted": False,
        "BootstrapExecuted": False,
        "ScienceContractChanged": False,
        "FullPytestPassed": pytest_result["passed"],
        "BlockingIssues": 0,
        "InitialAllowedDiff": initial_diff,
        "FinalZip": str(zip_path),
        "FinalZipSHA256": zip_sha,
        "StopAfterStage8": True,
    }


__all__ = [
    "BASELINE_STAGE7_ZIP_NAME",
    "BASELINE_STAGE7_ZIP_SHA256",
    "DECISION_CONTRACT_SHA256",
    "DECISION_INPUT_SHA256",
    "INDIVIDUAL_PRECISION_FIELDS",
    "METRICS",
    "ORACLE",
    "PAIRED_PRECISION_FIELDS",
    "STAGE_NAME",
    "STAGE_NUMBER",
    "ValidatedDecisionInput",
    "build_precision_warning",
    "build_r_decision",
    "build_stage8",
    "compute_individual_precision",
    "compute_paired_precision",
    "denominator_support_passes",
    "load_and_validate_decision_contract",
    "load_and_validate_decision_input",
    "package_stage8",
    "ratio_influence",
    "validate_frozen_oracle",
    "validate_stage1_through_stage7_tree_unchanged",
    "validate_stage7_status",
    "verify_stage7_baseline_zip",
]
