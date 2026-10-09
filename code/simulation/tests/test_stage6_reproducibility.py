from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from sal_stability_stage1.execution import (
    ExecutionOptions,
    ORDER_A,
    ORDER_B,
    ORDER_C,
    assert_scientific_results_equal,
    run_condition_repeats_serial,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import load_stage3_inputs
from sal_stability_stage1.stage6 import (
    FROZEN_METRICS_SHA256,
    FROZEN_RNG_SHA256,
    FROZEN_RNG_VECTORS_SHA256,
    FROZEN_SELECTORS_SHA256,
    FROZEN_STUDY_CONFIG_SHA256,
    ORACLE_MANIFEST_PATH,
    validate_frozen_contract,
    validate_source_permutation,
)


@pytest.fixture(scope="module")
def frozen_inputs():
    return load_stage3_inputs()


def test_source_metadata_permutation_does_not_change_any_selector_position():
    assert validate_source_permutation()


def test_method_order_a_b_c_is_exact_for_real_condition_repeat(frozen_inputs):
    config, reference = frozen_inputs
    base = ExecutionOptions()
    results = []
    for order in (ORDER_A, ORDER_B, ORDER_C):
        results.append(
            run_condition_repeats_serial(
                namespace=RandomNamespace.PILOT,
                condition_ordinal=7,
                repeat_ids=(10001,),
                study_config=config,
                encoding_reference=reference,
                execution_options=replace(base, method_execution_order=order),
            )
        )
    assert_scientific_results_equal(results[0], results[1], label="ORDER_A/B")
    assert_scientific_results_equal(results[0], results[2], label="ORDER_A/C")


def test_h_cache_on_off_is_exact_for_real_h300_repeat(frozen_inputs):
    config, reference = frozen_inputs
    cache_on = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(h_cache_enabled=True),
    )
    cache_off = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_ids=(10001,),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(h_cache_enabled=False),
    )
    assert_scientific_results_equal(cache_on, cache_off, label="cache ON/OFF")


def test_ni_identity_includes_source_reference_and_metrics_except_method_identity(
    frozen_inputs,
):
    config, reference = frozen_inputs
    result = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_ids=(10001, 10002),
        study_config=config,
        encoding_reference=reference,
        execution_options=ExecutionOptions(),
    )
    for repeat_position in range(2):
        for method_ordinal in range(1, 5):
            np.testing.assert_array_equal(
                result.source_state[0, repeat_position],
                result.source_state[method_ordinal, repeat_position],
            )
            np.testing.assert_array_equal(
                result.ref_toa_before_s[0, repeat_position],
                result.ref_toa_before_s[method_ordinal, repeat_position],
            )
    for repeat_id in result.repeat_ids:
        payloads = []
        for row in result.repeat_metrics:
            if row.RepeatID != repeat_id:
                continue
            payload = asdict(row)
            payload.pop("MethodOrdinal")
            payload.pop("MethodCode")
            payloads.append(payload)
        assert all(payload == payloads[0] for payload in payloads[1:])


def test_frozen_contract_hashes_and_stage_artifacts_remain_passed():
    validation = validate_frozen_contract()
    assert validation["StudyConfigSHA256"] == FROZEN_STUDY_CONFIG_SHA256
    assert validation["RNGSHA256"] == FROZEN_RNG_SHA256
    assert validation["RNGVectorsSHA256"] == FROZEN_RNG_VECTORS_SHA256
    assert validation["SelectorsSHA256"] == FROZEN_SELECTORS_SHA256
    assert validation["MetricsSHA256"] == FROZEN_METRICS_SHA256
    assert all(validation[f"Stage{stage}ArtifactPassed"] for stage in range(1, 6))


def test_oracle_is_locked_verify_metadata_with_all_recorded_file_hashes():
    manifest = json.loads(ORACLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["GeneratedBeforeStage6KernelRefactor"] is True
    assert manifest["OracleMode"] == "VERIFY"
    assert manifest["CreatedAsValidationOnly"] is True
    assert manifest["PilotR"] is None
    assert manifest["RepeatIDs"] == [10001, 10002, 10003]
    assert len(manifest["Files"]) == 21
