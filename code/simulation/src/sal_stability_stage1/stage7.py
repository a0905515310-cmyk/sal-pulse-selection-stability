from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np

from .build import CONFIG_DIR, ROOT
from .contracts import CONDITIONS, METHODS
from .execution import (
    ORDER_A,
    ConditionRunResult,
    ExecutionOptions,
    assert_scientific_results_equal,
    run_condition_repeats_parallel,
    run_condition_repeats_serial,
)
from .metrics import (
    AggregatedMetrics,
    MetricValue,
    RepeatMetrics,
    aggregate_repeat_metrics,
)
from .pilot_storage import (
    CODE_VERSION,
    PILOT_R,
    RUN_KIND,
    PilotChunkPaths,
    assemble_pilot_condition_chunks,
    compute_stage7_code_sha256,
    load_validated_pilot_chunk,
    partition_repeat_ids,
    pilot_chunk_paths,
    sha256_file,
    validate_pilot_chunk,
    write_csv_dicts_atomic,
    write_json_atomic,
    write_pilot_chunk_atomic,
    write_text_atomic,
)
from .rng import RandomNamespace
from .stage3 import load_stage3_inputs
from .stage5 import (
    H300WidthStratumMetrics,
    WIDTH_STRATA,
)
from .stage6 import run_condition_repeats_chunked as run_validation_chunked


STAGE_NUMBER = 7
STAGE_NAME = "PILOT_R200"
STAGE7_DIR = ROOT / "artifacts" / "stage7"
PREFLIGHT_DIR = STAGE7_DIR / "preflight"
PILOT_CHUNKS_DIR = STAGE7_DIR / "pilot_chunks"

BASELINE_ZIP_NAME = "sal_stability_stage6_20260823.zip"
BASELINE_ZIP_SHA256 = (
    "352285452827bb51f28f946dd737c88af4019abd41b9aead4097dc743931f895"
)

PILOT_REPEAT_IDS = tuple(range(1, PILOT_R + 1))
PREFLIGHT_CONDITIONS = ((3, "H300"), (7, "F2"), (14, "HF300-2"))
PREFLIGHT_REPEAT_IDS = (10001, 10002, 10003, 10004, 10005)
PILOT_WORKERS = 2
PILOT_CHUNK_SIZE = 20
EXPECTED_CHUNK_COUNT = 190
EXPECTED_METHOD_CYCLES = 3_800_000
HPRF_CONDITION_ORDINALS = frozenset(range(1, 6))
COMPOSITE_CONDITION_ORDINALS = frozenset(range(10, 19))

STAGE8_CONTRACT_PATH = STAGE7_DIR / "stage8_r_decision_contract.json"
PILOT_REPEAT_METRICS_PATH = STAGE7_DIR / "pilot_repeat_metrics.csv"
PILOT_AGGREGATED_METRICS_PATH = STAGE7_DIR / "pilot_aggregated_metrics.csv"
PILOT_H300_STRATA_PATH = STAGE7_DIR / "pilot_h300_width_strata.csv"
STAGE8_DECISION_INPUT_PATH = STAGE7_DIR / "stage8_decision_input.csv"

FROZEN_SCIENCE_HASHES = {
    "src/sal_stability_stage1/contracts.py": (
        "029a063dddae55ca66cf875528ab8db99343253ded7237026f6b2da242a9f594"
    ),
    "src/sal_stability_stage1/encoding.py": (
        "197eba52f68d2f91d9051f0ca70f83eda710595e98a949650232c882ea65a71f"
    ),
    "src/sal_stability_stage1/rng.py": (
        "36bedc40407912a68f34a2ccac090549e8b61e3d2c59f255904d34ac3fe1b70b"
    ),
    "src/sal_stability_stage1/events.py": (
        "ce1c7a54cd3d2a2219a782dfe0720118bf8d4f19cbabec765bd03916bee1cab2"
    ),
    "src/sal_stability_stage1/hprf.py": (
        "be09cf938230c24c3b4dfa480a1a5ca63d24e15924bd76451b16ee050d6dd883"
    ),
    "src/sal_stability_stage1/selectors.py": (
        "bc285cb5163dc1c0631a860f312a45c61738d6502e85e9c9779e5b9ddafd5fc6"
    ),
    "src/sal_stability_stage1/metrics.py": (
        "a4afd0f90c9d0d3dca08ac9b21310aaef5ac81c9aa669f449e013661a30b7a4e"
    ),
    "src/sal_stability_stage1/kernel.py": (
        "80bec3cde701c98491950dc40e6fd06def09c676a20c2ec28d1b884290be0e95"
    ),
    "src/sal_stability_stage1/diagnostics.py": (
        "fc44d12ee9029373f7c5c6a83bd9ea5303a69f36f15c433a98b0bb51d32f5a40"
    ),
    "src/sal_stability_stage1/execution.py": (
        "aa697603db3d4e539fafccd9fae3f46119a41a67adb75c729adfc2721e0dc7be"
    ),
    "src/sal_stability_stage1/storage.py": (
        "05770a8a751ce1ad6481efded78d72c3c4bf4de6e3033ccd30c57b212c4bd784"
    ),
    "src/sal_stability_stage1/stage3.py": (
        "59fbbf2cb6deedd2f42c2aad419c55b6df9f74231c9acb80a417377b5f516830"
    ),
    "src/sal_stability_stage1/stage4.py": (
        "a2c25cba787a657c5342773c7ddab0f907ded61a215c53c96222d90da9e3ffd9"
    ),
    "src/sal_stability_stage1/stage5.py": (
        "b76989512ba38661407a049b64ab18cf2af546ba977713f748d19f7a24baf4ac"
    ),
    "src/sal_stability_stage1/stage6.py": (
        "f404b44fa5f07de810a38a3271df09d0c86e50678baa39f5ef3adf38be0a81f1"
    ),
    "artifacts/config/study_config.json": (
        "4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb"
    ),
    "artifacts/config/encoding_reference.csv": (
        "aa319c482a3494d0cdece3bfa4aa28f4d919edaa5b0d2c45bb0d4918ee9647f6"
    ),
    "artifacts/config/encoding_reference.npz": (
        "5a058296f762477009c7d94a33d6d61a982c2d1dda7233db3f5949214ee3cd43"
    ),
    "artifacts/config/rng_test_vectors.csv": (
        "6f88198eca9ceaf882e0458d87b9986435316715fa91fd09617e0a3878db2274"
    ),
}

STAGE8_METRICS = ("P_cor", "P_C_given_C", "P_C_given_E", "Mean_L_NC")
AGGREGATED_METRIC_NAMES = (
    "P_cor",
    "P_C_given_C",
    "P_C_given_E",
    "Mean_L_NC",
    "P_E_given_C",
    "P_N_given_C",
    "P_E_given_E",
    "P_N_given_E",
    "P_N",
    "P_open_NC",
    "P_NC_end",
    "P_E_H",
    "P_E_F",
)
STAGE8_INPUT_FIELDS = (
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


def stage8_r_decision_contract() -> dict[str, object]:
    return {
        "ContractName": "STAGE8_FORMAL_R_AUTOMATIC_DECISION",
        "ContractVersion": 1.0,
        "LockedBeforePilot": True,
        "Rpre": 200,
        "InterferenceConditionOrdinals": list(range(1, 19)),
        "MethodOrdinals": [2, 4],
        "MethodCodes": ["T", "TW"],
        "Metrics": list(STAGE8_METRICS),
        "RatioDefinitions": {
            "P_cor": {"A_r": "N_C", "B_r": 200},
            "P_C_given_C": {"A_r": "N_CC", "B_r": "N_Cdot"},
            "P_C_given_E": {"A_r": "N_EC", "B_r": "N_Edot"},
            "Mean_L_NC": {"A_r": "Sum_L_NC_obs", "B_r": "N_run_NC"},
            "Estimator": "sum(A_r) / sum(B_r)",
            "ForbiddenEstimator": "mean(A_r / B_r)",
        },
        "IndependentUnit": "Repeat",
        "InfluenceQuantity": "psi_r=(A_r-theta_hat*B_r)/B_bar",
        "VarianceEstimator": "sum((psi_r-mean(psi))^2)/(R0-1)",
        "VarianceUCBMultiplier": 1.1890464657347386,
        "Z975": 1.96,
        "ProjectedHalfWidth": "Z975*sqrt(s_variance_upper_sq/R)",
        "ProbabilityHalfWidthTarget": 0.020,
        "MeanLncHalfWidthTarget": 0.25,
        "MinimumTotalDenominator": 100,
        "MinimumPositiveDenominatorRepeats": 30,
        "IndividualPrecisionCellCount": 144,
        "PairedPrecisionCellCount": 72,
        "TotalPrecisionCellCount": 216,
        "RformalCandidates": [1000, 2000],
        "IndividualRequiredRFormula": (
            "ceil(Z975^2*s_psi_upper_sq/TargetHalfWidth^2)"
        ),
        "PairedInfluenceQuantity": "psi_delta_r=psi_TW_r-psi_T_r",
        "PairedRequiredRFormula": (
            "ceil(Z975^2*s_delta_upper_sq/TargetHalfWidth^2)"
        ),
        "R1000Rule": {
            "Operator": "ALL",
            "RequiredChecks": [
                "Stage7PilotDataComplete",
                "R0Equals200",
                "Conditions1Through18Complete",
                "MethodsTAndTWComplete",
                "RepeatIDs1Through200UniqueAndComplete",
                "AllFourMetricsDefined",
                "AllNonfixedTotalDenominatorsAtLeast100",
                "AllNonfixedPositiveDenominatorRepeatCountsAtLeast30",
                "All144IndividualRequiredRValuesAtMost1000",
                "All72PairedRequiredRValuesAtMost1000",
            ],
        },
        "R2000Rule": "Use 2000 if any R1000Rule check fails",
        "ProjectedPrecisionWarningRule": (
            "If any required R exceeds 2000, retain 2000 and record warning"
        ),
        "NoOutcomeDirectionRule": True,
        "NoManualOverride": True,
        "NoConditionDropping": True,
        "NoParameterChange": True,
        "FormalUsesSingleRForAllConditionsAndMethods": True,
    }


def lock_stage8_r_decision_contract(path: Path = STAGE8_CONTRACT_PATH) -> str:
    path = Path(path)
    expected_text = (
        json.dumps(
            stage8_r_decision_contract(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != expected_text:
            raise AssertionError("existing Stage 8 decision contract is not the frozen contract")
    else:
        write_text_atomic(path, expected_text)
    return sha256_file(path)


def stage7_boundary_flags() -> dict[str, bool]:
    return {
        "r_decision_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "stop_after_stage7": True,
    }


def verify_baseline_zip(path: Path | None = None) -> str:
    baseline_path = Path(path) if path is not None else ROOT.parent / BASELINE_ZIP_NAME
    if not baseline_path.is_file():
        raise FileNotFoundError(f"unique Stage 6 baseline ZIP is missing: {baseline_path}")
    actual = sha256_file(baseline_path)
    if actual != BASELINE_ZIP_SHA256:
        raise AssertionError(
            f"baseline ZIP SHA256 mismatch: expected {BASELINE_ZIP_SHA256}, got {actual}"
        )
    return actual


def validate_frozen_science_hashes(root: Path = ROOT) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in FROZEN_SCIENCE_HASHES.items():
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen science file is missing: {relative}")
        file_sha = sha256_file(path)
        actual[relative] = file_sha
        if file_sha != expected:
            raise AssertionError(
                f"frozen science hash mismatch for {relative}: {file_sha}"
            )
    return actual


def validate_stage1_through_stage6_statuses(root: Path = ROOT) -> dict[int, dict]:
    statuses: dict[int, dict] = {}
    for stage in range(1, 7):
        path = Path(root) / "artifacts" / f"stage{stage}" / "status.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("status") != "PASS":
            raise AssertionError(f"Stage {stage} status is not PASS")
        statuses[stage] = payload
    stage6 = statuses[6]
    required_false = (
        "science_contract_changed",
        "pilot_executed",
        "formal_executed",
        "bootstrap_executed",
        "r_decision_executed",
    )
    for field_name in required_false:
        if stage6.get(field_name) is not False:
            raise AssertionError(f"Stage 6 {field_name} must be false")
    if int(stage6.get("total_test_count", 0)) < 316:
        raise AssertionError("Stage 6 baseline records fewer than 316 passing tests")
    return statuses


def _reset_preflight_root(path: Path) -> None:
    resolved = Path(path).resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError("refusing to reset an unsafe preflight path")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def run_combined_parallel_chunk_resume_preflight(
    *,
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    output_root: Path,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    output_root = Path(output_root)
    _reset_preflight_root(output_root)
    serial_options = ExecutionOptions(
        workers=1,
        chunk_size=2,
        resume=False,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    parallel_options = replace(serial_options, workers=2, resume=False)
    details: list[dict[str, object]] = []

    for condition_ordinal, condition_code in PREFLIGHT_CONDITIONS:
        progress(f"[Stage7 preflight] {condition_code}: serial reference")
        serial = run_condition_repeats_serial(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=PREFLIGHT_REPEAT_IDS,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=serial_options,
        )

        progress(f"[Stage7 preflight] {condition_code}: fresh parallel + chunk")
        fresh = run_validation_chunked(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=PREFLIGHT_REPEAT_IDS,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=parallel_options,
            output_root=output_root / "fresh_parallel_chunk" / condition_code,
        )
        if not fresh.completed or fresh.result is None or fresh.computed_chunks != 3:
            raise AssertionError(f"{condition_code} fresh preflight did not complete")

        interrupted_root = output_root / "interrupted_resume" / condition_code
        progress(f"[Stage7 preflight] {condition_code}: interrupt after one chunk")
        interrupted = run_validation_chunked(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=PREFLIGHT_REPEAT_IDS,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=parallel_options,
            output_root=interrupted_root,
            stop_after_chunks=1,
        )
        if interrupted.completed or interrupted.computed_chunks != 1:
            raise AssertionError(f"{condition_code} interruption preflight was not real")

        progress(f"[Stage7 preflight] {condition_code}: parallel chunk resume")
        resumed = run_validation_chunked(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=PREFLIGHT_REPEAT_IDS,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=replace(parallel_options, resume=True),
            output_root=interrupted_root,
        )
        if (
            not resumed.completed
            or resumed.result is None
            or resumed.skipped_chunks != 1
            or resumed.computed_chunks != 2
        ):
            raise AssertionError(f"{condition_code} combined resume preflight failed")

        assert_scientific_results_equal(
            serial,
            fresh.result,
            label=f"{condition_code} serial/fresh parallel chunk",
        )
        assert_scientific_results_equal(
            serial,
            resumed.result,
            label=f"{condition_code} serial/interrupted parallel chunk resume",
        )
        manifests = tuple(
            output_root.rglob(f"*{condition_code}*/**/chunk_manifest.json")
        )
        if not manifests:
            manifests = tuple(
                path
                for path in output_root.rglob("chunk_manifest.json")
                if condition_code.lower() in str(path).lower()
            )
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("RunKind") != "VALIDATION":
                raise AssertionError("preflight manifest is not VALIDATION identity")
        details.append(
            {
                "ConditionOrdinal": condition_ordinal,
                "ConditionCode": condition_code,
                "RepeatIDs": list(PREFLIGHT_REPEAT_IDS),
                "SerialEqualsFreshParallelChunk": True,
                "SerialEqualsInterruptedThenParallelChunkResume": True,
                "InterruptedCompleted": False,
                "ResumeSkippedChunks": resumed.skipped_chunks,
                "ResumeComputedChunks": resumed.computed_chunks,
                "RunKind": "VALIDATION",
                "Namespace": "PILOT",
            }
        )
        progress(f"[Stage7 preflight] {condition_code}: exact equality PASS")

    report = {
        "Passed": True,
        "Conditions": details,
        "RepeatIDs": list(PREFLIGHT_REPEAT_IDS),
        "Workers": 2,
        "ChunkSize": 2,
        "ExactEquality": True,
        "EqualNaNOnlyForSelectedObservedTOA": True,
        "RunKind": "VALIDATION",
        "Namespace": "PILOT",
    }
    write_json_atomic(output_root / "preflight_report.json", report)
    return report


@dataclass(frozen=True, slots=True)
class PilotChunkExecutionOutcome:
    result: ConditionRunResult | None
    completed: bool
    computed_chunks: int
    skipped_chunks: int
    recomputed_invalid_chunks: int
    invalidated_chunks: tuple[str, ...]
    chunk_paths: tuple[PilotChunkPaths, ...]
    chunk_wall_seconds: tuple[float, ...]


def _validate_pilot_science_options(options: ExecutionOptions) -> None:
    if not isinstance(options, ExecutionOptions):
        raise TypeError("execution_options must be ExecutionOptions")
    if options.diagnostic:
        raise ValueError("formal Pilot diagnostics must be OFF")
    if not options.h_cache_enabled:
        raise ValueError("formal Pilot H cache must be ON")
    if options.method_execution_order != ORDER_A:
        raise ValueError("formal Pilot method execution order must be ORDER_A")


def validate_official_pilot_execution_options(options: ExecutionOptions) -> None:
    _validate_pilot_science_options(options)
    if options.workers != PILOT_WORKERS:
        raise ValueError("official Stage 7 Pilot workers must equal 2")
    if options.chunk_size != PILOT_CHUNK_SIZE:
        raise ValueError("official Stage 7 Pilot ChunkSize must equal 20")
    if not options.resume:
        raise ValueError("official Stage 7 Pilot resume must be ON")


def run_pilot_condition_chunked(
    *,
    condition_ordinal: int,
    repeat_ids: Iterable[int],
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    execution_options: ExecutionOptions,
    output_root: Path,
    stop_after_chunks: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> PilotChunkExecutionOutcome:
    _validate_pilot_science_options(execution_options)
    chunks = partition_repeat_ids(repeat_ids, execution_options.chunk_size)
    all_repeat_ids = tuple(repeat_id for chunk in chunks for repeat_id in chunk)
    if all_repeat_ids[0] < 1 or all_repeat_ids[-1] > PILOT_R:
        raise ValueError("formal Pilot RepeatIDs must lie in 1..200")
    if stop_after_chunks is not None:
        if isinstance(stop_after_chunks, bool) or not isinstance(stop_after_chunks, int):
            raise TypeError("stop_after_chunks must be an integer or None")
        if stop_after_chunks < 1:
            raise ValueError("stop_after_chunks must be positive")
    try:
        condition = CONDITIONS[int(condition_ordinal)]
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("condition_ordinal must lie in 0..18") from exc
    if condition.ConditionOrdinal != condition_ordinal:
        raise AssertionError("frozen condition order is not canonical")

    config_sha = sha256_file(CONFIG_DIR / "study_config.json")
    code_sha = compute_stage7_code_sha256()
    computed = 0
    skipped = 0
    recomputed_invalid = 0
    invalidated: list[str] = []
    completed_results: list[ConditionRunResult] = []
    processed_paths: list[PilotChunkPaths] = []
    chunk_wall_seconds: list[float] = []

    for chunk_index, chunk_repeat_ids in enumerate(chunks):
        if stop_after_chunks is not None and chunk_index >= stop_after_chunks:
            break
        chunk_started = time.perf_counter()
        paths = pilot_chunk_paths(
            output_root,
            condition_ordinal=condition_ordinal,
            condition_code=condition.Code,
            repeat_ids=chunk_repeat_ids,
        )
        processed_paths.append(paths)
        loaded: ConditionRunResult | None = None
        action = "computed"
        if execution_options.resume:
            had_chunk_artifact = paths.directory.exists()
            validation = validate_pilot_chunk(
                paths,
                study_config_sha256=config_sha,
                code_sha256=code_sha,
                condition_ordinal=condition_ordinal,
                condition_code=condition.Code,
                repeat_ids=chunk_repeat_ids,
                pilot_r=PILOT_R,
            )
            if validation.valid:
                loaded = load_validated_pilot_chunk(
                    paths,
                    study_config_sha256=config_sha,
                    code_sha256=code_sha,
                    condition_ordinal=condition_ordinal,
                    condition_code=condition.Code,
                    repeat_ids=chunk_repeat_ids,
                    pilot_r=PILOT_R,
                )
                skipped += 1
                action = "skipped-valid"
            else:
                reason = (
                    f"{chunk_repeat_ids[0]}-{chunk_repeat_ids[-1]}:"
                    f"{validation.reason}"
                )
                invalidated.append(reason)
                if had_chunk_artifact:
                    recomputed_invalid += 1
                    action = f"recomputed-invalid:{validation.reason}"
        if loaded is None:
            runner = (
                run_condition_repeats_serial
                if execution_options.workers == 1
                else run_condition_repeats_parallel
            )
            loaded = runner(
                namespace=RandomNamespace.PILOT,
                condition_ordinal=condition_ordinal,
                repeat_ids=chunk_repeat_ids,
                study_config=study_config,
                encoding_reference=encoding_reference,
                execution_options=execution_options,
            )
            write_pilot_chunk_atomic(
                output_root,
                result=loaded,
                study_config_sha256=config_sha,
                code_sha256=code_sha,
                pilot_r=PILOT_R,
            )
            computed += 1
        elapsed = time.perf_counter() - chunk_started
        chunk_wall_seconds.append(elapsed)
        completed_results.append(loaded)
        if progress is not None:
            progress(
                f"[Stage7 Pilot] {condition.Code} chunk "
                f"{chunk_repeat_ids[0]}-{chunk_repeat_ids[-1]} {action} "
                f"({elapsed:.2f}s)"
            )

    completed = len(completed_results) == len(chunks)
    result: ConditionRunResult | None = None
    if completed:
        result = assemble_pilot_condition_chunks(
            completed_results,
            expected_repeat_ids=all_repeat_ids,
        )
    return PilotChunkExecutionOutcome(
        result=result,
        completed=completed,
        computed_chunks=computed,
        skipped_chunks=skipped,
        recomputed_invalid_chunks=recomputed_invalid,
        invalidated_chunks=tuple(invalidated),
        chunk_paths=tuple(processed_paths),
        chunk_wall_seconds=tuple(chunk_wall_seconds),
    )


def validate_and_load_all_pilot_chunks(
    *,
    output_root: Path,
    chunk_size: int = PILOT_CHUNK_SIZE,
) -> tuple[ConditionRunResult, ...]:
    config_sha = sha256_file(CONFIG_DIR / "study_config.json")
    code_sha = compute_stage7_code_sha256()
    chunks = partition_repeat_ids(PILOT_REPEAT_IDS, chunk_size)
    results: list[ConditionRunResult] = []
    valid_count = 0
    for condition in CONDITIONS:
        loaded_chunks: list[ConditionRunResult] = []
        for repeat_chunk in chunks:
            paths = pilot_chunk_paths(
                output_root,
                condition_ordinal=condition.ConditionOrdinal,
                condition_code=condition.Code,
                repeat_ids=repeat_chunk,
            )
            validation = validate_pilot_chunk(
                paths,
                study_config_sha256=config_sha,
                code_sha256=code_sha,
                condition_ordinal=condition.ConditionOrdinal,
                condition_code=condition.Code,
                repeat_ids=repeat_chunk,
                pilot_r=PILOT_R,
            )
            if not validation.valid:
                raise AssertionError(
                    f"invalid Pilot chunk {condition.Code} "
                    f"{repeat_chunk[0]}-{repeat_chunk[-1]}: {validation.reason}"
                )
            loaded_chunks.append(
                load_validated_pilot_chunk(
                    paths,
                    study_config_sha256=config_sha,
                    code_sha256=code_sha,
                    condition_ordinal=condition.ConditionOrdinal,
                    condition_code=condition.Code,
                    repeat_ids=repeat_chunk,
                    pilot_r=PILOT_R,
                )
            )
            valid_count += 1
        results.append(
            assemble_pilot_condition_chunks(
                loaded_chunks,
                expected_repeat_ids=PILOT_REPEAT_IDS,
            )
        )
    if valid_count != EXPECTED_CHUNK_COUNT:
        raise AssertionError("Pilot chunk validation did not cover exactly 190 chunks")
    return tuple(results)


def validate_ni_identity(result: ConditionRunResult) -> bool:
    if result.condition_ordinal != 0 or result.condition_code != "NI":
        raise AssertionError("NI identity validator received a non-NI condition")
    for position, repeat_id in enumerate(result.repeat_ids):
        for method_ordinal in range(1, 5):
            if not np.array_equal(
                result.source_state[0, position],
                result.source_state[method_ordinal, position],
            ):
                raise AssertionError(f"NI source identity failed for RepeatID={repeat_id}")
            if not np.array_equal(
                result.ref_toa_before_s[0, position],
                result.ref_toa_before_s[method_ordinal, position],
            ):
                raise AssertionError(f"NI reference identity failed for RepeatID={repeat_id}")
            if not np.array_equal(
                result.selected_observed_toa_s[0, position],
                result.selected_observed_toa_s[method_ordinal, position],
                equal_nan=True,
            ):
                raise AssertionError(f"NI selected TOA identity failed for RepeatID={repeat_id}")
            if not np.array_equal(
                result.has_selection[0, position],
                result.has_selection[method_ordinal, position],
            ):
                raise AssertionError(f"NI selection identity failed for RepeatID={repeat_id}")
        rows = tuple(row for row in result.repeat_metrics if row.RepeatID == repeat_id)
        if len(rows) != 5:
            raise AssertionError("NI RepeatMetrics are incomplete")
        payloads: list[dict[str, object]] = []
        for row in rows:
            if (row.N_C, row.N_E, row.N_N) != (200, 0, 0):
                raise AssertionError(f"NI scientific counts failed for RepeatID={repeat_id}")
            payload = asdict(row)
            payload.pop("MethodOrdinal")
            payload.pop("MethodCode")
            payloads.append(payload)
        if any(payload != payloads[0] for payload in payloads[1:]):
            raise AssertionError(f"NI RepeatMetrics identity failed for RepeatID={repeat_id}")
    return True


def validate_repeat_metrics_rows(
    rows: Iterable[RepeatMetrics],
    *,
    expected_repeat_ids: Sequence[int] = PILOT_REPEAT_IDS,
) -> dict[str, bool]:
    materialized = tuple(rows)
    expected_repeats = tuple(expected_repeat_ids)
    if any(not isinstance(row, RepeatMetrics) for row in materialized):
        raise TypeError("all Pilot RepeatMetrics rows must be RepeatMetrics")
    expected_keys = {
        (condition.ConditionOrdinal, method.MethodOrdinal, repeat_id)
        for condition in CONDITIONS
        for method in METHODS
        for repeat_id in expected_repeats
    }
    actual_keys = [
        (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID) for row in materialized
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise AssertionError("Pilot RepeatMetrics contain duplicate keys")
    if set(actual_keys) != expected_keys:
        raise AssertionError("Pilot RepeatMetrics contain a missing or unexpected key")
    condition_codes = {condition.ConditionOrdinal: condition.Code for condition in CONDITIONS}
    method_codes = {method.MethodOrdinal: method.Code for method in METHODS}
    repeat_conservation = True
    transition_conservation = True
    for row in materialized:
        if row.ConditionCode != condition_codes[row.ConditionOrdinal]:
            raise AssertionError("RepeatMetrics condition identity is not frozen")
        if row.MethodCode != method_codes[row.MethodOrdinal]:
            raise AssertionError("RepeatMetrics method identity is not frozen")
        repeat_conservation &= row.N_C + row.N_E + row.N_N == 200
        transitions = (
            row.N_CC
            + row.N_CE
            + row.N_CN
            + row.N_EC
            + row.N_EE
            + row.N_EN
            + row.N_NC
            + row.N_NE
            + row.N_NN
        )
        transition_conservation &= transitions == 199
        transition_conservation &= row.N_Cdot == row.N_CC + row.N_CE + row.N_CN
        transition_conservation &= row.N_Edot == row.N_EC + row.N_EE + row.N_EN
        transition_conservation &= row.N_Ndot == row.N_NC + row.N_NE + row.N_NN
        transition_conservation &= row.N_Cdot + row.N_Edot + row.N_Ndot == 199
        repeat_conservation &= row.Sum_L_NC_obs == row.N_E + row.N_N
        repeat_conservation &= row.N_E_H + row.N_E_F == row.N_E
    if not repeat_conservation:
        raise AssertionError("Pilot RepeatMetrics state/run conservation failed")
    if not transition_conservation:
        raise AssertionError("Pilot RepeatMetrics transition conservation failed")
    return {
        "repeat_conservation_passed": True,
        "transition_conservation_passed": True,
    }


def validate_n_state_assertions(rows: Iterable[RepeatMetrics]) -> tuple[bool, bool]:
    materialized = tuple(rows)
    hprf_rows = tuple(
        row for row in materialized if row.ConditionOrdinal in HPRF_CONDITION_ORDINALS
    )
    composite_rows = tuple(
        row
        for row in materialized
        if row.ConditionOrdinal in COMPOSITE_CONDITION_ORDINALS
    )
    expected_hprf = len(HPRF_CONDITION_ORDINALS) * 5 * PILOT_R
    expected_composite = len(COMPOSITE_CONDITION_ORDINALS) * 5 * PILOT_R
    if len(hprf_rows) != expected_hprf or len(composite_rows) != expected_composite:
        raise AssertionError("HPRF/composite N-state gate did not receive full Pilot data")
    hprf_passed = all(row.N_N == 0 for row in hprf_rows)
    composite_passed = all(row.N_N == 0 for row in composite_rows)
    if not hprf_passed:
        raise AssertionError("an HPRF Pilot row contains forbidden N state")
    if not composite_passed:
        raise AssertionError("a composite Pilot row contains forbidden N state")
    return True, True


def validate_h300_width_strata(
    rows: Iterable[H300WidthStratumMetrics],
    *,
    expected_repeat_ids: Sequence[int] = PILOT_REPEAT_IDS,
) -> bool:
    materialized = tuple(rows)
    expected_repeats = tuple(expected_repeat_ids)
    expected_keys = {
        (method.MethodOrdinal, repeat_id, width_stratum)
        for method in METHODS
        for repeat_id in expected_repeats
        for width_stratum in WIDTH_STRATA
    }
    actual_keys = [
        (row.MethodOrdinal, row.RepeatID, row.WidthStratum) for row in materialized
    ]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise AssertionError("H300 width strata are incomplete, duplicated, or noncanonical")
    if any(
        row.ConditionOrdinal != 3
        or row.ConditionCode != "H300"
        or row.WidthStratum not in WIDTH_STRATA
        for row in materialized
    ):
        raise AssertionError("H300 width stratum identity failed")
    for method in METHODS:
        for repeat_id in expected_repeats:
            subset = tuple(
                row
                for row in materialized
                if row.MethodOrdinal == method.MethodOrdinal
                and row.RepeatID == repeat_id
            )
            if sum(row.StratumCycleCount for row in subset) != 200:
                raise AssertionError("H300 width strata do not conserve 200 cycles")
    return True


def aggregate_pilot_repeat_metrics(
    rows: Iterable[RepeatMetrics],
    *,
    expected_repeat_ids: Sequence[int] = PILOT_REPEAT_IDS,
) -> tuple[AggregatedMetrics, ...]:
    materialized = tuple(rows)
    expected_repeats = set(expected_repeat_ids)
    result: list[AggregatedMetrics] = []
    for condition in CONDITIONS:
        for method in METHODS:
            group = tuple(
                row
                for row in materialized
                if row.ConditionOrdinal == condition.ConditionOrdinal
                and row.MethodOrdinal == method.MethodOrdinal
            )
            if {row.RepeatID for row in group} != expected_repeats or len(group) != len(
                expected_repeats
            ):
                raise AssertionError(
                    f"aggregate input is incomplete for {condition.Code}/{method.Code}"
                )
            result.append(aggregate_repeat_metrics(group))
    if len(result) != 95:
        raise AssertionError("Pilot aggregation did not produce exactly 95 rows")
    return tuple(result)


def aggregated_metrics_field_names() -> tuple[str, ...]:
    result: list[str] = []
    for field_info in fields(AggregatedMetrics):
        if field_info.name in AGGREGATED_METRIC_NAMES:
            result.extend(
                (
                    f"{field_info.name}_value",
                    f"{field_info.name}_defined",
                    f"{field_info.name}_undefined_reason",
                )
            )
        else:
            result.append(field_info.name)
    return tuple(result)


def flatten_aggregated_metrics(row: AggregatedMetrics) -> dict[str, object]:
    if not isinstance(row, AggregatedMetrics):
        raise TypeError("row must be AggregatedMetrics")
    result: dict[str, object] = {}
    for field_info in fields(AggregatedMetrics):
        value = getattr(row, field_info.name)
        if field_info.name in AGGREGATED_METRIC_NAMES:
            if not isinstance(value, MetricValue):
                raise TypeError(f"{field_info.name} is not MetricValue")
            result[f"{field_info.name}_value"] = value.value
            result[f"{field_info.name}_defined"] = value.defined
            result[f"{field_info.name}_undefined_reason"] = value.undefined_reason
        else:
            result[field_info.name] = value
    if tuple(result) != aggregated_metrics_field_names():
        raise AssertionError("flattened aggregate schema is not canonical")
    return result


def build_stage8_decision_rows(
    rows: Iterable[RepeatMetrics],
    *,
    expected_repeat_ids: Sequence[int] = PILOT_REPEAT_IDS,
) -> tuple[dict[str, object], ...]:
    expected_repeats = tuple(expected_repeat_ids)
    selected = tuple(
        sorted(
            (
                row
                for row in rows
                if 1 <= row.ConditionOrdinal <= 18 and row.MethodOrdinal in (2, 4)
            ),
            key=lambda row: (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID),
        )
    )
    expected_keys = {
        (condition_ordinal, method_ordinal, repeat_id)
        for condition_ordinal in range(1, 19)
        for method_ordinal in (2, 4)
        for repeat_id in expected_repeats
    }
    actual_keys = [
        (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID) for row in selected
    ]
    if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
        raise AssertionError("Stage 8 decision input is incomplete or contains duplicates")
    result = tuple(
        {field_name: getattr(row, field_name) for field_name in STAGE8_INPUT_FIELDS}
        for row in selected
    )
    if len(result) != 18 * 2 * len(expected_repeats):
        raise AssertionError("Stage 8 decision input row count is not canonical")
    return result


def _repeat_metric_rows_sorted(
    results: Iterable[ConditionRunResult],
) -> tuple[RepeatMetrics, ...]:
    return tuple(
        sorted(
            (row for result in results for row in result.repeat_metrics),
            key=lambda row: (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID),
        )
    )


def _h300_rows_sorted(
    results: Iterable[ConditionRunResult],
) -> tuple[H300WidthStratumMetrics, ...]:
    rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
    return tuple(
        sorted(
            (row for result in results for row in result.h300_width_strata),
            key=lambda row: (
                row.ConditionOrdinal,
                row.MethodOrdinal,
                row.RepeatID,
                rank[row.WidthStratum],
            ),
        )
    )


def write_pilot_repeat_metrics(
    path: Path, rows: Iterable[RepeatMetrics]
) -> tuple[RepeatMetrics, ...]:
    materialized = tuple(
        sorted(
            rows,
            key=lambda row: (row.ConditionOrdinal, row.MethodOrdinal, row.RepeatID),
        )
    )
    names = tuple(field.name for field in fields(RepeatMetrics))
    write_csv_dicts_atomic(
        path,
        field_names=names,
        rows=(asdict(row) for row in materialized),
    )
    return materialized


def write_pilot_aggregated_metrics(
    path: Path, rows: Iterable[AggregatedMetrics]
) -> tuple[AggregatedMetrics, ...]:
    materialized = tuple(rows)
    write_csv_dicts_atomic(
        path,
        field_names=aggregated_metrics_field_names(),
        rows=(flatten_aggregated_metrics(row) for row in materialized),
    )
    return materialized


def write_pilot_h300_width_strata(
    path: Path, rows: Iterable[H300WidthStratumMetrics]
) -> tuple[H300WidthStratumMetrics, ...]:
    rank = {name: index for index, name in enumerate(WIDTH_STRATA)}
    materialized = tuple(
        sorted(
            rows,
            key=lambda row: (
                row.ConditionOrdinal,
                row.MethodOrdinal,
                row.RepeatID,
                rank[row.WidthStratum],
            ),
        )
    )
    names = tuple(field.name for field in fields(H300WidthStratumMetrics))
    write_csv_dicts_atomic(
        path,
        field_names=names,
        rows=(asdict(row) for row in materialized),
    )
    return materialized


def write_stage8_decision_input(
    path: Path, rows: Iterable[Mapping[str, object]]
) -> tuple[dict[str, object], ...]:
    materialized = tuple(dict(row) for row in rows)
    write_csv_dicts_atomic(
        path,
        field_names=STAGE8_INPUT_FIELDS,
        rows=materialized,
    )
    return materialized


def build_pilot_manifest(
    *,
    frozen_hashes: Mapping[str, str],
    stage7_code_sha256: str,
    stage8_contract_sha256: str,
    stage8_input_sha256: str,
    repeat_metrics_sha256: str,
    aggregated_metrics_sha256: str,
    h300_strata_sha256: str,
    completed_chunk_count: int = EXPECTED_CHUNK_COUNT,
) -> dict[str, object]:
    return {
        "StageNumber": STAGE_NUMBER,
        "StageName": STAGE_NAME,
        "RunKind": RUN_KIND,
        "Namespace": RandomNamespace.PILOT.name,
        "PilotR": PILOT_R,
        "RepeatStart": 1,
        "RepeatEnd": PILOT_R,
        "ConditionCount": 19,
        "MethodCount": 5,
        "K": 200,
        "ExpectedMethodCycles": EXPECTED_METHOD_CYCLES,
        "ChunkSize": PILOT_CHUNK_SIZE,
        "ExpectedChunkCount": EXPECTED_CHUNK_COUNT,
        "CompletedChunkCount": int(completed_chunk_count),
        "StudyConfigSHA256": FROZEN_SCIENCE_HASHES[
            "artifacts/config/study_config.json"
        ],
        "FrozenScienceHashes": dict(frozen_hashes),
        "Stage7CodeSHA256": str(stage7_code_sha256),
        "Stage8DecisionContractSHA256": str(stage8_contract_sha256),
        "Stage8DecisionInputSHA256": str(stage8_input_sha256),
        "PilotRepeatMetricsSHA256": str(repeat_metrics_sha256),
        "PilotAggregatedMetricsSHA256": str(aggregated_metrics_sha256),
        "PilotH300WidthStrataSHA256": str(h300_strata_sha256),
        "PilotCompleted": True,
        "FormalExecuted": False,
        "BootstrapExecuted": False,
        "RDecisionExecuted": False,
    }


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
    matches = re.findall(r"(\d+) passed", output)
    passed_count = int(matches[-1]) if matches else 0
    if completed.returncode != 0 or passed_count < 316 or re.search(r"\d+ failed", output):
        raise AssertionError(f"full pytest failed\n{output}")
    return {
        "command": f"{sys.executable} -m pytest",
        "returncode": completed.returncode,
        "passed": passed_count,
        "failed": 0,
        "wall_seconds": elapsed,
        "output": output,
    }


def _write_validation_report(
    *,
    test_count: int,
    pilot_wall_seconds: float,
) -> None:
    text = f"""# Stage 7 Validation Report

## Baseline gate

- Unique Stage 6 ZIP SHA256: PASS.
- Unmodified Stage 1-6 regression baseline: 316 passed.
- Stage 1-6 artifact status gate: PASS.

## Frozen SHA gate

- All 19 frozen science/configuration file hashes: PASS.
- Science contract changed: no.

## Stage 8 contract lock

- Machine-readable decision contract was locked before Pilot RepeatID 1.
- The post-Pilot SHA256 exactly matches the pre-Pilot SHA256.
- No Stage 8 R decision was executed.

## Combined execution preflight

- Serial reference, fresh parallel+chunk, and interrupted parallel+chunk+resume are exactly equal for H300, F2, and HF300-2.
- Preflight RunKind is VALIDATION and is stored outside the formal Pilot chunk directory.

## Pilot execution completeness and chunk integrity

- Namespace=PILOT, RepeatID=1..200, 19 Conditions, and 5 Methods completed.
- 190/190 chunks passed identity, schema, file-hash, DataSHA256, and full-load validation.
- Resume eligibility requires every frozen identity and data hash; invalid chunks are recomputed as a whole.

## Global row counts and scientific conservation

- pilot_repeat_metrics.csv: 19000 rows.
- pilot_aggregated_metrics.csv: 95 rows.
- pilot_h300_width_strata.csv: 3000 rows.
- stage8_decision_input.csv: 7200 rows.
- State, transition, run-length, and error-source conservation: PASS.
- NI five-method identity: PASS.
- HPRF and Composite N-state assertions: PASS.
- Stage 8 input scope, method scope, RepeatID completeness, and ordering: PASS.

## Tests and performance

- Full pytest: {test_count} passed, 0 failed.
- Pilot wall time: {pilot_wall_seconds:.6f} seconds.
- Detailed engineering timing is recorded in performance.json.

## STOP status

- Formal execution: no.
- Bootstrap execution: no.
- Paper-result generation: no.
- Stage 8 R decision: no.
- Stop after Stage 7: yes.
"""
    write_text_atomic(STAGE7_DIR / "validation_report.md", text)


def _write_status(
    *,
    stage8_contract_sha256: str,
    stage8_input_sha256: str,
) -> dict[str, object]:
    status = {
        "stage": 7,
        "stage_name": STAGE_NAME,
        "status": "PASS",
        "baseline_zip_sha256_verified": True,
        "stage1_regression_passed": True,
        "stage2_regression_passed": True,
        "stage3_regression_passed": True,
        "stage4_regression_passed": True,
        "stage5_regression_passed": True,
        "stage6_regression_passed": True,
        "science_contract_changed": False,
        "frozen_hashes_passed": True,
        "stage8_decision_contract_locked_before_pilot": True,
        "stage8_decision_contract_sha256": stage8_contract_sha256,
        "combined_parallel_chunk_resume_preflight_passed": True,
        "pilot_executed": True,
        "pilot_r": PILOT_R,
        "pilot_namespace": "PILOT",
        "pilot_repeat_range": [1, PILOT_R],
        "pilot_condition_count": 19,
        "pilot_method_count": 5,
        "pilot_expected_chunk_count": EXPECTED_CHUNK_COUNT,
        "pilot_completed_chunk_count": EXPECTED_CHUNK_COUNT,
        "pilot_all_chunks_valid": True,
        "pilot_repeat_metrics_row_count": 19000,
        "pilot_aggregated_metrics_row_count": 95,
        "pilot_h300_strata_row_count": 3000,
        "stage8_decision_input_row_count": 7200,
        "ni_identity_passed": True,
        "hprf_no_n_passed": True,
        "composite_no_n_passed": True,
        "repeat_conservation_passed": True,
        "transition_conservation_passed": True,
        "stage8_decision_input_sha256": stage8_input_sha256,
        "r_decision_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "blocking_issue_count": 0,
        "stop_after_stage7": True,
    }
    write_json_atomic(STAGE7_DIR / "status.json", status)
    return status


def _excluded_packaging_part(path: Path) -> bool:
    return any(part in {"__pycache__", ".pytest_cache", ".venv"} for part in path.parts)


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
            if _excluded_packaging_part(relative):
                continue
            members[relative.as_posix()] = hashlib.sha256(archive.read(info)).hexdigest()
    return members


def _write_changed_files(baseline_zip: Path) -> None:
    baseline = _baseline_archive_members(baseline_zip)
    current: dict[str, str] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if _excluded_packaging_part(relative) or ".tmp-" in path.name:
            continue
        current[relative.as_posix()] = sha256_file(path)
    current.setdefault("artifacts/stage7/changed_files.txt", "SELF")
    added = sorted(set(current) - set(baseline))
    modified = sorted(
        relative
        for relative in set(current).intersection(baseline)
        if current[relative] != baseline[relative]
    )

    def reason(relative: str) -> str:
        if relative.startswith("artifacts/stage7/pilot_chunks/"):
            return "Stage 7 formal Pilot atomic chunk artifact"
        if relative.startswith("artifacts/stage7/preflight/"):
            return "Stage 7 combined parallel+chunk+resume preflight artifact"
        if relative.startswith("artifacts/stage7/"):
            return "Stage 7 contract, global output, validation, or status artifact"
        if relative.startswith("tests/test_stage7_"):
            return "Stage 7 contract and execution regression coverage"
        if relative.startswith("src/sal_stability_stage1/"):
            return "Stage 7 isolated implementation layer"
        if relative == "scripts/build_stage7.py":
            return "Stage 7 fixed-order build entry point"
        if relative == "pyproject.toml":
            return "Project version updated to 0.7.0 without dependency changes"
        if relative == "README.md":
            return "Stage 7 execution and STOP boundary documented"
        return "Stage 7 required project artifact"

    lines = ["STAGE 7 CHANGED FILES", "", "ADDED FILES:"]
    lines.extend(f"- {relative} | {reason(relative)}" for relative in added)
    lines.extend(("", "MODIFIED FILES:"))
    lines.extend(f"- {relative} | {reason(relative)}" for relative in modified)
    lines.extend(("", "FROZEN SCIENCE FILES:"))
    lines.extend(
        f"- {relative} | UNCHANGED" for relative in FROZEN_SCIENCE_HASHES
    )
    lines.extend(("", "DELETED FILES:", "- NONE", ""))
    write_text_atomic(STAGE7_DIR / "changed_files.txt", "\n".join(lines))


def build_stage7(
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    overall_started = time.perf_counter()
    baseline_zip = ROOT.parent / BASELINE_ZIP_NAME

    progress("[Stage7 STEP 1] Verify unique baseline ZIP and load Stage 6 inputs")
    baseline_sha = verify_baseline_zip(baseline_zip)
    study_config, encoding_reference = load_stage3_inputs()
    if study_config.get("PilotR") != PILOT_R:
        raise AssertionError("StudyConfig PilotR is not frozen at 200")

    progress("[Stage7 STEP 2] Verify all frozen science SHA256 values")
    frozen_hashes = validate_frozen_science_hashes()

    progress("[Stage7 STEP 3] Verify Stage 1-6 PASS artifacts")
    validate_stage1_through_stage6_statuses()

    progress("[Stage7 STEP 4] Lock Stage 8 automatic R-decision contract")
    STAGE7_DIR.mkdir(parents=True, exist_ok=True)
    contract_sha_before = lock_stage8_r_decision_contract()
    progress(f"[Stage7 STEP 5] Contract SHA256 cached: {contract_sha_before}")

    progress("[Stage7 STEP 6] Combined parallel + chunk + resume preflight")
    preflight = run_combined_parallel_chunk_resume_preflight(
        study_config=study_config,
        encoding_reference=encoding_reference,
        output_root=PREFLIGHT_DIR,
        progress=progress,
    )
    if preflight.get("Passed") is not True:
        raise AssertionError("combined Stage 7 preflight did not PASS")

    official_options = ExecutionOptions(
        workers=PILOT_WORKERS,
        chunk_size=PILOT_CHUNK_SIZE,
        resume=True,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    validate_official_pilot_execution_options(official_options)
    progress("[Stage7 STEP 7] Start complete formal Pilot R=200 in PILOT namespace")
    pilot_started = time.perf_counter()
    condition_wall_seconds: dict[str, float] = {}
    all_chunk_times: list[float] = []
    total_computed = 0
    total_skipped = 0
    total_recomputed_invalid = 0
    for condition in CONDITIONS:
        condition_started = time.perf_counter()
        progress(
            f"[Stage7 STEP 8] Condition {condition.ConditionOrdinal:02d} "
            f"{condition.Code}: RepeatID 1..200"
        )
        outcome = run_pilot_condition_chunked(
            condition_ordinal=condition.ConditionOrdinal,
            repeat_ids=PILOT_REPEAT_IDS,
            study_config=study_config,
            encoding_reference=encoding_reference,
            execution_options=official_options,
            output_root=PILOT_CHUNKS_DIR,
            progress=progress,
        )
        if not outcome.completed or outcome.result is None:
            raise AssertionError(f"Pilot condition {condition.Code} did not complete")
        if len(outcome.chunk_paths) != 10:
            raise AssertionError(f"Pilot condition {condition.Code} did not produce 10 chunks")
        condition_elapsed = time.perf_counter() - condition_started
        condition_wall_seconds[
            f"{condition.ConditionOrdinal:02d}_{condition.Code}"
        ] = condition_elapsed
        all_chunk_times.extend(outcome.chunk_wall_seconds)
        total_computed += outcome.computed_chunks
        total_skipped += outcome.skipped_chunks
        total_recomputed_invalid += outcome.recomputed_invalid_chunks
        progress(
            f"[Stage7 Pilot] {condition.Code} complete: 10/10 chunks "
            f"({condition_elapsed:.2f}s)"
        )
    pilot_wall_seconds = time.perf_counter() - pilot_started

    progress("[Stage7 STEP 9] Validate and fully load all 190 Pilot chunks")
    condition_results = validate_and_load_all_pilot_chunks(
        output_root=PILOT_CHUNKS_DIR,
        chunk_size=PILOT_CHUNK_SIZE,
    )
    repeat_rows = _repeat_metric_rows_sorted(condition_results)
    h300_rows = _h300_rows_sorted(condition_results)

    progress("[Stage7 STEP 10] Write pilot_repeat_metrics.csv")
    write_pilot_repeat_metrics(PILOT_REPEAT_METRICS_PATH, repeat_rows)

    progress("[Stage7 STEP 11] Aggregate total-count ratio point metrics")
    aggregated_rows = aggregate_pilot_repeat_metrics(repeat_rows)
    write_pilot_aggregated_metrics(PILOT_AGGREGATED_METRICS_PATH, aggregated_rows)

    progress("[Stage7 STEP 12] Write H300 three-level width support statistics")
    write_pilot_h300_width_strata(PILOT_H300_STRATA_PATH, h300_rows)

    progress("[Stage7 STEP 13] Write the unique Stage 8 Pilot decision input")
    stage8_rows = build_stage8_decision_rows(repeat_rows)
    write_stage8_decision_input(STAGE8_DECISION_INPUT_PATH, stage8_rows)

    progress("[Stage7 STEP 14] Re-read and verify immutable Stage 8 contract SHA256")
    contract_sha_after = sha256_file(STAGE8_CONTRACT_PATH)
    if contract_sha_after != contract_sha_before:
        raise AssertionError("Stage 8 contract changed after Pilot began")

    progress("[Stage7 STEP 15] Execute all Stage 7 scientific hard assertions")
    conservation = validate_repeat_metrics_rows(repeat_rows)
    ni_result = next(result for result in condition_results if result.condition_ordinal == 0)
    ni_passed = validate_ni_identity(ni_result)
    hprf_passed, composite_passed = validate_n_state_assertions(repeat_rows)
    h300_passed = validate_h300_width_strata(h300_rows)
    if len(repeat_rows) != 19000:
        raise AssertionError("pilot_repeat_metrics row count is not 19000")
    if len(aggregated_rows) != 95:
        raise AssertionError("pilot_aggregated_metrics row count is not 95")
    if len(h300_rows) != 3000:
        raise AssertionError("pilot_h300_width_strata row count is not 3000")
    if len(stage8_rows) != 7200:
        raise AssertionError("stage8_decision_input row count is not 7200")
    validate_frozen_science_hashes()

    progress("[Stage7 STEP 16] Run complete pytest regression suite")
    pytest_result = _run_full_pytest()
    progress(
        f"[Stage7 tests] {pytest_result['passed']} passed, "
        f"{pytest_result['failed']} failed"
    )

    progress("[Stage7 STEP 17] Write performance, manifests, reports, and PASS status")
    code_sha = compute_stage7_code_sha256()
    repeat_sha = sha256_file(PILOT_REPEAT_METRICS_PATH)
    aggregate_sha = sha256_file(PILOT_AGGREGATED_METRICS_PATH)
    h300_sha = sha256_file(PILOT_H300_STRATA_PATH)
    stage8_input_sha = sha256_file(STAGE8_DECISION_INPUT_PATH)
    performance = {
        "Workers": PILOT_WORKERS,
        "ChunkSize": PILOT_CHUNK_SIZE,
        "CPUCount": os.cpu_count(),
        "PythonVersion": platform.python_version(),
        "NumPyVersion": np.__version__,
        "OS": platform.platform(),
        "PilotWallSeconds": pilot_wall_seconds,
        "ConditionWallSeconds": condition_wall_seconds,
        "AverageChunkWallSeconds": (
            math.fsum(all_chunk_times) / len(all_chunk_times)
            if all_chunk_times
            else 0.0
        ),
        "ComputedChunks": total_computed,
        "SkippedChunks": total_skipped,
        "RecomputedInvalidChunks": total_recomputed_invalid,
        "CompletedChunks": EXPECTED_CHUNK_COUNT,
        "ExpectedChunks": EXPECTED_CHUNK_COUNT,
        "ResumeEnabled": True,
    }
    write_json_atomic(STAGE7_DIR / "performance.json", performance)
    manifest = build_pilot_manifest(
        frozen_hashes=frozen_hashes,
        stage7_code_sha256=code_sha,
        stage8_contract_sha256=contract_sha_before,
        stage8_input_sha256=stage8_input_sha,
        repeat_metrics_sha256=repeat_sha,
        aggregated_metrics_sha256=aggregate_sha,
        h300_strata_sha256=h300_sha,
    )
    write_json_atomic(STAGE7_DIR / "pilot_manifest.json", manifest)
    test_report = (
        f"Command: {pytest_result['command']}\n"
        f"Return code: {pytest_result['returncode']}\n"
        f"Result: {pytest_result['passed']} passed, 0 failed\n"
        f"Wall seconds: {pytest_result['wall_seconds']:.6f}\n\n"
        f"{pytest_result['output']}"
    )
    write_text_atomic(STAGE7_DIR / "test_report.txt", test_report)
    _write_validation_report(
        test_count=int(pytest_result["passed"]),
        pilot_wall_seconds=pilot_wall_seconds,
    )
    status = _write_status(
        stage8_contract_sha256=contract_sha_before,
        stage8_input_sha256=stage8_input_sha,
    )
    _write_changed_files(baseline_zip)
    validate_frozen_science_hashes()

    progress("[Stage7 STEP 18] STOP after Stage 7")
    return {
        "Stage7Status": status["status"],
        "BaselineZipSHA256": baseline_sha,
        "FullPytestPassed": pytest_result["passed"],
        "FrozenScienceHashesPassed": True,
        "Stage8DecisionContractLockedBeforePilot": True,
        "Stage8DecisionContractSHA256": contract_sha_before,
        "CombinedParallelChunkResumePreflightPassed": True,
        "PilotExecuted": True,
        "PilotNamespace": "PILOT",
        "PilotR": PILOT_R,
        "PilotRepeatIDs": [1, PILOT_R],
        "ConditionsCompleted": 19,
        "MethodsCompleted": 5,
        "PilotChunksValid": EXPECTED_CHUNK_COUNT,
        "PilotRepeatMetricsRows": len(repeat_rows),
        "PilotAggregatedMetricsRows": len(aggregated_rows),
        "PilotH300StrataRows": len(h300_rows),
        "Stage8DecisionInputRows": len(stage8_rows),
        "Stage8DecisionInputSHA256": stage8_input_sha,
        "NIIdentityPassed": ni_passed,
        "HPRFNoNPassed": hprf_passed,
        "CompositeNoNPassed": composite_passed,
        "H300StrataPassed": h300_passed,
        **conservation,
        **stage7_boundary_flags(),
        "ScienceContractChanged": False,
        "BlockingIssues": 0,
        "PilotWallSeconds": pilot_wall_seconds,
        "OverallWallSeconds": time.perf_counter() - overall_started,
    }


__all__ = [
    "AGGREGATED_METRIC_NAMES",
    "BASELINE_ZIP_NAME",
    "BASELINE_ZIP_SHA256",
    "COMPOSITE_CONDITION_ORDINALS",
    "EXPECTED_CHUNK_COUNT",
    "EXPECTED_METHOD_CYCLES",
    "FROZEN_SCIENCE_HASHES",
    "HPRF_CONDITION_ORDINALS",
    "PILOT_AGGREGATED_METRICS_PATH",
    "PILOT_CHUNKS_DIR",
    "PILOT_CHUNK_SIZE",
    "PILOT_H300_STRATA_PATH",
    "PILOT_REPEAT_IDS",
    "PILOT_REPEAT_METRICS_PATH",
    "PILOT_WORKERS",
    "PREFLIGHT_CONDITIONS",
    "PREFLIGHT_REPEAT_IDS",
    "STAGE7_DIR",
    "STAGE8_CONTRACT_PATH",
    "STAGE8_DECISION_INPUT_PATH",
    "STAGE8_INPUT_FIELDS",
    "PilotChunkExecutionOutcome",
    "aggregate_pilot_repeat_metrics",
    "aggregated_metrics_field_names",
    "build_pilot_manifest",
    "build_stage7",
    "build_stage8_decision_rows",
    "flatten_aggregated_metrics",
    "lock_stage8_r_decision_contract",
    "run_combined_parallel_chunk_resume_preflight",
    "run_pilot_condition_chunked",
    "stage7_boundary_flags",
    "stage8_r_decision_contract",
    "validate_and_load_all_pilot_chunks",
    "validate_frozen_science_hashes",
    "validate_h300_width_strata",
    "validate_ni_identity",
    "validate_n_state_assertions",
    "validate_official_pilot_execution_options",
    "validate_repeat_metrics_rows",
    "validate_stage1_through_stage6_statuses",
    "verify_baseline_zip",
    "write_pilot_aggregated_metrics",
    "write_pilot_h300_width_strata",
    "write_pilot_repeat_metrics",
    "write_stage8_decision_input",
]
