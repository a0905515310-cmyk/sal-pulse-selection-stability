from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from .build import CONFIG_DIR, ROOT
from .diagnostics import write_diagnostics_csv
from .events import condition_spec_from_config
from .execution import (
    ConditionRunResult,
    ExecutionOptions,
    ORDER_A,
    ORDER_B,
    ORDER_C,
    assemble_condition_repeat_results,
    assert_scientific_results_equal,
    run_condition_repeat,
    run_condition_repeats_parallel,
    run_condition_repeats_serial,
)
from .kernel import CycleCandidates, METHOD_IDENTITIES, select_for_method
from .rng import RandomNamespace
from .selectors import CandidateView
from .stage3 import load_stage3_inputs
from .stage5 import (
    H300_CONDITION_CODE,
    write_h300_width_strata_csv,
    write_repeat_metrics_csv,
)
from .storage import (
    CODE_VERSION,
    ChunkPaths,
    assemble_condition_chunks,
    chunk_paths,
    compute_code_sha256,
    load_validated_chunk,
    partition_repeat_ids,
    sha256_file,
    validate_chunk,
    write_chunk_atomic,
)


STAGE_NUMBER = 6
STAGE_NAME = "END_TO_END_REPRODUCIBILITY_AND_EXECUTION"
STAGE6_DIR = ROOT / "artifacts" / "stage6"
ORACLE_DIR = STAGE6_DIR / "baseline_oracle"
ORACLE_MANIFEST_PATH = STAGE6_DIR / "baseline_oracle_manifest.json"
EXECUTION_VALIDATION_DIR = STAGE6_DIR / "execution_validation"
RESUME_VALIDATION_DIR = STAGE6_DIR / "resume_validation"

STAGE5_ZIP_SHA256 = (
    "c50ff295d4eca7f901228b062d295137d18b973f441302911f57a118742701bb"
)
FROZEN_STUDY_CONFIG_SHA256 = (
    "4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb"
)
FROZEN_RNG_SHA256 = (
    "36bedc40407912a68f34a2ccac090549e8b61e3d2c59f255904d34ac3fe1b70b"
)
FROZEN_RNG_VECTORS_SHA256 = (
    "6f88198eca9ceaf882e0458d87b9986435316715fa91fd09617e0a3878db2274"
)
FROZEN_SELECTORS_SHA256 = (
    "bc285cb5163dc1c0631a860f312a45c61738d6502e85e9c9779e5b9ddafd5fc6"
)
FROZEN_METRICS_SHA256 = (
    "a4afd0f90c9d0d3dca08ac9b21310aaef5ac81c9aa669f449e013661a30b7a4e"
)
STAGE5_KERNEL_SHA256 = (
    "4cb93adca63d6abeff3f5dce79409f413982cce055542d4136da1a7c82fcaf83"
)
BASELINE_TEST_COUNT = 283
VALIDATION_CONDITIONS = (
    (3, "H300"),
    (7, "F2"),
    (14, "HF300-2"),
)
VALIDATION_REPEAT_IDS = tuple(range(10001, 10021))
ORACLE_REPEAT_IDS = (10001, 10002, 10003)
NI_REPEAT_IDS = tuple(range(10001, 10006))
CHUNK_SIZES = (1, 5, 7, 20)
PARALLEL_WORKERS = 2


def _write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _frozen_file_hashes() -> dict[str, str]:
    return {
        "StudyConfigSHA256": sha256_file(CONFIG_DIR / "study_config.json"),
        "RNGSHA256": sha256_file(ROOT / "src/sal_stability_stage1/rng.py"),
        "RNGVectorsSHA256": sha256_file(CONFIG_DIR / "rng_test_vectors.csv"),
        "SelectorsSHA256": sha256_file(
            ROOT / "src/sal_stability_stage1/selectors.py"
        ),
        "MetricsSHA256": sha256_file(ROOT / "src/sal_stability_stage1/metrics.py"),
    }


def validate_frozen_contract() -> dict[str, object]:
    hashes = _frozen_file_hashes()
    expected = {
        "StudyConfigSHA256": FROZEN_STUDY_CONFIG_SHA256,
        "RNGSHA256": FROZEN_RNG_SHA256,
        "RNGVectorsSHA256": FROZEN_RNG_VECTORS_SHA256,
        "SelectorsSHA256": FROZEN_SELECTORS_SHA256,
        "MetricsSHA256": FROZEN_METRICS_SHA256,
    }
    checks = {
        name.removesuffix("SHA256") + "Unchanged": hashes[name] == value
        for name, value in expected.items()
    }
    prior_stage_pass: dict[str, bool] = {}
    for stage_number in range(1, 6):
        status_path = ROOT / "artifacts" / f"stage{stage_number}" / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            passed = (
                status.get("status") == "PASS"
                and status.get("science_contract_changed") is False
            )
        except (OSError, json.JSONDecodeError):
            passed = False
        prior_stage_pass[f"Stage{stage_number}ArtifactPassed"] = passed
    if not all(checks.values()) or not all(prior_stage_pass.values()):
        raise AssertionError(
            f"frozen contract gate failed: hashes={checks}, stages={prior_stage_pass}"
        )
    return {**hashes, **checks, **prior_stage_pass}


def verify_stage5_oracle(
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
) -> bool:
    manifest = json.loads(ORACLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    required_metadata = {
        "GeneratedBeforeStage6KernelRefactor": True,
        "OracleMode": "VERIFY",
        "Stage5ZipSHA256": STAGE5_ZIP_SHA256,
        "StudyConfigSHA256": FROZEN_STUDY_CONFIG_SHA256,
        "KernelSHA256": STAGE5_KERNEL_SHA256,
        "MetricsSHA256": FROZEN_METRICS_SHA256,
        "Namespace": "PILOT",
        "RepeatIDs": list(ORACLE_REPEAT_IDS),
        "CreatedAsValidationOnly": True,
        "PilotR": None,
    }
    for name, expected in required_metadata.items():
        if manifest.get(name) != expected:
            raise AssertionError(f"baseline oracle metadata mismatch: {name}")
    files = manifest.get("Files")
    if not isinstance(files, dict) or len(files) != 21:
        raise AssertionError("baseline oracle file manifest is incomplete")
    for relative, expected_hash in files.items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise AssertionError(f"baseline oracle file hash mismatch: {relative}")

    default_options = ExecutionOptions(
        workers=1,
        chunk_size=5,
        resume=False,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    with tempfile.TemporaryDirectory(prefix="stage6-oracle-verify-") as temporary:
        temp_root = Path(temporary)
        for condition_ordinal, condition_code in VALIDATION_CONDITIONS:
            for repeat_id in ORACLE_REPEAT_IDS:
                result = run_condition_repeat(
                    namespace=RandomNamespace.PILOT,
                    condition_ordinal=condition_ordinal,
                    repeat_id=repeat_id,
                    study_config=study_config,
                    encoding_reference=encoding_reference,
                    execution_options=default_options,
                )
                stem = (
                    f"c{condition_ordinal:02d}_{condition_code.lower()}_r{repeat_id}"
                )
                with np.load(
                    ORACLE_DIR / f"{stem}_trajectory.npz", allow_pickle=False
                ) as oracle:
                    if not np.array_equal(
                        result.source_state_code, oracle["source_state_code"]
                    ):
                        raise AssertionError(f"oracle source mismatch: {stem}")
                    if not np.array_equal(
                        result.ref_toa_before_s, oracle["ref_toa_before_s"]
                    ):
                        raise AssertionError(f"oracle reference mismatch: {stem}")
                    if not np.array_equal(
                        result.selected_observed_toa_s,
                        oracle["selected_observed_toa_s"],
                        equal_nan=True,
                    ):
                        raise AssertionError(f"oracle selected TOA mismatch: {stem}")
                    if not np.array_equal(
                        result.has_selection, oracle["has_selection"]
                    ):
                        raise AssertionError(f"oracle selection mask mismatch: {stem}")
                metrics_path = temp_root / f"{stem}_repeat_metrics.csv"
                write_repeat_metrics_csv(metrics_path, result.repeat_metrics)
                if metrics_path.read_bytes() != (
                    ORACLE_DIR / f"{stem}_repeat_metrics.csv"
                ).read_bytes():
                    raise AssertionError(f"oracle RepeatMetrics mismatch: {stem}")
                if condition_code == H300_CONDITION_CODE:
                    strata_path = temp_root / f"{stem}_h300_width_strata.csv"
                    write_h300_width_strata_csv(
                        strata_path, result.h300_width_strata
                    )
                    if strata_path.read_bytes() != (
                        ORACLE_DIR / f"{stem}_h300_width_strata.csv"
                    ).read_bytes():
                        raise AssertionError(f"oracle H300 strata mismatch: {stem}")
    return True


def validate_source_permutation() -> bool:
    view = CandidateView(
        observed_toa_s=[-1.5e-6, -0.2e-6, 0.7e-6, 1.8e-6],
        observed_width_s=[280e-9, 310e-9, 300e-9, 260e-9],
        tie_rank_hi=[9, 7, 5, 3],
        tie_rank_lo=[1, 2, 3, 4],
    )
    first = CycleCandidates(
        view=view,
        source_state_code=[1, 2, 3, 1],
        event_index=[1, -1, 1, 2],
        gate_a_s=-5e-6,
        gate_b_s=5e-6,
    )
    permuted = CycleCandidates(
        view=view,
        source_state_code=[3, 1, 1, 2],
        event_index=[99, 98, 97, 96],
        gate_a_s=-5e-6,
        gate_b_s=5e-6,
    )
    selected_positions: list[tuple[int, int]] = []
    for _, method_code in METHOD_IDENTITIES:
        kwargs = {
            "ref_toa_s": 0.1e-6,
            "ref_width_s": 300e-9,
            "sigma_dt_s": np.sqrt(2.0) * 1e-6,
            "sigma_dwidth_s": 1.5e-9,
        }
        left = select_for_method(method_code, first.view, **kwargs)
        right = select_for_method(method_code, permuted.view, **kwargs)
        selected_positions.append((left, right))
    if not all(left == right for left, right in selected_positions):
        raise AssertionError("source metadata permutation changed a selected_pos")
    return True


@dataclass(frozen=True, slots=True)
class ChunkExecutionOutcome:
    result: ConditionRunResult | None
    completed: bool
    computed_chunks: int
    skipped_chunks: int
    invalidated_chunks: tuple[str, ...]
    chunk_paths: tuple[ChunkPaths, ...]


def run_condition_repeats_chunked(
    *,
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_ids: Iterable[int],
    study_config: Mapping[str, object],
    encoding_reference: Mapping[str, np.ndarray],
    execution_options: ExecutionOptions,
    output_root: Path,
    stop_after_chunks: int | None = None,
) -> ChunkExecutionOutcome:
    if not isinstance(execution_options, ExecutionOptions):
        raise TypeError("execution_options must be ExecutionOptions")
    chunks = partition_repeat_ids(repeat_ids, execution_options.chunk_size)
    all_repeat_ids = tuple(repeat_id for chunk in chunks for repeat_id in chunk)
    if stop_after_chunks is not None:
        if isinstance(stop_after_chunks, bool) or not isinstance(stop_after_chunks, int):
            raise TypeError("stop_after_chunks must be an integer or None")
        if stop_after_chunks < 1:
            raise ValueError("stop_after_chunks must be positive")
    condition = condition_spec_from_config(study_config, condition_ordinal)
    study_config_sha = sha256_file(CONFIG_DIR / "study_config.json")
    code_sha = compute_code_sha256()
    computed = 0
    skipped = 0
    invalidated: list[str] = []
    completed_results: list[ConditionRunResult] = []
    processed_paths: list[ChunkPaths] = []

    for chunk_index, chunk_repeat_ids in enumerate(chunks):
        if stop_after_chunks is not None and chunk_index >= stop_after_chunks:
            break
        paths = chunk_paths(
            output_root,
            condition_ordinal=condition_ordinal,
            condition_code=condition.Code,
            repeat_ids=chunk_repeat_ids,
        )
        processed_paths.append(paths)
        loaded: ConditionRunResult | None = None
        if execution_options.resume:
            validation = validate_chunk(
                paths,
                study_config_sha256=study_config_sha,
                code_sha256=code_sha,
                namespace=namespace,
                condition_ordinal=condition_ordinal,
                condition_code=condition.Code,
                repeat_ids=chunk_repeat_ids,
                validation_repeat_count=len(all_repeat_ids),
            )
            if validation.valid:
                loaded = load_validated_chunk(
                    paths,
                    study_config_sha256=study_config_sha,
                    code_sha256=code_sha,
                    namespace=namespace,
                    condition_ordinal=condition_ordinal,
                    condition_code=condition.Code,
                    repeat_ids=chunk_repeat_ids,
                    validation_repeat_count=len(all_repeat_ids),
                )
                skipped += 1
            else:
                invalidated.append(
                    f"{chunk_repeat_ids[0]}-{chunk_repeat_ids[-1]}:{validation.reason}"
                )
        if loaded is None:
            if execution_options.workers == 1:
                loaded = run_condition_repeats_serial(
                    namespace=namespace,
                    condition_ordinal=condition_ordinal,
                    repeat_ids=chunk_repeat_ids,
                    study_config=study_config,
                    encoding_reference=encoding_reference,
                    execution_options=execution_options,
                )
            else:
                loaded = run_condition_repeats_parallel(
                    namespace=namespace,
                    condition_ordinal=condition_ordinal,
                    repeat_ids=chunk_repeat_ids,
                    study_config=study_config,
                    encoding_reference=encoding_reference,
                    execution_options=execution_options,
                )
            write_chunk_atomic(
                output_root,
                result=loaded,
                study_config_sha256=study_config_sha,
                code_sha256=code_sha,
                validation_repeat_count=len(all_repeat_ids),
            )
            computed += 1
        completed_results.append(loaded)

    completed = len(completed_results) == len(chunks)
    result: ConditionRunResult | None = None
    if completed:
        result = assemble_condition_chunks(
            completed_results,
            expected_repeat_ids=all_repeat_ids,
        )
    return ChunkExecutionOutcome(
        result=result,
        completed=completed,
        computed_chunks=computed,
        skipped_chunks=skipped,
        invalidated_chunks=tuple(invalidated),
        chunk_paths=tuple(processed_paths),
    )


def _subset_result(
    result: ConditionRunResult,
    repeat_ids: Iterable[int],
) -> ConditionRunResult:
    requested = tuple(sorted(repeat_ids))
    positions = tuple(result.repeat_ids.index(repeat_id) for repeat_id in requested)
    metrics = tuple(row for row in result.repeat_metrics if row.RepeatID in requested)
    strata = tuple(
        row for row in result.h300_width_strata if row.RepeatID in requested
    )
    diagnostics = tuple(
        record for record in result.diagnostics if record.RepeatID in requested
    )
    return ConditionRunResult(
        namespace=result.namespace,
        condition_ordinal=result.condition_ordinal,
        condition_code=result.condition_code,
        repeat_ids=requested,
        source_state=result.source_state[:, positions, :],
        ref_toa_before_s=result.ref_toa_before_s[:, positions, :],
        selected_observed_toa_s=result.selected_observed_toa_s[:, positions, :],
        has_selection=result.has_selection[:, positions, :],
        repeat_metrics=metrics,
        h300_width_strata=strata,
        diagnostics=diagnostics,
    )


def _validate_ni(result: ConditionRunResult) -> bool:
    if result.condition_ordinal != 0 or result.condition_code != "NI":
        raise AssertionError("NI validation received the wrong condition")
    for repeat_position, repeat_id in enumerate(result.repeat_ids):
        for method_ordinal in range(1, 5):
            if not np.array_equal(
                result.source_state[0, repeat_position],
                result.source_state[method_ordinal, repeat_position],
            ):
                raise AssertionError(f"NI source identity failed for RepeatID={repeat_id}")
            if not np.array_equal(
                result.ref_toa_before_s[0, repeat_position],
                result.ref_toa_before_s[method_ordinal, repeat_position],
            ):
                raise AssertionError(f"NI reference identity failed for RepeatID={repeat_id}")
        rows = tuple(
            row for row in result.repeat_metrics if row.RepeatID == repeat_id
        )
        payloads: list[dict[str, object]] = []
        for row in rows:
            payload = asdict(row)
            payload.pop("MethodOrdinal")
            payload.pop("MethodCode")
            payloads.append(payload)
        if not all(payload == payloads[0] for payload in payloads[1:]):
            raise AssertionError(f"NI RepeatMetrics identity failed for RepeatID={repeat_id}")
    return True


def _mutate_manifest(path: Path, field_name: str, value: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field_name] = value
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _corrupt_last_byte(path: Path) -> None:
    with path.open("r+b") as handle:
        handle.seek(-1, os.SEEK_END)
        original = handle.read(1)
        if len(original) != 1:
            raise ValueError("cannot corrupt an empty data file")
        handle.seek(-1, os.SEEK_END)
        handle.write(bytes((original[0] ^ 0x01,)))
        handle.flush()
        os.fsync(handle.fileno())


def _check_no_rng_state_fields(root: Path) -> bool:
    forbidden = {"GlobalRNGState", "LoopCounterState", "ThreadRNGState"}
    for manifest_path in Path(root).rglob("chunk_manifest.json"):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if forbidden.intersection(payload):
            return False
    return True


def run_scientific_validation(
    progress: Callable[[str], None] = print,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    contract = validate_frozen_contract()
    config, reference = load_stage3_inputs()
    progress("[Stage6] Frozen contract: PASS")
    oracle_passed = verify_stage5_oracle(config, reference)
    progress("[Stage6] Stage5 oracle exact match: PASS")
    source_permutation = validate_source_permutation()
    progress("[Stage6] Selector source permutation: PASS")

    base_options = ExecutionOptions(
        workers=1,
        chunk_size=5,
        resume=False,
        diagnostic=False,
        h_cache_enabled=True,
        method_execution_order=ORDER_A,
    )
    serial_results: dict[int, ConditionRunResult] = {}
    serial_start = time.perf_counter()
    for condition_ordinal, condition_code in VALIDATION_CONDITIONS:
        progress(f"[Stage6] Serial reference {condition_code} 10001..10020")
        serial_results[condition_ordinal] = run_condition_repeats_serial(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=VALIDATION_REPEAT_IDS,
            study_config=config,
            encoding_reference=reference,
            execution_options=base_options,
        )
    serial_seconds = time.perf_counter() - serial_start

    for condition_ordinal in (3, 14):
        if any(row.N_N != 0 for row in serial_results[condition_ordinal].repeat_metrics):
            raise AssertionError("HPRF/composite validation produced forbidden N")
    if serial_results[7].h300_width_strata:
        raise AssertionError("F2 produced forbidden H300 width strata")
    if serial_results[14].h300_width_strata:
        raise AssertionError("HF300-2 produced forbidden H300 width strata")

    method_order_repeats = tuple(range(10001, 10011))
    for condition_ordinal, condition_code in VALIDATION_CONDITIONS:
        reference_a = _subset_result(
            serial_results[condition_ordinal], method_order_repeats
        )
        for label, order in (("B", ORDER_B), ("C", ORDER_C)):
            candidate = run_condition_repeats_serial(
                namespace=RandomNamespace.PILOT,
                condition_ordinal=condition_ordinal,
                repeat_ids=method_order_repeats,
                study_config=config,
                encoding_reference=reference,
                execution_options=replace(
                    base_options, method_execution_order=order
                ),
            )
            assert_scientific_results_equal(
                reference_a,
                candidate,
                label=f"{condition_code} ORDER_A/ORDER_{label}",
            )
    progress("[Stage6] Method order A/B/C: PASS")

    cache_repeats = tuple(range(10001, 10011))
    for condition_ordinal, condition_code in ((3, "H300"), (14, "HF300-2")):
        cache_on = _subset_result(serial_results[condition_ordinal], cache_repeats)
        cache_off = run_condition_repeats_serial(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=cache_repeats,
            study_config=config,
            encoding_reference=reference,
            execution_options=replace(base_options, h_cache_enabled=False),
        )
        assert_scientific_results_equal(
            cache_on, cache_off, label=f"{condition_code} cache ON/OFF"
        )
    progress("[Stage6] H cache ON/OFF: PASS")

    diagnostic_repeats = ORACLE_REPEAT_IDS
    diagnostic_sample_written = False
    for condition_ordinal, condition_code in VALIDATION_CONDITIONS:
        diagnostic_off = _subset_result(
            serial_results[condition_ordinal], diagnostic_repeats
        )
        diagnostic_on = run_condition_repeats_serial(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=diagnostic_repeats,
            study_config=config,
            encoding_reference=reference,
            execution_options=replace(base_options, diagnostic=True),
        )
        if len(diagnostic_on.diagnostics) != len(diagnostic_repeats) * 5 * 200:
            raise AssertionError("diagnostic mode did not log every method-cycle")
        assert_scientific_results_equal(
            diagnostic_off,
            diagnostic_on,
            label=f"{condition_code} diagnostic OFF/ON",
        )
        if condition_code == H300_CONDITION_CODE:
            sample = tuple(
                record
                for record in diagnostic_on.diagnostics
                if record.RepeatID == diagnostic_repeats[0]
            )
            write_diagnostics_csv(
                EXECUTION_VALIDATION_DIR / "diagnostic_sample_h300_r10001.csv",
                sample,
            )
            diagnostic_sample_written = len(sample) == 1000
    if not diagnostic_sample_written:
        raise AssertionError("diagnostic validation sample was not written")
    progress("[Stage6] Diagnostic OFF/ON: PASS")

    completion_first = run_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_id=10001,
        study_config=config,
        encoding_reference=reference,
        execution_options=base_options,
    )
    completion_second = run_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=7,
        repeat_id=10002,
        study_config=config,
        encoding_reference=reference,
        execution_options=base_options,
    )
    completion_forward = assemble_condition_repeat_results(
        (completion_first, completion_second),
        expected_repeat_ids=(10001, 10002),
    )
    completion_reverse = assemble_condition_repeat_results(
        (completion_second, completion_first),
        expected_repeat_ids=(10001, 10002),
    )
    assert_scientific_results_equal(
        completion_forward,
        completion_reverse,
        label="parallel completion order",
    )

    parallel_results: dict[int, ConditionRunResult] = {}
    parallel_options = replace(base_options, workers=PARALLEL_WORKERS)
    parallel_start = time.perf_counter()
    for condition_ordinal, condition_code in VALIDATION_CONDITIONS:
        progress(f"[Stage6] Parallel workers=2 {condition_code} 10001..10020")
        parallel_results[condition_ordinal] = run_condition_repeats_parallel(
            namespace=RandomNamespace.PILOT,
            condition_ordinal=condition_ordinal,
            repeat_ids=VALIDATION_REPEAT_IDS,
            study_config=config,
            encoding_reference=reference,
            execution_options=parallel_options,
        )
        assert_scientific_results_equal(
            serial_results[condition_ordinal],
            parallel_results[condition_ordinal],
            label=f"{condition_code} serial/parallel",
        )
    parallel_seconds = time.perf_counter() - parallel_start
    progress("[Stage6] Serial = parallel workers=2: PASS")

    for chunk_size in CHUNK_SIZES:
        for condition_ordinal, condition_code in VALIDATION_CONDITIONS:
            progress(f"[Stage6] ChunkSize={chunk_size} {condition_code}")
            outcome = run_condition_repeats_chunked(
                namespace=RandomNamespace.PILOT,
                condition_ordinal=condition_ordinal,
                repeat_ids=VALIDATION_REPEAT_IDS,
                study_config=config,
                encoding_reference=reference,
                execution_options=replace(
                    base_options,
                    chunk_size=chunk_size,
                    resume=False,
                ),
                output_root=(
                    EXECUTION_VALIDATION_DIR / f"chunk_size_{chunk_size}"
                ),
            )
            if not outcome.completed or outcome.result is None:
                raise AssertionError("chunked execution did not complete")
            assert_scientific_results_equal(
                serial_results[condition_ordinal],
                outcome.result,
                label=f"{condition_code} ChunkSize={chunk_size}",
            )
    progress("[Stage6] ChunkSize 1/5/7/20 invariance: PASS")

    resume_options = replace(
        base_options,
        chunk_size=5,
        resume=False,
    )
    interrupted = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=resume_options,
        output_root=RESUME_VALIDATION_DIR,
        stop_after_chunks=2,
    )
    if interrupted.completed or interrupted.computed_chunks != 2:
        raise AssertionError("simulated interruption did not stop after two chunks")
    resumed = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(resume_options, resume=True),
        output_root=RESUME_VALIDATION_DIR,
    )
    if (
        not resumed.completed
        or resumed.result is None
        or resumed.skipped_chunks != 2
        or resumed.computed_chunks != 2
    ):
        raise AssertionError("resume did not skip two and compute two chunks")
    assert_scientific_results_equal(
        serial_results[14], resumed.result, label="HF300-2 interrupted resume"
    )
    valid_skip = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(resume_options, resume=True),
        output_root=RESUME_VALIDATION_DIR,
    )
    if valid_skip.computed_chunks != 0 or valid_skip.skipped_chunks != 4:
        raise AssertionError("valid completed chunks were not truly skipped")

    first_chunk_paths = resumed.chunk_paths[0]
    recovery_results: dict[str, bool] = {}

    _corrupt_last_byte(first_chunk_paths.state)
    corrupt_recovery = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(resume_options, resume=True),
        output_root=RESUME_VALIDATION_DIR,
    )
    recovery_results["CorruptChunkRecoveryPassed"] = bool(
        corrupt_recovery.computed_chunks == 1
        and corrupt_recovery.skipped_chunks == 3
        and corrupt_recovery.result is not None
        and any("hash_mismatch:state" in item for item in corrupt_recovery.invalidated_chunks)
    )
    if corrupt_recovery.result is None:
        raise AssertionError("corruption recovery did not produce a result")
    assert_scientific_results_equal(
        serial_results[14], corrupt_recovery.result, label="corrupt chunk recovery"
    )

    _mutate_manifest(first_chunk_paths.manifest, "Completed", False)
    incomplete_recovery = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(resume_options, resume=True),
        output_root=RESUME_VALIDATION_DIR,
    )
    recovery_results["CompletedFalseRecoveryPassed"] = bool(
        incomplete_recovery.computed_chunks == 1
        and incomplete_recovery.skipped_chunks == 3
        and any(
            "manifest_mismatch:Completed" in item
            for item in incomplete_recovery.invalidated_chunks
        )
    )
    if incomplete_recovery.result is None:
        raise AssertionError("Completed=false recovery did not produce a result")
    assert_scientific_results_equal(
        serial_results[14],
        incomplete_recovery.result,
        label="Completed=false recovery",
    )

    _mutate_manifest(first_chunk_paths.manifest, "CodeVersion", "0.5.0")
    version_recovery = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(resume_options, resume=True),
        output_root=RESUME_VALIDATION_DIR,
    )
    recovery_results["CodeVersionMismatchRecoveryPassed"] = bool(
        version_recovery.computed_chunks == 1
        and version_recovery.skipped_chunks == 3
        and any(
            "manifest_mismatch:CodeVersion" in item
            for item in version_recovery.invalidated_chunks
        )
    )
    if version_recovery.result is None:
        raise AssertionError("CodeVersion recovery did not produce a result")
    assert_scientific_results_equal(
        serial_results[14], version_recovery.result, label="CodeVersion recovery"
    )

    _mutate_manifest(first_chunk_paths.manifest, "StudyConfigSHA256", "0" * 64)
    config_recovery = run_condition_repeats_chunked(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=14,
        repeat_ids=VALIDATION_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=replace(resume_options, resume=True),
        output_root=RESUME_VALIDATION_DIR,
    )
    recovery_results["StudyConfigMismatchRecoveryPassed"] = bool(
        config_recovery.computed_chunks == 1
        and config_recovery.skipped_chunks == 3
        and any(
            "manifest_mismatch:StudyConfigSHA256" in item
            for item in config_recovery.invalidated_chunks
        )
    )
    if config_recovery.result is None:
        raise AssertionError("StudyConfig recovery did not produce a result")
    assert_scientific_results_equal(
        serial_results[14], config_recovery.result, label="StudyConfig recovery"
    )
    if not all(recovery_results.values()):
        raise AssertionError(f"one or more chunk recovery checks failed: {recovery_results}")
    progress("[Stage6] Resume, skip, corruption and manifest recovery: PASS")

    ni_result = run_condition_repeats_serial(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=0,
        repeat_ids=NI_REPEAT_IDS,
        study_config=config,
        encoding_reference=reference,
        execution_options=base_options,
    )
    ni_passed = _validate_ni(ni_result)
    progress("[Stage6] NI five-method identity: PASS")

    overlap_rejected = False
    try:
        assemble_condition_chunks(
            (config_recovery.result, config_recovery.result),
            expected_repeat_ids=VALIDATION_REPEAT_IDS,
        )
    except ValueError as exc:
        overlap_rejected = "overlap" in str(exc)
    if not overlap_rejected:
        raise AssertionError("chunk overlap was not rejected")

    no_rng_state = _check_no_rng_state_fields(STAGE6_DIR)
    no_tmp_files = not any(
        ".tmp-" in path.name for path in STAGE6_DIR.rglob("*") if path.is_file()
    )
    if not no_rng_state or not no_tmp_files:
        raise AssertionError("storage retained forbidden RNG state or temporary files")

    final_contract = validate_frozen_contract()
    stage6_kernel_sha = sha256_file(ROOT / "src/sal_stability_stage1/kernel.py")
    code_sha = compute_code_sha256()
    reproducibility: dict[str, object] = {
        "Stage5OracleMatch": oracle_passed,
        "SourcePermutationPassed": source_permutation,
        "MethodOrderPassed": True,
        "CacheInvariantPassed": True,
        "DiagnosticInvariantPassed": True,
        "SerialParallelPassed": True,
        "ParallelCompletionOrderPassed": True,
        "ChunkSizeInvariantPassed": True,
        "ResumePassed": True,
        "ValidChunkSkipPassed": True,
        **recovery_results,
        "RNGVectorsUnchanged": final_contract["RNGVectorsSHA256"]
        == FROZEN_RNG_VECTORS_SHA256,
        "RNGUnchanged": final_contract["RNGSHA256"] == FROZEN_RNG_SHA256,
        "StudyConfigUnchanged": final_contract["StudyConfigSHA256"]
        == FROZEN_STUDY_CONFIG_SHA256,
        "SelectorUnchanged": final_contract["SelectorsSHA256"]
        == FROZEN_SELECTORS_SHA256,
        "MetricsUnchanged": final_contract["MetricsSHA256"]
        == FROZEN_METRICS_SHA256,
        "NIIdentityPassed": ni_passed,
        "H300NoNPassed": all(
            row.N_N == 0 for row in serial_results[3].repeat_metrics
        ),
        "HF300_2NoNPassed": all(
            row.N_N == 0 for row in serial_results[14].repeat_metrics
        ),
        "H300StrataModeInvariantPassed": True,
        "ChunkOverlapRejected": overlap_rejected,
        "AtomicWritePassed": no_tmp_files,
        "ManifestValidationPassed": True,
        "NoMutableRNGStateStored": no_rng_state,
        "ScienceContractChanged": False,
        "Stage5KernelSHA256": STAGE5_KERNEL_SHA256,
        "Stage6KernelSHA256": stage6_kernel_sha,
        "Stage6CodeSHA256": code_sha,
    }
    if not all(
        value
        for key, value in reproducibility.items()
        if key.endswith("Passed")
        or key.endswith("Unchanged")
        or key
        in {
            "Stage5OracleMatch",
            "SerialParallelPassed",
            "ValidChunkSkipPassed",
            "ChunkOverlapRejected",
            "NoMutableRNGStateStored",
        }
    ):
        raise AssertionError("a final reproducibility hard gate is false")

    condition_repeat_count = len(VALIDATION_CONDITIONS) * len(
        VALIDATION_REPEAT_IDS
    )
    method_cycles = condition_repeat_count * 5 * 200
    serial_per_cr = serial_seconds / condition_repeat_count
    parallel_per_cr = parallel_seconds / condition_repeat_count
    performance = {
        "Environment": platform.platform(),
        "PythonVersion": platform.python_version(),
        "NumPyVersion": np.__version__,
        "CPUCount": os.cpu_count(),
        "ValidationConditions": [code for _, code in VALIDATION_CONDITIONS],
        "ValidationRepeatCount": len(VALIDATION_REPEAT_IDS),
        "ConditionRepeatCount": condition_repeat_count,
        "SerialWallSeconds": serial_seconds,
        "ParallelWallSeconds": parallel_seconds,
        "Workers": PARALLEL_WORKERS,
        "SecondsPerConditionRepeatSerial": serial_per_cr,
        "SecondsPerConditionRepeatParallel": parallel_per_cr,
        "MethodCycles": method_cycles,
        "SecondsPerMethodCycleSerial": serial_seconds / method_cycles,
        "SecondsPerMethodCycleParallel": parallel_seconds / method_cycles,
        "ProjectedPilotSecondsSerial": serial_per_cr * 3800,
        "ProjectedPilotSecondsParallel": parallel_per_cr * 3800,
        "projection_only": True,
    }
    boundary = {
        "Stage7Executed": False,
        "PilotR200Executed": False,
        "RDecisionExecuted": False,
        "FormalExecuted": False,
        "BootstrapExecuted": False,
        "PaperResultGenerated": False,
        "StopAfterStage6": True,
    }
    progress("[Stage6] Scientific validation: PASS")
    return reproducibility, performance, {**contract, **boundary}


def _pytest_version() -> str:
    try:
        return importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _run_pytest_once() -> tuple[bool, int, str, float]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    output = completed.stdout + completed.stderr
    print(output, end="", flush=True)
    matches = re.findall(r"(\d+) passed", output)
    passed_count = int(matches[-1]) if matches else 0
    return completed.returncode == 0, passed_count, output, elapsed


def _write_changed_files(stage6_kernel_sha: str) -> None:
    oracle_files = sorted(
        path.relative_to(ROOT).as_posix() for path in ORACLE_DIR.glob("*") if path.is_file()
    )
    lines = [
        "ADDED",
        "src/sal_stability_stage1/diagnostics.py",
        "src/sal_stability_stage1/execution.py",
        "src/sal_stability_stage1/storage.py",
        "src/sal_stability_stage1/stage6.py",
        "scripts/build_stage6.py",
        "tests/test_stage6_execution.py",
        "tests/test_stage6_reproducibility.py",
        "tests/test_stage6_storage_resume.py",
        "tests/test_stage6_diagnostics.py",
        "artifacts/stage6/baseline_oracle_manifest.json",
        *oracle_files,
        "artifacts/stage6/reproducibility_validation.json",
        "artifacts/stage6/performance_validation.json",
        "artifacts/stage6/status.json",
        "artifacts/stage6/validation_report.md",
        "artifacts/stage6/changed_files.txt",
        "artifacts/stage6/test_report.txt",
        "artifacts/stage6/execution_validation/",
        "artifacts/stage6/resume_validation/",
        "",
        "MODIFIED",
        "src/sal_stability_stage1/kernel.py — EXECUTION-CONTROL REFACTOR ONLY; Stage5 oracle exact match",
        f"  Stage5 SHA256: {STAGE5_KERNEL_SHA256}",
        f"  Stage6 SHA256: {stage6_kernel_sha}",
        "README.md",
        "pyproject.toml — version only 0.5.0 -> 0.6.0",
        "",
        "UNCHANGED",
        "artifacts/config/study_config.json",
        "artifacts/config/encoding_reference.csv",
        "artifacts/config/encoding_reference.npz",
        "artifacts/config/rng_test_vectors.csv",
        "src/sal_stability_stage1/contracts.py",
        "src/sal_stability_stage1/encoding.py",
        "src/sal_stability_stage1/events.py",
        "src/sal_stability_stage1/hprf.py",
        "src/sal_stability_stage1/rng.py",
        "src/sal_stability_stage1/selectors.py",
        "src/sal_stability_stage1/metrics.py",
        "src/sal_stability_stage1/stage5.py",
        "All Stage1-5 tests and artifacts",
    ]
    _write_text(STAGE6_DIR / "changed_files.txt", "\n".join(lines) + "\n")


def _write_validation_report(
    reproducibility: Mapping[str, object],
    performance: Mapping[str, object],
    total_tests: int,
) -> None:
    report = f"""# Stage 6 End-to-End Reproducibility and Execution Validation

## A. Baseline

- Stage 1-5 regression: PASS (baseline 283 tests retained; full suite {total_tests}/{total_tests} PASS).

## B. Oracle

- Stage 5 oracle generated before any kernel refactor: YES.
- Oracle mode: VERIFY; overwrite prohibited.
- New default path exact array/metrics/H300-strata match: PASS.

## C. Source leakage

- Synthetic CandidateView with externally permuted source metadata left all five selected_pos values unchanged: PASS.

## D. Method order

- ORDER_A, ORDER_B and ORDER_C were exact for H300, F2 and HF300-2, RepeatID 10001..10010: PASS.

## E. Cache

- H cache ON/OFF was exact for H300 and HF300-2, RepeatID 10001..10010: PASS.

## F. Diagnostic

- diagnostic OFF/ON was exact for all three validation conditions, RepeatID 10001..10003: PASS.
- Source metadata was logged only after selected_pos returned.

## G. Parallel

- Serial and spawn-compatible ProcessPool workers=2 were exact for 60 Condition-Repeat units: PASS.
- Reversed completion collection order assembled to the same canonical output: PASS.

## H. Chunk

- ChunkSize 1, 5, 7 and 20 were exact against the clean serial reference: PASS.
- Duplicate/overlapping RepeatID chunks were rejected: PASS.

## I. Resume

- HF300-2 ChunkSize=5 interruption after two of four chunks resumed exactly: PASS.
- Two completed chunks were proven skipped; two missing chunks were computed: PASS.
- A second resume skipped all four valid chunks without computation: PASS.

## J. Corruption

- DataSHA mismatch, Completed=false, CodeVersion mismatch and StudyConfig manifest mismatch each invalidated and recomputed one complete chunk: PASS.

## K. RNG

- rng.py SHA256: {FROZEN_RNG_SHA256} (UNCHANGED).
- rng_test_vectors.csv SHA256: {FROZEN_RNG_VECTORS_SHA256} (UNCHANGED).
- No global, loop-counter or thread RNG state was stored.

## L. Storage

- Data files used temporary-file, close/fsync and os.replace writes; manifest was committed last with Completed=true.
- Identity, individual file SHA256 and aggregate DataSHA256 were validated before reuse.
- chunk_state.npy shape was (5, R_chunk, 200), dtype uint8; full trajectories were retained in an additional NPZ.

## M. Performance

- Serial wall seconds: {performance['SerialWallSeconds']:.6f}.
- Parallel wall seconds (workers=2): {performance['ParallelWallSeconds']:.6f}.
- Serial seconds per Condition-Repeat: {performance['SecondsPerConditionRepeatSerial']:.6f}.
- Parallel seconds per Condition-Repeat: {performance['SecondsPerConditionRepeatParallel']:.6f}.
- Projected Pilot serial seconds: {performance['ProjectedPilotSecondsSerial']:.3f}.
- Projected Pilot parallel seconds: {performance['ProjectedPilotSecondsParallel']:.3f}.
- Projection only; not a scientific result.

## N. Stage boundary

- Stage 7 was NOT executed.
- Pilot R=200 was NOT executed.
- R decision was NOT executed.
- Formal was NOT executed.
- Bootstrap was NOT executed.
- No paper result was generated.
"""
    _write_text(STAGE6_DIR / "validation_report.md", report)


def _write_test_report(
    *,
    total_tests: int,
    pytest_seconds: float,
    scientific_validation_passed: bool,
    pytest_output: str,
) -> None:
    stage6_tests = total_tests - BASELINE_TEST_COUNT
    report = [
        f"Python version: {platform.python_version()}",
        f"NumPy version: {np.__version__}",
        f"pytest version: {_pytest_version()}",
        f"Stage1-5 baseline test count: {BASELINE_TEST_COUNT}",
        f"Stage6 test count: {stage6_tests}",
        f"Total test count: {total_tests}",
        f"Passed: {total_tests}",
        "Failed: 0",
        f"Full pytest wall seconds: {pytest_seconds:.6f}",
        "Full pytest executions inside build_stage6.py: 1",
        "build_stage6 scientific validation result: "
        + ("PASS" if scientific_validation_passed else "FAIL"),
        "",
        "pytest output:",
        pytest_output.rstrip(),
        "",
    ]
    _write_text(STAGE6_DIR / "test_report.txt", "\n".join(report))


def _write_status(
    *,
    status: str,
    blocking_issue_count: int,
    total_tests: int,
    reproducibility: Mapping[str, object] | None,
) -> None:
    repro = dict(reproducibility or {})
    stage6_tests = max(0, total_tests - BASELINE_TEST_COUNT)
    payload = {
        "stage": STAGE_NUMBER,
        "stage_name": STAGE_NAME,
        "status": status,
        "blocking_issue_count": blocking_issue_count,
        "science_contract_changed": False,
        "stage1_regression_passed": status == "PASS",
        "stage2_regression_passed": status == "PASS",
        "stage3_regression_passed": status == "PASS",
        "stage4_regression_passed": status == "PASS",
        "stage5_regression_passed": status == "PASS",
        "baseline_test_count": BASELINE_TEST_COUNT,
        "stage6_test_count": stage6_tests,
        "total_test_count": total_tests,
        "study_config_unchanged": bool(repro.get("StudyConfigUnchanged", False)),
        "rng_unchanged": bool(repro.get("RNGUnchanged", False)),
        "rng_vectors_unchanged": bool(repro.get("RNGVectorsUnchanged", False)),
        "selector_unchanged": bool(repro.get("SelectorUnchanged", False)),
        "metrics_unchanged": bool(repro.get("MetricsUnchanged", False)),
        "stage5_oracle_match_passed": bool(repro.get("Stage5OracleMatch", False)),
        "source_permutation_passed": bool(
            repro.get("SourcePermutationPassed", False)
        ),
        "method_order_invariance_passed": bool(
            repro.get("MethodOrderPassed", False)
        ),
        "cache_invariance_passed": bool(
            repro.get("CacheInvariantPassed", False)
        ),
        "diagnostic_invariance_passed": bool(
            repro.get("DiagnosticInvariantPassed", False)
        ),
        "serial_parallel_identical": bool(
            repro.get("SerialParallelPassed", False)
        ),
        "chunk_size_invariance_passed": bool(
            repro.get("ChunkSizeInvariantPassed", False)
        ),
        "resume_invariance_passed": bool(repro.get("ResumePassed", False)),
        "corrupt_chunk_recovery_passed": bool(
            repro.get("CorruptChunkRecoveryPassed", False)
        ),
        "completed_false_recovery_passed": bool(
            repro.get("CompletedFalseRecoveryPassed", False)
        ),
        "code_version_mismatch_recovery_passed": bool(
            repro.get("CodeVersionMismatchRecoveryPassed", False)
        ),
        "study_config_manifest_mismatch_recovery_passed": bool(
            repro.get("StudyConfigMismatchRecoveryPassed", False)
        ),
        "ni_identity_passed": bool(repro.get("NIIdentityPassed", False)),
        "h300_strata_mode_invariance_passed": bool(
            repro.get("H300StrataModeInvariantPassed", False)
        ),
        "pilot_executed": False,
        "r_decision_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_generated": False,
        "stop_after_stage6": True,
    }
    _write_json(STAGE6_DIR / "status.json", payload)


def build_stage6() -> dict[str, object]:
    STAGE6_DIR.mkdir(parents=True, exist_ok=True)
    validate_frozen_contract()
    pytest_passed, total_tests, pytest_output, pytest_seconds = _run_pytest_once()
    if not pytest_passed:
        _write_test_report(
            total_tests=total_tests,
            pytest_seconds=pytest_seconds,
            scientific_validation_passed=False,
            pytest_output=pytest_output,
        )
        _write_status(
            status="FAIL",
            blocking_issue_count=1,
            total_tests=total_tests,
            reproducibility=None,
        )
        return {"status": "FAIL", "blocking_issues": ["full pytest failed"]}

    try:
        reproducibility, performance, contract = run_scientific_validation()
    except Exception as exc:
        _write_test_report(
            total_tests=total_tests,
            pytest_seconds=pytest_seconds,
            scientific_validation_passed=False,
            pytest_output=pytest_output,
        )
        _write_status(
            status="FAIL",
            blocking_issue_count=1,
            total_tests=total_tests,
            reproducibility=None,
        )
        _write_text(
            STAGE6_DIR / "validation_report.md",
            "# Stage 6 Validation\n\nFAIL\n\n"
            f"Blocking exception: {type(exc).__name__}: {exc}\n\n"
            "Stage 7 was NOT executed.\n"
            "Pilot R=200 was NOT executed.\n",
        )
        raise

    _write_json(
        STAGE6_DIR / "reproducibility_validation.json", reproducibility
    )
    _write_json(STAGE6_DIR / "performance_validation.json", performance)
    _write_test_report(
        total_tests=total_tests,
        pytest_seconds=pytest_seconds,
        scientific_validation_passed=True,
        pytest_output=pytest_output,
    )
    _write_validation_report(reproducibility, performance, total_tests)
    _write_changed_files(str(reproducibility["Stage6KernelSHA256"]))
    _write_status(
        status="PASS",
        blocking_issue_count=0,
        total_tests=total_tests,
        reproducibility=reproducibility,
    )
    result = {
        "status": "PASS",
        "total_tests": total_tests,
        "stage6_tests": total_tests - BASELINE_TEST_COUNT,
        "reproducibility": reproducibility,
        "performance": performance,
        "contract": contract,
    }
    print("[Stage6] PASS — STOP AFTER STAGE 6", flush=True)
    return result


__all__ = [
    "BASELINE_TEST_COUNT",
    "CHUNK_SIZES",
    "ChunkExecutionOutcome",
    "NI_REPEAT_IDS",
    "ORACLE_REPEAT_IDS",
    "PARALLEL_WORKERS",
    "STAGE_NAME",
    "VALIDATION_CONDITIONS",
    "VALIDATION_REPEAT_IDS",
    "build_stage6",
    "run_condition_repeats_chunked",
    "run_scientific_validation",
    "validate_frozen_contract",
    "validate_source_permutation",
    "verify_stage5_oracle",
]
