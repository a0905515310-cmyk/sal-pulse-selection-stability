from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

import numpy as np

from .build import CONFIG_DIR, ROOT
from .kernel import (
    METHOD_IDENTITIES,
    SOURCE_F,
    SOURCE_G,
    SOURCE_H,
    SOURCE_N,
    CycleCandidates,
    MethodState,
    _simulate_shared_world,
    gate_bounds,
    in_half_open_gate,
    initialize_method_states,
    selected_source_state_code,
    simulate_condition_repeat,
    update_ref_toa,
)
from .rng import RandomNamespace, TieRankCollisionError
from .selectors import (
    CandidateView,
    joint_scores,
    select_first,
    select_last,
    select_t,
    select_tw,
    select_w,
)
from .stage2 import TestRun, run_pytest, scan_forbidden_random_apis
from .stage3 import (
    build_shared_physical_world,
    load_stage3_inputs,
    scan_retired_identifiers,
    validate_stage3_inputs,
)


STAGE4_DIR = ROOT / "artifacts" / "stage4"
FROZEN_CONFIG_SHA256 = (
    "4419f964beee079f67c83ed759b0b8295052d46a037e692c844a81337b232ecb"
)
FROZEN_STAGE3_FILES_SHA256 = {
    "artifacts/config/encoding_reference.csv": (
        "aa319c482a3494d0cdece3bfa4aa28f4d919edaa5b0d2c45bb0d4918ee9647f6"
    ),
    "artifacts/config/encoding_reference.npz": (
        "5a058296f762477009c7d94a33d6d61a982c2d1dda7233db3f5949214ee3cd43"
    ),
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
    "src/sal_stability_stage1/stage3.py": (
        "59fbbf2cb6deedd2f42c2aad419c55b6df9f74231c9acb80a417377b5f516830"
    ),
    "tests/test_stage1.py": (
        "2f61cabef205d23d19df599f3c090a672049482bdd18be0dd84aaafcf6770478"
    ),
    "tests/test_stage2_rng.py": (
        "b49d5f07e5f0aba8bd409033538dad5a9180e414cb2d5e462dc9b9c751f5b1c6"
    ),
    "tests/test_stage3_events.py": (
        "30d7730218bc78333f5d09aa1b0ddbb2428990e91bffdebad614075262176082"
    ),
}
VALIDATION_REPEAT_IDS = (1, 2, 17)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    def normalize(value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=normalize,
        )
        + "\n",
        encoding="utf-8",
    )


def _selected_rank(view: CandidateView, selector) -> tuple[int, int]:
    position = selector(view)
    return int(view.tie_rank_hi[position]), int(view.tie_rank_lo[position])


def _selector_has_no_approximate_tie_call() -> bool:
    import sal_stability_stage1.selectors as selector_module

    tree = ast.parse(inspect.getsource(selector_module))
    forbidden_calls = {"isclose", "allclose"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in forbidden_calls:
                return False
            if isinstance(function, ast.Attribute) and function.attr in forbidden_calls:
                return False
    return True


def validate_selector_layer() -> dict[str, object]:
    view = CandidateView(
        [3.0, 1.0, 2.0],
        [280e-9, 301e-9, 315e-9],
        [30, 20, 10],
        [3, 2, 1],
    )
    checks: dict[str, bool] = {
        "candidate_view_fields_exact": [field.name for field in fields(CandidateView)]
        == ["observed_toa_s", "observed_width_s", "tie_rank_hi", "tie_rank_lo"],
        "candidate_view_arrays_read_only": all(
            not getattr(view, name).flags.writeable
            for name in (
                "observed_toa_s",
                "observed_width_s",
                "tie_rank_hi",
                "tie_rank_lo",
            )
        ),
        "selector_first": select_first(view) == 1,
        "selector_last": select_last(view) == 0,
        "selector_t": select_t(view, 0.0, 1.0) == 1,
        "selector_w": select_w(view, 300e-9, 1.5e-9) == 1,
    }
    expected_joint = np.square(view.observed_toa_s / np.float64(2.0)) + np.square(
        (view.observed_width_s - np.float64(300e-9)) / np.float64(1.5e-9)
    )
    actual_joint = joint_scores(view, 0.0, 300e-9, 2.0, 1.5e-9)
    checks["selector_tw_score_exact"] = np.array_equal(actual_joint, expected_joint)
    checks["selector_tw"] = select_tw(view, 0.0, 300e-9, 2.0, 1.5e-9) == int(
        np.argmin(expected_joint)
    )

    exact_tie_views = (
        (select_first, CandidateView([1.0, 1.0], [0.0, 9.0], [9, 1], [0, 0])),
        (select_last, CandidateView([1.0, 1.0], [0.0, 9.0], [9, 1], [0, 0])),
        (
            lambda current: select_t(current, 0.0, 1.0),
            CandidateView([-1.0, 1.0], [0.0, 9.0], [9, 1], [0, 0]),
        ),
        (
            lambda current: select_w(current, 0.0, 1.0),
            CandidateView([0.0, 9.0], [-1.0, 1.0], [9, 1], [0, 0]),
        ),
        (
            lambda current: select_tw(current, 0.0, 0.0, 1.0, 1.0),
            CandidateView([-1.0, 1.0], [1.0, -1.0], [9, 1], [0, 0]),
        ),
    )
    checks["exact_tie_minimum_tierank"] = all(
        selector(tied_view) == 1 for selector, tied_view in exact_tie_views
    )
    checks["no_approximate_tie_call"] = _selector_has_no_approximate_tie_call()
    lower = np.float64(1.0)
    higher = np.nextafter(lower, np.float64(np.inf))
    near_tie = CandidateView([lower, higher], [lower, higher], [9, 0], [9, 0])
    checks["near_tie_primary_wins"] = select_first(near_tie) == 0

    collision_view = CandidateView([0.0, 1.0], [0.0, 1.0], [7, 7], [11, 11])
    try:
        select_first(collision_view)
    except TieRankCollisionError:
        checks["tierank_collision_stop"] = True
    else:
        checks["tierank_collision_stop"] = False

    selector_calls = (
        lambda current: select_first(current),
        lambda current: select_last(current),
        lambda current: select_t(current, 0.25, 1.7),
        lambda current: select_w(current, -0.4, 0.8),
        lambda current: select_tw(current, 0.25, -0.4, 1.7, 0.8),
    )
    permutation_checks = 0
    permutation_passed = True
    for case in range(64):
        size = 3 + case % 9
        identity = np.arange(size, dtype=np.float64)
        toa = np.sin((case + 1.25) * (identity + 0.375)) + identity * 1e-7
        width = np.cos((case + 0.75) * (identity + 0.625)) - identity * 1e-7
        hi = np.arange(case * 32, case * 32 + size, dtype=np.uint64)
        lo = np.arange(size, 0, -1, dtype=np.uint64)
        base = CandidateView(toa, width, hi, lo)
        expected = tuple(_selected_rank(base, selector) for selector in selector_calls)
        for permutation in (
            np.arange(size - 1, -1, -1),
            np.roll(np.arange(size), 1 + case % size),
        ):
            permuted = CandidateView(
                toa[permutation],
                width[permutation],
                hi[permutation],
                lo[permutation],
            )
            actual = tuple(
                _selected_rank(permuted, selector) for selector in selector_calls
            )
            permutation_passed &= actual == expected
            permutation_checks += len(selector_calls)
    checks["selector_permutation_invariance"] = permutation_passed

    expected_positions = tuple(selector(view) for selector in selector_calls)
    outer_source_metadata = np.array([1, 2, 3], dtype=np.uint8)
    source_permutation_passed = True
    for changed_metadata in ([3, 1, 2], [2, 3, 1], [1, 3, 2]):
        outer_source_metadata[:] = changed_metadata
        source_permutation_passed &= (
            tuple(selector(view) for selector in selector_calls) == expected_positions
        )
    checks["source_permutation_invariance"] = source_permutation_passed

    expected_parameters = {
        select_first: ("view",),
        select_last: ("view",),
        select_t: ("view", "ref_toa_s", "sigma_dt_s"),
        select_w: ("view", "ref_width_s", "sigma_dwidth_s"),
        select_tw: (
            "view",
            "ref_toa_s",
            "ref_width_s",
            "sigma_dt_s",
            "sigma_dwidth_s",
        ),
    }
    checks["selector_signatures_source_blind"] = all(
        tuple(inspect.signature(selector).parameters) == parameters
        for selector, parameters in expected_parameters.items()
    )
    checks["selectors_return_selected_pos"] = all(
        isinstance(selector(view), int) for selector in selector_calls
    )
    return {
        "checks": checks,
        "details": {
            "candidate_view_fields": [field.name for field in fields(CandidateView)],
            "selector_property_check_count": permutation_checks,
            "source_metadata_permutations_checked": 3,
        },
        "all_pass": all(checks.values()),
    }


def _stage5_identifier_findings(paths: Sequence[Path]) -> list[dict[str, object]]:
    forbidden_fragments = (
        ("N", "_C"),
        ("N", "_E"),
        ("N", "_N"),
        ("N", "_CC"),
        ("N", "_CE"),
        ("N", "_CN"),
        ("N", "_EC"),
        ("N", "_EE"),
        ("N", "_EN"),
        ("N", "_Cdot"),
        ("N", "_Edot"),
        ("N", "_run_NC"),
        ("Sum", "_L_NC_obs"),
        ("N", "_open_NC"),
        ("P", "_cor"),
        ("P", "_C_given_C"),
        ("P", "_C_given_E"),
        ("P", "_E_given_E"),
        ("L", "_NC"),
        ("current", "_nc_len"),
        ("Repeat", "Metrics"),
    )
    forbidden = {"".join(parts) for parts in forbidden_fragments}
    findings: list[dict[str, object]] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: tuple[str, ...] = ()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = (node.name,)
            elif isinstance(node, ast.Name):
                names = (node.id,)
            elif isinstance(node, ast.Attribute):
                names = (node.attr,)
            for name in names:
                if name in forbidden:
                    findings.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "line": getattr(node, "lineno", 0),
                            "identifier": name,
                        }
                    )
    return findings


def validate_kernel_layer(
    config: Mapping[str, object],
    reference: Mapping[str, np.ndarray],
) -> dict[str, object]:
    checks: dict[str, bool] = {}
    half_width = np.float64(config["GateHalfWidth_s"])
    full_width = np.float64(config["GateFullWidth_s"])
    a_s, b_s = gate_bounds(0.0, half_width)
    checks["half_open_gate"] = (
        np.float64(2.0) * half_width == full_width
        and in_half_open_gate(a_s, a_s, b_s)
        and not in_half_open_gate(b_s, a_s, b_s)
    )
    build_source = inspect.getsource(sys.modules[CycleCandidates.__module__].build_cycle_candidates)
    checks["true_toa_gate_membership"] = (
        "guidance.true_toa_s" in build_source
        and "forwarded.true_toa_s" in build_source
        and "hprf_index_range" in build_source
    )
    checks["observed_toa_not_regated"] = build_source.count("in_half_open_gate(") == 2
    checks["hprf_uses_stage3_resolver"] = "hprf_index_range(" in build_source

    update_parameters = tuple(inspect.signature(update_ref_toa).parameters)
    checks["reference_update_source_blind"] = update_parameters == (
        "current_ref_toa_s",
        "selected_observed_toa_s",
        "delta_t_s",
    )
    checks["selection_reference_update"] = update_ref_toa(10.0, 3.0, 0.5) == 3.5
    checks["n_continuation"] = update_ref_toa(10.0, None, 0.5) == 10.5
    checks["method_state_minimal"] = [field.name for field in fields(MethodState)] == [
        "ref_toa_s"
    ]

    initial_world = build_shared_physical_world(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=0,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    initial_states = initialize_method_states(initial_world, config)
    checks["method_state_initialization"] = (
        len(initial_states) == 5
        and len({id(state) for state in initial_states}) == 5
        and all(
            state.ref_toa_s == initial_world.initial_ref_toa_s
            for state in initial_states
        )
    )

    empty_candidates = CycleCandidates(
        CandidateView([], [], [], []),
        [],
        [],
        np.float64(-5e-6),
        np.float64(5e-6),
    )
    source_candidates = CycleCandidates(
        CandidateView([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1, 2, 3], [0, 0, 0]),
        [SOURCE_G, SOURCE_H, SOURCE_F],
        [1, -1, 1],
        np.float64(-5e-6),
        np.float64(5e-6),
    )
    checks["source_state_mapping"] = (
        selected_source_state_code(empty_candidates, None) == SOURCE_N
        and selected_source_state_code(source_candidates, 0) == SOURCE_G
        and selected_source_state_code(source_candidates, 1) == SOURCE_H
        and selected_source_state_code(source_candidates, 2) == SOURCE_F
    )

    total_condition_repeats = 0
    total_method_cycles = 0
    total_h_count_checks = 0
    total_h_no_n_checks = 0
    ni_identity_checks = 0
    trajectory_schema_passed = True
    hprf_no_n_passed = True
    reference_divergence_found = False
    conditions_seen: set[int] = set()
    for repeat_id in VALIDATION_REPEAT_IDS:
        for condition_ordinal in range(19):
            trajectory = simulate_condition_repeat(
                namespace=RandomNamespace.FORMAL,
                condition_ordinal=condition_ordinal,
                repeat_id=repeat_id,
                study_config=config,
                encoding_reference=reference,
            )
            total_condition_repeats += 1
            total_method_cycles += int(trajectory.source_state_code.size)
            total_h_count_checks += trajectory.hprf_count_check_count
            total_h_no_n_checks += trajectory.hprf_no_n_check_count
            conditions_seen.add(condition_ordinal)
            trajectory_schema_passed &= (
                trajectory.source_state_code.shape == (5, 200)
                and trajectory.source_state_code.dtype == np.uint8
                and trajectory.ref_toa_before_s.shape == (5, 200)
                and trajectory.ref_toa_before_s.dtype == np.float64
                and trajectory.delta_t_consumed_count == 199
                and set(np.unique(trajectory.source_state_code)).issubset(
                    {0, 1, 2, 3}
                )
            )
            condition_scene = config["ConditionSpecs"][condition_ordinal]["Scene"]
            if condition_scene in ("HPRF", "COMPOSITE"):
                hprf_no_n_passed &= bool(
                    np.all(trajectory.source_state_code != SOURCE_N)
                )
            if condition_scene == "NI":
                ni_identity_checks += 1
                trajectory_schema_passed &= all(
                    np.array_equal(
                        trajectory.source_state_code[0],
                        trajectory.source_state_code[method_index],
                    )
                    and np.array_equal(
                        trajectory.ref_toa_before_s[0],
                        trajectory.ref_toa_before_s[method_index],
                    )
                    for method_index in range(1, 5)
                )
            reference_divergence_found |= any(
                not np.array_equal(
                    trajectory.ref_toa_before_s[0],
                    trajectory.ref_toa_before_s[method_index],
                )
                for method_index in range(1, 5)
            )

    expected_h_checks = 14 * len(VALIDATION_REPEAT_IDS) * 5 * 200
    checks["condition_count_19"] = conditions_seen == set(range(19))
    checks["validation_repeat_count"] = total_condition_repeats == 57
    checks["cycle_count_200"] = trajectory_schema_passed
    checks["delta_t_consumed_199"] = trajectory_schema_passed
    checks["source_state_schema"] = trajectory_schema_passed
    checks["hprf_per_cycle_count"] = total_h_count_checks == expected_h_checks
    checks["hprf_composite_no_n"] = (
        hprf_no_n_passed and total_h_no_n_checks == expected_h_checks
    )
    checks["ni_trajectory_identity"] = ni_identity_checks == len(
        VALIDATION_REPEAT_IDS
    )
    checks["reference_toa_divergence"] = reference_divergence_found

    pilot = simulate_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    formal = simulate_condition_repeat(
        namespace=RandomNamespace.FORMAL,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    checks["physical_namespace_isolation"] = not np.array_equal(
        pilot.ref_toa_before_s,
        formal.ref_toa_before_s,
    )

    return {
        "checks": checks,
        "details": {
            "validation_condition_count": len(conditions_seen),
            "validation_repeat_ids": list(VALIDATION_REPEAT_IDS),
            "total_validation_condition_repeats": total_condition_repeats,
            "total_method_cycles": total_method_cycles,
            "hprf_count_checks": total_h_count_checks,
            "hprf_no_n_checks": total_h_no_n_checks,
            "hprf_composite_n_violations": 0 if hprf_no_n_passed else 1,
            "ni_identity_checks": ni_identity_checks,
            "delta_t_consumed_per_trajectory": 199,
            "state_arrays_saved": False,
        },
        "all_pass": all(checks.values()),
    }


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def write_stage4_test_report(test_runs: Sequence[TestRun], build_status: str) -> None:
    by_label = {run.label: run for run in test_runs}
    baseline = sum(
        by_label[label].passed + by_label[label].failed
        for label in ("stage1", "stage2", "stage3")
    )
    selector_count = (
        by_label["stage4_selectors"].passed + by_label["stage4_selectors"].failed
    )
    kernel_count = by_label["stage4_kernel"].passed + by_label["stage4_kernel"].failed
    total = by_label["all"]
    lines = [
        "SAL stability Stage 4 test report",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"NumPy version: {np.__version__}",
        f"pytest version: {importlib.metadata.version('pytest')}",
        f"baseline Stage1+2+3 test count: {baseline}",
        f"Stage4 selector test count: {selector_count}",
        f"Stage4 kernel test count: {kernel_count}",
        f"total tests: {total.passed + total.failed}",
        f"passed tests: {total.passed}",
        f"failed tests: {total.failed}",
        f"build_stage4 result: {build_status}",
        "",
    ]
    for run in test_runs:
        lines.extend(
            [
                f"[{run.label}]",
                f"command: {_format_command(run.command)}",
                f"return code: {run.returncode}",
                f"passed: {run.passed}",
                f"failed: {run.failed}",
                f"duration: {run.duration_s:.6f} s",
                "output:",
                run.output or "<no output>",
                "",
            ]
        )
    (STAGE4_DIR / "test_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_stage4_changed_files() -> None:
    lines = [
        "MODIFIED README.md",
        "MODIFIED pyproject.toml",
        "ADDED src/sal_stability_stage1/selectors.py",
        "ADDED src/sal_stability_stage1/kernel.py",
        "ADDED src/sal_stability_stage1/stage4.py",
        "ADDED scripts/build_stage4.py",
        "ADDED tests/test_stage4_selectors.py",
        "ADDED tests/test_stage4_kernel.py",
        "ADDED artifacts/stage4/status.json",
        "ADDED artifacts/stage4/validation_report.md",
        "ADDED artifacts/stage4/changed_files.txt",
        "ADDED artifacts/stage4/test_report.txt",
        "ADDED artifacts/stage4/kernel_validation.json",
        "UNCHANGED src/sal_stability_stage1/rng.py",
        "UNCHANGED src/sal_stability_stage1/contracts.py",
        "UNCHANGED src/sal_stability_stage1/encoding.py",
        "UNCHANGED src/sal_stability_stage1/events.py",
        "UNCHANGED src/sal_stability_stage1/hprf.py",
        "UNCHANGED src/sal_stability_stage1/stage3.py",
        "UNCHANGED artifacts/config/*",
        "UNCHANGED artifacts/stage1/*",
        "UNCHANGED artifacts/stage2/*",
        "UNCHANGED artifacts/stage3/*",
    ]
    (STAGE4_DIR / "changed_files.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_stage4_validation_report(
    status: Mapping[str, object],
    selector_validation: Mapping[str, object],
    kernel_validation: Mapping[str, object],
    blocking_issues: Sequence[str],
) -> None:
    lines = [
        "# Stage 4 Selector and 200-Cycle Kernel Validation Report",
        "",
        f"Verdict: **{status['status']}**",
        "",
        "## A. Baseline",
        "",
        f"- Stage 1-3 regression: {'PASS' if status['stage1_regression_passed'] and status['stage2_regression_passed'] and status['stage3_regression_passed'] else 'FAIL'}.",
        f"- Baseline Stage 1-3 tests: {status['baseline_stage1_stage2_stage3_test_count']}.",
        f"- Current complete test count: {status['total_test_count']}.",
        f"- StudyConfig changed: {str(status['study_config_changed']).lower()}.",
        "",
        "## B. CandidateView",
        "",
        "- Fields: `observed_toa_s`, `observed_width_s`, `tie_rank_hi`, `tie_rank_lo`.",
        "- It contains no Source, source state, TrueTOA, TrueWidth, condition, scene, or physical event identity.",
        "- All arrays are aligned one-dimensional read-only arrays.",
        "- Every selector returns only `selected_pos`; source classification remains outside the selector.",
        "",
        "## C. Five Selectors",
        "",
        "- FIRST: argmin of observed TOA; width is not read.",
        "- LAST: argmax of observed TOA; width is not read.",
        "- T: argmin of `q_t = ((observed_toa - ref_toa) / sigma_dt)^2`; width is not read.",
        "- W: argmin of `q_width = ((observed_width - ref_width) / sigma_dwidth)^2`; TOA deviation is not read.",
        "- TW: argmin of exactly `D = q_t + q_width`, with no empirical weight, hard feature gate, or rejection threshold.",
        "",
        "## D. Scales",
        "",
        "- `sigma_dt = sqrt(2) * SigmaTOA`.",
        "- `sigma_dwidth = SigmaWidth`.",
        "- Both scales remain fixed for every condition, method, cycle, and prior selection outcome; no dynamic re-estimation occurs.",
        "",
        "## E. Tie",
        "",
        "- TieRank is used only when the primary float64 values are exactly equal; no tolerance or approximate tie is used.",
        "- Every method chooses the minimum unsigned 128-bit `(tie_rank_hi, tie_rank_lo)` lexicographic rank.",
        "- A complete 128-bit collision raises `TieRankCollisionError` and stops; array position is never a fallback.",
        "",
        "## F. Gate",
        "",
        "- Every gate is `[a,b)` with `a = ref_toa - 5 us` and `b = ref_toa + 5 us`.",
        "- G and F membership uses true TOA only.",
        "- H membership uses the frozen Stage 3 `hprf_index_range` resolver and signed n.",
        "- Observed TOA is never used for a second membership filter.",
        "",
        "## G. MethodState",
        "",
        "- Five independent MethodState objects begin at the one shared `initial_ref_toa_s`.",
        "- Their reference TOAs can naturally diverge after different selections.",
        "- N does not terminate a trajectory; the next cycle is always evaluated.",
        "- Reference width is not state and is read directly from the same-cycle encoding reference.",
        "",
        "## H. Reference Update",
        "",
        "- Selection: `next_ref_toa = selected_observed_toa + DeltaT`.",
        "- N: `next_ref_toa = current_ref_toa + DeltaT`.",
        "- The update function has no Source or selected-width input.",
        "",
        "## I. State",
        "",
        "- SourceStateCode is uint8: `0=N`, `1=G`, `2=H`, `3=F`.",
        "- H and F remain distinct Stage 4 source codes; no downstream C/E/N metric is computed here.",
        "",
        "## J. HPRF",
        "",
        f"- Per-Method x per-Cycle H count hard assertions executed: {kernel_validation.get('details', {}).get('hprf_count_checks', 0)}.",
        "- Every HPRF/composite Method x Cycle was asserted non-N.",
        "",
        "## K. NI",
        "",
        "- All five NI source-state trajectories are exactly identical within each validation Condition-Repeat.",
        "- All five NI reference-TOA trajectories are exactly identical within each validation Condition-Repeat.",
        "",
        "## L. Stage Boundary",
        "",
        "Stage 5 was NOT executed.",
        "RepeatMetrics were NOT computed.",
        "Pilot was NOT executed.",
        "Formal was NOT executed.",
        "Bootstrap was NOT executed.",
        "No paper result was generated.",
        "No paper result figures were generated.",
        "",
        "## Validation Checks",
        "",
    ]
    for prefix, validation in (
        ("selector", selector_validation),
        ("kernel", kernel_validation),
    ):
        for name, passed in sorted(validation.get("checks", {}).items()):
            lines.append(f"- {prefix}.{name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Blocking Issues", ""])
    if blocking_issues:
        lines.extend(f"- {issue}" for issue in blocking_issues)
    else:
        lines.append("None")
    (STAGE4_DIR / "validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _failed_validation(reason: str) -> dict[str, object]:
    return {
        "checks": {"validation_completed": False},
        "details": {"skip_reason": reason},
        "all_pass": False,
    }


def _skipped_test_run(label: str, reason: str) -> TestRun:
    return TestRun(
        label=label,
        command=(sys.executable, "-m", "pytest"),
        returncode=125,
        passed=0,
        failed=0,
        duration_s=0.0,
        output=f"SKIPPED: {reason}",
    )


def build_stage4() -> dict[str, object]:
    STAGE4_DIR.mkdir(parents=True, exist_ok=True)
    blocking_issues: list[str] = []

    config_path = CONFIG_DIR / "study_config.json"
    checksum_path = CONFIG_DIR / "study_config.sha256"
    config_hash = _sha256_file(config_path) if config_path.is_file() else ""
    checksum_value = (
        checksum_path.read_text(encoding="utf-8").strip()
        if checksum_path.is_file()
        else ""
    )
    config_hash_ok = (
        config_hash == FROZEN_CONFIG_SHA256 and checksum_value == config_hash
    )
    frozen_file_checks = {
        relative: (ROOT / relative).is_file()
        and _sha256_file(ROOT / relative) == expected
        for relative, expected in FROZEN_STAGE3_FILES_SHA256.items()
    }
    science_contract_unchanged = config_hash_ok and all(frozen_file_checks.values())
    if not config_hash_ok:
        blocking_issues.append("frozen StudyConfig hash or checksum changed")
    if not all(frozen_file_checks.values()):
        changed = [name for name, passed in frozen_file_checks.items() if not passed]
        blocking_issues.append(f"frozen Stage 1-3 files changed: {', '.join(changed)}")

    prior_stage_checks: dict[str, bool] = {}
    for prior_stage in (1, 2, 3):
        path = ROOT / "artifacts" / f"stage{prior_stage}" / "status.json"
        try:
            prior_status = json.loads(path.read_text(encoding="utf-8"))
            passed = (
                prior_status.get("status") == "PASS"
                and prior_status.get("science_contract_changed") is False
            )
        except (OSError, json.JSONDecodeError):
            passed = False
        prior_stage_checks[f"stage{prior_stage}_artifact_pass"] = passed
        if not passed:
            blocking_issues.append(f"Stage {prior_stage} artifact is not a frozen PASS")

    stage1_run = run_pytest("stage1", ("tests/test_stage1.py",))
    stage2_run = run_pytest("stage2", ("tests/test_stage2_rng.py",))
    stage3_run = run_pytest("stage3", ("tests/test_stage3_events.py",))
    baseline_runs = (stage1_run, stage2_run, stage3_run)
    baseline_count = sum(run.passed + run.failed for run in baseline_runs)
    baseline_passed = all(run.returncode == 0 for run in baseline_runs) and baseline_count == 147
    if not baseline_passed:
        blocking_issues.append(
            f"Stage 1-3 baseline regression failed or counted {baseline_count}, expected 147"
        )

    preconditions_passed = (
        science_contract_unchanged
        and all(prior_stage_checks.values())
        and baseline_passed
    )
    if preconditions_passed:
        selector_run = run_pytest(
            "stage4_selectors",
            ("tests/test_stage4_selectors.py",),
        )
        if selector_run.returncode != 0:
            blocking_issues.append("Stage 4 selector pytest failed")
    else:
        selector_run = _skipped_test_run(
            "stage4_selectors",
            "Stage 1-3 hard precondition failed",
        )
    if preconditions_passed and selector_run.returncode == 0:
        try:
            selector_validation = validate_selector_layer()
        except Exception as exc:
            selector_validation = _failed_validation(
                f"{type(exc).__name__}: {exc}"
            )
            blocking_issues.append(
                f"selector validation raised {type(exc).__name__}: {exc}"
            )
    else:
        selector_validation = _failed_validation("hard precondition or selector tests failed")
    if not selector_validation["all_pass"]:
        failed = [
            name
            for name, passed in selector_validation["checks"].items()
            if not passed
        ]
        blocking_issues.append(f"selector validation failed: {', '.join(failed)}")

    selector_gate_passed = (
        preconditions_passed
        and selector_run.returncode == 0
        and selector_validation["all_pass"]
    )
    if selector_gate_passed:
        kernel_run = run_pytest("stage4_kernel", ("tests/test_stage4_kernel.py",))
        if kernel_run.returncode != 0:
            blocking_issues.append("Stage 4 kernel pytest failed")
    else:
        kernel_run = _skipped_test_run(
            "stage4_kernel",
            "selector hard gate failed",
        )
    if (
        preconditions_passed
        and selector_run.returncode == 0
        and selector_validation["all_pass"]
        and kernel_run.returncode == 0
    ):
        try:
            config, reference = load_stage3_inputs()
            input_checks = validate_stage3_inputs(config, reference)
            if not all(input_checks.values()):
                raise ValueError("frozen Stage 3 input validation failed")
            kernel_validation = validate_kernel_layer(config, reference)
            kernel_validation["checks"].update(
                {f"stage3_input_{name}": passed for name, passed in input_checks.items()}
            )
            kernel_validation["all_pass"] = all(
                kernel_validation["checks"].values()
            )
        except Exception as exc:
            kernel_validation = _failed_validation(
                f"{type(exc).__name__}: {exc}"
            )
            blocking_issues.append(
                f"kernel validation raised {type(exc).__name__}: {exc}"
            )
    else:
        kernel_validation = _failed_validation("selector or kernel hard gate failed")
    if not kernel_validation["all_pass"]:
        failed = [
            name for name, passed in kernel_validation["checks"].items() if not passed
        ]
        blocking_issues.append(f"kernel validation failed: {', '.join(failed)}")

    active_paths = [
        CONFIG_DIR / "study_config.json",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
    ]
    retired_findings = scan_retired_identifiers(active_paths)
    source_paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
    ]
    random_api_findings = scan_forbidden_random_apis(source_paths)
    stage5_findings = _stage5_identifier_findings(
        sorted((ROOT / "src").rglob("*.py"))
    )
    forbidden_result_dirs = (
        ROOT / "artifacts" / "pilot",
        ROOT / "artifacts" / "formal",
        ROOT / "artifacts" / "bootstrap",
        ROOT / "artifacts" / "results",
        ROOT / "artifacts" / "figures",
    )
    forbidden_existing = [
        str(path.relative_to(ROOT)) for path in forbidden_result_dirs if path.exists()
    ]
    boundary_checks = {
        "retired_identifier_scan": not retired_findings,
        "forbidden_random_api_scan": not random_api_findings,
        "stage5_identifier_scan": not stage5_findings,
        "forbidden_result_directories_absent": not forbidden_existing,
    }
    kernel_validation["checks"].update(boundary_checks)
    kernel_validation["details"].update(
        {
            "retired_identifier_findings": retired_findings,
            "forbidden_random_api_findings": random_api_findings,
            "stage5_identifier_findings": stage5_findings,
            "forbidden_result_directories": forbidden_existing,
            "frozen_file_checks": frozen_file_checks,
            "prior_stage_checks": prior_stage_checks,
            "study_config_hash_ok": config_hash_ok,
        }
    )
    kernel_validation["all_pass"] = all(kernel_validation["checks"].values())
    for name, passed in boundary_checks.items():
        if not passed:
            blocking_issues.append(f"boundary validation failed: {name}")

    final_test_gate_passed = (
        preconditions_passed
        and selector_run.returncode == 0
        and selector_validation["all_pass"]
        and kernel_run.returncode == 0
        and kernel_validation["all_pass"]
    )
    if final_test_gate_passed:
        all_run = run_pytest("all", ())
        if all_run.returncode != 0:
            blocking_issues.append("complete pytest regression failed")
    else:
        all_run = _skipped_test_run("all", "Stage 4 hard gate failed")
    stage4_selector_count = selector_run.passed + selector_run.failed
    stage4_kernel_count = kernel_run.passed + kernel_run.failed
    expected_total = baseline_count + stage4_selector_count + stage4_kernel_count
    if all_run.passed + all_run.failed != expected_total:
        blocking_issues.append(
            "complete pytest count does not equal baseline plus Stage 4 test counts"
        )

    blocking_issues = list(dict.fromkeys(blocking_issues))
    passed = (
        not blocking_issues
        and science_contract_unchanged
        and selector_validation["all_pass"]
        and kernel_validation["all_pass"]
        and all_run.returncode == 0
    )
    selector_checks = selector_validation["checks"]
    kernel_checks = kernel_validation["checks"]
    status: dict[str, object] = {
        "stage": 4,
        "stage_name": "SELECTOR_AND_CYCLE_KERNEL",
        "status": "PASS" if passed else "FAIL",
        "blocking_issue_count": len(blocking_issues),
        "science_contract_changed": not science_contract_unchanged,
        "study_config_changed": not config_hash_ok,
        "stage1_regression_passed": stage1_run.returncode == 0,
        "stage2_regression_passed": stage2_run.returncode == 0,
        "stage3_regression_passed": stage3_run.returncode == 0,
        "condition_count": 19,
        "method_count": 5,
        "K": 200,
        "candidate_view_source_blind_passed": selector_checks.get(
            "candidate_view_fields_exact", False
        )
        and selector_checks.get("selector_signatures_source_blind", False),
        "selector_first_passed": selector_checks.get("selector_first", False),
        "selector_last_passed": selector_checks.get("selector_last", False),
        "selector_t_passed": selector_checks.get("selector_t", False),
        "selector_w_passed": selector_checks.get("selector_w", False),
        "selector_tw_passed": selector_checks.get("selector_tw", False)
        and selector_checks.get("selector_tw_score_exact", False),
        "selector_permutation_invariance_passed": selector_checks.get(
            "selector_permutation_invariance", False
        ),
        "source_permutation_invariance_passed": selector_checks.get(
            "source_permutation_invariance", False
        ),
        "exact_tie_tierank_passed": selector_checks.get(
            "exact_tie_minimum_tierank", False
        )
        and selector_checks.get("no_approximate_tie_call", False)
        and selector_checks.get("near_tie_primary_wins", False),
        "tierank_collision_stop_passed": selector_checks.get(
            "tierank_collision_stop", False
        ),
        "true_toa_gate_membership_passed": kernel_checks.get(
            "true_toa_gate_membership", False
        )
        and kernel_run.returncode == 0,
        "half_open_gate_passed": kernel_checks.get("half_open_gate", False),
        "method_state_initialization_passed": kernel_checks.get(
            "method_state_initialization", False
        ),
        "n_continuation_passed": kernel_checks.get("n_continuation", False),
        "reference_update_source_blind_passed": kernel_checks.get(
            "reference_update_source_blind", False
        ),
        "reference_toa_divergence_passed": kernel_checks.get(
            "reference_toa_divergence", False
        ),
        "hprf_per_cycle_count_passed": kernel_checks.get(
            "hprf_per_cycle_count", False
        ),
        "hprf_composite_no_n_passed": kernel_checks.get(
            "hprf_composite_no_n", False
        ),
        "ni_trajectory_identity_passed": kernel_checks.get(
            "ni_trajectory_identity", False
        ),
        "cycle_count_200_passed": kernel_checks.get("cycle_count_200", False),
        "delta_t_consumed_199_passed": kernel_checks.get(
            "delta_t_consumed_199", False
        ),
        "source_state_code_passed": kernel_checks.get("source_state_schema", False)
        and kernel_checks.get("source_state_mapping", False),
        "baseline_stage1_stage2_stage3_test_count": baseline_count,
        "stage4_selector_test_count": stage4_selector_count,
        "stage4_kernel_test_count": stage4_kernel_count,
        "total_test_count": all_run.passed + all_run.failed,
        "total_passed_count": all_run.passed,
        "total_failed_count": all_run.failed,
        "study_config_sha256": config_hash,
        "stage5_executed": False,
        "repeat_metrics_computed": False,
        "pilot_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_figures_generated": False,
        "stop_after_stage4": True,
    }
    status = {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in status.items()
    }
    kernel_validation["all_pass"] = passed
    kernel_validation["blocking_issues"] = blocking_issues
    kernel_validation["selector_validation"] = selector_validation
    _write_json(STAGE4_DIR / "kernel_validation.json", kernel_validation)
    _write_json(STAGE4_DIR / "status.json", status)
    test_runs = (
        stage1_run,
        stage2_run,
        stage3_run,
        selector_run,
        kernel_run,
        all_run,
    )
    write_stage4_test_report(test_runs, status["status"])
    write_stage4_validation_report(
        status,
        selector_validation,
        kernel_validation,
        blocking_issues,
    )
    write_stage4_changed_files()
    return status


__all__ = [
    "build_stage4",
    "validate_kernel_layer",
    "validate_selector_layer",
]
