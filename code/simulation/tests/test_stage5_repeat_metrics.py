from __future__ import annotations

import csv
import hashlib
import inspect
from dataclasses import fields, replace

import numpy as np
import pytest

import sal_stability_stage1.stage5 as stage5_module
from sal_stability_stage1.kernel import (
    ConditionRepeatTrajectory,
    METHOD_IDENTITIES,
    simulate_condition_repeat,
    simulate_shared_world,
)
from sal_stability_stage1.metrics import (
    EFFECTIVE_INTERVAL_COUNT,
    K,
    TRANSITION_COUNT_FIELDS,
    compute_repeat_metrics,
    reference_repeat_counts,
    repeat_metrics_from_trajectory,
    streaming_repeat_counts,
)
from sal_stability_stage1.rng import RandomNamespace
from sal_stability_stage1.stage3 import build_shared_physical_world, load_stage3_inputs
from sal_stability_stage1.stage5 import (
    H300WidthStratumMetrics,
    WIDTH_STRATA,
    WIDTH_STRATUM_0NS,
    WIDTH_STRATUM_10NS,
    WIDTH_STRATUM_GE20NS,
    compute_h300_width_strata,
    level_strata_match_physical_width_difference,
    simulate_condition_repeat_with_metrics,
    width_strata_from_levels,
    write_h300_width_strata_csv,
    write_repeat_metrics_csv,
)


def _source(prefix, *, suffix_code=1):
    values = list(prefix)
    if len(values) > K:
        raise ValueError("prefix is longer than 200")
    values.extend([suffix_code] * (K - len(values)))
    return np.asarray(values, dtype=np.uint8)


def _metrics(source, repeat_id=1):
    return compute_repeat_metrics(
        source_state_code_1d=source,
        condition_ordinal=0,
        condition_code="NI",
        method_ordinal=0,
        method_code="FIRST",
        repeat_id=repeat_id,
    )


def _trajectory(source_by_method, condition_ordinal=0, condition_code="NI"):
    source = np.asarray(source_by_method, dtype=np.uint8)
    has_selection = source != 0
    selected = np.where(has_selection, 0.0, np.nan).astype(np.float64)
    return ConditionRepeatTrajectory(
        condition_ordinal=condition_ordinal,
        condition_code=condition_code,
        repeat_id=1,
        namespace=RandomNamespace.PILOT,
        method_codes=tuple(code for _, code in METHOD_IDENTITIES),
        source_state_code=source,
        ref_toa_before_s=np.zeros((5, K), dtype=np.float64),
        selected_observed_toa_s=selected,
        has_selection=has_selection,
        delta_t_consumed_count=EFFECTIVE_INTERVAL_COUNT,
        hprf_count_check_count=0,
        hprf_no_n_check_count=0,
    )


def test_all_c_repeat_counts_and_end_state():
    row = _metrics(np.ones(K, dtype=np.uint8))
    assert (row.N_C, row.N_E, row.N_N) == (200, 0, 0)
    assert row.N_CC == 199
    assert sum(getattr(row, name) for name in TRANSITION_COUNT_FIELDS[1:]) == 0
    assert (row.N_Cdot, row.N_Edot, row.N_Ndot) == (199, 0, 0)
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (0, 0, 0)
    assert row.EndState == "C"


def test_all_h_repeat_counts_one_open_nc_run():
    row = _metrics(np.full(K, 2, dtype=np.uint8))
    assert (row.N_C, row.N_E, row.N_N) == (0, 200, 0)
    assert row.N_EE == 199
    assert (row.N_Cdot, row.N_Edot, row.N_Ndot) == (0, 199, 0)
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (1, 200, 1)
    assert row.EndState == "E"
    assert (row.N_E_H, row.N_E_F) == (200, 0)


def test_all_f_is_the_same_evaluation_state_but_a_distinct_error_source():
    row = _metrics(np.full(K, 3, dtype=np.uint8))
    assert (row.N_E, row.N_EE, row.N_Edot) == (200, 199, 199)
    assert (row.N_E_H, row.N_E_F) == (0, 200)
    assert row.EndState == "E"


def test_all_n_repeat_counts_one_open_nc_run():
    row = _metrics(np.zeros(K, dtype=np.uint8))
    assert (row.N_C, row.N_E, row.N_N) == (0, 0, 200)
    assert row.N_NN == 199
    assert (row.N_Cdot, row.N_Edot, row.N_Ndot) == (0, 0, 199)
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (1, 200, 1)
    assert row.EndState == "N"


def test_controlled_sequence_exercises_all_nine_transition_directions():
    source = _source([1, 1, 2, 1, 0, 2, 2, 0, 0, 1])
    row = _metrics(source)
    assert all(getattr(row, name) >= 1 for name in TRANSITION_COUNT_FIELDS)
    assert sum(getattr(row, name) for name in TRANSITION_COUNT_FIELDS) == 199
    assert row.N_Cdot == row.N_CC + row.N_CE + row.N_CN
    assert row.N_Edot == row.N_EC + row.N_EE + row.N_EN
    assert row.N_Ndot == row.N_NC + row.N_NE + row.N_NN


def test_e_e_n_e_c_is_one_nc_run_of_length_four():
    row = _metrics(_source([2, 2, 0, 2, 1]))
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (1, 4, 0)


def test_c_e_n_e_c_keeps_n_inside_one_nc_run_and_counts_directions():
    row = _metrics(_source([1, 2, 0, 2, 1]))
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (1, 3, 0)
    assert row.N_CE == 1
    assert row.N_EN == 1
    assert row.N_NE == 1
    assert row.N_EC == 1


def test_right_edge_e_n_e_is_retained_without_a_cycle_201():
    source = np.asarray([1] * 197 + [2, 0, 2], dtype=np.uint8)
    row = _metrics(source)
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (1, 3, 1)
    assert row.EndState == "E"
    assert sum(getattr(row, name) for name in TRANSITION_COUNT_FIELDS) == 199


def test_left_edge_e_n_does_not_require_a_preceding_c():
    row = _metrics(_source([2, 0, 1]))
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (1, 2, 0)


def test_multiple_maximal_nc_runs_are_counted_separately():
    row = _metrics(_source([1, 2, 1, 0, 2, 1, 2, 2, 1]))
    assert (row.N_run_NC, row.Sum_L_NC_obs, row.N_open_NC) == (3, 5, 0)


def test_h_and_f_are_unified_for_core_transitions():
    row = _metrics(_source([1, 2, 3, 1]))
    assert (row.N_CE, row.N_EE, row.N_EC) == (1, 1, 1)
    assert (row.N_E_H, row.N_E_F) == (1, 1)


@pytest.mark.parametrize(
    "source",
    (
        np.ones(K, dtype=np.uint8),
        np.zeros(K, dtype=np.uint8),
        np.full(K, 2, dtype=np.uint8),
        np.resize(np.array([1, 2, 0, 3, 1, 0], dtype=np.uint8), K),
        np.asarray([1] * 197 + [2, 0, 3], dtype=np.uint8),
    ),
)
def test_streaming_and_independent_reference_reconstruction_are_identical(source):
    assert streaming_repeat_counts(
        source_state_code_1d=source
    ) == reference_repeat_counts(source_state_code_1d=source)


def test_end_state_origin_count_identity_is_hard_enforced():
    row = _metrics(np.ones(K, dtype=np.uint8))
    with pytest.raises(ValueError, match="origin"):
        replace(row, N_Cdot=198, N_CC=198, N_NN=1)


def test_nc_length_conservation_is_hard_enforced():
    row = _metrics(_source([1, 2, 0, 2, 1]))
    with pytest.raises(ValueError, match="Sum_L_NC_obs"):
        replace(row, Sum_L_NC_obs=row.Sum_L_NC_obs - 1)


def test_repeat_metrics_from_trajectory_preserves_frozen_method_order():
    source = np.vstack(
        [
            np.ones(K, dtype=np.uint8),
            np.full(K, 2, dtype=np.uint8),
            np.full(K, 3, dtype=np.uint8),
            np.zeros(K, dtype=np.uint8),
            np.resize(np.array([1, 2, 3], dtype=np.uint8), K),
        ]
    )
    trajectory = _trajectory(source)
    before = trajectory.source_state_code.copy()
    rows = repeat_metrics_from_trajectory(trajectory)
    assert len(rows) == 5
    assert tuple((row.MethodOrdinal, row.MethodCode) for row in rows) == METHOD_IDENTITIES
    assert [row.EndState for row in rows] == ["C", "E", "E", "N", "E"]
    np.testing.assert_array_equal(trajectory.source_state_code, before)


def test_repeat_metrics_rejects_a_float_count_even_when_integral():
    row = _metrics(np.ones(K, dtype=np.uint8))
    with pytest.raises(TypeError):
        replace(row, N_C=200.0)


def test_public_shared_world_wrapper_does_not_change_private_cycle_kernel_source():
    import sal_stability_stage1.kernel as kernel

    source = inspect.getsource(kernel._simulate_shared_world).encode("utf-8")
    assert hashlib.sha256(source).hexdigest() == (
        "e680f3852b1e7cbfa70f8b641e921c1e31695e2de5805fb1de6648c64dae5630"
    )


def test_public_shared_world_and_condition_entry_points_are_array_identical():
    config, reference = load_stage3_inputs()
    world = build_shared_physical_world(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    via_world = simulate_shared_world(
        world=world,
        study_config=config,
        encoding_reference=reference,
    )
    legacy = simulate_condition_repeat(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_id=17,
        study_config=config,
        encoding_reference=reference,
    )
    assert via_world.condition_ordinal == legacy.condition_ordinal
    assert via_world.condition_code == legacy.condition_code
    assert via_world.repeat_id == legacy.repeat_id
    assert via_world.method_codes == legacy.method_codes
    np.testing.assert_array_equal(via_world.source_state_code, legacy.source_state_code)
    np.testing.assert_array_equal(via_world.ref_toa_before_s, legacy.ref_toa_before_s)
    np.testing.assert_array_equal(
        via_world.selected_observed_toa_s,
        legacy.selected_observed_toa_s,
    )
    np.testing.assert_array_equal(via_world.has_selection, legacy.has_selection)
    assert via_world.delta_t_consumed_count == legacy.delta_t_consumed_count
    assert via_world.hprf_count_check_count == legacy.hprf_count_check_count
    assert via_world.hprf_no_n_check_count == legacy.hprf_no_n_check_count


def test_h300_fixed_level_differences_map_to_exact_three_strata():
    reference_levels = np.full(K, 5, dtype=np.uint8)
    reference_levels[:3] = [5, 4, 3]
    strata = width_strata_from_levels(
        h_true_width_level=np.uint8(5),
        reference_width_level=reference_levels,
    )
    assert strata[:3].tolist() == [
        WIDTH_STRATUM_0NS,
        WIDTH_STRATUM_10NS,
        WIDTH_STRATUM_GE20NS,
    ]
    assert set(strata) == set(WIDTH_STRATA)


def test_h300_level_classification_matches_frozen_physical_widths():
    _, reference = load_stage3_inputs()
    for h_level in (0, 5, 15):
        assert level_strata_match_physical_width_difference(
            h_true_width_level=np.uint8(h_level),
            reference_width_level=reference["width_level"],
            reference_width_s=reference["ref_width_s"],
        )


def test_h300_c_to_e_is_anchored_to_current_cycle_stratum():
    source = np.ones((5, K), dtype=np.uint8)
    source[:, 1] = 2
    trajectory = _trajectory(source, condition_ordinal=3, condition_code="H300")
    reference_levels = np.full(K, 5, dtype=np.uint8)
    reference_levels[1] = 3
    rows = compute_h300_width_strata(
        trajectory=trajectory,
        h_true_width_level=np.uint8(5),
        reference_width_level=reference_levels,
    )
    for method_ordinal in range(5):
        method_rows = {
            row.WidthStratum: row
            for row in rows
            if row.MethodOrdinal == method_ordinal
        }
        assert method_rows[WIDTH_STRATUM_0NS].Stratum_C_to_E_Count == 1
        assert method_rows[WIDTH_STRATUM_GE20NS].Stratum_C_to_E_Count == 0


def test_h300_strata_conserve_cycles_and_c_e_per_method():
    source = np.vstack(
        [np.resize(np.array([1, 2, 1, 2, 2], dtype=np.uint8), K) for _ in range(5)]
    )
    trajectory = _trajectory(source, condition_ordinal=3, condition_code="H300")
    levels = np.resize(np.array([5, 4, 3], dtype=np.uint8), K)
    rows = compute_h300_width_strata(
        trajectory=trajectory,
        h_true_width_level=np.uint8(5),
        reference_width_level=levels,
    )
    assert len(rows) == 15
    expected_cycles = tuple(row.StratumCycleCount for row in rows[:3])
    for method_ordinal in range(5):
        method_rows = rows[method_ordinal * 3 : method_ordinal * 3 + 3]
        assert tuple(row.StratumCycleCount for row in method_rows) == expected_cycles
        assert sum(row.StratumCycleCount for row in method_rows) == K
        assert all(
            row.StratumCorrectCount + row.StratumErrorCount
            == row.StratumCycleCount
            for row in method_rows
        )


@pytest.mark.parametrize("forbidden_source", (0, 3))
def test_h300_rejects_n_and_f_sources(forbidden_source):
    source = np.ones((5, K), dtype=np.uint8)
    source[0, 0] = forbidden_source
    trajectory = _trajectory(source, condition_ordinal=3, condition_code="H300")
    with pytest.raises(ValueError, match="H300"):
        compute_h300_width_strata(
            trajectory=trajectory,
            h_true_width_level=np.uint8(5),
            reference_width_level=np.full(K, 5, dtype=np.uint8),
        )


def test_h300_schema_contains_no_l_nc_stratification_fields():
    field_names = tuple(field_info.name for field_info in fields(H300WidthStratumMetrics))
    assert field_names == (
        "ConditionOrdinal",
        "ConditionCode",
        "MethodOrdinal",
        "MethodCode",
        "RepeatID",
        "WidthStratum",
        "StratumCycleCount",
        "StratumCorrectCount",
        "StratumErrorCount",
        "Stratum_C_to_E_Count",
    )
    assert not any("L_NC" in name for name in field_names)


def test_real_h300_metric_bundle_has_5_repeat_rows_and_15_stratum_rows():
    config, reference = load_stage3_inputs()
    bundle = simulate_condition_repeat_with_metrics(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    assert bundle.trajectory.condition_code == "H300"
    assert len(bundle.repeat_metrics) == 5
    assert len(bundle.h300_width_strata) == 15
    assert all(row.N_N == 0 for row in bundle.repeat_metrics)
    assert all(row.N_E_H == row.N_E and row.N_E_F == 0 for row in bundle.repeat_metrics)


def test_non_h300_metric_bundle_has_no_width_stratum_rows():
    config, reference = load_stage3_inputs()
    bundle = simulate_condition_repeat_with_metrics(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=6,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    assert bundle.trajectory.condition_code == "F1"
    assert len(bundle.repeat_metrics) == 5
    assert bundle.h300_width_strata == ()
    assert all(row.N_E_H == 0 and row.N_E_F == row.N_E for row in bundle.repeat_metrics)


def test_high_level_metric_execution_builds_exactly_one_world(monkeypatch):
    config, reference = load_stage3_inputs()
    world = build_shared_physical_world(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_id=2,
        study_config=config,
        encoding_reference=reference,
    )
    calls = []

    def one_world_builder(**kwargs):
        calls.append(kwargs)
        return world

    monkeypatch.setattr(stage5_module, "build_shared_physical_world", one_world_builder)
    bundle = simulate_condition_repeat_with_metrics(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_id=2,
        study_config=config,
        encoding_reference=reference,
    )
    assert len(calls) == 1
    assert bundle.trajectory.repeat_id == 2


def test_validation_csv_serializes_counts_as_integers_and_rejects_duplicate_keys(
    tmp_path,
):
    config, reference = load_stage3_inputs()
    bundle = simulate_condition_repeat_with_metrics(
        namespace=RandomNamespace.PILOT,
        condition_ordinal=3,
        repeat_id=1,
        study_config=config,
        encoding_reference=reference,
    )
    repeat_path = tmp_path / "repeat.csv"
    strata_path = tmp_path / "strata.csv"
    write_repeat_metrics_csv(repeat_path, bundle.repeat_metrics)
    write_h300_width_strata_csv(strata_path, bundle.h300_width_strata)
    with repeat_path.open(encoding="utf-8", newline="") as handle:
        repeat_rows = list(csv.DictReader(handle))
    with strata_path.open(encoding="utf-8", newline="") as handle:
        strata_rows = list(csv.DictReader(handle))
    assert len(repeat_rows) == 5
    assert len(strata_rows) == 15
    assert all("." not in row["N_C"] for row in repeat_rows)
    assert all("." not in row["StratumCycleCount"] for row in strata_rows)
    with pytest.raises(ValueError, match="duplicate"):
        write_repeat_metrics_csv(repeat_path, bundle.repeat_metrics * 2)
