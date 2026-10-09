import json
from pathlib import Path

import numpy as np

from sal_stability_stage1.build import ROOT, build_study_config, scan_active_files, validate
from sal_stability_stage1.encoding import (
    EXPECTED_199_INTERVAL_SUM_S,
    INITIAL_STATE,
    generate_reference_arrays,
    next_state,
    state_integer,
    verify_full_period,
    width_level,
)


def test_config_contract_counts_and_order():
    cfg = build_study_config()
    assert cfg["K"] == 200
    assert len(cfg["ConditionSpecs"]) == 19
    assert [x["ConditionOrdinal"] for x in cfg["ConditionSpecs"]] == list(range(19))
    assert len(cfg["MethodSpecs"]) == 5
    assert [x["Code"] for x in cfg["MethodSpecs"]] == ["FIRST", "LAST", "T", "W", "TW"]


def test_lfsr_initial_and_first_transition():
    assert state_integer(INITIAL_STATE) == 42393
    s2 = next_state(INITIAL_STATE)
    assert state_integer(s2) == 21196


def test_first_width_levels_are_deterministic():
    a = generate_reference_arrays(10)
    assert a["width_level"].tolist() == [3, 4, 12, 3, 13, 7, 13, 13, 12, 12]


def test_reference_array_sizes_dtypes_and_sum():
    a = generate_reference_arrays(200)
    assert a["state_bits"].shape == (200, 16)
    assert a["state_bits"].dtype == np.uint8
    assert a["state_u16"].dtype == np.uint16
    assert a["interval_level"].shape == (199,)
    assert a["interval_level"].dtype == np.uint16
    assert a["delta_t_s"].shape == (199,)
    assert a["delta_t_s"].dtype == np.float64
    assert a["width_level"].shape == (200,)
    assert a["width_level"].dtype == np.uint8
    assert a["ref_width_s"].shape == (200,)
    assert a["ref_width_s"].dtype == np.float64
    assert np.isclose(a["delta_t_s"].sum(), EXPECTED_199_INTERVAL_SUM_S, rtol=0.0, atol=5e-15)


def test_width_values_are_16_level_grid():
    a = generate_reference_arrays(200)
    expected = 250e-9 + a["width_level"].astype(np.float64) * 10e-9
    assert np.array_equal(a["ref_width_s"], expected)
    assert np.all((a["width_level"] >= 0) & (a["width_level"] <= 15))


def test_full_lfsr_period():
    assert verify_full_period()


def test_static_validation_passes():
    cfg = build_study_config()
    a = generate_reference_arrays(cfg["K"])
    result = validate(cfg, a)
    assert result["all_pass"]


def test_active_files_have_no_retired_identifiers():
    active = [
        ROOT / "artifacts" / "config" / "study_config.json",
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
    ]
    assert scan_active_files(active) == []
