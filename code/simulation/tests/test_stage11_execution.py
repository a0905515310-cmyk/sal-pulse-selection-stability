from __future__ import annotations

from sal_stability_stage1 import bootstrap, execution, stage9
from sal_stability_stage1.results_export import build_evidence_exports
from sal_stability_stage1.stage11 import (
    STAGE12_EXECUTED,
    assert_no_forbidden_execution_path,
)


def test_stage11_code_has_no_rng_bootstrap_formal_physics_plotting_or_new_ci_path():
    gate = assert_no_forbidden_execution_path()
    assert all(gate.values())
    assert STAGE12_EXECUTED is False


def test_export_does_not_call_formal_bootstrap_or_physical_execution(monkeypatch, tmp_path):
    def prohibited(*args, **kwargs):
        raise AssertionError("prohibited inference or physical execution path called")

    monkeypatch.setattr(stage9, "run_formal_condition_chunked", prohibited)
    monkeypatch.setattr(bootstrap, "run_condition_bootstrap", prohibited)
    monkeypatch.setattr(execution, "run_condition_repeats_serial", prohibited)
    monkeypatch.setattr(execution, "run_condition_repeats_parallel", prohibited)
    result = build_evidence_exports(tmp_path)
    assert result["EvidenceArtifactCount"] == 16
    assert result["EvidenceRegistryRows"] == 16


def test_repeated_stage11_export_is_byte_identical(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_evidence_exports(first)
    build_evidence_exports(second)
    first_files = sorted(path.relative_to(first) for path in first.rglob("*.csv"))
    second_files = sorted(path.relative_to(second) for path in second.rglob("*.csv"))
    assert first_files == second_files
    assert len(first_files) == 17
    for relative in first_files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
