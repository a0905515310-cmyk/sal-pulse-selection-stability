"""Verify frozen study data, then regenerate every paper figure."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    root = args.root.resolve()
    code, data, results = root / "code", root / "data", root / "results"

    inventory = json.loads((code / "data_manifest.json").read_text(encoding="utf-8"))
    for name, expected in inventory.items():
        path = data / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Data checksum mismatch: {name}")

    stage11_root = data / "simulation"
    stage11 = stage11_root / "artifacts" / "stage11"
    status = json.loads((stage11 / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "PASS" or status["blocking_issue_count"] != 0:
        raise RuntimeError("Stage 11 frozen status failed")
    stage11_manifest = json.loads((stage11 / "stage11_manifest.json").read_text(encoding="utf-8"))
    for name, expected in stage11_manifest["ArtifactSHA256"].items():
        if sha256(stage11_root / name) != expected:
            raise RuntimeError(f"Stage 11 artifact mismatch: {name}")

    figure_data = data / "figure_source_data"
    figure_manifest = json.loads((figure_data / "manifest.json").read_text(encoding="utf-8"))
    frozen_hashes = set(stage11_manifest["ArtifactSHA256"].values())
    for name, info in figure_manifest["datasets"].items():
        if sha256(figure_data / name) != info["sha256"]:
            raise RuntimeError(f"Figure source mismatch: {name}")
        if info["source_sha256"] not in frozen_hashes:
            raise RuntimeError(f"Figure source lacks a Stage 11 provenance match: {name}")

    results.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["CO_SOURCE_DATA"] = str(figure_data)
    env["CO_RESULTS_DIR"] = str(results)
    subprocess.run([sys.executable, str(code / "plot_results.py")], check=True, env=env)

    figures = sorted((results / "figures").glob("*.pdf"))
    if len(figures) != 7:
        raise RuntimeError(f"Expected 7 figure PDFs, found {len(figures)}")
    report = {
        "stage11_status": "PASS",
        "data_files_verified": len(inventory),
        "stage11_artifacts_verified": len(stage11_manifest["ArtifactSHA256"]),
        "figure_tables_verified": len(figure_manifest["datasets"]),
        "figure_pdfs_generated": len(figures),
        "scope": "Frozen data integrity and figure regeneration; no Formal or Bootstrap rerun",
        "legacy_test_failures_recorded_in_frozen_status": status["full_pytest_raw_failed"],
    }
    (results / "capsule_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
