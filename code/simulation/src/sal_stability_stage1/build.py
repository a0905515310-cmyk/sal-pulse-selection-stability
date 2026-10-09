from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np

from .contracts import CONDITIONS, METHODS, condition_dicts, method_dicts
from .encoding import (
    EXPECTED_199_INTERVAL_SUM_S,
    K_DEFAULT,
    generate_reference_arrays,
    verify_full_period,
)
from .rng import (
    MASTER_SEED_IDENTIFIER,
    MASTER_SEED_SHA256,
    PHILOX_ROUNDS,
    RandomNamespace,
    VariableFamily,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "artifacts" / "config"
STAGE_DIR = ROOT / "artifacts" / "stage1"


def build_rng_spec() -> dict:
    return {
        "Algorithm": "Philox-4x32-10",
        "Rounds": PHILOX_ROUNDS,
        "MasterSeedIdentifier": MASTER_SEED_IDENTIFIER,
        "MasterSeedSHA256": MASTER_SEED_SHA256,
        "NamespaceIDs": {member.name: int(member) for member in RandomNamespace},
        "VariableFamilyIDs": {member.name: int(member) for member in VariableFamily},
        "CounterLayout": [
            "ConditionOrdinal_u32",
            "RepeatID_u32",
            "EventIndex_low32_twos_complement",
            "EventIndex_high32_twos_complement",
        ],
        "UniformTransform": "(x+0.5)/2^32",
        "NormalTransform": "BoxMullerCos_word0_word1",
        "TieRankLayout": "w0w1_hi64_w2w3_lo64",
    }


def build_study_config() -> dict:
    return {
        "StudyConfigVersion": "SAL_STABILITY_19C5M_V1",
        "K": 200,
        "EffectiveIntervalCount": 199,
        "ExpectedEncodedDuration_s": 29.5308825,
        "GateHalfWidth_s": 5e-6,
        "GateFullWidth_s": 10e-6,
        "SigmaTOA_s": 1e-6,
        "SigmaSync_s": 1e-6,
        "SigmaWidth_s": 1.5e-9,
        "Xi": 0.0,
        "PilotR": 200,
        "CandidateFormalR": [1000, 2000],
        "BootstrapB": 2000,
        "ConditionSpecs": condition_dicts(),
        "MethodSpecs": method_dicts(),
        "RandomNamespaces": ["PILOT", "FORMAL", "BOOTSTRAP"],
        "RNGSpec": build_rng_spec(),
        "DiagnosticConditionCodes": [],
        "DiagnosticRepeatIDs": [],
        "DataSchemaVersion": "SAL_STABILITY_SCHEMA_V1",
        "DTypes": {
            "time_s": "float64",
            "frequency_hz": "float64",
            "cycle_index": "int64",
            "hprf_pulse_index": "int64",
            "source_state_code": "uint8",
            "lfsr_state_bit": "uint8",
            "lfsr_state_integer": "uint16",
            "interval_level": "uint16",
            "width_level": "uint8"
        },
    }


def canonical_json_bytes(data: dict) -> bytes:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return text.encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_reference_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "k", "state_hex", "state_u16", "interval_level", "delta_t_s",
            "width_level", "ref_width_s"
        ])
        for i in range(len(arrays["state_u16"])):
            has_interval = i < len(arrays["delta_t_s"])
            w.writerow([
                i + 1,
                f"0x{int(arrays['state_u16'][i]):04X}",
                int(arrays["state_u16"][i]),
                int(arrays["interval_level"][i]) if has_interval else "",
                f"{float(arrays['delta_t_s'][i]):.16g}" if has_interval else "",
                int(arrays["width_level"][i]),
                f"{float(arrays['ref_width_s'][i]):.16g}",
            ])


def _blocked_fragments() -> list[tuple[str, ...]]:
    # Fragments are assembled so retired identifiers do not themselves become
    # active literal identifiers in this fresh project.
    return [
        ("Pairing", "BlockID"),
        ("Pairing", "Registry"),
        ("Condition", "Registry"),
        ("U", "0"),
        ("U", "T"),
        ("Act", "ive"),
        ("Termin", "ated"),
        ("M", "1"), ("M", "2"), ("M", "3"), ("M", "4"), ("M", "5"), ("M", "6"),
        ("H", "1"), ("H", "2"), ("H", "3"),
    ]


def scan_active_files(paths: Iterable[Path]) -> list[dict]:
    findings: list[dict] = []
    blocked = ["".join(parts) for parts in _blocked_fragments()]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in blocked:
            pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])")
            if pattern.search(text):
                findings.append({"path": str(path.relative_to(ROOT)), "token": token})
    return findings


def validate(config: dict, arrays: dict[str, np.ndarray]) -> dict:
    checks: dict[str, bool] = {}
    checks["K_200"] = config["K"] == K_DEFAULT == 200
    checks["condition_count_19"] = len(config["ConditionSpecs"]) == len(CONDITIONS) == 19
    checks["condition_ordinals"] = [x["ConditionOrdinal"] for x in config["ConditionSpecs"]] == list(range(19))
    checks["condition_codes_unique"] = len({x["Code"] for x in config["ConditionSpecs"]}) == 19
    checks["method_count_5"] = len(config["MethodSpecs"]) == len(METHODS) == 5
    checks["method_ordinals"] = [x["MethodOrdinal"] for x in config["MethodSpecs"]] == list(range(5))
    checks["method_codes"] = [x["Code"] for x in config["MethodSpecs"]] == ["FIRST", "LAST", "T", "W", "TW"]
    checks["state_shape"] = arrays["state_bits"].shape == (200, 16)
    checks["ref_width_count_200"] = arrays["ref_width_s"].shape == (200,)
    checks["delta_t_count_199"] = arrays["delta_t_s"].shape == (199,)
    checks["encoded_duration"] = bool(np.isclose(arrays["delta_t_s"].sum(dtype=np.float64), EXPECTED_199_INTERVAL_SUM_S, rtol=0.0, atol=5e-15))
    checks["state_dtype"] = arrays["state_bits"].dtype == np.uint8
    checks["interval_dtype"] = arrays["delta_t_s"].dtype == np.float64
    checks["width_dtype"] = arrays["ref_width_s"].dtype == np.float64
    checks["width_levels_range"] = int(arrays["width_level"].min()) >= 0 and int(arrays["width_level"].max()) <= 15
    checks["interval_levels_range"] = int(arrays["interval_level"].min()) >= 0 and int(arrays["interval_level"].max()) <= 65534
    checks["lfsr_full_period"] = verify_full_period()
    checks["gate_width_identity"] = config["GateFullWidth_s"] == 2.0 * config["GateHalfWidth_s"]
    checks["namespace_count_3"] = config["RandomNamespaces"] == ["PILOT", "FORMAL", "BOOTSTRAP"]
    checks["rng_spec"] = config.get("RNGSpec") == build_rng_spec()
    return {"checks": checks, "all_pass": all(checks.values())}


def build() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    config = build_study_config()
    config_bytes = canonical_json_bytes(config)
    config_path = CONFIG_DIR / "study_config.json"
    config_path.write_bytes(config_bytes)
    config_sha = sha256_bytes(config_bytes)
    (CONFIG_DIR / "study_config.sha256").write_text(config_sha + "\n", encoding="utf-8")

    arrays = generate_reference_arrays(config["K"])
    np.savez_compressed(CONFIG_DIR / "encoding_reference.npz", **arrays)
    write_reference_csv(CONFIG_DIR / "encoding_reference.csv", arrays)

    validation = validate(config, arrays)

    active_scan_paths = [
        config_path,
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
    ]
    legacy_findings = scan_active_files(active_scan_paths)
    validation["checks"]["retired_identifier_scan"] = len(legacy_findings) == 0
    validation["all_pass"] = all(validation["checks"].values())
    validation["retired_identifier_findings"] = legacy_findings
    validation["study_config_sha256"] = config_sha
    validation["encoded_duration_s"] = float(arrays["delta_t_s"].sum(dtype=np.float64))

    return validation
