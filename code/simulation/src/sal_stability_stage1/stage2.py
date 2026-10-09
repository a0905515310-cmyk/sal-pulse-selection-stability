from __future__ import annotations

import ast
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .build import (
    CONFIG_DIR,
    ROOT,
    build_rng_spec,
    build_study_config,
    canonical_json_bytes,
    scan_active_files,
    sha256_bytes,
    validate,
)
from .encoding import generate_reference_arrays
from .rng import (
    MASK32,
    MASK64,
    MASTER_SEED_IDENTIFIER,
    MASTER_SEED_SHA256,
    PHILOX_ROUNDS,
    TIE_FAMILIES,
    RandomAddress,
    RandomNamespace,
    TieRankCollisionError,
    VariableFamily,
    assert_unique_tieranks,
    counter_words,
    derive_philox_key,
    encode_event_index,
    philox4x32_10,
    random_normal,
    random_tierank,
    random_uniform,
    raw_words,
    uniform_from_uint32,
)


STAGE2_DIR = ROOT / "artifacts" / "stage2"
GOLDEN_PATH = CONFIG_DIR / "rng_test_vectors.csv"
FROZEN_STAGE1_CONFIG_SHA256 = (
    "e4ad227d9fe619b98a12208b957699a24d51ef5a6533fa6eabf790e9f171f72b"
)
FROZEN_UNCHANGED_SHA256 = {
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
    "tests/test_stage1.py": (
        "2f61cabef205d23d19df599f3c090a672049482bdd18be0dd84aaafcf6770478"
    ),
}

GOLDEN_FIELDS = (
    "VectorID",
    "Namespace",
    "NamespaceID",
    "ConditionOrdinal",
    "RepeatID",
    "VariableFamily",
    "VariableFamilyID",
    "EventIndex",
    "Counter0_hex",
    "Counter1_hex",
    "Counter2_hex",
    "Counter3_hex",
    "Key0_hex",
    "Key1_hex",
    "Raw0_hex",
    "Raw1_hex",
    "Raw2_hex",
    "Raw3_hex",
    "Uniform0",
    "Uniform1",
    "NormalZ",
    "Uniform0_hex",
    "Uniform1_hex",
    "NormalZ_hex",
    "TieRankHi_hex",
    "TieRankLo_hex",
)


def _address(
    namespace: RandomNamespace,
    condition_ordinal: int,
    repeat_id: int,
    variable_family: VariableFamily,
    event_index: int,
) -> RandomAddress:
    return RandomAddress(
        namespace=namespace,
        condition_ordinal=condition_ordinal,
        repeat_id=repeat_id,
        variable_family=variable_family,
        event_index=event_index,
    )


GOLDEN_ADDRESSES = (
    ("V001", _address(RandomNamespace.PILOT, 0, 1, VariableFamily.INITIAL_SYNC, 0)),
    ("V002", _address(RandomNamespace.FORMAL, 18, 200, VariableFamily.G_TOA_ERROR, 1)),
    ("V003", _address(RandomNamespace.PILOT, 1, 200, VariableFamily.G_WIDTH_ERROR, 200)),
    ("V004", _address(RandomNamespace.FORMAL, 2, 1, VariableFamily.H_PHASE, 0)),
    ("V005", _address(RandomNamespace.PILOT, 18, 200, VariableFamily.H_TRUE_WIDTH_LEVEL, 0)),
    ("V006", _address(RandomNamespace.FORMAL, 3, 1, VariableFamily.H_TOA_ERROR, -1)),
    ("V007", _address(RandomNamespace.FORMAL, 4, 200, VariableFamily.H_WIDTH_ERROR, -2)),
    ("V008", _address(RandomNamespace.PILOT, 5, 1, VariableFamily.F_TOA_ERROR, 1)),
    ("V009", _address(RandomNamespace.FORMAL, 6, 200, VariableFamily.F_WIDTH_ERROR, 200)),
    ("V010", _address(RandomNamespace.PILOT, 7, 1, VariableFamily.TIE_G, 1)),
    ("V011", _address(RandomNamespace.FORMAL, 8, 200, VariableFamily.TIE_H, -1)),
    ("V012", _address(RandomNamespace.FORMAL, 9, 1, VariableFamily.TIE_F, 200)),
    ("V013", _address(RandomNamespace.BOOTSTRAP, 0, 1, VariableFamily.BOOTSTRAP_INDEX, 0)),
    ("V014", _address(RandomNamespace.FORMAL, 10, 1, VariableFamily.H_TOA_ERROR, 1 << 40)),
    ("V015", _address(RandomNamespace.FORMAL, 11, 1, VariableFamily.H_WIDTH_ERROR, -(1 << 40))),
    ("V016", _address(RandomNamespace.FORMAL, 18, MASK32, VariableFamily.H_TOA_ERROR, (1 << 63) - 1)),
    ("V017", _address(RandomNamespace.PILOT, 0, 1, VariableFamily.H_WIDTH_ERROR, -(1 << 63))),
    ("V018", _address(RandomNamespace.PILOT, 1, 1, VariableFamily.INITIAL_SYNC, 0)),
    ("V019", _address(RandomNamespace.FORMAL, 1, 1, VariableFamily.INITIAL_SYNC, 0)),
    ("V020", _address(RandomNamespace.BOOTSTRAP, 1, 1, VariableFamily.INITIAL_SYNC, 0)),
    ("V021", _address(RandomNamespace.FORMAL, 1, 1, VariableFamily.G_TOA_ERROR, 1)),
    ("V022", _address(RandomNamespace.FORMAL, 2, 1, VariableFamily.G_TOA_ERROR, 1)),
    ("V023", _address(RandomNamespace.BOOTSTRAP, 18, 2000, VariableFamily.BOOTSTRAP_INDEX, 1999)),
    ("V024", _address(RandomNamespace.BOOTSTRAP, 18, 200, VariableFamily.BOOTSTRAP_INDEX, 200)),
    ("V025", _address(RandomNamespace.PILOT, 0, 200, VariableFamily.F_TOA_ERROR, -2)),
    ("V026", _address(RandomNamespace.FORMAL, 18, 1, VariableFamily.TIE_G, -2)),
    ("V027", _address(RandomNamespace.PILOT, 18, 200, VariableFamily.TIE_H, (1 << 32) + 7)),
    ("V028", _address(RandomNamespace.FORMAL, 0, 200, VariableFamily.TIE_F, -(1 << 32) - 7)),
)


def _hex32(value: int) -> str:
    return f"0x{value:08X}"


def _hex64(value: int) -> str:
    return f"0x{value:016X}"


def golden_row(vector_id: str, address: RandomAddress) -> dict[str, str]:
    counter = counter_words(address)
    key = derive_philox_key(address.namespace, address.variable_family)
    words = raw_words(address)
    uniform0 = uniform_from_uint32(words[0])
    uniform1 = uniform_from_uint32(words[1])
    normal = random_normal(address)
    if address.variable_family in TIE_FAMILIES:
        tie_hi, tie_lo = random_tierank(address)
        tie_hi_hex = _hex64(tie_hi)
        tie_lo_hex = _hex64(tie_lo)
    else:
        tie_hi_hex = ""
        tie_lo_hex = ""
    return {
        "VectorID": vector_id,
        "Namespace": address.namespace.name,
        "NamespaceID": str(int(address.namespace)),
        "ConditionOrdinal": str(address.condition_ordinal),
        "RepeatID": str(address.repeat_id),
        "VariableFamily": address.variable_family.name,
        "VariableFamilyID": str(int(address.variable_family)),
        "EventIndex": str(address.event_index),
        "Counter0_hex": _hex32(counter[0]),
        "Counter1_hex": _hex32(counter[1]),
        "Counter2_hex": _hex32(counter[2]),
        "Counter3_hex": _hex32(counter[3]),
        "Key0_hex": _hex32(key[0]),
        "Key1_hex": _hex32(key[1]),
        "Raw0_hex": _hex32(words[0]),
        "Raw1_hex": _hex32(words[1]),
        "Raw2_hex": _hex32(words[2]),
        "Raw3_hex": _hex32(words[3]),
        "Uniform0": format(uniform0, ".17g"),
        "Uniform1": format(uniform1, ".17g"),
        "NormalZ": format(normal, ".17g"),
        "Uniform0_hex": uniform0.hex(),
        "Uniform1_hex": uniform1.hex(),
        "NormalZ_hex": normal.hex(),
        "TieRankHi_hex": tie_hi_hex,
        "TieRankLo_hex": tie_lo_hex,
    }


def expected_golden_rows() -> list[dict[str, str]]:
    return [golden_row(vector_id, address) for vector_id, address in GOLDEN_ADDRESSES]


def generate_golden_vectors(path: Path = GOLDEN_PATH) -> int:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing golden vectors: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = expected_golden_rows()
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLDEN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_golden_vectors(path: Path = GOLDEN_PATH) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def verify_golden_vectors(path: Path = GOLDEN_PATH) -> dict:
    issues: list[str] = []
    if not path.is_file():
        return {
            "passed": False,
            "count": 0,
            "issues": [f"missing golden vector file: {path}"],
        }
    header, actual = read_golden_vectors(path)
    expected = expected_golden_rows()
    if header != list(GOLDEN_FIELDS):
        issues.append("golden vector header differs from the frozen field order")
    if len(actual) != len(expected):
        issues.append(
            f"golden vector count differs: expected {len(expected)}, found {len(actual)}"
        )
    for row_index, (actual_row, expected_row) in enumerate(
        zip(actual, expected, strict=False), start=2
    ):
        for column in GOLDEN_FIELDS:
            if actual_row.get(column) != expected_row[column]:
                issues.append(
                    f"row {row_index} column {column} drifted: "
                    f"expected {expected_row[column]!r}, found {actual_row.get(column)!r}"
                )
    return {"passed": not issues, "count": len(actual), "issues": issues}


def golden_coverage_checks() -> dict[str, bool]:
    addresses = [address for _, address in GOLDEN_ADDRESSES]
    events = {address.event_index for address in addresses}
    return {
        "golden_count_at_least_20": len(addresses) >= 20,
        "golden_namespaces_complete": {address.namespace for address in addresses}
        == set(RandomNamespace),
        "golden_families_complete": {address.variable_family for address in addresses}
        == set(VariableFamily),
        "golden_condition_boundaries": {0, 18}.issubset(
            {address.condition_ordinal for address in addresses}
        ),
        "golden_repeat_examples": {1, 200}.issubset(
            {address.repeat_id for address in addresses}
        ),
        "golden_event_examples": {0, 1, 200, -1, -2}.issubset(events),
        "golden_large_positive_h_index": any(event > 200 for event in events),
        "golden_large_negative_h_index": any(event < -2 for event in events),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage1_config_hash(config: dict) -> str:
    stage1_view = dict(config)
    stage1_view.pop("RNGSpec", None)
    return sha256_bytes(canonical_json_bytes(stage1_view))


def _call_path(node: ast.AST) -> str | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _resolve_call_path(path: str, aliases: dict[str, str]) -> str:
    first, separator, remainder = path.partition(".")
    resolved = aliases.get(first, first)
    return resolved + (separator + remainder if separator else "")


def scan_forbidden_random_apis(paths: Iterable[Path]) -> list[dict]:
    prohibited_exact = {
        ".".join(("numpy", "random", "seed")),
        ".".join(("numpy", "random", "default_rng")),
        ".".join(("numpy", "random", "RandomState")),
        ".".join(("random", "seed")),
        ".".join(("random", "random")),
        ".".join(("os", "urandom")),
    }
    prohibited_root = "".join(("sec", "rets"))
    findings: list[dict] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "line": exc.lineno,
                    "call": "SYNTAX_ERROR",
                }
            )
            continue
        aliases = _import_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            raw_path = _call_path(node.func)
            if raw_path is None:
                continue
            resolved = _resolve_call_path(raw_path, aliases)
            if resolved in prohibited_exact or resolved.split(".", 1)[0] == prohibited_root:
                findings.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "line": node.lineno,
                        "call": resolved,
                    }
                )
    return findings


def _evaluate(addresses: Sequence[RandomAddress]) -> dict[RandomAddress, tuple[int, ...]]:
    return {address: raw_words(address) for address in addresses}


def _fixed_permutation(addresses: Sequence[RandomAddress]) -> list[RandomAddress]:
    def key(address: RandomAddress) -> bytes:
        payload = (
            f"{int(address.namespace)}:{address.condition_ordinal}:"
            f"{address.repeat_id}:{int(address.variable_family)}:{address.event_index}"
        ).encode("ascii")
        return hashlib.sha256(payload).digest()

    return sorted(addresses, key=key)


def _evaluate_chunks(
    addresses: Sequence[RandomAddress], chunk_size: int
) -> dict[RandomAddress, tuple[int, ...]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    result: dict[RandomAddress, tuple[int, ...]] = {}
    for start in range(0, len(addresses), chunk_size):
        result.update(_evaluate(addresses[start : start + chunk_size]))
    return result


def validate_rng_layer() -> dict:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    official_vectors = (
        (
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0x00000000, 0x00000000),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        ),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    )
    checks["philox_official_known_answers"] = all(
        philox4x32_10(counter, key) == expected
        for counter, key, expected in official_vectors
    )
    checks["philox_round_count_10"] = PHILOX_ROUNDS == 10
    checks["namespace_ids"] = {member.name: int(member) for member in RandomNamespace} == {
        "PILOT": 1,
        "FORMAL": 2,
        "BOOTSTRAP": 3,
    }
    checks["variable_family_ids"] = {
        member.name: int(member) for member in VariableFamily
    } == {
        "INITIAL_SYNC": 1,
        "G_TOA_ERROR": 2,
        "G_WIDTH_ERROR": 3,
        "H_PHASE": 4,
        "H_TRUE_WIDTH_LEVEL": 5,
        "H_TOA_ERROR": 6,
        "H_WIDTH_ERROR": 7,
        "F_TOA_ERROR": 8,
        "F_WIDTH_ERROR": 9,
        "TIE_G": 10,
        "TIE_H": 11,
        "TIE_F": 12,
        "BOOTSTRAP_INDEX": 13,
    }
    address_field_names = [field.name for field in fields(RandomAddress)]
    checks["random_address_fields"] = address_field_names == [
        "namespace",
        "condition_ordinal",
        "repeat_id",
        "variable_family",
        "event_index",
    ]
    details["random_address_fields"] = address_field_names

    signed_examples = {
        event: encode_event_index(event)
        for event in (0, 1, -1, -2, (1 << 63) - 1, -(1 << 63))
    }
    checks["signed_event_index"] = signed_examples == {
        0: (0x00000000, 0x00000000),
        1: (0x00000001, 0x00000000),
        -1: (0xFFFFFFFF, 0xFFFFFFFF),
        -2: (0xFFFFFFFE, 0xFFFFFFFF),
        (1 << 63) - 1: (0xFFFFFFFF, 0x7FFFFFFF),
        -(1 << 63): (0x00000000, 0x80000000),
    }
    rejected_bounds = 0
    for event in (1 << 63, -(1 << 63) - 1):
        try:
            encode_event_index(event)
        except ValueError:
            rejected_bounds += 1
    checks["signed_event_index_bounds"] = rejected_bounds == 2
    details["event_minus_one_counter_words"] = [
        _hex32(word) for word in signed_examples[-1]
    ]

    base_fields = {
        "condition_ordinal": 1,
        "repeat_id": 1,
        "variable_family": VariableFamily.G_TOA_ERROR,
        "event_index": 1,
    }
    namespace_blocks = {
        namespace: raw_words(_address(namespace=namespace, **base_fields))
        for namespace in RandomNamespace
    }
    checks["namespace_isolation"] = len(set(namespace_blocks.values())) == 3
    details["namespace_blocks"] = {
        namespace.name: [_hex32(word) for word in words]
        for namespace, words in namespace_blocks.items()
    }

    family_keys = {
        family: derive_philox_key(RandomNamespace.FORMAL, family)
        for family in VariableFamily
    }
    checks["variable_family_isolation"] = len(set(family_keys.values())) == len(
        VariableFamily
    )
    condition_a = _address(
        RandomNamespace.FORMAL, 1, 1, VariableFamily.G_TOA_ERROR, 1
    )
    condition_b = _address(
        RandomNamespace.FORMAL, 2, 1, VariableFamily.G_TOA_ERROR, 1
    )
    checks["condition_isolation"] = (
        counter_words(condition_a) != counter_words(condition_b)
        and raw_words(condition_a) != raw_words(condition_b)
    )
    checks["key_excludes_counter_fields"] = derive_philox_key(
        condition_a.namespace, condition_a.variable_family
    ) == derive_philox_key(condition_b.namespace, condition_b.variable_family)

    api_functions = (counter_words, raw_words, random_uniform, random_normal, random_tierank)
    api_parameters = {
        function.__name__: list(inspect.signature(function).parameters)
        for function in api_functions
    }
    checks["method_absent_from_rng_api"] = all(
        parameters == ["address"] for parameters in api_parameters.values()
    ) and all("method" not in name.lower() for name in address_field_names)
    details["rng_api_parameters"] = api_parameters

    sample = _address(RandomNamespace.FORMAL, 18, 200, VariableFamily.TIE_G, -1)
    sample_words = raw_words(sample)
    checks["raw_words_uint32"] = len(sample_words) == 4 and all(
        0 <= word <= MASK32 for word in sample_words
    )
    uniform = random_uniform(sample)
    checks["uniform_open_interval_and_formula"] = (
        0.0 < uniform < 1.0
        and uniform == (sample_words[0] + 0.5) / (1 << 32)
    )
    u1 = (sample_words[0] + 0.5) / (1 << 32)
    u2 = (sample_words[1] + 0.5) / (1 << 32)
    expected_normal = math.sqrt(-2.0 * math.log(u1)) * math.cos(
        2.0 * math.pi * u2
    )
    checks["box_muller_fixed_slots"] = random_normal(sample) == expected_normal
    expected_tie = (
        (sample_words[0] << 32) | sample_words[1],
        (sample_words[2] << 32) | sample_words[3],
    )
    checks["tierank_128_layout"] = random_tierank(sample) == expected_tie and all(
        0 <= word <= MASK64 for word in expected_tie
    )
    try:
        assert_unique_tieranks((expected_tie, expected_tie))
    except TieRankCollisionError:
        collision_stops = True
    else:
        collision_stops = False
    checks["tierank_collision_stops"] = collision_stops

    derive_philox_key.cache_clear()
    derive_philox_key(RandomNamespace.FORMAL, VariableFamily.G_TOA_ERROR)
    derive_philox_key(RandomNamespace.FORMAL, VariableFamily.G_TOA_ERROR)
    cache_info = derive_philox_key.cache_info()
    checks["key_derivation_cached"] = cache_info.misses == 1 and cache_info.hits == 1
    details["key_cache_info"] = {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "currsize": cache_info.currsize,
    }

    addresses = [
        _address(
            RandomNamespace.FORMAL,
            index % 19,
            (index % 2000) + 1,
            VariableFamily.H_TOA_ERROR,
            index - 500,
        )
        for index in range(1000)
    ]
    reference = _evaluate(addresses)
    checks["order_invariance"] = (
        _evaluate(list(reversed(addresses))) == reference
        and _evaluate(_fixed_permutation(addresses)) == reference
    )
    checks["chunk_invariance"] = all(
        _evaluate_chunks(addresses, chunk_size) == reference
        for chunk_size in (1, 7, 64, len(addresses))
    )
    parallel_order = _fixed_permutation(addresses)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        parallel_words = list(executor.map(raw_words, parallel_order))
    checks["serial_parallel_invariance"] = dict(
        zip(parallel_order, parallel_words, strict=True)
    ) == reference
    address_a = addresses[0]
    address_b = addresses[-1]
    first_a = raw_words(address_a)
    repeated = (raw_words(address_a), raw_words(address_a), raw_words(address_b), raw_words(address_a))
    checks["repeat_calls_are_stateless"] = repeated[0] == repeated[1] == repeated[3] == first_a
    for address in addresses[1:]:
        raw_words(address)
    checks["other_addresses_do_not_mutate_result"] = raw_words(address_a) == first_a
    negative_addresses = [
        _address(
            RandomNamespace.FORMAL,
            3,
            1,
            VariableFamily.H_WIDTH_ERROR,
            event_index,
        )
        for event_index in range(-100, 101)
    ]
    checks["negative_event_order_invariance"] = _evaluate(
        negative_addresses
    ) == _evaluate(list(reversed(negative_addresses)))

    source_paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").rglob("*.py")),
        *sorted((ROOT / "tests").rglob("*.py")),
    ]
    legacy_paths = [ROOT / "artifacts" / "config" / "study_config.json", *source_paths]
    legacy_findings = scan_active_files(legacy_paths)
    random_api_findings = scan_forbidden_random_apis(source_paths)
    checks["legacy_scan"] = len(legacy_findings) == 0
    checks["forbidden_random_api_scan"] = len(random_api_findings) == 0
    details["legacy_findings"] = legacy_findings
    details["forbidden_random_api_findings"] = random_api_findings
    checks.update(golden_coverage_checks())
    return {"checks": checks, "details": details, "all_pass": all(checks.values())}


@dataclass(frozen=True)
class TestRun:
    label: str
    command: tuple[str, ...]
    returncode: int
    passed: int
    failed: int
    duration_s: float
    output: str


def _parse_pytest_counts(output: str) -> tuple[int, int]:
    passed_matches = re.findall(r"(\d+) passed", output)
    failed_matches = re.findall(r"(\d+) failed", output)
    passed = int(passed_matches[-1]) if passed_matches else 0
    failed = int(failed_matches[-1]) if failed_matches else 0
    return passed, failed


def run_pytest(label: str, arguments: Sequence[str]) -> TestRun:
    command = (sys.executable, "-m", "pytest", *arguments)
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    duration = time.perf_counter() - start
    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    passed, failed = _parse_pytest_counts(output)
    return TestRun(
        label=label,
        command=command,
        returncode=completed.returncode,
        passed=passed,
        failed=failed,
        duration_s=duration,
        output=output.rstrip(),
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


def write_test_report(test_runs: Sequence[TestRun]) -> None:
    total_run = next(run for run in test_runs if run.label == "all")
    lines = [
        "SAL stability Stage 2 test report",
        f"Python version: {platform.python_version()}",
        f"Python executable: {sys.executable}",
        f"NumPy version: {np.__version__}",
        f"pytest version: {importlib.metadata.version('pytest')}",
        f"pytest command: {_format_command(total_run.command)}",
        f"test count: {total_run.passed + total_run.failed}",
        f"passed count: {total_run.passed}",
        f"failed count: {total_run.failed}",
        f"duration: {total_run.duration_s:.6f} s",
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
    (STAGE2_DIR / "test_report.txt").write_text("\n".join(lines), encoding="utf-8")


def write_changed_files() -> None:
    lines = [
        "MODIFIED README.md",
        "MODIFIED pyproject.toml",
        "MODIFIED src/sal_stability_stage1/build.py",
        "ADDED src/sal_stability_stage1/rng.py",
        "ADDED src/sal_stability_stage1/stage2.py",
        "ADDED scripts/build_stage2.py",
        "ADDED tests/test_stage2_rng.py",
        "MODIFIED artifacts/config/study_config.json",
        "MODIFIED artifacts/config/study_config.sha256",
        "UNCHANGED artifacts/config/encoding_reference.csv",
        "UNCHANGED artifacts/config/encoding_reference.npz",
        "ADDED artifacts/config/rng_test_vectors.csv",
        "UNCHANGED src/sal_stability_stage1/contracts.py",
        "UNCHANGED src/sal_stability_stage1/encoding.py",
        "UNCHANGED scripts/build_stage1.py",
        "UNCHANGED tests/test_stage1.py",
        "UNCHANGED artifacts/stage1/*",
        "ADDED artifacts/stage2/status.json",
        "ADDED artifacts/stage2/validation_report.md",
        "ADDED artifacts/stage2/changed_files.txt",
        "ADDED artifacts/stage2/test_report.txt",
        "ADDED artifacts/stage2/rng_validation.json",
    ]
    (STAGE2_DIR / "changed_files.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_validation_report(
    status: dict,
    config: dict,
    rng_validation: dict,
    golden_result: dict,
    golden_mode: str,
    stage1_run: TestRun,
    stage2_run: TestRun,
    total_run: TestRun,
    blocking_issues: Sequence[str],
) -> None:
    minus_one_low, minus_one_high = encode_event_index(-1)
    lines = [
        "# Stage 2 Random Layer Validation Report",
        "",
        f"Verdict: **{status['status']}**",
        "",
        "## 1. Current PRNG",
        "",
        "Philox-4x32-10. The scalar implementation passed all three official Random123 known-answer vectors.",
        "",
        "## 2. Master Seed",
        "",
        f"- MasterSeedIdentifier: `{MASTER_SEED_IDENTIFIER}`",
        f"- MasterSeedSHA256: `{MASTER_SEED_SHA256}`",
        "",
        "## 3. Key Derivation",
        "",
        "The two Philox key words are derived only from `MasterSeed + Namespace + VariableFamily` by SHA-256 and are cached per namespace/family pair.",
        "",
        "## 4. Counter Layout",
        "",
        "`(ConditionOrdinal_u32, RepeatID_u32, EventIndex_low32_twos_complement, EventIndex_high32_twos_complement)`",
        "",
        "## 5. Signed EventIndex",
        "",
        f"`EventIndex=-1` encodes as `Ctr2={_hex32(minus_one_low)}` and `Ctr3={_hex32(minus_one_high)}`. The full signed-int64 boundary tests passed.",
        "",
        "## 6. Uniform Transform",
        "",
        "`word0` is transformed as `(x+0.5)/2^32`, producing a float64 strictly inside `(0,1)`.",
        "",
        "## 7. Normal Transform",
        "",
        "`word0 -> u1` and `word1 -> u2`; the returned value is the cosine component of the fixed Box-Muller transform.",
        "",
        "## 8. TieRank",
        "",
        "`word0/word1 -> TieRankHi` and `word2/word3 -> TieRankLo`. Both halves are unsigned 64-bit integers. A complete 128-bit collision raises and has no fallback ordering.",
        "",
        "## 9. Method Isolation",
        "",
        "Method is structurally absent from RandomAddress. The physical random APIs accept only a `RandomAddress`; the five future methods therefore cannot alter the address.",
        "",
        "## 10. Namespace Isolation",
        "",
        "`PILOT != FORMAL != BOOTSTRAP` for otherwise identical address fields; the fixed isolation sample produced three distinct raw blocks.",
        "",
        "## 11. Golden Vectors",
        "",
        f"- Current mode: `{golden_mode}`",
        "- Creation record: the file was explicitly created once with `--generate-golden-vectors`; normal execution defaults to verification and refuses implicit replacement.",
        f"- Vector count: {golden_result['count']}",
        f"- All verified: {str(golden_result['passed']).lower()}",
        "",
        "## 12. Stage 1 Regression",
        "",
        f"- Stage 1 tests: {stage1_run.passed}/{stage1_run.passed + stage1_run.failed} passed",
        f"- Stage 2 tests: {stage2_run.passed}/{stage2_run.passed + stage2_run.failed} passed",
        f"- Full pytest: {total_run.passed}/{total_run.passed + total_run.failed} passed",
        f"- Frozen Stage 1 scientific config hash preserved: {str(not status['science_contract_changed']).lower()}",
        "",
        "## 13. Static and Invariance Checks",
        "",
    ]
    for name, passed in sorted(rng_validation["checks"].items()):
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## 14. Unexecuted Content",
            "",
            "Stage 3 not executed.",
            "Pilot not executed.",
            "Formal not executed.",
            "Bootstrap not executed.",
            "No paper result generated.",
            "",
            "## 15. Blocking Issues",
            "",
        ]
    )
    if blocking_issues:
        lines.extend(f"- {issue}" for issue in blocking_issues)
    else:
        lines.append("None")
    (STAGE2_DIR / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_stage2(generate_golden: bool = False) -> dict:
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / "study_config.json"
    checksum_path = CONFIG_DIR / "study_config.sha256"
    blocking_issues: list[str] = []

    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    existing_checksum = checksum_path.read_text(encoding="utf-8").strip()
    existing_checksum_valid = existing_checksum == _sha256_file(config_path)
    existing_science_hash = _stage1_config_hash(existing_config)

    config = build_study_config()
    builder_science_hash = _stage1_config_hash(config)
    arrays = generate_reference_arrays(config["K"])
    stage1_static = validate(config, arrays)
    science_contract_unchanged = (
        existing_checksum_valid
        and existing_science_hash == FROZEN_STAGE1_CONFIG_SHA256
        and builder_science_hash == FROZEN_STAGE1_CONFIG_SHA256
    )
    if not existing_checksum_valid:
        blocking_issues.append("pre-build study_config.sha256 did not match study_config.json")
    if not science_contract_unchanged:
        blocking_issues.append("frozen Stage 1 scientific configuration changed")
    if not stage1_static["all_pass"]:
        failed = [name for name, passed in stage1_static["checks"].items() if not passed]
        blocking_issues.append(f"Stage 1 static validation failed: {', '.join(failed)}")

    stage1_status_path = ROOT / "artifacts" / "stage1" / "status.json"
    stage1_status = json.loads(stage1_status_path.read_text(encoding="utf-8"))
    stage1_artifact_status_ok = (
        stage1_status.get("status") == "PASS"
        and stage1_status.get("science_contract_changed") is False
    )
    if not stage1_artifact_status_ok:
        blocking_issues.append("Stage 1 status artifact is not a frozen PASS")

    unchanged_file_checks: dict[str, bool] = {}
    for relative_path, expected_sha in FROZEN_UNCHANGED_SHA256.items():
        path = ROOT / relative_path
        unchanged_file_checks[relative_path] = (
            path.is_file() and _sha256_file(path) == expected_sha
        )
    if not all(unchanged_file_checks.values()):
        changed = [name for name, passed in unchanged_file_checks.items() if not passed]
        blocking_issues.append(
            f"files required to remain byte-identical changed: {', '.join(changed)}"
        )

    hard_precondition_passed = not blocking_issues
    if hard_precondition_passed:
        config_bytes = canonical_json_bytes(config)
        config_path.write_bytes(config_bytes)
        config_sha = sha256_bytes(config_bytes)
        checksum_path.write_text(config_sha + "\n", encoding="utf-8")
    else:
        config_sha = _sha256_file(config_path)

    golden_mode = (
        "GENERATE_GOLDEN_VECTORS" if generate_golden else "VERIFY_GOLDEN_VECTORS"
    )
    if hard_precondition_passed and generate_golden:
        try:
            generate_golden_vectors(GOLDEN_PATH)
        except FileExistsError as exc:
            blocking_issues.append(str(exc))
    golden_result = verify_golden_vectors(GOLDEN_PATH)
    if not golden_result["passed"]:
        blocking_issues.extend(golden_result["issues"])

    rng_validation = validate_rng_layer()
    rng_validation["checks"]["rng_spec_matches_config"] = (
        config.get("RNGSpec") == build_rng_spec()
    )
    rng_validation["checks"]["stage1_static_contract"] = stage1_static["all_pass"]
    rng_validation["checks"]["science_contract_unchanged"] = science_contract_unchanged
    rng_validation["checks"]["stage1_artifact_status"] = stage1_artifact_status_ok
    rng_validation["checks"]["frozen_files_unchanged"] = all(
        unchanged_file_checks.values()
    )
    rng_validation["checks"]["golden_vectors_verified"] = golden_result["passed"]
    rng_validation["all_pass"] = all(rng_validation["checks"].values())
    rng_validation["details"]["frozen_file_checks"] = unchanged_file_checks
    rng_validation["details"]["golden_issues"] = golden_result["issues"]
    for name, passed in rng_validation["checks"].items():
        if not passed:
            issue = f"validation check failed: {name}"
            if issue not in blocking_issues:
                blocking_issues.append(issue)

    stage1_run = run_pytest("stage1", ("tests/test_stage1.py",))
    if stage1_run.returncode != 0:
        blocking_issues.append("Stage 1 pytest regression failed")
    stage2_run = run_pytest("stage2", ("tests/test_stage2_rng.py",))
    if stage2_run.returncode != 0:
        blocking_issues.append("Stage 2 RNG pytest failed")
    total_run = run_pytest("all", ())
    if total_run.returncode != 0:
        blocking_issues.append("full pytest suite failed")
    test_runs = (stage1_run, stage2_run, total_run)
    write_test_report(test_runs)

    forbidden_result_dirs = (
        ROOT / "artifacts" / "pilot",
        ROOT / "artifacts" / "formal",
        ROOT / "artifacts" / "bootstrap",
        ROOT / "artifacts" / "figure_data",
        ROOT / "artifacts" / "table_data",
    )
    if any(path.exists() for path in forbidden_result_dirs):
        blocking_issues.append("a forbidden post-Stage-2 result directory exists")

    blocking_issues = list(dict.fromkeys(blocking_issues))
    passed = len(blocking_issues) == 0
    status = {
        "stage": 2,
        "stage_name": "RANDOM_LAYER",
        "status": "PASS" if passed else "FAIL",
        "blocking_issue_count": len(blocking_issues),
        "science_contract_changed": not science_contract_unchanged,
        "stage1_regression_passed": stage1_run.returncode == 0,
        "rng_address_tests_passed": (
            stage2_run.returncode == 0 and rng_validation["all_pass"]
        ),
        "golden_vector_count": golden_result["count"],
        "golden_vectors_passed": golden_result["passed"],
        "negative_event_index_passed": rng_validation["checks"]["signed_event_index"]
        and rng_validation["checks"]["signed_event_index_bounds"],
        "namespace_isolation_passed": rng_validation["checks"]["namespace_isolation"],
        "method_not_in_random_address": rng_validation["checks"][
            "method_absent_from_rng_api"
        ],
        "order_invariance_passed": rng_validation["checks"]["order_invariance"],
        "chunk_invariance_passed": rng_validation["checks"]["chunk_invariance"],
        "serial_parallel_invariance_passed": rng_validation["checks"][
            "serial_parallel_invariance"
        ],
        "stage3_executed": False,
        "pilot_executed": False,
        "formal_executed": False,
        "bootstrap_executed": False,
        "paper_result_figures_generated": False,
        "study_config_sha256": config_sha,
        "golden_vector_mode": golden_mode,
        "stage1_test_count": stage1_run.passed + stage1_run.failed,
        "stage1_passed_count": stage1_run.passed,
        "stage2_test_count": stage2_run.passed + stage2_run.failed,
        "stage2_passed_count": stage2_run.passed,
        "total_test_count": total_run.passed + total_run.failed,
        "total_passed_count": total_run.passed,
        "stop_after_stage2": True,
    }
    rng_validation["all_pass"] = passed
    rng_validation["golden_vector_mode"] = golden_mode
    rng_validation["golden_vector_count"] = golden_result["count"]
    rng_validation["study_config_sha256"] = config_sha
    rng_validation["blocking_issues"] = blocking_issues
    _write_json(STAGE2_DIR / "rng_validation.json", rng_validation)
    _write_json(STAGE2_DIR / "status.json", status)
    write_validation_report(
        status,
        config,
        rng_validation,
        golden_result,
        golden_mode,
        stage1_run,
        stage2_run,
        total_run,
        blocking_issues,
    )
    write_changed_files()
    return status
