from __future__ import annotations

import csv
import operator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np


def _strict_integer(value: object, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _finite_float(value: object, field_name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CycleDiagnostic:
    """Read-only post-selection execution log; never an input to a selector."""

    Namespace: str
    ConditionOrdinal: int
    ConditionCode: str
    RepeatID: int
    Cycle: int
    MethodOrdinal: int
    MethodCode: str
    GateLeft: float
    GateRight: float
    CandidateCount: int
    SelectedPos: int | None
    SelectedSourceAfterSelection: int
    RefTOABefore: float
    RefTOAAfter: float

    def __post_init__(self) -> None:
        namespace = str(self.Namespace)
        if namespace not in {"PILOT", "FORMAL"}:
            raise ValueError("Namespace must be PILOT or FORMAL")
        condition_ordinal = _strict_integer(
            self.ConditionOrdinal, "ConditionOrdinal"
        )
        repeat_id = _strict_integer(self.RepeatID, "RepeatID")
        cycle = _strict_integer(self.Cycle, "Cycle")
        method_ordinal = _strict_integer(self.MethodOrdinal, "MethodOrdinal")
        candidate_count = _strict_integer(self.CandidateCount, "CandidateCount")
        source = _strict_integer(
            self.SelectedSourceAfterSelection,
            "SelectedSourceAfterSelection",
        )
        if not 0 <= condition_ordinal < 19:
            raise ValueError("ConditionOrdinal must lie in 0..18")
        if repeat_id < 1:
            raise ValueError("RepeatID must be positive")
        if not 1 <= cycle <= 200:
            raise ValueError("Cycle must lie in 1..200")
        if not 0 <= method_ordinal < 5:
            raise ValueError("MethodOrdinal must lie in 0..4")
        if candidate_count < 0:
            raise ValueError("CandidateCount cannot be negative")
        if source not in {0, 1, 2, 3}:
            raise ValueError("SelectedSourceAfterSelection must lie in 0..3")

        selected_pos: int | None
        if self.SelectedPos is None:
            selected_pos = None
            if candidate_count != 0 or source != 0:
                raise ValueError("an absent selection requires zero candidates and source N")
        else:
            selected_pos = _strict_integer(self.SelectedPos, "SelectedPos")
            if not 0 <= selected_pos < candidate_count:
                raise ValueError("SelectedPos is outside the candidate view")
            if source == 0:
                raise ValueError("a present selection cannot have source N")

        gate_left = _finite_float(self.GateLeft, "GateLeft")
        gate_right = _finite_float(self.GateRight, "GateRight")
        if gate_right <= gate_left:
            raise ValueError("diagnostic gate must have positive width")
        ref_before = _finite_float(self.RefTOABefore, "RefTOABefore")
        ref_after = _finite_float(self.RefTOAAfter, "RefTOAAfter")
        condition_code = str(self.ConditionCode)
        method_code = str(self.MethodCode)
        if not condition_code or not method_code:
            raise ValueError("condition and method codes cannot be empty")

        object.__setattr__(self, "Namespace", namespace)
        object.__setattr__(self, "ConditionOrdinal", condition_ordinal)
        object.__setattr__(self, "ConditionCode", condition_code)
        object.__setattr__(self, "RepeatID", repeat_id)
        object.__setattr__(self, "Cycle", cycle)
        object.__setattr__(self, "MethodOrdinal", method_ordinal)
        object.__setattr__(self, "MethodCode", method_code)
        object.__setattr__(self, "GateLeft", gate_left)
        object.__setattr__(self, "GateRight", gate_right)
        object.__setattr__(self, "CandidateCount", candidate_count)
        object.__setattr__(self, "SelectedPos", selected_pos)
        object.__setattr__(self, "SelectedSourceAfterSelection", source)
        object.__setattr__(self, "RefTOABefore", ref_before)
        object.__setattr__(self, "RefTOAAfter", ref_after)


DIAGNOSTIC_FIELD_NAMES = tuple(field.name for field in fields(CycleDiagnostic))


def canonical_diagnostics(
    records: Iterable[CycleDiagnostic],
) -> tuple[CycleDiagnostic, ...]:
    materialized = tuple(records)
    if any(not isinstance(record, CycleDiagnostic) for record in materialized):
        raise TypeError("all diagnostic records must be CycleDiagnostic values")
    keys = [
        (
            record.ConditionOrdinal,
            record.RepeatID,
            record.Cycle,
            record.MethodOrdinal,
        )
        for record in materialized
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("diagnostic records contain duplicate cycle-method keys")
    return tuple(
        sorted(
            materialized,
            key=lambda record: (
                record.ConditionOrdinal,
                record.RepeatID,
                record.Cycle,
                record.MethodOrdinal,
            ),
        )
    )


def write_diagnostics_csv(
    path: Path, records: Iterable[CycleDiagnostic]
) -> None:
    path = Path(path)
    rows = canonical_diagnostics(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(DIAGNOSTIC_FIELD_NAMES),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in rows:
            writer.writerow(asdict(record))


__all__ = [
    "CycleDiagnostic",
    "DIAGNOSTIC_FIELD_NAMES",
    "canonical_diagnostics",
    "write_diagnostics_csv",
]
