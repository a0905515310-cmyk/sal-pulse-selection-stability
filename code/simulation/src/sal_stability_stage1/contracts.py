from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class ConditionSpec:
    ConditionOrdinal: int
    Code: str
    Scene: str
    HPRF_Hz: Optional[float] = None
    FDelay_s: Optional[float] = None


@dataclass(frozen=True)
class MethodSpec:
    MethodOrdinal: int
    Code: str
    Name: str


CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec(0, "NI", "NI"),
    ConditionSpec(1, "H100", "HPRF", 100_000.0),
    ConditionSpec(2, "H200", "HPRF", 200_000.0),
    ConditionSpec(3, "H300", "HPRF", 300_000.0),
    ConditionSpec(4, "H400", "HPRF", 400_000.0),
    ConditionSpec(5, "H500", "HPRF", 500_000.0),
    ConditionSpec(6, "F1", "IDF", FDelay_s=1e-6),
    ConditionSpec(7, "F2", "IDF", FDelay_s=2e-6),
    ConditionSpec(8, "F3", "IDF", FDelay_s=3e-6),
    ConditionSpec(9, "F4", "IDF", FDelay_s=4e-6),
    ConditionSpec(10, "HF200-1", "COMPOSITE", 200_000.0, 1e-6),
    ConditionSpec(11, "HF200-2", "COMPOSITE", 200_000.0, 2e-6),
    ConditionSpec(12, "HF200-3", "COMPOSITE", 200_000.0, 3e-6),
    ConditionSpec(13, "HF300-1", "COMPOSITE", 300_000.0, 1e-6),
    ConditionSpec(14, "HF300-2", "COMPOSITE", 300_000.0, 2e-6),
    ConditionSpec(15, "HF300-3", "COMPOSITE", 300_000.0, 3e-6),
    ConditionSpec(16, "HF500-1", "COMPOSITE", 500_000.0, 1e-6),
    ConditionSpec(17, "HF500-2", "COMPOSITE", 500_000.0, 2e-6),
    ConditionSpec(18, "HF500-3", "COMPOSITE", 500_000.0, 3e-6),
)

METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(0, "FIRST", "首脉冲准则"),
    MethodSpec(1, "LAST", "末脉冲准则"),
    MethodSpec(2, "T", "最优时序准则"),
    MethodSpec(3, "W", "脉宽单特征准则"),
    MethodSpec(4, "TW", "双特征联合准则"),
)


def condition_dicts() -> list[dict]:
    return [asdict(x) for x in CONDITIONS]


def method_dicts() -> list[dict]:
    return [asdict(x) for x in METHODS]
