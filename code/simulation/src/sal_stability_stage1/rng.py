from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Iterable, Sequence


MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
UINT32_SCALE = 1 << 32

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
PHILOX_ROUNDS = 10

MASTER_SEED_IDENTIFIER = "SAL_STABILITY_MASTER_SEED_V1"
MASTER_SEED_BYTES = hashlib.sha256(MASTER_SEED_IDENTIFIER.encode("ascii")).digest()
MASTER_SEED_SHA256 = MASTER_SEED_BYTES.hex()


class RandomNamespace(IntEnum):
    PILOT = 1
    FORMAL = 2
    BOOTSTRAP = 3


class VariableFamily(IntEnum):
    INITIAL_SYNC = 1
    G_TOA_ERROR = 2
    G_WIDTH_ERROR = 3
    H_PHASE = 4
    H_TRUE_WIDTH_LEVEL = 5
    H_TOA_ERROR = 6
    H_WIDTH_ERROR = 7
    F_TOA_ERROR = 8
    F_WIDTH_ERROR = 9
    TIE_G = 10
    TIE_H = 11
    TIE_F = 12
    BOOTSTRAP_INDEX = 13


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an integer") from exc


def _enum_member(enum_type: type[IntEnum], value: object, field_name: str) -> IntEnum:
    integer = _integer(value, field_name)
    try:
        return enum_type(integer)
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {integer}") from exc


@dataclass(frozen=True, slots=True)
class RandomAddress:
    namespace: RandomNamespace
    condition_ordinal: int
    repeat_id: int
    variable_family: VariableFamily
    event_index: int

    def __post_init__(self) -> None:
        namespace = _enum_member(RandomNamespace, self.namespace, "namespace")
        variable_family = _enum_member(
            VariableFamily, self.variable_family, "variable_family"
        )
        condition_ordinal = _integer(self.condition_ordinal, "condition_ordinal")
        repeat_id = _integer(self.repeat_id, "repeat_id")
        event_index = _integer(self.event_index, "event_index")

        if not 0 <= condition_ordinal <= 18:
            raise ValueError("condition_ordinal must be in [0, 18]")
        if not 1 <= repeat_id <= MASK32:
            raise ValueError("repeat_id must be in [1, 2^32-1]")
        if not -(1 << 63) <= event_index <= (1 << 63) - 1:
            raise ValueError("event_index must be a signed int64")

        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "condition_ordinal", condition_ordinal)
        object.__setattr__(self, "repeat_id", repeat_id)
        object.__setattr__(self, "variable_family", variable_family)
        object.__setattr__(self, "event_index", event_index)


def _uint32(value: object, field_name: str) -> int:
    integer = _integer(value, field_name)
    if not 0 <= integer <= MASK32:
        raise ValueError(f"{field_name} must be a uint32")
    return integer


def _uint64(value: object, field_name: str) -> int:
    integer = _integer(value, field_name)
    if not 0 <= integer <= MASK64:
        raise ValueError(f"{field_name} must be a uint64")
    return integer


def _word_tuple(values: Sequence[int], count: int, field_name: str) -> tuple[int, ...]:
    try:
        words = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence of {count} uint32 words") from exc
    if len(words) != count:
        raise ValueError(f"{field_name} must contain exactly {count} words")
    return tuple(_uint32(word, f"{field_name}[{index}]") for index, word in enumerate(words))


def mulhilo32(a: int, b: int) -> tuple[int, int]:
    """Return the explicit low and high uint32 halves of a 32x32 product."""
    a_u32 = _uint32(a, "a")
    b_u32 = _uint32(b, "b")
    product = a_u32 * b_u32
    low32 = product & MASK32
    high32 = (product >> 32) & MASK32
    return low32, high32


def philox4x32_10(
    counter: Sequence[int], key: Sequence[int]
) -> tuple[int, int, int, int]:
    """Canonical scalar Random123 Philox-4x32 with exactly ten rounds."""
    ctr = _word_tuple(counter, 4, "counter")
    key_words = _word_tuple(key, 2, "key")
    c0, c1, c2, c3 = ctr
    k0, k1 = key_words

    for round_index in range(PHILOX_ROUNDS):
        lo0, hi0 = mulhilo32(PHILOX_M0, c0)
        lo1, hi1 = mulhilo32(PHILOX_M1, c2)
        c0, c1, c2, c3 = (
            (hi1 ^ c1 ^ k0) & MASK32,
            lo1,
            (hi0 ^ c3 ^ k1) & MASK32,
            lo0,
        )
        if round_index + 1 < PHILOX_ROUNDS:
            k0 = (k0 + PHILOX_W0) & MASK32
            k1 = (k1 + PHILOX_W1) & MASK32

    return c0, c1, c2, c3


@lru_cache(maxsize=None)
def derive_philox_key(
    namespace: RandomNamespace, variable_family: VariableFamily
) -> tuple[int, int]:
    """Derive and cache the two-word key for one namespace/family pair."""
    namespace_member = _enum_member(RandomNamespace, namespace, "namespace")
    family_member = _enum_member(
        VariableFamily, variable_family, "variable_family"
    )
    payload = (
        MASTER_SEED_BYTES
        + int(namespace_member).to_bytes(4, "little", signed=False)
        + int(family_member).to_bytes(4, "little", signed=False)
    )
    digest = hashlib.sha256(payload).digest()
    return (
        int.from_bytes(digest[0:4], "little", signed=False),
        int.from_bytes(digest[4:8], "little", signed=False),
    )


def encode_event_index(event_index: int) -> tuple[int, int]:
    """Encode a signed int64 as low/high uint32 words in two's complement."""
    event = _integer(event_index, "event_index")
    if not -(1 << 63) <= event <= (1 << 63) - 1:
        raise ValueError("event_index must be a signed int64")
    event_u64 = event & MASK64
    return event_u64 & MASK32, (event_u64 >> 32) & MASK32


def counter_words(address: RandomAddress) -> tuple[int, int, int, int]:
    if not isinstance(address, RandomAddress):
        raise TypeError("address must be a RandomAddress")
    event_low, event_high = encode_event_index(address.event_index)
    return (
        address.condition_ordinal & MASK32,
        address.repeat_id & MASK32,
        event_low,
        event_high,
    )


def raw_words(address: RandomAddress) -> tuple[int, int, int, int]:
    if not isinstance(address, RandomAddress):
        raise TypeError("address must be a RandomAddress")
    key = derive_philox_key(address.namespace, address.variable_family)
    return philox4x32_10(counter_words(address), key)


def uniform_from_uint32(word: int) -> float:
    word_u32 = _uint32(word, "word")
    uniform = (float(word_u32) + 0.5) / UINT32_SCALE
    if not 0.0 < uniform < 1.0:
        raise ArithmeticError("uint32 midpoint conversion left the open unit interval")
    return uniform


def random_uniform(address: RandomAddress) -> float:
    return uniform_from_uint32(raw_words(address)[0])


def random_normal(address: RandomAddress) -> float:
    words = raw_words(address)
    u1 = uniform_from_uint32(words[0])
    u2 = uniform_from_uint32(words[1])
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


TieRank = tuple[int, int]
TIE_FAMILIES = frozenset(
    (VariableFamily.TIE_G, VariableFamily.TIE_H, VariableFamily.TIE_F)
)


def random_tierank(address: RandomAddress) -> TieRank:
    if not isinstance(address, RandomAddress):
        raise TypeError("address must be a RandomAddress")
    if address.variable_family not in TIE_FAMILIES:
        raise ValueError("TieRank requires TIE_G, TIE_H, or TIE_F")
    w0, w1, w2, w3 = raw_words(address)
    tie_hi = ((w0 << 32) | w1) & MASK64
    tie_lo = ((w2 << 32) | w3) & MASK64
    return tie_hi, tie_lo


class TieRankCollisionError(RuntimeError):
    pass


def assert_unique_tieranks(tieranks: Iterable[TieRank]) -> None:
    """Stop on a complete 128-bit collision; no fallback ordering is allowed."""
    seen: set[TieRank] = set()
    for index, rank in enumerate(tieranks):
        try:
            hi, lo = rank
        except (TypeError, ValueError) as exc:
            raise ValueError("each TieRank must contain exactly two uint64 words") from exc
        normalized = (
            _uint64(hi, f"tieranks[{index}][0]"),
            _uint64(lo, f"tieranks[{index}][1]"),
        )
        if normalized in seen:
            raise TieRankCollisionError(
                f"complete 128-bit TieRank collision at input index {index}"
            )
        seen.add(normalized)
