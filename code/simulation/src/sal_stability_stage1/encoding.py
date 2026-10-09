from __future__ import annotations

import numpy as np

K_DEFAULT = 200
INITIAL_STATE = np.array(
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    dtype=np.uint8,
)
DELTA_T_BASE_S = np.float64(70e-3)
DELTA_T_STEP_S = np.float64(2.5e-6)
WIDTH_BASE_S = np.float64(250e-9)
WIDTH_STEP_S = np.float64(10e-9)
EXPECTED_199_INTERVAL_SUM_S = np.float64(29.5308825)


def _bit(state: np.ndarray, s_index: int) -> int:
    """Return s_{s_index}; state order is [s16, ..., s1]."""
    return int(state[16 - s_index])


def next_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.uint8)
    if state.shape != (16,):
        raise ValueError("state must contain exactly 16 bits")
    if np.any((state != 0) & (state != 1)):
        raise ValueError("state entries must be binary")
    feedback = _bit(state, 15) ^ _bit(state, 14) ^ _bit(state, 12) ^ _bit(state, 1)
    return np.concatenate((np.array([feedback], dtype=np.uint8), state[:-1]))


def state_integer(state: np.ndarray) -> np.uint16:
    value = 0
    for bit in np.asarray(state, dtype=np.uint8):
        value = (value << 1) | int(bit)
    if value == 0:
        raise ValueError("all-zero LFSR state is invalid")
    return np.uint16(value)


def width_level(state: np.ndarray) -> np.uint8:
    v3 = _bit(state, 15) ^ _bit(state, 14) ^ _bit(state, 10) ^ _bit(state, 5)
    v2 = _bit(state, 16) ^ _bit(state, 6) ^ _bit(state, 3) ^ _bit(state, 1)
    v1 = _bit(state, 11) ^ _bit(state, 9) ^ _bit(state, 7) ^ _bit(state, 4)
    v0 = _bit(state, 13) ^ _bit(state, 12) ^ _bit(state, 8) ^ _bit(state, 2)
    return np.uint8(8 * v3 + 4 * v2 + 2 * v1 + v0)


def generate_reference_arrays(k_count: int = K_DEFAULT) -> dict[str, np.ndarray]:
    if k_count < 1:
        raise ValueError("k_count must be positive")

    states = np.empty((k_count, 16), dtype=np.uint8)
    state_u16 = np.empty(k_count, dtype=np.uint16)
    width_levels = np.empty(k_count, dtype=np.uint8)
    ref_width_s = np.empty(k_count, dtype=np.float64)
    interval_levels = np.empty(max(0, k_count - 1), dtype=np.uint16)
    delta_t_s = np.empty(max(0, k_count - 1), dtype=np.float64)

    state = INITIAL_STATE.copy()
    for k in range(k_count):
        states[k] = state
        u = state_integer(state)
        state_u16[k] = u
        q = width_level(state)
        width_levels[k] = q
        ref_width_s[k] = WIDTH_BASE_S + np.float64(q) * WIDTH_STEP_S
        if k < k_count - 1:
            m = np.uint16(int(u) - 1)
            interval_levels[k] = m
            delta_t_s[k] = DELTA_T_BASE_S + np.float64(m) * DELTA_T_STEP_S
            state = next_state(state)

    return {
        "state_bits": states,
        "state_u16": state_u16,
        "interval_level": interval_levels,
        "delta_t_s": delta_t_s,
        "width_level": width_levels,
        "ref_width_s": ref_width_s,
    }


def verify_full_period() -> bool:
    state = INITIAL_STATE.copy()
    seen: set[bytes] = set()
    for _ in range(65535):
        key = state.tobytes()
        if key in seen:
            return False
        seen.add(key)
        state = next_state(state)
    return len(seen) == 65535 and np.array_equal(state, INITIAL_STATE)
