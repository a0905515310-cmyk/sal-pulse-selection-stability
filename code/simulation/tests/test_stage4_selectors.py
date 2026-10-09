from __future__ import annotations

import inspect
from dataclasses import fields

import numpy as np
import pytest

from sal_stability_stage1.selectors import (
    CandidateView,
    TieRankCollisionError,
    joint_scores,
    select_first,
    select_last,
    select_t,
    select_tw,
    select_w,
    temporal_scores,
    width_scores,
)


def make_view(
    toa=(3.0, 1.0, 2.0),
    width=(10.0, 20.0, 30.0),
    hi=(30, 20, 10),
    lo=(3, 2, 1),
) -> CandidateView:
    return CandidateView(toa, width, hi, lo)


SELECTOR_CALLS = (
    lambda view: select_first(view),
    lambda view: select_last(view),
    lambda view: select_t(view, 0.0, 1.0),
    lambda view: select_w(view, 0.0, 1.0),
    lambda view: select_tw(view, 0.0, 0.0, 1.0, 1.0),
)


def test_candidate_view_has_exact_source_blind_fields_and_read_only_arrays():
    view = make_view()
    assert [field.name for field in fields(CandidateView)] == [
        "observed_toa_s",
        "observed_width_s",
        "tie_rank_hi",
        "tie_rank_lo",
    ]
    assert view.observed_toa_s.dtype == np.float64
    assert view.observed_width_s.dtype == np.float64
    assert view.tie_rank_hi.dtype == np.uint64
    assert view.tie_rank_lo.dtype == np.uint64
    assert all(
        not getattr(view, name).flags.writeable
        for name in (
            "observed_toa_s",
            "observed_width_s",
            "tie_rank_hi",
            "tie_rank_lo",
        )
    )
    with pytest.raises(ValueError):
        view.observed_toa_s[0] = 0.0


def test_candidate_view_rejects_misaligned_non_1d_and_nonfinite_arrays():
    with pytest.raises(ValueError, match="same length"):
        CandidateView([1.0], [1.0, 2.0], [1], [1])
    with pytest.raises(ValueError, match="one-dimensional"):
        CandidateView([[1.0]], [1.0], [1], [1])
    with pytest.raises(ValueError, match="finite"):
        CandidateView([np.nan], [1.0], [1], [1])


def test_selector_signatures_and_source_blind_module_surface():
    banned = (
        "source",
        "state_code",
        "true_toa",
        "true_width",
        "condition",
        "scene",
        "is_guidance",
        "is_h",
        "is_f",
    )
    for selector in (select_first, select_last, select_t, select_w, select_tw):
        names = tuple(inspect.signature(selector).parameters)
        assert names[0] == "view"
        assert all(fragment not in name.lower() for name in names for fragment in banned)


@pytest.mark.parametrize("selector", SELECTOR_CALLS)
def test_single_candidate_is_always_selected(selector):
    assert selector(make_view((9.0,), (999.0,), (4,), (8,))) == 0


def test_first_and_last_use_only_observed_toa():
    original = make_view()
    changed_width = make_view(width=(1e99, -1e99, 0.0))
    assert select_first(original) == select_first(changed_width) == 1
    assert select_last(original) == select_last(changed_width) == 0


def test_t_uses_squared_normalized_toa_and_ignores_width():
    view = make_view(toa=(-3.0, 1.0, 2.0), width=(1.0, 2.0, 3.0))
    changed_width = make_view(toa=(-3.0, 1.0, 2.0), width=(9e8, -4e8, 7e8))
    np.testing.assert_array_equal(temporal_scores(view, 0.0, 2.0), [2.25, 0.25, 1.0])
    assert select_t(view, 0.0, 2.0) == 1
    assert select_t(changed_width, 0.0, 2.0) == 1


def test_w_uses_squared_normalized_width_and_ignores_toa():
    widths = np.array([280e-9, 301e-9, 315e-9], dtype=np.float64)
    view = make_view(toa=(1.0, 2.0, 3.0), width=widths)
    changed_toa = make_view(toa=(9e8, -4e8, 7e8), width=widths)
    expected = np.square((widths - np.float64(300e-9)) / np.float64(1.5e-9))
    np.testing.assert_array_equal(width_scores(view, 300e-9, 1.5e-9), expected)
    assert select_w(view, 300e-9, 1.5e-9) == 1
    assert select_w(changed_toa, 300e-9, 1.5e-9) == 1


def test_tw_score_is_exactly_q_t_plus_q_width():
    view = make_view(toa=(-2.0, 1.0, 3.0), width=(0.0, 4.0, 1.0))
    q_t = np.square((view.observed_toa_s - 0.5) / 2.0)
    q_width = np.square((view.observed_width_s - 1.5) / 0.5)
    expected = q_t + q_width
    actual = joint_scores(view, 0.5, 1.5, 2.0, 0.5)
    np.testing.assert_array_equal(actual, expected)
    assert select_tw(view, 0.5, 1.5, 2.0, 0.5) == int(np.argmin(expected))


def test_tw_uses_both_features_and_can_change_selection():
    base = make_view(toa=(0.0, 2.0), width=(3.0, 0.0), hi=(2, 1), lo=(0, 0))
    assert select_tw(base, 0.0, 0.0, 1.0, 1.0) == 1
    width_changed = make_view(
        toa=(0.0, 2.0), width=(0.0, 3.0), hi=(2, 1), lo=(0, 0)
    )
    assert select_tw(width_changed, 0.0, 0.0, 1.0, 1.0) == 0
    toa_changed = make_view(
        toa=(3.0, 0.0), width=(0.0, 3.0), hi=(2, 1), lo=(0, 0)
    )
    assert select_tw(toa_changed, 0.0, 0.0, 1.0, 1.0) == 1


@pytest.mark.parametrize(
    ("selector", "view"),
    (
        (select_first, make_view((1.0, 1.0), (7.0, 8.0), (9, 1), (0, 0))),
        (select_last, make_view((1.0, 1.0), (7.0, 8.0), (9, 1), (0, 0))),
        (
            lambda view: select_t(view, 0.0, 1.0),
            make_view((-1.0, 1.0), (7.0, 8.0), (9, 1), (0, 0)),
        ),
        (
            lambda view: select_w(view, 0.0, 1.0),
            make_view((7.0, 8.0), (-1.0, 1.0), (9, 1), (0, 0)),
        ),
        (
            lambda view: select_tw(view, 0.0, 0.0, 1.0, 1.0),
            make_view((-1.0, 1.0), (1.0, -1.0), (9, 1), (0, 0)),
        ),
    ),
)
def test_exact_primary_ties_choose_minimum_128_bit_tierank(selector, view):
    assert selector(view) == 1


@pytest.mark.parametrize("selector", SELECTOR_CALLS)
def test_complete_tierank_collision_stops_every_selector(selector):
    view = make_view((1.0, 2.0), (3.0, 4.0), (7, 7), (11, 11))
    with pytest.raises(TieRankCollisionError):
        selector(view)


def test_near_but_unequal_primary_values_are_not_ties():
    lower = np.float64(1.0)
    higher = np.nextafter(lower, np.float64(np.inf))
    view = make_view((lower, higher), (lower, higher), (9, 0), (9, 0))
    assert select_first(view) == 0
    assert select_last(view) == 1
    assert select_w(view, 0.0, 1.0) == 0
    joint = make_view((0.0, 0.0), (lower, higher), (9, 0), (9, 0))
    assert select_tw(joint, 0.0, 0.0, 1.0, 1.0) == 0


def _selected_rank(view: CandidateView, selector) -> tuple[int, int]:
    position = selector(view)
    return int(view.tie_rank_hi[position]), int(view.tie_rank_lo[position])


def _deterministic_permutation(size: int, seed: int) -> np.ndarray:
    positions = list(range(size))
    state = (seed + 1) & 0xFFFFFFFF
    for upper in range(size - 1, 0, -1):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        swap = state % (upper + 1)
        positions[upper], positions[swap] = positions[swap], positions[upper]
    return np.asarray(positions, dtype=np.int64)


def test_all_selectors_are_invariant_to_hundreds_of_input_permutations():
    selectors = (
        lambda view: select_first(view),
        lambda view: select_last(view),
        lambda view: select_t(view, 0.25, 1.7),
        lambda view: select_w(view, -0.4, 0.8),
        lambda view: select_tw(view, 0.25, -0.4, 1.7, 0.8),
    )
    for case in range(240):
        size = 2 + case % 11
        identity = np.arange(size, dtype=np.float64)
        toa = np.sin((case + 1.25) * (identity + 0.375)) + identity * 1e-7
        width = np.cos((case + 0.75) * (identity + 0.625)) - identity * 1e-7
        hi = np.arange(case * 32, case * 32 + size, dtype=np.uint64)
        lo = np.arange(size, 0, -1, dtype=np.uint64)
        view = CandidateView(toa, width, hi, lo)
        expected = tuple(_selected_rank(view, selector) for selector in selectors)
        for permutation in (
            np.arange(size - 1, -1, -1),
            _deterministic_permutation(size, case + 20260823),
        ):
            permuted = CandidateView(
                toa[permutation],
                width[permutation],
                hi[permutation],
                lo[permutation],
            )
            actual = tuple(
                _selected_rank(permuted, selector) for selector in selectors
            )
            assert actual == expected


def test_source_metadata_permutation_cannot_change_selector_positions():
    view = make_view(toa=(2.0, -1.0, 4.0), width=(1.0, 3.0, -2.0))
    source_metadata = np.array([1, 2, 3], dtype=np.uint8)
    expected_positions = tuple(selector(view) for selector in SELECTOR_CALLS)
    for metadata_only_permutation in ([3, 1, 2], [2, 3, 1], [1, 3, 2]):
        source_metadata[:] = metadata_only_permutation
        assert tuple(selector(view) for selector in SELECTOR_CALLS) == expected_positions


@pytest.mark.parametrize("selector", SELECTOR_CALLS)
def test_selectors_reject_empty_views_instead_of_returning_n(selector):
    empty = CandidateView([], [], [], [])
    with pytest.raises(ValueError, match="nonempty"):
        selector(empty)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_score_scales_must_be_finite_and_positive(scale):
    view = make_view()
    with pytest.raises(ValueError):
        temporal_scores(view, 0.0, scale)
    with pytest.raises(ValueError):
        width_scores(view, 0.0, scale)
