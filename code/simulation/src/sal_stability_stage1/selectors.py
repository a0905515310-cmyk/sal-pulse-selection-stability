from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import TieRankCollisionError, assert_unique_tieranks


@dataclass(frozen=True, slots=True)
class CandidateView:
    """The complete, source-blind input visible to every Stage 4 selector."""

    observed_toa_s: np.ndarray
    observed_width_s: np.ndarray
    tie_rank_hi: np.ndarray
    tie_rank_lo: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            _readonly_1d(self.observed_toa_s, np.dtype(np.float64), "observed_toa_s"),
            _readonly_1d(
                self.observed_width_s,
                np.dtype(np.float64),
                "observed_width_s",
            ),
            _readonly_1d(self.tie_rank_hi, np.dtype(np.uint64), "tie_rank_hi"),
            _readonly_1d(self.tie_rank_lo, np.dtype(np.uint64), "tie_rank_lo"),
        )
        lengths = {len(values) for values in arrays}
        if len(lengths) != 1:
            raise ValueError("all CandidateView arrays must have the same length")
        if not np.all(np.isfinite(arrays[0])):
            raise ValueError("observed_toa_s must contain only finite values")
        if not np.all(np.isfinite(arrays[1])):
            raise ValueError("observed_width_s must contain only finite values")
        for name, values in zip(
            ("observed_toa_s", "observed_width_s", "tie_rank_hi", "tie_rank_lo"),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, values)

    def __len__(self) -> int:
        return len(self.observed_toa_s)


def _readonly_1d(values: object, dtype: np.dtype, field_name: str) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    result.setflags(write=False)
    return result


def _finite_scalar(value: object, field_name: str) -> np.float64:
    result = np.float64(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_scale(value: object, field_name: str) -> np.float64:
    result = _finite_scalar(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _validate_nonempty_unique(view: CandidateView) -> None:
    if not isinstance(view, CandidateView):
        raise TypeError("view must be a CandidateView")
    if len(view) == 0:
        raise ValueError("selectors require a nonempty CandidateView")
    assert_unique_tieranks(zip(view.tie_rank_hi, view.tie_rank_lo, strict=True))


def _finite_scores(scores: np.ndarray, score_name: str) -> np.ndarray:
    result = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{score_name} contains a non-finite float64 value")
    return result


def temporal_scores(
    view: CandidateView,
    ref_toa_s: float,
    sigma_dt_s: float,
) -> np.ndarray:
    """Return the frozen float64 q_t values without applying a threshold."""
    if not isinstance(view, CandidateView):
        raise TypeError("view must be a CandidateView")
    reference = _finite_scalar(ref_toa_s, "ref_toa_s")
    scale = _positive_scale(sigma_dt_s, "sigma_dt_s")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.square((view.observed_toa_s - reference) / scale)
    return _finite_scores(scores, "q_t")


def width_scores(
    view: CandidateView,
    ref_width_s: float,
    sigma_dwidth_s: float,
) -> np.ndarray:
    """Return the frozen float64 q_width values without applying a threshold."""
    if not isinstance(view, CandidateView):
        raise TypeError("view must be a CandidateView")
    reference = _finite_scalar(ref_width_s, "ref_width_s")
    scale = _positive_scale(sigma_dwidth_s, "sigma_dwidth_s")
    with np.errstate(over="ignore", invalid="ignore"):
        scores = np.square((view.observed_width_s - reference) / scale)
    return _finite_scores(scores, "q_width")


def joint_scores(
    view: CandidateView,
    ref_toa_s: float,
    ref_width_s: float,
    sigma_dt_s: float,
    sigma_dwidth_s: float,
) -> np.ndarray:
    """Return the frozen float64 joint score D = q_t + q_width."""
    with np.errstate(over="ignore", invalid="ignore"):
        scores = temporal_scores(view, ref_toa_s, sigma_dt_s) + width_scores(
            view,
            ref_width_s,
            sigma_dwidth_s,
        )
    return _finite_scores(scores, "D")


def _minimum_tierank_position(view: CandidateView, positions: np.ndarray) -> int:
    return int(
        min(
            (int(position) for position in positions),
            key=lambda position: (
                int(view.tie_rank_hi[position]),
                int(view.tie_rank_lo[position]),
            ),
        )
    )


def _select_primary_min(view: CandidateView, primary: np.ndarray) -> int:
    _validate_nonempty_unique(view)
    scores = _finite_scores(primary, "primary score")
    if scores.shape != (len(view),):
        raise ValueError("primary score must align with CandidateView")
    best = np.min(scores)
    tied_positions = np.flatnonzero(scores == best)
    return _minimum_tierank_position(view, tied_positions)


def _select_primary_max(view: CandidateView, primary: np.ndarray) -> int:
    _validate_nonempty_unique(view)
    scores = _finite_scores(primary, "primary score")
    if scores.shape != (len(view),):
        raise ValueError("primary score must align with CandidateView")
    best = np.max(scores)
    tied_positions = np.flatnonzero(scores == best)
    return _minimum_tierank_position(view, tied_positions)


def select_first(view: CandidateView) -> int:
    return _select_primary_min(view, view.observed_toa_s)


def select_last(view: CandidateView) -> int:
    return _select_primary_max(view, view.observed_toa_s)


def select_t(
    view: CandidateView,
    ref_toa_s: float,
    sigma_dt_s: float,
) -> int:
    _validate_nonempty_unique(view)
    return _select_primary_min(
        view,
        temporal_scores(view, ref_toa_s, sigma_dt_s),
    )


def select_w(
    view: CandidateView,
    ref_width_s: float,
    sigma_dwidth_s: float,
) -> int:
    _validate_nonempty_unique(view)
    return _select_primary_min(
        view,
        width_scores(view, ref_width_s, sigma_dwidth_s),
    )


def select_tw(
    view: CandidateView,
    ref_toa_s: float,
    ref_width_s: float,
    sigma_dt_s: float,
    sigma_dwidth_s: float,
) -> int:
    _validate_nonempty_unique(view)
    return _select_primary_min(
        view,
        joint_scores(
            view,
            ref_toa_s,
            ref_width_s,
            sigma_dt_s,
            sigma_dwidth_s,
        ),
    )


__all__ = [
    "CandidateView",
    "TieRankCollisionError",
    "joint_scores",
    "select_first",
    "select_last",
    "select_t",
    "select_tw",
    "select_w",
    "temporal_scores",
    "width_scores",
]
