"""Unit conversion and biological normalisation utilities."""

from __future__ import annotations

import numpy as np

from .skeleton import SkeletonDefinition

# Common length unit factors (relative to metres)
UNIT_FACTORS: dict[str, float] = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
}


def unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """Return the multiplicative factor to convert *from_unit* to *to_unit*.

    Parameters
    ----------
    from_unit, to_unit : str
        Unit strings — one of ``'mm'``, ``'cm'``, ``'m'``.

    Returns
    -------
    float
        The conversion factor.

    Raises
    ------
    ValueError
        If either unit is not recognised.
    """
    if from_unit not in UNIT_FACTORS:
        msg = f"Unknown unit '{from_unit}'. Known: {list(UNIT_FACTORS)}"
        raise ValueError(msg)
    if to_unit not in UNIT_FACTORS:
        msg = f"Unknown unit '{to_unit}'. Known: {list(UNIT_FACTORS)}"
        raise ValueError(msg)
    return UNIT_FACTORS[from_unit] / UNIT_FACTORS[to_unit]


def compute_wingspan(shape: np.ndarray, skeleton: SkeletonDefinition) -> float:
    """Compute wingspan from left/right wingtip markers.

    Parameters
    ----------
    shape : np.ndarray
        Marker positions, shape ``(n_markers, 3)`` or ``(1, n_markers, 3)``.
    skeleton : SkeletonDefinition
        The skeleton definition (used to look up wingtip marker names).

    Returns
    -------
    float
        Euclidean distance between the left and right wingtip markers.

    Raises
    ------
    ValueError
        If wingtip markers cannot be found.
    """
    if shape.ndim == 3:
        shape = shape[0]

    # Try common wingtip naming conventions
    wingtip_candidates = [
        ("left_wingtip", "right_wingtip"),
        ("left_firstprimary_tip", "right_firstprimary_tip"),
    ]

    names = skeleton.all_marker_names
    for left_name, right_name in wingtip_candidates:
        if left_name in names and right_name in names:
            left_idx = names.index(left_name)
            right_idx = names.index(right_name)
            return float(np.linalg.norm(shape[left_idx] - shape[right_idx]))

    msg = "Could not find wingtip markers in the skeleton definition."
    raise ValueError(msg)


def compute_body_length(shape: np.ndarray, skeleton: SkeletonDefinition) -> float:
    """Compute body length from head to tail markers.

    Parameters
    ----------
    shape : np.ndarray
        Marker positions, shape ``(n_markers, 3)`` or ``(1, n_markers, 3)``.
    skeleton : SkeletonDefinition
        The skeleton definition.

    Returns
    -------
    float
        Euclidean distance between head and tail markers.

    Raises
    ------
    ValueError
        If head/tail markers cannot be found.
    """
    if shape.ndim == 3:
        shape = shape[0]

    names = skeleton.all_marker_names

    # Head candidates
    head_candidates = ["hood", "head", "clypeus"]
    # Tail candidates
    tail_candidates = ["tailpack", "centre_body_base", "spinneret"]

    head_idx = None
    for h in head_candidates:
        if h in names:
            head_idx = names.index(h)
            break

    tail_idx = None
    for t in tail_candidates:
        if t in names:
            tail_idx = names.index(t)
            break

    if head_idx is None or tail_idx is None:
        msg = "Could not find head/tail markers in the skeleton definition."
        raise ValueError(msg)

    return float(np.linalg.norm(shape[head_idx] - shape[tail_idx]))
