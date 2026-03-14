"""Bilateral / unilateral conversion and mirroring.

Pure functions operating on numpy arrays + a SkeletonDefinition.
No Animal3D coupling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .skeleton import SkeletonDefinition


def mirror_to_bilateral(keypoints: np.ndarray) -> np.ndarray:
    """Mirror right-side markers across the x-axis to create the left side.

    Assumes alternating ``left, right, left, right, ...`` marker order.

    Parameters
    ----------
    keypoints : np.ndarray
        Right-side only keypoints, shape ``(n_frames, n_right, 3)``.

    Returns
    -------
    np.ndarray
        Full bilateral keypoints, shape ``(n_frames, n_right * 2, 3)``.
    """
    mirrored = np.copy(keypoints)
    mirrored[:, :, 0] *= -1

    n_frames, n_markers, n_coords = keypoints.shape
    full = np.empty((n_frames, n_markers * 2, n_coords), dtype=keypoints.dtype)
    full[:, 0::2, :] = mirrored   # left at even indices
    full[:, 1::2, :] = keypoints  # right at odd indices
    return full


def make_unilateral(
    motion_data: np.ndarray,
    skeleton: SkeletonDefinition,
    info_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame | None]:
    """Split bilateral data into doubled unilateral observations.

    Left markers are mirrored across the x-axis so all observations
    look like right-side data.  Frame count doubles.

    Parameters
    ----------
    motion_data : np.ndarray
        Bilateral motion data, shape ``(n_frames, n_markers, 3)``.
    skeleton : SkeletonDefinition
        Skeleton definition (used for marker pairing).
    info_df : pd.DataFrame, optional
        Per-frame metadata.  If provided, rows are doubled to match.

    Returns
    -------
    tuple
        ``(unilateral_data, is_left, unilateral_info_df)``

        - ``unilateral_data``: shape ``(n_frames * 2, n_right + n_centre, 3)``
        - ``is_left``: boolean array, ``True`` for frames from the left side
        - ``unilateral_info_df``: doubled info DataFrame, or ``None``
    """
    motion_data_copy = np.copy(motion_data)
    info_df_copy = info_df.copy() if info_df is not None else None

    # Get marker pairs and centres
    pairs = skeleton.get_marker_pairs()
    left_names = [p[0] for p in pairs]
    right_names = [p[1] for p in pairs]
    centre_names = skeleton.get_centre_markers()
    # Filter to markers actually in the skeleton
    centre_names = [c for c in centre_names if c in skeleton.all_marker_names]

    all_names = skeleton.all_marker_names
    left_indices = [all_names.index(m) for m in left_names]
    right_indices = [all_names.index(m) for m in right_names]
    centre_indices = [all_names.index(m) for m in centre_names] if centre_names else []

    # Validate left-right positions
    valid_frames = validate_left_right(motion_data_copy, left_indices, right_indices)
    motion_data_copy = motion_data_copy[valid_frames]

    if len(motion_data_copy) == 0:
        msg = "No valid frames found after left-right validation."
        raise ValueError(msg)

    if info_df_copy is not None:
        if len(info_df_copy) != len(motion_data):
            msg = "info_df must have same number of rows as motion_data frames."
            raise ValueError(msg)
        info_df_copy = info_df_copy.iloc[valid_frames]

    # Extract left and right data
    left_data = motion_data_copy[:, left_indices, :]
    right_data = motion_data_copy[:, right_indices, :]
    centre_data = (
        motion_data_copy[:, centre_indices, :] if centre_indices else None
    )

    # Mirror left side across x-axis
    left_mirrored = np.copy(left_data)
    left_mirrored[..., 0] *= -1

    # Stack: mirrored left first, then right
    paired_data = np.concatenate((left_mirrored, right_data), axis=0)

    if centre_data is not None:
        centre_doubled = np.concatenate((centre_data, centre_data), axis=0)
        unilateral_data = np.concatenate((paired_data, centre_doubled), axis=1)
    else:
        unilateral_data = paired_data

    # Double the info DataFrame
    if info_df_copy is not None:
        unilateral_info_df = pd.concat(
            [info_df_copy, info_df_copy], axis=0, ignore_index=True
        )
    else:
        unilateral_info_df = None

    # Create is_left mask
    n_valid = len(motion_data_copy)
    is_left = np.zeros(n_valid * 2, dtype=bool)
    is_left[:n_valid] = True

    return unilateral_data, is_left, unilateral_info_df


def make_bilateral(
    unilateral_data: np.ndarray,
    skeleton: SkeletonDefinition,
    is_left: np.ndarray,
) -> np.ndarray:
    """Reconstruct bilateral data from unilateral observations.

    Must be perfectly invertible:
    ``make_bilateral(make_unilateral(data)) == data``

    Parameters
    ----------
    unilateral_data : np.ndarray
        Unilateral data, shape ``(n_frames * 2, n_right + n_centre, 3)``.
    skeleton : SkeletonDefinition
        Skeleton definition.
    is_left : np.ndarray
        Boolean or integer array indicating which frames were originally left.

    Returns
    -------
    np.ndarray
        Reconstructed bilateral data, shape ``(n_frames, n_markers, 3)``.
    """
    if is_left.dtype.kind in "iu":
        is_left = is_left.astype(bool)

    pairs = skeleton.get_marker_pairs()
    left_names = [p[0] for p in pairs]
    right_names = [p[1] for p in pairs]
    centre_names = skeleton.get_centre_markers()
    centre_names = [c for c in centre_names if c in skeleton.all_marker_names]

    n_pairs = len(pairs)
    n_centres = len(centre_names)

    data = np.copy(unilateral_data)
    n_frames = np.sum(is_left)  # number of original frames

    if n_centres > 0:
        paired_data = data[:, :n_pairs, :]
        centre_data = data[:, n_pairs:, :]

        left_data = paired_data[is_left]
        right_data = paired_data[~is_left]
        centre_data = centre_data[:n_frames]
    else:
        left_data = data[is_left]
        right_data = data[~is_left]
        centre_data = None

    # Un-mirror the left data
    left_data[:, :, 0] *= -1

    # Build bilateral output
    all_names = skeleton.all_marker_names
    total_markers = len(all_names)
    bilateral = np.zeros((n_frames, total_markers, 3))

    left_indices = [all_names.index(m) for m in left_names]
    right_indices = [all_names.index(m) for m in right_names]
    centre_indices = [all_names.index(m) for m in centre_names] if centre_names else []

    bilateral[:, left_indices, :] = left_data
    bilateral[:, right_indices, :] = right_data
    if centre_data is not None and centre_indices:
        bilateral[:, centre_indices, :] = centre_data

    return bilateral


def validate_left_right(
    motion_data: np.ndarray,
    left_indices: list[int],
    right_indices: list[int],
    min_separation: float = 0.001,
) -> np.ndarray:
    """Validate that left markers are to the left of right markers.

    Parameters
    ----------
    motion_data : np.ndarray
        Shape ``(n_frames, n_markers, 3)``.
    left_indices : list[int]
        Indices of left markers.
    right_indices : list[int]
        Indices of corresponding right markers.
    min_separation : float
        Minimum x-axis separation (default: 1mm).

    Returns
    -------
    np.ndarray
        Boolean mask of valid frames.
    """
    left_x = motion_data[:, left_indices, 0]
    right_x = motion_data[:, right_indices, 0]
    x_diff = right_x - left_x
    return np.all(x_diff >= min_separation, axis=1)
