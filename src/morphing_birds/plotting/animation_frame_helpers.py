"""Helpers for formatting animation frame data."""

import numpy as np


def format_keypoint_frames(animal3d_instance, keypoints_frames):
    """Format keypoints_frames to ``[n_frames, n_markers, 3]``.

    If only right-side markers are given, mirrors them to create the
    left side.
    """
    if len(np.shape(keypoints_frames)) == 2:
        keypoints_frames = keypoints_frames.reshape(1, -1, 3)

    # Mirror if only right side given
    right_markers = animal3d_instance.skeleton.get_right_markers()
    # Filter to analysis markers
    right_analysis = [
        n for n in right_markers
        if n not in animal3d_instance._analysis_exclude
    ]
    if keypoints_frames.shape[1] == len(right_analysis):
        keypoints_frames = animal3d_instance.mirror_keypoints(keypoints_frames)

    return keypoints_frames


def check_transformation_frames(num_frames, transformation_frames):
    """Validate transformation frames or create zeros if None."""
    if transformation_frames is None:
        return np.zeros(num_frames)

    if len(transformation_frames) != num_frames:
        raise ValueError(
            "Transformation frames must be the same length as keypoints_frames."
        )

    return transformation_frames
