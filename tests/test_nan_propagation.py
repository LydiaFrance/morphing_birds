"""Tests verifying NaN propagation through the data pipeline.

Documents current behaviour at each stage: loading, transforms,
bilateral conversion, and plotting. These tests do not fix bugs —
they record what actually happens when NaN values are present.
"""


import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pytest

from morphing_birds import Animal3D, SkeletonDefinition, plot_plotly
from morphing_birds.bilateral import (
    make_unilateral,
    mirror_to_bilateral,
    validate_left_right,
)
from morphing_birds.data_loading import (
    load_from_csv,
    load_from_dataframe,
)
from morphing_birds.plotting.plotly_plots import (
    plot_sections_plotly,
)
from morphing_birds.transforms import TransformState

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _hawk_skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_builtin("hawk")


def _make_csv(tmp_path, marker_names, n_frames=5, *, nan_indices=None, suffix="_"):
    """Create a CSV with columns ``<name>_x``, ``<name>_y``, ``<name>_z``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory.
    marker_names : list[str]
        Marker names to include.
    n_frames : int
        Number of rows.
    nan_indices : dict, optional
        ``{marker_name: [(row, axis), ...]}`` to inject NaN values.
    suffix : str
        Separator between name and axis (``"_"`` or ``""``).
    """
    rng = np.random.default_rng(42)
    data = {}
    for name in marker_names:
        for axis in ("x", "y", "z"):
            col = f"{name}{suffix}{axis}"
            data[col] = rng.standard_normal(n_frames)

    df = pd.DataFrame(data)

    if nan_indices:
        for marker, positions in nan_indices.items():
            for row, axis in positions:
                col = f"{marker}{suffix}{axis}"
                df.loc[row, col] = np.nan

    path = tmp_path / "nan_test_data.csv"
    df.to_csv(path, index=False)
    return path


def _make_valid_bilateral(skeleton, n_frames=10, seed=42):
    """Create synthetic bilateral data with correct left/right x-positions."""
    rng = np.random.default_rng(seed)
    data = rng.random((n_frames, skeleton.n_markers, 3))

    for pair_l, pair_r in skeleton.get_marker_pairs():
        l_idx = skeleton.marker_index(pair_l)
        r_idx = skeleton.marker_index(pair_r)
        data[:, l_idx, 0] = -np.abs(data[:, l_idx, 0]) - 0.01
        data[:, r_idx, 0] = np.abs(data[:, r_idx, 0]) + 0.01

    return data


# ==================================================================
# 1. NaN in loaded data
# ==================================================================


class TestNanInLoadedData:
    """Verify NaN values in CSV propagate correctly through the loading pipeline."""

    def test_single_nan_coordinate_propagates(self, tmp_path):
        """A single NaN coordinate in the CSV appears in the output array."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {"left_wingtip": [(0, "x")]}
        path = _make_csv(tmp_path, names, n_frames=3, nan_indices=nan_positions)
        result = load_from_csv(str(path), skel)

        lw_idx = skel.marker_index("left_wingtip")
        assert np.isnan(result[0, lw_idx, 0]), "NaN should appear at frame 0, x"
        assert np.isfinite(result[0, lw_idx, 1]), "y should remain finite"
        assert np.isfinite(result[0, lw_idx, 2]), "z should remain finite"

    def test_nan_does_not_contaminate_other_markers(self, tmp_path):
        """NaN in one marker should not affect neighbouring markers."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {"left_wingtip": [(0, "x"), (0, "y"), (0, "z")]}
        path = _make_csv(tmp_path, names, n_frames=3, nan_indices=nan_positions)
        result = load_from_csv(str(path), skel)

        lw_idx = skel.marker_index("left_wingtip")
        # The NaN marker should be all-NaN on frame 0
        assert np.all(np.isnan(result[0, lw_idx, :]))

        # Other markers on the same frame should be finite
        rw_idx = skel.marker_index("right_wingtip")
        assert np.all(np.isfinite(result[0, rw_idx, :]))

    def test_nan_does_not_contaminate_other_frames(self, tmp_path):
        """NaN on frame 0 should not affect frame 1."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {"left_wingtip": [(0, "x")]}
        path = _make_csv(tmp_path, names, n_frames=3, nan_indices=nan_positions)
        result = load_from_csv(str(path), skel)

        lw_idx = skel.marker_index("left_wingtip")
        assert np.isfinite(result[1, lw_idx, 0]), "Frame 1 should be unaffected"
        assert np.isfinite(result[2, lw_idx, 0]), "Frame 2 should be unaffected"

    def test_multiple_nan_markers_same_frame(self, tmp_path):
        """Multiple markers with NaN on the same frame are all preserved."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {
            "left_wingtip": [(1, "x")],
            "hood": [(1, "y")],
            "right_wingtip": [(1, "z")],
        }
        path = _make_csv(tmp_path, names, n_frames=3, nan_indices=nan_positions)
        result = load_from_csv(str(path), skel)

        assert np.isnan(result[1, skel.marker_index("left_wingtip"), 0])
        assert np.isnan(result[1, skel.marker_index("hood"), 1])
        assert np.isnan(result[1, skel.marker_index("right_wingtip"), 2])

    def test_load_from_dataframe_preserves_nan(self):
        """NaN values in a DataFrame propagate through load_from_dataframe."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        rng = np.random.default_rng(99)
        data = {}
        for name in names:
            for axis in ("x", "y", "z"):
                data[f"{name}_{axis}"] = rng.standard_normal(4)

        df = pd.DataFrame(data)
        lw_col = "left_wingtip_x"
        df.loc[2, lw_col] = np.nan

        result = load_from_dataframe(df, skel)
        lw_idx = skel.marker_index("left_wingtip")
        assert np.isnan(result[2, lw_idx, 0])

    def test_missing_columns_produce_nan(self, tmp_path):
        """Markers with no matching CSV columns should be NaN (the default)."""
        skel = _hawk_skeleton()
        # Only provide data for one marker
        path = _make_csv(tmp_path, ["left_wingtip"], n_frames=3)
        result = load_from_csv(str(path), skel)

        lw_idx = skel.marker_index("left_wingtip")
        hood_idx = skel.marker_index("hood")

        assert np.all(np.isfinite(result[:, lw_idx, :]))
        assert np.all(np.isnan(result[:, hood_idx, :]))


# ==================================================================
# 2. NaN after transform
# ==================================================================


class TestNanAfterTransform:
    """Verify NaN propagation through TransformState operations."""

    def test_nan_in_any_coord_taints_all_after_rotation(self):
        """NaN in any single coordinate taints ALL output coordinates.

        The homogeneous transform multiplies ``[x, y, z, 1] @ M.T``.
        Because ``nan * 0.0 == nan`` in IEEE 754 arithmetic, even a
        zero coefficient in the rotation matrix cannot prevent the NaN
        from spreading to every output coordinate.
        """
        t = TransformState()
        t.add_rotation(45, axis="z")

        # NaN in x
        coords = np.array([[np.nan, 2.0, 3.0]])
        result = t.apply_to(coords)
        assert np.all(
            np.isnan(result[0])
        ), "NaN in x should taint all output coordinates via homogeneous multiply"

        # NaN in z — same behaviour
        coords_z = np.array([[1.0, 2.0, np.nan]])
        result_z = t.apply_to(coords_z)
        assert np.all(
            np.isnan(result_z[0])
        ), "NaN in z should also taint all output coordinates"

    def test_nan_propagates_through_translation(self):
        """NaN in any coordinate taints all outputs even for pure translation.

        The homogeneous representation ``[nan, y, z, 1] @ T.T`` causes
        ``nan * 0.0 = nan`` to contaminate every output coordinate,
        even though intuitively a pure translation should not mix axes.
        """
        t = TransformState()
        t.add_translation(x=5.0, y=10.0, z=15.0)

        coords = np.array([[np.nan, 2.0, 3.0]])
        result = t.apply_to(coords)

        # All coordinates are tainted because of nan * 0 in homogeneous multiply
        assert np.all(
            np.isnan(result[0])
        ), "NaN in one coord taints all via homogeneous multiply"

    def test_nan_does_not_corrupt_other_points(self):
        """NaN in one point should not affect other points in the same batch.

        While NaN taints all coordinates of its *own* row, it should
        not spread to other rows in the (n, 3) input array.
        """
        t = TransformState()
        t.add_rotation(30, axis="x")
        t.add_translation(x=1.0)

        coords = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, np.nan, np.nan],
                [4.0, 5.0, 6.0],
            ]
        )
        result = t.apply_to(coords)

        assert np.all(np.isfinite(result[0])), "Clean point 0 should stay finite"
        assert np.all(np.isnan(result[1])), "All-NaN point should stay all-NaN"
        assert np.all(np.isfinite(result[2])), "Clean point 2 should stay finite"

    def test_single_nan_coord_taints_own_row_only(self):
        """A single NaN in one point taints that entire row but not others."""
        t = TransformState()
        t.add_rotation(45, axis="z")

        coords = np.array(
            [
                [1.0, 2.0, 3.0],
                [np.nan, 2.0, 3.0],  # Single NaN in x
                [4.0, 5.0, 6.0],
            ]
        )
        result = t.apply_to(coords)

        assert np.all(np.isfinite(result[0])), "Row 0 should be clean"
        assert np.all(np.isnan(result[1])), "Row 1 all tainted by single NaN"
        assert np.all(np.isfinite(result[2])), "Row 2 should be clean"

    def test_identity_transform_taints_all_from_single_nan(self):
        """Even the identity transform taints all coordinates if any is NaN.

        The identity matrix has zeros off-diagonal, and ``nan * 0 = nan``
        in IEEE 754, so ``[nan, 1, 2, 1] @ I`` still produces all-NaN
        output coordinates.
        """
        t = TransformState()

        coords = np.array([[np.nan, 1.0, 2.0]])
        result = t.apply_to(coords)

        assert np.all(
            np.isnan(result[0])
        ), "Identity transform still taints all coords via nan * 0"

    def test_all_nan_point_stays_all_nan(self):
        """A fully NaN point remains fully NaN after any transform."""
        t = TransformState()
        t.add_rotation(30, axis="x")
        t.add_translation(x=1.0)

        coords = np.array([[np.nan, np.nan, np.nan]])
        result = t.apply_to(coords)
        assert np.all(np.isnan(result[0]))

    def test_combined_rotation_translation_nan_propagation(self):
        """NaN in any input coordinate taints all output coordinates.

        This holds for any combination of rotations and translations due
        to the homogeneous matrix multiply where ``nan * 0.0 == nan``.
        """
        t = TransformState()
        t.add_rotation(45, axis="y")
        t.add_translation(x=1.0, y=2.0, z=3.0)

        coords = np.array([[np.nan, 5.0, 5.0]])
        result = t.apply_to(coords)

        assert np.all(
            np.isnan(result[0])
        ), "NaN in one coord taints all outputs via homogeneous multiply"


# ==================================================================
# 3. NaN in bilateral conversion
# ==================================================================


class TestNanInBilateralConversion:
    """Verify NaN behaviour through bilateral/unilateral conversion."""

    def test_nan_frame_dropped_by_validate_left_right(self):
        """NaN frames fail left-right validation (NaN comparisons are False)."""
        skel = _hawk_skeleton()
        data = _make_valid_bilateral(skel, n_frames=5)

        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        # Inject NaN into frame 2
        data[2, :, :] = np.nan

        valid = validate_left_right(data, left_indices, right_indices)
        assert not valid[2], "NaN frame should be marked invalid"
        assert valid.sum() == 4, "Only 4 of 5 frames should be valid"

    def test_nan_frame_excluded_from_unilateral(self):
        """make_unilateral drops NaN frames via its internal validation."""
        skel = _hawk_skeleton()
        data = _make_valid_bilateral(skel, n_frames=5)
        data[3, :, :] = np.nan

        uni, is_left, _ = make_unilateral(data, skel)
        # 4 valid frames, doubled
        assert uni.shape[0] == 4 * 2

    def test_all_nan_frames_raises_in_make_unilateral(self):
        """All-NaN input raises ValueError in make_unilateral."""
        skel = _hawk_skeleton()
        data = np.full((3, skel.n_markers, 3), np.nan)

        with pytest.raises(ValueError, match="No valid frames"):
            make_unilateral(data, skel)

    def test_partial_nan_in_single_marker_drops_frame(self):
        """A single NaN coordinate on a paired marker drops the whole frame.

        NaN in x-coordinate of a left marker means the x-difference
        comparison returns NaN (not >= min_separation), so the frame
        is invalid.
        """
        skel = _hawk_skeleton()
        data = _make_valid_bilateral(skel, n_frames=5)

        # Inject NaN into just one coordinate of one paired marker
        pair = skel.get_marker_pairs()[0]
        l_idx = skel.marker_index(pair[0])
        data[1, l_idx, 0] = np.nan  # NaN in x of left marker

        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        valid = validate_left_right(data, left_indices, right_indices)
        assert not valid[1], "Frame with NaN in paired marker x should be invalid"

    def test_nan_in_centre_marker_does_not_drop_frame(self):
        """NaN in a centre marker does not cause validate_left_right to drop
        the frame, because validation only checks paired markers.

        However, the NaN will propagate into the unilateral output.
        """
        skel = _hawk_skeleton()
        data = _make_valid_bilateral(skel, n_frames=5)

        centres = skel.get_centre_markers()
        if not centres:
            pytest.skip("No centre markers in hawk skeleton")

        centre_idx = skel.marker_index(centres[0])
        data[2, centre_idx, :] = np.nan

        # Validation only checks paired markers — frame 2 should survive
        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        valid = validate_left_right(data, left_indices, right_indices)
        assert valid[2], "NaN in centre marker should not invalidate the frame"

        # The NaN should propagate into the unilateral output
        uni, is_left, _ = make_unilateral(data, skel)
        assert uni.shape[0] == 5 * 2, "All 5 frames should survive"

    def test_mirror_to_bilateral_preserves_nan(self):
        """mirror_to_bilateral should propagate NaN without corruption."""
        # 2 frames, 3 markers, 3 coords
        right_data = np.array(
            [
                [[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, np.nan, 18.0]],
            ]
        )

        result = mirror_to_bilateral(right_data)
        assert result.shape == (2, 6, 3)

        # NaN in right_data[0, 1, 0] should appear in both mirrored (left) and original (right)
        # Mirrored left at index 2 (even), original right at index 3 (odd)
        assert np.isnan(result[0, 3, 0]), "Original NaN should be at right position"
        assert np.isnan(result[0, 2, 0]), "Mirrored NaN (* -1) should still be NaN"

        # Clean values at other positions
        assert np.isfinite(result[0, 0, 0]), "Mirrored clean value should be finite"
        assert np.isfinite(result[0, 1, 0]), "Original clean value should be finite"

    def test_nan_in_non_x_coordinate_preserved_in_bilateral(self):
        """NaN in y or z should propagate identically through mirroring.

        mirror_to_bilateral only negates x; y and z are just copied.
        """
        right_data = np.array([[[1.0, np.nan, 3.0]]])
        result = mirror_to_bilateral(right_data)

        # Left (mirrored): y should still be NaN
        assert np.isnan(result[0, 0, 1])
        # Right (original): y should still be NaN
        assert np.isnan(result[0, 1, 1])


# ==================================================================
# 4. NaN in plotting
# ==================================================================


class TestNanInPlotting:
    """Verify that plotting functions handle NaN data gracefully.

    Plotly generally tolerates NaN in trace data — it simply skips
    those points. These tests document that no exceptions are raised.
    """

    @pytest.fixture
    def hawk_with_nan(self):
        """Create a hawk Animal3D with NaN injected into one marker."""
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        # Inject NaN into a single analysis marker
        lw_idx = hawk.skeleton.marker_index("left_wingtip")
        hawk.current_shape[0, lw_idx, :] = np.nan
        hawk.untransformed_shape = hawk.current_shape.copy()
        return hawk

    @pytest.fixture
    def hawk_all_nan(self):
        """Create a hawk Animal3D with all markers set to NaN."""
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        hawk.current_shape[:] = np.nan
        hawk.default_shape[:] = np.nan
        hawk.untransformed_shape[:] = np.nan
        return hawk

    def test_plot_plotly_with_single_nan_marker(self, hawk_with_nan):
        """plot_plotly should not raise when one marker is NaN."""
        fig = plot_plotly(hawk_with_nan, colour="steelblue")
        assert isinstance(fig, go.Figure)
        # The figure should have traces (sections + keypoints)
        assert len(fig.data) > 0

    def test_plot_sections_plotly_with_nan(self, hawk_with_nan):
        """plot_sections_plotly should not raise when section coords contain NaN."""
        fig = go.Figure()
        # This should not raise — Plotly handles NaN in coordinates
        result = plot_sections_plotly(fig, hawk_with_nan, colour="blue", alpha=0.5)
        assert isinstance(result, go.Figure)

    def test_plot_plotly_all_nan_does_not_raise(self, hawk_all_nan):
        """plot_plotly with all-NaN data should not raise.

        The axis limits will be NaN, and Plotly will render an empty
        scene, but no Python exception should occur.
        """
        fig = plot_plotly(hawk_all_nan, colour="grey")
        assert isinstance(fig, go.Figure)

    def test_plot_plotly_with_transform_and_nan(self, hawk_with_nan):
        """Applying a transform via plot_plotly with NaN data should not raise.

        The NaN marker will produce NaN after transformation, but Plotly
        will silently skip it.
        """
        fig = plot_plotly(
            hawk_with_nan,
            colour="red",
            bodypitch=30,
            bodyyaw=15,
        )
        assert isinstance(fig, go.Figure)

    def test_plot_plotly_preserves_nan_after_call(self, hawk_with_nan):
        """plot_plotly should not alter NaN values in current_shape."""
        before = hawk_with_nan.current_shape.copy()
        plot_plotly(hawk_with_nan, colour="blue")
        np.testing.assert_array_equal(
            np.isnan(hawk_with_nan.current_shape),
            np.isnan(before),
            err_msg="plot_plotly changed the NaN pattern in current_shape",
        )

    def test_nan_trace_data_in_scatter(self, hawk_with_nan):
        """Scatter3d traces should contain NaN where the marker data has NaN."""
        fig = go.Figure()
        from morphing_birds.plotting.plotly_plots import plot_keypoints_plotly

        fig = plot_keypoints_plotly(fig, hawk_with_nan, colour="black")
        # At least one trace should exist
        assert len(fig.data) > 0

        # The scatter trace should include the NaN coordinate
        scatter = fig.data[0]
        x_vals = np.array(scatter.x, dtype=float)
        # The NaN marker's x coordinate should produce a NaN in the trace
        assert np.any(
            np.isnan(x_vals)
        ), "Scatter trace should contain NaN from the NaN marker"
