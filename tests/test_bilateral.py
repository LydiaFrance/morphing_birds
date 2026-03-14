"""Tests for bilateral/unilateral conversion."""

import numpy as np
import pytest

from morphing_birds import SkeletonDefinition
from morphing_birds.bilateral import make_bilateral, make_unilateral


class TestBilateralRoundTrip:
    """Test that make_unilateral -> make_bilateral is invertible."""

    @pytest.fixture
    def hawk_skel(self):
        return SkeletonDefinition.from_builtin("hawk")

    def test_round_trip_hawk(self, hawk_skel):
        """Bilateral -> unilateral -> bilateral should be identity."""
        # Create synthetic bilateral data
        np.random.seed(42)
        n_frames = 10
        n_markers = hawk_skel.n_markers
        data = np.random.rand(n_frames, n_markers, 3)

        # Ensure left is to the left of right (negative x for left)
        for pair_l, pair_r in hawk_skel.get_marker_pairs():
            l_idx = hawk_skel.marker_index(pair_l)
            r_idx = hawk_skel.marker_index(pair_r)
            data[:, l_idx, 0] = -np.abs(data[:, l_idx, 0])  # negative x
            data[:, r_idx, 0] = np.abs(data[:, r_idx, 0])   # positive x

        uni, is_left, _ = make_unilateral(data, hawk_skel)
        reconstructed = make_bilateral(uni, hawk_skel, is_left)

        # Only check paired and centre markers (these are the ones that get round-tripped)
        pairs = hawk_skel.get_marker_pairs()
        centres = hawk_skel.get_centre_markers()

        for left, right in pairs:
            l_idx = hawk_skel.marker_index(left)
            r_idx = hawk_skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :], data[:, l_idx, :],
                err_msg=f"Left marker {left} mismatch after round trip",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :], data[:, r_idx, :],
                err_msg=f"Right marker {right} mismatch after round trip",
            )

        for c in centres:
            if c in hawk_skel.all_marker_names:
                c_idx = hawk_skel.marker_index(c)
                np.testing.assert_array_almost_equal(
                    reconstructed[:, c_idx, :], data[:, c_idx, :],
                    err_msg=f"Centre marker {c} mismatch after round trip",
                )

    def test_unilateral_doubles_frames(self, hawk_skel):
        """Unilateral data should have 2x the valid frames."""
        np.random.seed(42)
        n_frames = 5
        data = np.random.rand(n_frames, hawk_skel.n_markers, 3)

        # Ensure valid left-right positions
        for pair_l, pair_r in hawk_skel.get_marker_pairs():
            l_idx = hawk_skel.marker_index(pair_l)
            r_idx = hawk_skel.marker_index(pair_r)
            data[:, l_idx, 0] = -np.abs(data[:, l_idx, 0])
            data[:, r_idx, 0] = np.abs(data[:, r_idx, 0])

        uni, is_left, _ = make_unilateral(data, hawk_skel)
        assert uni.shape[0] == n_frames * 2
        assert is_left.sum() == n_frames  # First half is left
