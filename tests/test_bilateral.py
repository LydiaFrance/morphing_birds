"""Tests for bilateral/unilateral conversion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from morphing_birds import Animal3D, SkeletonDefinition
from morphing_birds.bilateral import (
    _analysis_pairs_and_centres,
    make_bilateral,
    make_unilateral,
    validate_left_right,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ------------------------------------------------------------------
# Helper: create valid bilateral data for a given skeleton
# ------------------------------------------------------------------


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
# Round-trip tests for multiple species
# ==================================================================


class TestBilateralRoundTrip:
    """Test that make_unilateral -> make_bilateral is invertible."""

    @pytest.fixture
    def hawk_skel(self):
        return SkeletonDefinition.from_builtin("hawk")

    @pytest.fixture
    def pigeon_skel(self):
        return SkeletonDefinition.from_builtin("pigeon")

    @pytest.fixture
    def kestrel_skel(self):
        return SkeletonDefinition.from_builtin("kestrel")

    @pytest.fixture
    def spider_skel(self):
        return SkeletonDefinition.from_builtin("spider")

    # -- Original hawk tests (kept) --

    def test_round_trip_hawk(self, hawk_skel):
        """Bilateral -> unilateral -> bilateral should be identity for hawk."""
        data = _make_valid_bilateral(hawk_skel)

        uni, is_left, _ = make_unilateral(data, hawk_skel)
        reconstructed = make_bilateral(uni, hawk_skel, is_left)

        pairs, centres = _analysis_pairs_and_centres(hawk_skel)

        for left, right in pairs:
            l_idx = hawk_skel.marker_index(left)
            r_idx = hawk_skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
                err_msg=f"Left marker {left} mismatch after round trip",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
                err_msg=f"Right marker {right} mismatch after round trip",
            )

        for c in centres:
            if c in hawk_skel.all_marker_names:
                c_idx = hawk_skel.marker_index(c)
                np.testing.assert_array_almost_equal(
                    reconstructed[:, c_idx, :],
                    data[:, c_idx, :],
                    err_msg=f"Centre marker {c} mismatch after round trip",
                )

    def test_unilateral_doubles_frames(self, hawk_skel):
        """Unilateral data should have 2x the valid frames."""
        data = _make_valid_bilateral(hawk_skel, n_frames=5)

        uni, is_left, _ = make_unilateral(data, hawk_skel)
        assert uni.shape[0] == 5 * 2
        assert is_left.sum() == 5  # First half is left

    # -- Pigeon round-trip --

    def test_round_trip_pigeon(self, pigeon_skel):
        """Pigeon round-trip preserves all paired and centre markers."""
        data = _make_valid_bilateral(pigeon_skel)

        uni, is_left, _ = make_unilateral(data, pigeon_skel)
        reconstructed = make_bilateral(uni, pigeon_skel, is_left)

        pairs, _ = _analysis_pairs_and_centres(pigeon_skel)
        for left, right in pairs:
            l_idx = pigeon_skel.marker_index(left)
            r_idx = pigeon_skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
                err_msg=f"Pigeon left marker {left} mismatch",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
                err_msg=f"Pigeon right marker {right} mismatch",
            )

    def test_round_trip_kestrel(self, kestrel_skel):
        """Kestrel round-trip preserves all paired and centre markers."""
        data = _make_valid_bilateral(kestrel_skel)

        uni, is_left, _ = make_unilateral(data, kestrel_skel)
        reconstructed = make_bilateral(uni, kestrel_skel, is_left)

        pairs, _ = _analysis_pairs_and_centres(kestrel_skel)
        for left, right in pairs:
            l_idx = kestrel_skel.marker_index(left)
            r_idx = kestrel_skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
                err_msg=f"Kestrel left marker {left} mismatch",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
                err_msg=f"Kestrel right marker {right} mismatch",
            )

    def test_round_trip_spider(self, spider_skel):
        """Spider (dict-based laterality) round-trip is invertible."""
        data = _make_valid_bilateral(spider_skel)

        uni, is_left, _ = make_unilateral(data, spider_skel)
        reconstructed = make_bilateral(uni, spider_skel, is_left)

        pairs, _ = _analysis_pairs_and_centres(spider_skel)
        for left, right in pairs:
            l_idx = spider_skel.marker_index(left)
            r_idx = spider_skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
                err_msg=f"Spider left marker {left} mismatch",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
                err_msg=f"Spider right marker {right} mismatch",
            )

    # -- Round-trip using real mean-shape CSVs via Animal3D --

    @pytest.mark.parametrize(
        ("species", "csv_name"),
        [
            ("pigeon", "mean_pigeon_shape.csv"),
            ("kestrel", "mean_kestrel_shape.csv"),
            ("hawk", "mean_hawk_shape.csv"),
        ],
    )
    def test_round_trip_real_mean_shape(self, species, csv_name):
        """Round-trip with real mean-shape data loaded via Animal3D."""
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists():
            pytest.skip(f"{csv_name} not found in data directory")

        animal = Animal3D(species, data=str(csv_path))
        # current_shape is (1, n_markers, 3) — use it directly
        data = animal.current_shape.copy()
        skel = animal.skeleton

        uni, is_left, _ = make_unilateral(data, skel)
        reconstructed = make_bilateral(uni, skel, is_left)

        pairs, _ = _analysis_pairs_and_centres(skel)
        for left, right in pairs:
            l_idx = skel.marker_index(left)
            r_idx = skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
                err_msg=f"{species} real-data left marker {left} mismatch",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
                err_msg=f"{species} real-data right marker {right} mismatch",
            )


# ==================================================================
# Centre markers
# ==================================================================


class TestCentreMarkerRoundTrip:
    """Verify centre (unpaired) markers survive the round-trip."""

    def test_pigeon_has_no_analysis_centres(self):
        """Pigeon's centre markers are all display-only (in analysis_exclude)."""
        skel = SkeletonDefinition.from_builtin("pigeon")
        _, centres = _analysis_pairs_and_centres(skel)
        assert centres == [], (
            "Pigeon centres (centre_body_base, head, centre_backpack) are "
            "all display-only and should be excluded from bilateral analysis"
        )

    def test_spider_centre_markers_preserved(self):
        """Spider has clypeus, pedicel, spinneret, center as centres."""
        skel = SkeletonDefinition.from_builtin("spider")
        data = _make_valid_bilateral(skel)

        uni, is_left, _ = make_unilateral(data, skel)
        reconstructed = make_bilateral(uni, skel, is_left)

        _, centre_in_all = _analysis_pairs_and_centres(skel)
        assert len(centre_in_all) > 0

        for c in centre_in_all:
            c_idx = skel.marker_index(c)
            np.testing.assert_array_almost_equal(
                reconstructed[:, c_idx, :],
                data[:, c_idx, :],
                err_msg=f"Spider centre marker {c} not preserved",
            )

    def test_hawk_no_centre_markers(self):
        """Hawk has hood and tailpack as centre markers."""
        skel = SkeletonDefinition.from_builtin("hawk")
        centres = skel.get_centre_markers()
        # hood and tailpack should be centre markers
        assert "hood" in centres
        assert "tailpack" in centres


# ==================================================================
# validate_left_right
# ==================================================================


class TestValidateLeftRight:
    """Tests for the validate_left_right function."""

    def test_all_valid_frames(self):
        """All frames valid when left x < right x with sufficient separation."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=5)

        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        valid = validate_left_right(data, left_indices, right_indices)
        assert valid.all(), "All frames should be valid"
        assert valid.shape == (5,)

    def test_swapped_markers_dropped(self):
        """Frames where left x > right x should be dropped."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=5)

        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        # Swap left/right x on frame 2 for the first marker pair
        l0, r0 = left_indices[0], right_indices[0]
        data[2, l0, 0], data[2, r0, 0] = data[2, r0, 0], data[2, l0, 0]

        valid = validate_left_right(data, left_indices, right_indices)
        assert not valid[2], "Frame 2 should be invalid after swap"
        assert valid.sum() == 4

    def test_overlapping_markers_dropped(self):
        """Frames with left x == right x (no separation) should be dropped."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        # Set left x == right x on frame 1 (below min_separation)
        for li, ri in zip(left_indices, right_indices, strict=False):
            data[1, li, 0] = 0.0
            data[1, ri, 0] = 0.0

        valid = validate_left_right(data, left_indices, right_indices)
        assert not valid[1], "Frame with zero separation should be invalid"

    def test_custom_min_separation(self):
        """Custom min_separation threshold is respected."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        pairs = skel.get_marker_pairs()
        left_indices = [skel.marker_index(p[0]) for p in pairs]
        right_indices = [skel.marker_index(p[1]) for p in pairs]

        # Set small but non-zero separation on all frames
        for li, ri in zip(left_indices, right_indices, strict=False):
            data[:, li, 0] = -0.0001
            data[:, ri, 0] = 0.0001

        # Default threshold (0.001) should reject (separation = 0.0002)
        valid_default = validate_left_right(data, left_indices, right_indices)
        assert not valid_default.any()

        # Lower threshold should accept
        valid_low = validate_left_right(
            data,
            left_indices,
            right_indices,
            min_separation=0.0001,
        )
        assert valid_low.all()

    def test_all_frames_invalid_raises_in_make_unilateral(self):
        """make_unilateral raises ValueError when all frames are invalid."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        # Make all frames invalid by swapping left/right
        for pair_l, pair_r in skel.get_marker_pairs():
            l_idx = skel.marker_index(pair_l)
            r_idx = skel.marker_index(pair_r)
            data[:, l_idx, 0] = np.abs(data[:, l_idx, 0]) + 0.1
            data[:, r_idx, 0] = -np.abs(data[:, r_idx, 0]) - 0.1

        with pytest.raises(ValueError, match="No valid frames"):
            make_unilateral(data, skel)


# ==================================================================
# Edge cases
# ==================================================================


class TestEdgeCases:
    """Edge-case tests for bilateral conversion."""

    def test_single_frame_round_trip(self):
        """Single-frame input should round-trip correctly."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=1)

        uni, is_left, _ = make_unilateral(data, skel)
        assert uni.shape[0] == 2  # doubled

        reconstructed = make_bilateral(uni, skel, is_left)
        assert reconstructed.shape[0] == 1

        pairs, _ = _analysis_pairs_and_centres(skel)
        for left, right in pairs:
            l_idx = skel.marker_index(left)
            r_idx = skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
            )

    def test_nan_frames_dropped_before_conversion(self):
        """Frames containing NaN should be removed before make_unilateral.

        NaN frames violate left-right validation, so they get dropped
        by validate_left_right (NaN comparisons return False).
        """
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=5)

        # Inject NaN into frame 3
        data[3, :, :] = np.nan

        uni, is_left, _ = make_unilateral(data, skel)
        # Frame 3 should have been dropped by validation
        assert uni.shape[0] == 4 * 2  # 4 valid frames doubled

    def test_all_nan_frames_raises(self):
        """All-NaN input should raise ValueError."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = np.full((3, skel.n_markers, 3), np.nan)

        with pytest.raises(ValueError, match="No valid frames"):
            make_unilateral(data, skel)

    def test_partial_invalid_frames(self):
        """Only invalid frames are dropped; valid ones survive."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=4)

        # Invalidate frames 0 and 3
        pairs = skel.get_marker_pairs()
        l0 = skel.marker_index(pairs[0][0])
        r0 = skel.marker_index(pairs[0][1])
        data[0, l0, 0] = data[0, r0, 0] + 1.0  # left > right
        data[3, l0, 0] = data[3, r0, 0] + 1.0

        uni, is_left, _ = make_unilateral(data, skel)
        # Only frames 1 and 2 should survive
        assert uni.shape[0] == 2 * 2


# ==================================================================
# is_left as integer array vs boolean
# ==================================================================


class TestIsLeftTypes:
    """Verify make_bilateral accepts both boolean and integer is_left."""

    def test_is_left_boolean(self):
        """Boolean is_left array works correctly."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        uni, is_left, _ = make_unilateral(data, skel)
        assert is_left.dtype == bool

        reconstructed = make_bilateral(uni, skel, is_left)
        assert reconstructed.shape[0] == 3

    def test_is_left_integer(self):
        """Integer is_left array (0/1) should work identically to boolean."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        uni, is_left_bool, _ = make_unilateral(data, skel)
        is_left_int = is_left_bool.astype(int)

        reconstructed_bool = make_bilateral(uni, skel, is_left_bool)
        reconstructed_int = make_bilateral(uni.copy(), skel, is_left_int)

        np.testing.assert_array_almost_equal(
            reconstructed_bool,
            reconstructed_int,
            err_msg="Integer and boolean is_left should produce identical output",
        )

    def test_is_left_uint8(self):
        """Unsigned integer is_left should also be accepted."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=2)

        uni, is_left_bool, _ = make_unilateral(data, skel)
        is_left_uint8 = is_left_bool.astype(np.uint8)

        reconstructed = make_bilateral(uni, skel, is_left_uint8)
        assert reconstructed.shape[0] == 2


# ==================================================================
# info_df doubling
# ==================================================================


class TestInfoDfDoubling:
    """Verify info_df is correctly doubled and ordered."""

    def test_info_df_row_count_doubled(self):
        """info_df should have 2x rows after make_unilateral."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=4)

        info = pd.DataFrame(
            {
                "frame": [10, 20, 30, 40],
                "label": ["a", "b", "c", "d"],
            }
        )

        _, _, info_out = make_unilateral(data, skel, info_df=info)

        assert info_out is not None
        assert len(info_out) == 4 * 2

    def test_info_df_metadata_preserved(self):
        """Metadata values should be correctly duplicated."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        info = pd.DataFrame(
            {
                "frame": [1, 2, 3],
                "trial": ["x", "y", "z"],
            }
        )

        _, _, info_out = make_unilateral(data, skel, info_df=info)

        # First half and second half should have the same values
        n = 3
        first_half = info_out.iloc[:n].reset_index(drop=True)
        second_half = info_out.iloc[n:].reset_index(drop=True)
        pd.testing.assert_frame_equal(first_half, second_half)

    def test_info_df_none_returns_none(self):
        """When info_df is None, output info should also be None."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=2)

        _, _, info_out = make_unilateral(data, skel, info_df=None)
        assert info_out is None

    def test_info_df_invalid_frames_removed(self):
        """info_df rows for invalid frames should be dropped."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=4)

        # Invalidate frame 1
        pairs = skel.get_marker_pairs()
        l0 = skel.marker_index(pairs[0][0])
        r0 = skel.marker_index(pairs[0][1])
        data[1, l0, 0] = data[1, r0, 0] + 1.0

        info = pd.DataFrame(
            {
                "frame": [10, 20, 30, 40],
                "label": ["a", "b", "c", "d"],
            }
        )

        _, _, info_out = make_unilateral(data, skel, info_df=info)

        # 3 valid frames -> 6 rows in output
        assert len(info_out) == 3 * 2
        # Frame 20 (index 1) should not appear
        assert 20 not in info_out["frame"].values

    def test_info_df_wrong_length_raises(self):
        """info_df with mismatched row count should raise ValueError."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=3)

        bad_info = pd.DataFrame({"frame": [1, 2]})  # wrong length

        with pytest.raises(ValueError, match="same number of rows"):
            make_unilateral(data, skel, info_df=bad_info)

    def test_info_df_index_is_reset(self):
        """Output info_df should have a clean integer index."""
        skel = SkeletonDefinition.from_builtin("hawk")
        data = _make_valid_bilateral(skel, n_frames=2)

        info = pd.DataFrame(
            {"val": [100, 200]},
            index=[5, 10],  # non-default index
        )

        _, _, info_out = make_unilateral(data, skel, info_df=info)

        expected_index = list(range(2 * 2))
        assert list(info_out.index) == expected_index


# ==================================================================
# Animal3D wrapper methods
# ==================================================================


class TestAnimal3DWrapperMethods:
    """Tests for Animal3D.make_unilateral() and make_bilateral().

    These wrapper methods exist so that runtime marker exclusions
    (via exclude_markers()) are respected by bilateral conversion.
    """

    @pytest.fixture
    def pigeon_csv(self):
        csv_path = DATA_DIR / "mean_pigeon_shape.csv"
        if not csv_path.exists():
            pytest.skip("mean_pigeon_shape.csv not found")
        return str(csv_path)

    def test_wrapper_respects_runtime_exclusions(self, pigeon_csv):
        """Animal3D.make_unilateral() uses current _analysis_exclude.

        This is the bug Charlie hit: after calling exclude_markers(),
        the direct make_unilateral() function ignores the exclusions,
        but the Animal3D wrapper method should respect them.
        """
        pigeon = Animal3D("pigeon", data=pigeon_csv)

        # Get initial marker counts
        initial_analysis = len(pigeon.analysis_indices)
        assert initial_analysis == 14, "Pigeon should have 14 analysis markers"

        # Exclude some markers
        pigeon.exclude_markers(
            [
                "left_shoulder",
                "right_shoulder",
                "left_lastsecondary_tip",
                "right_lastsecondary_tip",
            ]
        )
        reduced_analysis = len(pigeon.analysis_indices)
        assert reduced_analysis == 10, "Should have 10 markers after exclusion"

        # Create fake motion data
        data = _make_valid_bilateral(pigeon.skeleton, n_frames=5)

        # Using the wrapper method should respect exclusions
        uni_wrapper, is_left, _ = pigeon.make_unilateral(data)

        # Using the direct function should use skeleton's original excludes
        uni_direct, _, _ = make_unilateral(data, pigeon.skeleton)

        # The wrapper should produce fewer unilateral markers
        # because it excludes the 4 extra markers (2 pairs)
        assert (
            uni_wrapper.shape[1] < uni_direct.shape[1]
        ), "Wrapper should produce fewer markers due to exclusions"

        # Specifically: 10 bilateral markers / 2 = 5 unilateral pairs
        # (pigeon has no centre markers in analysis set)
        assert (
            uni_wrapper.shape[1] == 5
        ), f"Expected 5 unilateral markers from 10 bilateral, got {uni_wrapper.shape[1]}"

    def test_wrapper_round_trip_with_exclusions(self, pigeon_csv):
        """Wrapper methods round-trip correctly after exclude_markers()."""
        pigeon = Animal3D("pigeon", data=pigeon_csv)

        pigeon.exclude_markers(
            [
                "left_shoulder",
                "right_shoulder",
                "left_lastsecondary_tip",
                "right_lastsecondary_tip",
            ]
        )

        data = _make_valid_bilateral(pigeon.skeleton, n_frames=5)
        uni, is_left, _ = pigeon.make_unilateral(data)
        reconstructed = pigeon.make_bilateral(uni, is_left)

        # Reconstructed should have same shape as original
        assert reconstructed.shape == data.shape

        # Check that non-excluded pairs round-trip correctly
        pairs, _ = _analysis_pairs_and_centres(
            pigeon.skeleton, exclude=pigeon._analysis_exclude
        )
        for left, right in pairs:
            l_idx = pigeon.skeleton.marker_index(left)
            r_idx = pigeon.skeleton.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :],
                data[:, l_idx, :],
                err_msg=f"Left marker {left} mismatch after wrapper round trip",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :],
                data[:, r_idx, :],
                err_msg=f"Right marker {right} mismatch after wrapper round trip",
            )

    def test_pca_animation_after_exclude_markers(self, pigeon_csv):
        """E2E test: PCA + animation works after exclude_markers().

        This is the exact workflow that was broken:
        1. Create Animal3D
        2. exclude_markers()
        3. make_unilateral() for PCA
        4. Run PCA
        5. Reconstruct and animate

        The bug was that step 3 ignored the exclusions, causing
        marker count mismatch at step 5.
        """
        from sklearn.decomposition import PCA

        from morphing_birds.bilateral import mirror_to_bilateral

        pigeon = Animal3D("pigeon", data=pigeon_csv)

        # Exclude markers (like Charlie did)
        pigeon.exclude_markers(
            [
                "left_shoulder",
                "right_shoulder",
                "left_lastsecondary_tip",
                "right_lastsecondary_tip",
            ]
        )
        n_analysis = len(pigeon.analysis_indices)
        assert n_analysis == 10

        # Create fake motion and convert to unilateral
        data = _make_valid_bilateral(pigeon.skeleton, n_frames=50)
        uni, is_left, _ = pigeon.make_unilateral(data)

        # Run PCA on flattened unilateral data
        flat = uni.reshape(uni.shape[0], -1)
        pca = PCA(n_components=3)
        scores = pca.fit_transform(flat)
        _mu = pca.mean_.reshape(1, -1, 3)

        # Reconstruct a frame from PCA
        recon_flat = pca.inverse_transform(scores[:1])
        recon_uni = recon_flat.reshape(1, -1, 3)

        # Mirror to bilateral (this is what animate_mode does)
        recon_bilateral = mirror_to_bilateral(recon_uni)

        # The critical check: marker count must match what pigeon expects
        assert recon_bilateral.shape[1] == n_analysis, (
            f"Mirrored reconstruction has {recon_bilateral.shape[1]} markers, "
            f"but Animal3D expects {n_analysis}. "
            "This is the bug where exclude_markers() wasn't respected."
        )

        # Verify we can update keypoints without error
        # (this was raising ValueError before the fix)
        pigeon.update_keypoints(recon_bilateral[0])

    def test_direct_function_with_exclude_override(self, pigeon_csv):
        """Direct functions accept exclude parameter for advanced use."""
        pigeon = Animal3D("pigeon", data=pigeon_csv)

        custom_exclude = pigeon._analysis_exclude | {
            "left_shoulder",
            "right_shoulder",
        }

        data = _make_valid_bilateral(pigeon.skeleton, n_frames=3)

        uni_with_exclude, _, _ = make_unilateral(
            data, pigeon.skeleton, exclude=custom_exclude
        )
        uni_without, _, _ = make_unilateral(data, pigeon.skeleton)

        # With custom exclude should have fewer markers
        assert uni_with_exclude.shape[1] < uni_without.shape[1]
