"""Tests for the new Animal3D class."""

import numpy as np
import pytest
from morphing_birds import Animal3D, SkeletonDefinition


class TestAnimalLoading:
    """Test loading animals from various data sources."""

    def test_load_hawk_csv(self):
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        assert hawk.current_shape.shape == (1, 14, 3)
        assert hawk.markers.shape == (1, 8, 3)
        # Check no NaN values
        assert not np.isnan(hawk.current_shape).any()

    def test_load_pigeon_csv(self):
        pigeon = Animal3D("pigeon", data="data/mean_pigeon_shape.csv")
        assert pigeon.current_shape.shape == (1, 19, 3)

    def test_load_kestrel_csv(self):
        kestrel = Animal3D("kestrel", data="data/mean_kestrel_shape.csv")
        assert kestrel.current_shape.shape == (1, 35, 3)

    def test_load_spider_csv(self):
        spider = Animal3D("spider", data="data/mean_spider_shape_carolina.csv")
        assert spider.current_shape.shape == (1, 36, 3)

    def test_load_from_dict(self):
        positions = {
            "left_wingtip": [-0.3, 0.04, -0.04],
            "right_wingtip": [0.3, 0.04, -0.04],
        }
        hawk = Animal3D("hawk", data=positions)
        idx_lw = hawk.skeleton.marker_index("left_wingtip")
        np.testing.assert_array_almost_equal(
            hawk.current_shape[0, idx_lw], [-0.3, 0.04, -0.04]
        )

    def test_load_from_skeleton_definition(self):
        skel = SkeletonDefinition.from_builtin("hawk")
        hawk = Animal3D(skel, data="data/mean_hawk_shape.csv")
        assert hawk.current_shape.shape == (1, 14, 3)

    def test_variant(self):
        pigeon = Animal3D("pigeon", data="data/mean_pigeon_shape.csv", variant="simple")
        assert len(pigeon.analysis_indices) == 8


class TestAnimalMarkers:
    """Test marker access and indexing."""

    @pytest.fixture
    def hawk(self):
        return Animal3D("hawk", data="data/mean_hawk_shape.csv")

    def test_marker_index(self, hawk):
        assert len(hawk.marker_index) == 8
        assert len(hawk.fixed_marker_index) == 6

    def test_markers_property(self, hawk):
        assert hawk.markers.shape == (1, 8, 3)
        assert hawk.fixed_markers.shape == (1, 6, 3)

    def test_right_left_markers(self, hawk):
        # Right analysis markers (4 of 8)
        assert hawk.right_markers.shape[1] == 4
        assert hawk.left_markers.shape[1] == 4

    def test_marker_name_at(self, hawk):
        assert hawk.marker_name_at(0) == "left_wingtip"
        assert hawk.marker_name_at(1) == "right_wingtip"

    def test_marker_index_of(self, hawk):
        assert hawk.marker_index_of("left_wingtip") == 0
        assert hawk.marker_index_of("hood") == 12

    def test_analysis_marker_labels(self, hawk):
        labels = hawk.analysis_marker_labels()
        assert len(labels) == 24  # 8 * 3

    def test_exclude_include_markers(self, hawk):
        n_before = len(hawk.analysis_indices)
        hawk.exclude_markers(["left_wingtip"])
        assert len(hawk.analysis_indices) == n_before - 1
        hawk.include_markers(["left_wingtip"])
        assert len(hawk.analysis_indices) == n_before


class TestAnimalTransforms:
    """Test transformation modes."""

    @pytest.fixture
    def hawk(self):
        return Animal3D("hawk", data="data/mean_hawk_shape.csv")

    def test_transform_display_only(self, hawk):
        before = hawk.current_shape.copy()
        hawk.transform_display_only(bodypitch=25)

        # Analysis markers unchanged
        np.testing.assert_array_almost_equal(
            hawk.current_shape[0, hawk.analysis_indices],
            before[0, hawk.analysis_indices],
        )
        # Display-only markers changed
        assert not np.allclose(
            hawk.current_shape[0, hawk.display_only_indices],
            before[0, hawk.display_only_indices],
        )

    def test_transform_all(self, hawk):
        before = hawk.current_shape.copy()
        hawk.transform_all(bodypitch=25)

        # ALL markers should change
        assert not np.allclose(hawk.current_shape, before)

    def test_backward_compat_transform_keypoints(self, hawk):
        """transform_keypoints should transform display-only only."""
        before = hawk.current_shape.copy()
        hawk.transform_keypoints(bodypitch=25)

        # Analysis markers unchanged
        np.testing.assert_array_almost_equal(
            hawk.current_shape[0, hawk.analysis_indices],
            before[0, hawk.analysis_indices],
        )

    def test_reset_transformation(self, hawk):
        before = hawk.current_shape.copy()
        hawk.transform_keypoints(bodypitch=25, horzDist=10)
        hawk.reset_transformation()
        np.testing.assert_array_almost_equal(hawk.current_shape, before)

    def test_restore_default(self, hawk):
        original = hawk.default_shape.copy()
        hawk.update_keypoints(np.random.rand(1, 8, 3))
        hawk.restore_default()
        np.testing.assert_array_almost_equal(hawk.current_shape, original)


class TestAnimalPolygons:
    """Test polygon coordinate retrieval."""

    @pytest.fixture
    def hawk(self):
        return Animal3D("hawk", data="data/mean_hawk_shape.csv")

    def test_get_polygon_coords(self, hawk):
        for section in hawk.polygons:
            coords = hawk.get_polygon_coords(section)
            assert coords.shape[1] == 3
            assert coords.shape[0] > 0

    def test_invalid_section_raises(self, hawk):
        with pytest.raises(ValueError, match="not recognised"):
            hawk.get_polygon_coords("nonexistent_section")

    def test_bounding_box(self, hawk):
        min_c, max_c = hawk.get_bounding_box()
        assert min_c.shape == (3,)
        assert max_c.shape == (3,)
        assert all(min_c <= max_c)


class TestAnimalCopy:
    """Test deep copy."""

    def test_copy_independence(self):
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        hawk2 = hawk.copy()

        hawk.transform_all(bodypitch=45)
        # hawk2 should be unaffected
        np.testing.assert_array_almost_equal(hawk2.current_shape, hawk2.default_shape)


class TestAnimalScaling:
    """Test scaling functionality."""

    def test_unit_scale(self):
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        before_max = hawk.current_shape.max()
        hawk.set_scale(factor=2.0)
        assert np.isclose(hawk.current_shape.max(), before_max * 2.0)

    def test_wingspan_normalisation(self):
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        hawk.set_scale(normalise_by="wingspan")
        # After normalisation, wingspan should be ~1.0
        from morphing_birds import compute_wingspan
        ws = compute_wingspan(hawk.current_shape, hawk.skeleton)
        assert np.isclose(ws, 1.0, atol=0.01)
