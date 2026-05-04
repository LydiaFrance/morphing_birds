"""Tests for the new Animal3D class."""

import tempfile

import numpy as np
import pandas as pd
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

    def test_analysis_indices(self, hawk):
        assert len(hawk.analysis_indices) == 8
        assert len(hawk.display_only_indices) == 6

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

    def test_transform_display_only_leaves_analysis_unchanged(self, hawk):
        """transform_display_only should leave analysis markers unchanged."""
        before = hawk.current_shape.copy()
        hawk.transform_display_only(bodypitch=25)

        # Analysis markers unchanged
        np.testing.assert_array_almost_equal(
            hawk.current_shape[0, hawk.analysis_indices],
            before[0, hawk.analysis_indices],
        )

    def test_reset_transformation(self, hawk):
        before = hawk.current_shape.copy()
        hawk.transform_display_only(bodypitch=25, horzDist=10)
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


class TestGetAnalysisData:
    """Test get_analysis_data slicing."""

    @pytest.fixture
    def hawk(self):
        return Animal3D("hawk", data="data/mean_hawk_shape.csv")

    def test_include_all_returns_full_array(self, hawk):
        n_markers = hawk.skeleton.n_markers
        motion = np.random.rand(5, n_markers, 3)
        result = hawk.get_analysis_data(motion, include_all=True)
        assert result.shape == (5, n_markers, 3)
        np.testing.assert_array_equal(result, motion)

    def test_default_slices_to_analysis_indices(self, hawk):
        n_markers = hawk.skeleton.n_markers
        motion = np.random.rand(5, n_markers, 3)
        result = hawk.get_analysis_data(motion)
        n_analysis = len(hawk.analysis_indices)
        assert result.shape == (5, n_analysis, 3)
        # Verify it matches manual slicing
        expected = motion[:, hawk.analysis_indices, :]
        np.testing.assert_array_equal(result, expected)

    def test_respects_exclude_markers(self, hawk):
        n_markers = hawk.skeleton.n_markers
        motion = np.random.rand(5, n_markers, 3)
        n_before = len(hawk.analysis_indices)
        hawk.exclude_markers(["left_wingtip"])
        result = hawk.get_analysis_data(motion)
        assert result.shape == (5, n_before - 1, 3)

    def test_prints_excluded_marker_names(self, hawk, capsys):
        n_markers = hawk.skeleton.n_markers
        motion = np.random.rand(2, n_markers, 3)
        hawk.exclude_markers(["left_wingtip"])
        hawk.get_analysis_data(motion)
        captured = capsys.readouterr()
        assert "left_wingtip" in captured.out
        assert "excluded:" in captured.out

    def test_prints_all_included_message(self, hawk, capsys):
        n_markers = hawk.skeleton.n_markers
        motion = np.random.rand(2, n_markers, 3)
        hawk.get_analysis_data(motion, include_all=True)
        captured = capsys.readouterr()
        assert "all included" in captured.out


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


# ------------------------------------------------------------------
# Helper to create a temporary multi-frame CSV for hawk markers
# ------------------------------------------------------------------


def _make_hawk_csv(
    n_frames: int = 5,
    extra_columns: dict[str, list] | None = None,
    drop_markers: list[str] | None = None,
) -> str:
    """Create a temporary CSV file with hawk marker data.

    Returns the path to the temporary file.
    """
    skel = SkeletonDefinition.from_builtin("hawk")
    marker_names = skel.all_marker_names

    rng = np.random.default_rng(42)
    data = {}
    for name in marker_names:
        if drop_markers and name in drop_markers:
            continue
        for axis in ("x", "y", "z"):
            data[f"{name}_{axis}"] = rng.standard_normal(n_frames)

    if extra_columns:
        data.update(extra_columns)

    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df.to_csv(tmp.name, index=False)
    return tmp.name


class TestLoadMotionDataFeedback:
    """Test that load_motion_data prints informative feedback."""

    def test_prints_summary(self, capsys):
        path = _make_hawk_csv(n_frames=10)
        hawk = Animal3D("hawk")
        hawk.load_motion_data(path)
        captured = capsys.readouterr()
        assert "Loaded 10 frames with 14 markers" in captured.out
        assert "'hawk'" in captured.out

    def test_warns_zero_data_markers(self, capsys):
        # Drop left_wingtip and right_wingtip from the CSV
        path = _make_hawk_csv(drop_markers=["left_wingtip", "right_wingtip"])
        hawk = Animal3D("hawk")
        hawk.load_motion_data(path)
        captured = capsys.readouterr()
        assert "Warning:" in captured.out
        assert "left_wingtip" in captured.out
        assert "right_wingtip" in captured.out
        assert "column mapping mismatch" in captured.out

    def test_notes_unmatched_csv_columns(self, capsys):
        extra = {
            "extra_point_x": [1.0] * 5,
            "extra_point_y": [2.0] * 5,
            "extra_point_z": [3.0] * 5,
        }
        path = _make_hawk_csv(extra_columns=extra)
        hawk = Animal3D("hawk")
        hawk.load_motion_data(path)
        captured = capsys.readouterr()
        assert "Note:" in captured.out
        assert "extra_point_x" in captured.out

    def test_no_warnings_on_clean_load(self, capsys):
        path = _make_hawk_csv()
        hawk = Animal3D("hawk")
        hawk.load_motion_data(path)
        captured = capsys.readouterr()
        assert "Warning:" not in captured.out
        assert "Note:" not in captured.out
        assert "Loaded 5 frames" in captured.out

    def test_custom_column_mapping_label(self, capsys):
        path = _make_hawk_csv()
        hawk = Animal3D("hawk")
        hawk.load_motion_data(path, column_mapping={"left_wingtip": "left_wingtip"})
        captured = capsys.readouterr()
        assert "custom" in captured.out


class TestAnimalRepr:
    """Test __repr__ on Animal3D."""

    def test_repr_no_motion_data(self):
        hawk = Animal3D("hawk")
        r = repr(hawk)
        assert "Animal3D('hawk'" in r
        assert "14 markers" in r
        assert "no motion data loaded" in r

    def test_repr_with_motion_data(self):
        path = _make_hawk_csv(n_frames=20)
        hawk = Animal3D("hawk")
        hawk.load_motion_data(path)
        r = repr(hawk)
        assert "Animal3D('hawk'" in r
        assert "20 frames loaded" in r
        assert "8 in analysis set" in r

    def test_repr_with_initial_data(self):
        hawk = Animal3D("hawk", data="data/mean_hawk_shape.csv")
        r = repr(hawk)
        # Initial data via constructor doesn't set motion frames
        assert "no motion data loaded" in r
