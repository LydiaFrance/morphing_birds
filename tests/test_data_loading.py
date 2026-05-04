"""Tests for data loading error handling and edge cases."""

import numpy as np
import pandas as pd
import pytest

from morphing_birds import Animal3D, SkeletonDefinition
from morphing_birds.data_loading import (
    load_from_csv,
    load_from_dict,
    load_mean_shape_csv,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _hawk_skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_builtin("hawk")


def _kestrel_skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_builtin("kestrel")


def _pigeon_skeleton() -> SkeletonDefinition:
    return SkeletonDefinition.from_builtin("pigeon")


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

    Returns
    -------
    pathlib.Path
        Path to the created CSV file.
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

    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


# ------------------------------------------------------------------
# 1. Missing CSV file
# ------------------------------------------------------------------


class TestMissingCsvFile:
    """Loading a non-existent CSV should raise a clear error."""

    def test_load_from_csv_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            load_from_csv(str(missing), _hawk_skeleton())

    def test_animal_load_data_missing_file(self, tmp_path):
        missing = tmp_path / "nope.csv"
        hawk = Animal3D("hawk")
        with pytest.raises(FileNotFoundError):
            hawk.load_data(str(missing))

    def test_animal_load_motion_data_missing_file(self, tmp_path):
        missing = tmp_path / "gone.csv"
        hawk = Animal3D("hawk")
        with pytest.raises(FileNotFoundError):
            hawk.load_motion_data(str(missing))

    def test_animal_constructor_missing_file(self, tmp_path):
        missing = tmp_path / "nowhere.csv"
        with pytest.raises(FileNotFoundError):
            Animal3D("hawk", data=str(missing))


# ------------------------------------------------------------------
# 2. CSV with wrong column names
# ------------------------------------------------------------------


class TestWrongColumnNames:
    """Columns that don't match any marker mapping should produce all-NaN data."""

    def test_completely_unrelated_columns(self, tmp_path):
        bogus_names = ["alpha", "beta", "gamma", "delta"]
        path = _make_csv(tmp_path, bogus_names)
        skel = _hawk_skeleton()
        result = load_from_csv(str(path), skel)

        # Every marker slot should be NaN (no columns matched)
        assert result.shape == (5, skel.n_markers, 3)
        assert np.all(np.isnan(result))

    def test_wrong_columns_via_animal(self, tmp_path, capsys):
        bogus_names = ["foo", "bar", "baz"]
        path = _make_csv(tmp_path, bogus_names)
        hawk = Animal3D("hawk")
        data = hawk.load_motion_data(str(path))

        # All analysis markers should be NaN
        assert np.all(np.isnan(data))

        # Should print a warning about markers with no data
        captured = capsys.readouterr()
        assert "Warning:" in captured.out


# ------------------------------------------------------------------
# 3. CSV with partial marker data
# ------------------------------------------------------------------


class TestPartialMarkerData:
    """Some markers present, some missing — missing ones should be NaN."""

    def test_partial_hawk_markers(self, tmp_path):
        skel = _hawk_skeleton()
        present = ["left_wingtip", "right_wingtip"]
        path = _make_csv(tmp_path, present)
        result = load_from_csv(str(path), skel)

        # Present markers should have real values
        lw_idx = skel.all_marker_names.index("left_wingtip")
        rw_idx = skel.all_marker_names.index("right_wingtip")
        assert not np.any(np.isnan(result[:, lw_idx, :]))
        assert not np.any(np.isnan(result[:, rw_idx, :]))

        # Missing markers should be NaN
        missing_idx = skel.all_marker_names.index("hood")
        assert np.all(np.isnan(result[:, missing_idx, :]))

    def test_partial_load_warns(self, tmp_path, capsys):
        present = ["left_wingtip", "right_wingtip"]
        path = _make_csv(tmp_path, present)
        hawk = Animal3D("hawk")
        hawk.load_motion_data(str(path))
        captured = capsys.readouterr()
        assert "Warning:" in captured.out
        # Should list a missing marker name
        assert "hood" in captured.out


# ------------------------------------------------------------------
# 4. Mismatched skeleton — hawk CSV with pigeon skeleton
# ------------------------------------------------------------------


class TestMismatchedSkeleton:
    """Loading hawk-formatted CSV with a pigeon skeleton should yield NaN."""

    def test_hawk_csv_pigeon_skeleton(self, tmp_path):
        hawk_skel = _hawk_skeleton()
        hawk_names = hawk_skel.all_marker_names
        path = _make_csv(tmp_path, hawk_names)

        pigeon_skel = _pigeon_skeleton()
        result = load_from_csv(str(path), pigeon_skel)

        # Pigeon markers that don't overlap with hawk names should be NaN
        pigeon_only = set(pigeon_skel.all_marker_names) - set(hawk_names)
        for name in pigeon_only:
            idx = pigeon_skel.all_marker_names.index(name)
            assert np.all(
                np.isnan(result[:, idx, :])
            ), f"Expected NaN for pigeon-only marker '{name}'"

    def test_hawk_csv_pigeon_skeleton_shared_markers_load(self, tmp_path):
        """Shared markers without column remapping should load fine.

        Pigeon remaps most marker names (e.g. left_secondary ->
        Left_Wrist_Trailing_Edge), so only markers whose pigeon
        column_mapping resolves to the hawk canonical name will match.
        """
        hawk_skel = _hawk_skeleton()
        hawk_names = hawk_skel.all_marker_names
        path = _make_csv(tmp_path, hawk_names)

        pigeon_skel = _pigeon_skeleton()
        result = load_from_csv(str(path), pigeon_skel)

        # Only markers that aren't remapped by pigeon's column_mapping
        # will match the hawk CSV columns (which use canonical names).
        shared = set(pigeon_skel.all_marker_names) & set(hawk_names)
        unmapped_shared = {
            n for n in shared if pigeon_skel.column_mapping.get(n, n) == n
        }
        # If no unmapped shared markers exist, the test is still valid
        for name in unmapped_shared:
            idx = pigeon_skel.all_marker_names.index(name)
            assert not np.all(
                np.isnan(result[:, idx, :])
            ), f"Shared unmapped marker '{name}' should have loaded data"


# ------------------------------------------------------------------
# 5. Empty CSV or single-row CSV
# ------------------------------------------------------------------


class TestEmptyAndSingleRowCsv:
    """Edge cases for CSVs with very few rows."""

    def test_empty_csv_no_rows(self, tmp_path):
        """A CSV with headers but zero data rows should produce a 0-frame array."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        # Write headers only
        cols = []
        for n in names:
            cols.extend([f"{n}_x", f"{n}_y", f"{n}_z"])
        header = ",".join(cols)
        path = tmp_path / "empty.csv"
        path.write_text(header + "\n")

        result = load_from_csv(str(path), skel)
        assert result.shape == (0, skel.n_markers, 3)

    def test_single_row_csv(self, tmp_path):
        """A CSV with exactly one data row should produce a 1-frame array."""
        skel = _hawk_skeleton()
        path = _make_csv(tmp_path, skel.all_marker_names, n_frames=1)
        result = load_from_csv(str(path), skel)
        assert result.shape == (1, skel.n_markers, 3)
        assert not np.all(np.isnan(result))

    def test_completely_empty_csv(self, tmp_path):
        """A completely empty CSV (no headers) should raise an error."""
        path = tmp_path / "blank.csv"
        path.write_text("")
        skel = _hawk_skeleton()
        with pytest.raises(pd.errors.EmptyDataError):
            load_from_csv(str(path), skel)


# ------------------------------------------------------------------
# 6. NaN values in CSV
# ------------------------------------------------------------------


class TestNanValuesInCsv:
    """NaN values in the CSV should propagate into the output array."""

    def test_nan_propagates_to_output(self, tmp_path):
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {
            "left_wingtip": [(0, "x"), (2, "z")],
            "hood": [(1, "y")],
        }
        path = _make_csv(tmp_path, names, n_frames=5, nan_indices=nan_positions)
        result = load_from_csv(str(path), skel)

        lw_idx = skel.all_marker_names.index("left_wingtip")
        hood_idx = skel.all_marker_names.index("hood")

        # Check specific NaN positions
        assert np.isnan(result[0, lw_idx, 0])  # frame 0, x
        assert np.isnan(result[2, lw_idx, 2])  # frame 2, z
        assert np.isnan(result[1, hood_idx, 1])  # frame 1, y

        # Non-NaN positions should be finite
        assert np.isfinite(result[0, lw_idx, 1])  # frame 0, y
        assert np.isfinite(result[0, lw_idx, 2])  # frame 0, z

    def test_remove_nan_frames(self, tmp_path):
        """Animal3D.remove_nan_frames should strip frames containing NaN."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {
            "left_wingtip": [(1, "x"), (3, "y")],
        }
        path = _make_csv(tmp_path, names, n_frames=5, nan_indices=nan_positions)
        result = load_from_csv(str(path), skel)

        animal = Animal3D(skel)
        clean, mask = animal.remove_nan_frames(result, analysis_only=False)
        assert clean.shape[0] == 3  # frames 0, 2, 4 survive
        assert mask.sum() == 3

    def test_mean_shape_csv_replaces_nan_with_zero(self, tmp_path):
        """load_mean_shape_csv should replace NaN with 0.0."""
        skel = _hawk_skeleton()
        names = skel.all_marker_names
        nan_positions = {"left_wingtip": [(0, "x")]}
        path = _make_csv(tmp_path, names, n_frames=1, nan_indices=nan_positions)
        result = load_mean_shape_csv(str(path), skel)

        lw_idx = skel.all_marker_names.index("left_wingtip")
        # NaN should have been replaced with 0.0
        assert result[0, lw_idx, 0] == 0.0
        assert not np.isnan(result).any()


# ------------------------------------------------------------------
# 7. column_mapping override parameter
# ------------------------------------------------------------------


class TestColumnMappingOverride:
    """The column_mapping parameter should remap CSV column names."""

    def test_custom_mapping_remaps_columns(self, tmp_path):
        """A custom mapping should allow non-standard CSV column names."""
        skel = _hawk_skeleton()
        # Create CSV with non-standard column names
        custom_names = ["wing_L", "wing_R"]
        path = _make_csv(tmp_path, custom_names)

        # Map canonical names to the custom CSV names
        mapping = {
            "left_wingtip": "wing_L",
            "right_wingtip": "wing_R",
        }
        result = load_from_csv(str(path), skel, column_mapping=mapping)

        lw_idx = skel.all_marker_names.index("left_wingtip")
        rw_idx = skel.all_marker_names.index("right_wingtip")

        # Mapped markers should have real data
        assert not np.any(np.isnan(result[:, lw_idx, :]))
        assert not np.any(np.isnan(result[:, rw_idx, :]))

        # Unmapped markers should be NaN
        hood_idx = skel.all_marker_names.index("hood")
        assert np.all(np.isnan(result[:, hood_idx, :]))

    def test_kestrel_column_mapping_works(self, tmp_path):
        """Kestrel has extensive column mapping — verify it resolves correctly."""
        skel = _kestrel_skeleton()
        # Create CSV using the kestrel's mapped column names (e.g. h1, l_sh, r_sh)
        mapped_names = list(skel.column_mapping.values())
        path = _make_csv(tmp_path, mapped_names)

        result = load_from_csv(str(path), skel)

        # All non-ignored markers should have data
        for idx, name in enumerate(skel.all_marker_names):
            if name in skel.ignored_columns:
                continue
            marker_data = result[:, idx, :]
            assert not np.all(np.isnan(marker_data)), (
                f"Kestrel marker '{name}' (mapped to "
                f"'{skel.column_mapping.get(name, name)}') should have data"
            )

    def test_custom_mapping_overrides_skeleton_default(self, tmp_path):
        """A custom mapping should override the skeleton's built-in mapping."""
        skel = _kestrel_skeleton()

        # Create CSV with a completely different name for the head marker
        # Kestrel maps head -> h1, but we override to head -> my_head
        custom_csv_names = ["my_head"]
        path = _make_csv(tmp_path, custom_csv_names)

        mapping = {"head": "my_head"}
        result = load_from_csv(str(path), skel, column_mapping=mapping)

        head_idx = skel.all_marker_names.index("head")
        assert not np.any(
            np.isnan(result[:, head_idx, :])
        ), "Custom mapping should override skeleton default"

    def test_column_mapping_via_animal_constructor(self, tmp_path):
        """column_mapping passed to Animal3D constructor should be used."""
        custom_names = ["my_lwing", "my_rwing"]
        path = _make_csv(tmp_path, custom_names, n_frames=1)
        mapping = {
            "left_wingtip": "my_lwing",
            "right_wingtip": "my_rwing",
        }
        hawk = Animal3D("hawk", data=str(path), column_mapping=mapping)

        lw_idx = hawk.skeleton.marker_index("left_wingtip")
        # Should have loaded real data for left_wingtip
        assert hawk.current_shape[0, lw_idx, 0] != 0.0

    def test_column_mapping_via_load_motion_data(self, tmp_path, capsys):
        """column_mapping passed to load_motion_data should be used."""
        custom_names = ["lwt", "rwt"]
        path = _make_csv(tmp_path, custom_names, n_frames=3)

        mapping = {
            "left_wingtip": "lwt",
            "right_wingtip": "rwt",
        }
        hawk = Animal3D("hawk")
        data = hawk.load_motion_data(str(path), column_mapping=mapping)

        lw_idx = hawk.skeleton.all_marker_names.index("left_wingtip")
        assert not np.any(np.isnan(data[:, lw_idx, :]))

        captured = capsys.readouterr()
        assert "custom" in captured.out


# ------------------------------------------------------------------
# 8. load_from_dict edge cases
# ------------------------------------------------------------------


class TestLoadFromDictEdgeCases:
    """Edge cases for the dictionary-based loading path."""

    def test_unknown_marker_name_raises(self):
        skel = _hawk_skeleton()
        positions = {"nonexistent_marker": [1.0, 2.0, 3.0]}
        with pytest.raises(ValueError, match="Unknown marker name"):
            load_from_dict(positions, skel)

    def test_empty_dict_returns_zeros(self):
        skel = _hawk_skeleton()
        result = load_from_dict({}, skel)
        assert result.shape == (1, skel.n_markers, 3)
        assert np.all(result == 0)

    def test_unsupported_data_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            Animal3D("hawk", data=12345)


# ------------------------------------------------------------------
# 9. No-underscore column format
# ------------------------------------------------------------------


class TestNoUnderscoreColumnFormat:
    """CSV columns can use ``markerx`` instead of ``marker_x``."""

    def test_no_underscore_format(self, tmp_path):
        skel = _hawk_skeleton()
        path = _make_csv(tmp_path, skel.all_marker_names, suffix="")
        result = load_from_csv(str(path), skel)

        # Should still load correctly
        lw_idx = skel.all_marker_names.index("left_wingtip")
        assert not np.any(np.isnan(result[:, lw_idx, :]))
