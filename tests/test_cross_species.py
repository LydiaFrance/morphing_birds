"""Cross-species integration tests for the bilateral and data loading pipeline.

These tests verify that the end-to-end pipeline — loading real mean-shape
CSVs, creating Animal3D instances, and performing bilateral round-trips —
works correctly for pigeon, spider, and kestrel species.
"""

from pathlib import Path

import numpy as np
import pytest

from morphing_birds import Animal3D, SkeletonDefinition
from morphing_birds.bilateral import (
    _analysis_pairs_and_centres,
    make_bilateral,
    make_unilateral,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ------------------------------------------------------------------
# Helpers
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


def _load_animal(species: str, csv_name: str) -> Animal3D:
    """Load an Animal3D from a mean-shape CSV, skipping if file is absent."""
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        pytest.skip(f"{csv_name} not found in data directory")
    return Animal3D(species, data=str(csv_path))


# ==================================================================
# 1. Pigeon end-to-end
# ==================================================================


class TestPigeonEndToEnd:
    """Full pipeline test for pigeon: load CSV, round-trip bilateral."""

    @pytest.fixture
    def pigeon(self):
        return _load_animal("pigeon", "mean_pigeon_shape.csv")

    def test_load_pigeon_mean_shape(self, pigeon):
        """Pigeon mean shape loads without NaN values."""
        assert pigeon.current_shape.ndim == 3
        assert pigeon.current_shape.shape[0] == 1
        assert not np.isnan(pigeon.current_shape).any()

    def test_pigeon_marker_count(self, pigeon):
        """Pigeon has the expected number of markers."""
        assert pigeon.skeleton.n_markers == 19

    def test_pigeon_round_trip_preserves_shape(self, pigeon):
        """make_unilateral -> make_bilateral round-trip preserves pigeon shape."""
        data = pigeon.current_shape.copy()
        skel = pigeon.skeleton

        uni, is_left, _ = make_unilateral(data, skel)
        reconstructed = make_bilateral(uni, skel, is_left)

        pairs, centres = _analysis_pairs_and_centres(skel)

        # Check all paired markers survive the round-trip
        for left, right in pairs:
            l_idx = skel.marker_index(left)
            r_idx = skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :], data[:, l_idx, :],
                err_msg=f"Pigeon left marker {left} changed after round-trip",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :], data[:, r_idx, :],
                err_msg=f"Pigeon right marker {right} changed after round-trip",
            )

        # Check centre markers survive
        for c in centres:
            if c in skel.all_marker_names:
                c_idx = skel.marker_index(c)
                np.testing.assert_array_almost_equal(
                    reconstructed[:, c_idx, :], data[:, c_idx, :],
                    err_msg=f"Pigeon centre marker {c} changed after round-trip",
                )

    def test_pigeon_unilateral_doubles_frames(self, pigeon):
        """Unilateral conversion doubles the frame count for pigeon."""
        data = pigeon.current_shape.copy()
        uni, is_left, _ = make_unilateral(data, pigeon.skeleton)
        assert uni.shape[0] == data.shape[0] * 2

    def test_pigeon_analysis_indices_within_bounds(self, pigeon):
        """Analysis indices should be valid indices into the marker array."""
        indices = pigeon.analysis_indices
        assert len(indices) > 0
        assert all(0 <= i < pigeon.skeleton.n_markers for i in indices)


# ==================================================================
# 2. Spider end-to-end (dict-based laterality with suffix pairing)
# ==================================================================


class TestSpiderEndToEnd:
    """Full pipeline test for spider: load CSV, round-trip bilateral."""

    @pytest.fixture
    def spider(self):
        return _load_animal("spider", "mean_spider_shape_carolina.csv")

    def test_load_spider_mean_shape(self, spider):
        """Spider mean shape loads without NaN values."""
        assert spider.current_shape.ndim == 3
        assert spider.current_shape.shape[0] == 1
        assert not np.isnan(spider.current_shape).any()

    def test_spider_marker_count(self, spider):
        """Spider has the expected number of markers."""
        assert spider.skeleton.n_markers == 36

    def test_spider_round_trip_preserves_shape(self, spider):
        """Round-trip with synthetic valid bilateral data preserves spider shape.

        The real spider mean shape does not satisfy the x-axis left/right
        separation convention (spiders are oriented differently to birds),
        so we use synthetically valid bilateral data based on the spider
        skeleton definition.
        """
        skel = spider.skeleton
        data = _make_valid_bilateral(skel, n_frames=5)

        uni, is_left, _ = make_unilateral(data, skel)
        reconstructed = make_bilateral(uni, skel, is_left)

        pairs, centres = _analysis_pairs_and_centres(skel)

        for left, right in pairs:
            l_idx = skel.marker_index(left)
            r_idx = skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :], data[:, l_idx, :],
                err_msg=f"Spider left marker {left} changed after round-trip",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :], data[:, r_idx, :],
                err_msg=f"Spider right marker {right} changed after round-trip",
            )

        # Check centre markers survive
        for c in centres:
            if c in skel.all_marker_names:
                c_idx = skel.marker_index(c)
                np.testing.assert_array_almost_equal(
                    reconstructed[:, c_idx, :], data[:, c_idx, :],
                    err_msg=f"Spider centre marker {c} changed after round-trip",
                )

    def test_spider_has_dict_laterality_pairs(self, spider):
        """Spider uses dict-based laterality — verify it has marker pairs."""
        pairs = spider.skeleton.get_marker_pairs()
        assert len(pairs) > 0, "Spider should have paired markers"

    def test_spider_analysis_indices_within_bounds(self, spider):
        """Analysis indices should be valid indices into the marker array."""
        indices = spider.analysis_indices
        assert len(indices) > 0
        assert all(0 <= i < spider.skeleton.n_markers for i in indices)


# ==================================================================
# 3. Kestrel end-to-end (complex column mapping)
# ==================================================================


class TestKestrelEndToEnd:
    """Full pipeline test for kestrel: load CSV, round-trip bilateral."""

    @pytest.fixture
    def kestrel(self):
        return _load_animal("kestrel", "mean_kestrel_shape.csv")

    def test_load_kestrel_mean_shape(self, kestrel):
        """Kestrel mean shape loads without NaN values."""
        assert kestrel.current_shape.ndim == 3
        assert kestrel.current_shape.shape[0] == 1
        assert not np.isnan(kestrel.current_shape).any()

    def test_kestrel_marker_count(self, kestrel):
        """Kestrel has the expected number of markers."""
        assert kestrel.skeleton.n_markers == 35

    def test_kestrel_round_trip_preserves_shape(self, kestrel):
        """make_unilateral -> make_bilateral round-trip preserves kestrel shape."""
        data = kestrel.current_shape.copy()
        skel = kestrel.skeleton

        uni, is_left, _ = make_unilateral(data, skel)
        reconstructed = make_bilateral(uni, skel, is_left)

        pairs, _centres = _analysis_pairs_and_centres(skel)

        for left, right in pairs:
            l_idx = skel.marker_index(left)
            r_idx = skel.marker_index(right)
            np.testing.assert_array_almost_equal(
                reconstructed[:, l_idx, :], data[:, l_idx, :],
                err_msg=f"Kestrel left marker {left} changed after round-trip",
            )
            np.testing.assert_array_almost_equal(
                reconstructed[:, r_idx, :], data[:, r_idx, :],
                err_msg=f"Kestrel right marker {right} changed after round-trip",
            )

    def test_kestrel_has_column_mapping(self, kestrel):
        """Kestrel should have a non-trivial column mapping."""
        mapping = kestrel.skeleton.column_mapping
        assert len(mapping) > 0, "Kestrel should have column_mapping entries"
        # At least some mappings should differ from canonical names
        remapped = {k: v for k, v in mapping.items() if k != v}
        assert len(remapped) > 0, "Kestrel should remap some column names"

    def test_kestrel_analysis_indices_within_bounds(self, kestrel):
        """Analysis indices should be valid indices into the marker array."""
        indices = kestrel.analysis_indices
        assert len(indices) > 0
        assert all(0 <= i < kestrel.skeleton.n_markers for i in indices)


# ==================================================================
# 4. Analysis indices consistency through the pipeline
# ==================================================================


class TestAnalysisIndicesConsistency:
    """Verify analysis_indices remain consistent through bilateral conversion."""

    @pytest.mark.parametrize(
        "species,csv_name",
        [
            ("pigeon", "mean_pigeon_shape.csv"),
            ("kestrel", "mean_kestrel_shape.csv"),
            ("spider", "mean_spider_shape_carolina.csv"),
        ],
    )
    def test_analysis_slice_survives_round_trip(self, species, csv_name):
        """Analysis-indexed data should be identical after round-trip.

        Uses synthetic valid bilateral data because some species (e.g.
        spider) have mean shapes that do not satisfy the x-axis left/right
        separation assumption required by ``validate_left_right``.
        """
        animal = _load_animal(species, csv_name)
        skel = animal.skeleton
        indices = animal.analysis_indices

        # Use synthetic data with valid left/right separation
        data = _make_valid_bilateral(skel, n_frames=5)

        # Extract analysis data before round-trip
        analysis_before = data[:, indices, :].copy()

        # Round-trip
        uni, is_left, _ = make_unilateral(data, skel)
        reconstructed = make_bilateral(uni, skel, is_left)

        # Extract analysis data after round-trip
        analysis_after = reconstructed[:, indices, :]

        np.testing.assert_array_almost_equal(
            analysis_after, analysis_before,
            err_msg=f"{species} analysis data changed after bilateral round-trip",
        )

    @pytest.mark.parametrize(
        "species,csv_name",
        [
            ("pigeon", "mean_pigeon_shape.csv"),
            ("kestrel", "mean_kestrel_shape.csv"),
            ("spider", "mean_spider_shape_carolina.csv"),
        ],
    )
    def test_analysis_indices_match_skeleton(self, species, csv_name):
        """Animal3D.analysis_indices should match skeleton.analysis_indices."""
        animal = _load_animal(species, csv_name)
        assert animal.analysis_indices == animal.skeleton.analysis_indices

    @pytest.mark.parametrize(
        "species,csv_name",
        [
            ("pigeon", "mean_pigeon_shape.csv"),
            ("kestrel", "mean_kestrel_shape.csv"),
            ("spider", "mean_spider_shape_carolina.csv"),
        ],
    )
    def test_analysis_and_display_indices_partition_all_markers(
        self, species, csv_name
    ):
        """Analysis + display-only indices should cover all markers exactly once."""
        animal = _load_animal(species, csv_name)
        analysis = set(animal.analysis_indices)
        display = set(animal.skeleton.display_only_indices)

        all_indices = set(range(animal.skeleton.n_markers))
        assert analysis | display == all_indices, (
            f"{species}: analysis + display indices do not cover all markers"
        )
        assert analysis & display == set(), (
            f"{species}: analysis and display indices overlap"
        )


# ==================================================================
# 5. Cross-species comparison
# ==================================================================


class TestCrossSpeciesComparison:
    """Verify that different species produce different marker counts and sections."""

    @pytest.fixture
    def all_species(self):
        """Load all four species and return as a dict."""
        species_map = {
            "hawk": ("hawk", "mean_hawk_shape.csv"),
            "pigeon": ("pigeon", "mean_pigeon_shape.csv"),
            "kestrel": ("kestrel", "mean_kestrel_shape.csv"),
            "spider": ("spider", "mean_spider_shape_carolina.csv"),
        }
        animals = {}
        for key, (species, csv) in species_map.items():
            csv_path = DATA_DIR / csv
            if not csv_path.exists():
                pytest.skip(f"{csv} not found")
            animals[key] = Animal3D(species, data=str(csv_path))
        return animals

    def test_marker_counts_differ_across_species(self, all_species):
        """Each species should have a different total marker count."""
        counts = {
            name: animal.skeleton.n_markers
            for name, animal in all_species.items()
        }
        # At least some species should differ in marker count
        unique_counts = set(counts.values())
        assert len(unique_counts) > 1, (
            f"Expected different marker counts across species, got {counts}"
        )

    def test_section_names_differ_across_species(self, all_species):
        """Different species should have different body section layouts."""
        section_sets = {
            name: frozenset(animal.skeleton.body_sections.keys())
            for name, animal in all_species.items()
        }
        unique_sections = set(section_sets.values())
        assert len(unique_sections) > 1, (
            "Expected different body sections across species"
        )

    def test_analysis_marker_counts_differ(self, all_species):
        """Different species should have different analysis marker counts."""
        analysis_counts = {
            name: len(animal.analysis_indices)
            for name, animal in all_species.items()
        }
        unique_counts = set(analysis_counts.values())
        assert len(unique_counts) > 1, (
            f"Expected different analysis marker counts, got {analysis_counts}"
        )

    def test_each_species_has_body_sections(self, all_species):
        """Every species should define at least one body section."""
        for name, animal in all_species.items():
            sections = animal.skeleton.body_sections
            assert len(sections) > 0, f"{name} has no body sections defined"

    def test_each_species_has_paired_markers(self, all_species):
        """Every species should have at least one left-right marker pair."""
        for name, animal in all_species.items():
            pairs = animal.skeleton.get_marker_pairs()
            assert len(pairs) > 0, f"{name} has no marker pairs"

    def test_polygon_sections_accessible_for_all_species(self, all_species):
        """Every body section should yield valid polygon coordinates."""
        for name, animal in all_species.items():
            for section in animal.polygons:
                coords = animal.get_polygon_coords(section)
                assert coords.ndim == 2, (
                    f"{name} section '{section}' has wrong dimensions"
                )
                assert coords.shape[1] == 3, (
                    f"{name} section '{section}' should have xyz columns"
                )
                assert coords.shape[0] > 0, (
                    f"{name} section '{section}' has no vertices"
                )
