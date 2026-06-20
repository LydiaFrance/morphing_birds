"""Tests for SkeletonDefinition config loading."""

import numpy as np
import pytest

from morphing_birds import Animal3D, SkeletonDefinition, animate_plotly
from morphing_birds.configs import list_configs


class TestSkeletonFromBuiltin:
    """Test loading builtin configs."""

    @pytest.mark.parametrize("name", list_configs())
    def test_load_builtin(self, name):
        skel = SkeletonDefinition.from_builtin(name)
        assert skel.name == name
        assert skel.n_markers > 0
        assert len(skel.all_marker_names) == skel.n_markers

    def test_unknown_builtin_raises(self):
        with pytest.raises(ValueError, match="Unknown builtin config"):
            SkeletonDefinition.from_builtin("unicorn")


class TestCustomSkeleton:
    """Test workshop-style custom skeleton construction."""

    def test_from_markers_with_explicit_pairs(self):
        skel = SkeletonDefinition.from_markers(
            "toy",
            markers=["nose", "left_paw", "right_paw", "tail_base"],
            body_sections={
                "body": ["nose", "tail_base"],
                "front": ["left_paw", "nose", "right_paw"],
            },
            analysis_exclude=["tail_base"],
            marker_pairs=[("left_paw", "right_paw")],
            centre_markers=["nose", "tail_base"],
        )

        assert skel.name == "toy"
        assert skel.analysis_markers == ["nose", "left_paw", "right_paw"]
        assert skel.get_marker_pairs() == [("left_paw", "right_paw")]
        assert skel.get_centre_markers() == ["nose", "tail_base"]
        assert skel.get_body_section_indices()["front"] == [1, 0, 2]

    def test_from_markers_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="Unknown marker"):
            SkeletonDefinition.from_markers(
                "toy",
                markers=["nose"],
                body_sections={"body": ["nose", "tail"]},
            )

    def test_from_markers_animates_full_keypoint_frames(self):
        skel = SkeletonDefinition.from_markers(
            "toy",
            markers=["nose", "left_paw", "right_paw", "tail_base"],
            body_sections={"front": ["left_paw", "nose", "right_paw"]},
            analysis_exclude=["tail_base"],
            marker_pairs=[("left_paw", "right_paw")],
            centre_markers=["nose", "tail_base"],
        )
        frames = np.array(
            [
                [[0, 1, 0], [-1, 0, 0], [1, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [-1, 0, 0.1], [1, 0, -0.1], [0, -1, 0]],
            ],
            dtype=float,
        )

        animal = Animal3D(skel, data=frames[0])
        fig = animate_plotly(animal, frames, axes_visible=False)

        assert len(fig.frames) == 2


class TestHawkSkeleton:
    """Test hawk-specific skeleton properties."""

    @pytest.fixture
    def hawk(self):
        return SkeletonDefinition.from_builtin("hawk")

    def test_marker_count(self, hawk):
        assert hawk.n_markers == 14
        assert hawk.n_analysis_markers == 8

    def test_analysis_exclude(self, hawk):
        excluded = hawk.analysis_exclude
        assert "left_shoulder" in excluded
        assert "hood" in excluded
        assert "left_wingtip" not in excluded

    def test_analysis_markers_order(self, hawk):
        analysis = hawk.analysis_markers
        assert analysis[0] == "left_wingtip"
        assert analysis[1] == "right_wingtip"
        assert len(analysis) == 8

    def test_body_sections(self, hawk):
        assert "left_handwing" in hawk.body_sections
        assert "right_armwing" in hawk.body_sections
        assert "tail" in hawk.body_sections
        assert len(hawk.body_sections) == 7

    def test_left_right_markers(self, hawk):
        left = hawk.get_left_markers()
        right = hawk.get_right_markers()
        assert len(left) == 6
        assert len(right) == 6
        assert all(n.startswith("left_") for n in left)
        assert all(n.startswith("right_") for n in right)

    def test_marker_pairs(self, hawk):
        pairs = hawk.get_marker_pairs()
        assert len(pairs) == 6
        for left, right in pairs:
            assert left.startswith("left_")
            assert right.startswith("right_")
            assert left[5:] == right[6:]

    def test_centre_markers(self, hawk):
        centres = hawk.get_centre_markers()
        assert "hood" in centres
        assert "tailpack" in centres

    def test_display_names_fallback(self, hawk):
        assert hawk.get_display_name("left_wingtip") == "left_wingtip"

    def test_analysis_marker_labels(self, hawk):
        labels = hawk.analysis_marker_labels()
        assert len(labels) == 24  # 8 markers * 3 axes
        assert labels[0] == "left_wingtip_x"

    def test_body_section_indices(self, hawk):
        indices = hawk.get_body_section_indices()
        assert "left_handwing" in indices
        # left_handwing has 3 markers
        assert len(indices["left_handwing"]) == 3


class TestPigeonSkeleton:
    """Test pigeon-specific skeleton properties."""

    @pytest.fixture
    def pigeon(self):
        return SkeletonDefinition.from_builtin("pigeon")

    def test_marker_count(self, pigeon):
        assert pigeon.n_markers == 19

    def test_column_mapping(self, pigeon):
        assert pigeon.column_mapping["left_wingtip"] == "Left_Tip"
        assert pigeon.column_mapping["right_wrist"] == "Right_Wrist_Leading_Edge"

    def test_simple_variant(self, pigeon):
        simple = pigeon.get_variant("simple")
        assert len(simple.analysis_exclude) > len(pigeon.analysis_exclude)
        # Simple mode has 7 body sections
        assert len(simple.body_sections) == 7

    def test_unknown_variant_raises(self, pigeon):
        with pytest.raises(ValueError, match="Unknown variant"):
            pigeon.get_variant("nonexistent")


class TestKestrelSkeleton:
    """Test kestrel skeleton with lab code column mapping."""

    @pytest.fixture
    def kestrel(self):
        return SkeletonDefinition.from_builtin("kestrel")

    def test_column_mapping(self, kestrel):
        assert kestrel.column_mapping["right_firstprimary_tip"] == "r_p1_3"
        assert kestrel.column_mapping["head"] == "h1"

    def test_display_names(self, kestrel):
        assert kestrel.get_display_name("right_firstprimary_tip") == "R 1st Primary Tip"
        assert kestrel.get_display_name("left_elbow") == "left_elbow"  # fallback

    def test_simple_variant(self, kestrel):
        simple = kestrel.get_variant("simple")
        assert "left_lastsecondary_tip" in simple.analysis_exclude


class TestSpiderSkeleton:
    """Test spider skeleton with dict-based laterality."""

    @pytest.fixture
    def spider(self):
        return SkeletonDefinition.from_builtin("spider")

    def test_marker_count(self, spider):
        assert spider.n_markers == 36  # 8 legs * 4 + 3 body + 1 centre

    def test_laterality_dict(self, spider):
        right = spider.get_right_markers()
        left = spider.get_left_markers()
        assert len(right) == 16  # 4 markers * 4 right legs
        assert len(left) == 16  # 4 markers * 4 left legs

    def test_centre_markers(self, spider):
        centres = spider.get_centre_markers()
        assert "clypeus" in centres
        assert "spinneret" in centres

    def test_leg_surface_false(self, spider):
        """Spider leg section_styles should have surface: false."""
        leg_style = spider.section_styles.get("leg", {})
        assert leg_style.get("surface") is False
