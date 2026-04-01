"""Tests for plotting functions — especially that they don't clobber
pre-applied transforms.
"""

import numpy as np
import plotly.graph_objs as go
import pytest

from morphing_birds import (
    Animal3D,
    plot_compare_plotly,
    plot_partial_plotly,
    plot_plotly,
    plot_plotly_compare,
    plot_plotly_with_trace,
)
from morphing_birds.plotting.plotly_helpers import is_surface_section
from morphing_birds.plotting.plotly_plots import get_polygon_plotly


@pytest.fixture
def hawk():
    return Animal3D("hawk", data="data/mean_hawk_shape.csv")


@pytest.fixture
def spider():
    return Animal3D("spider", data="data/mean_spider_shape_carolina.csv")


# ------------------------------------------------------------------
# plot_plotly
# ------------------------------------------------------------------

class TestPlotPlotlyPreservesTransform:
    """plot_plotly must not reset a transform applied before the call."""

    def test_display_only_transform_preserved(self, hawk):
        before = hawk.current_shape.copy()
        hawk.transform_display_only(bodypitch=30)
        transformed = hawk.current_shape.copy()

        # Sanity: display-only markers actually moved
        assert not np.allclose(
            before[0, hawk.display_only_indices],
            transformed[0, hawk.display_only_indices],
        )

        plot_plotly(hawk, colour="steelblue")

        # After plotting without transform args, shape must be unchanged
        np.testing.assert_array_almost_equal(
            hawk.current_shape, transformed,
            err_msg="plot_plotly clobbered a pre-applied transform_display_only",
        )

    def test_transform_all_preserved(self, hawk):
        hawk.transform_all(bodypitch=25, bodyyaw=10)
        transformed = hawk.current_shape.copy()

        plot_plotly(hawk, colour="red")

        np.testing.assert_array_almost_equal(
            hawk.current_shape, transformed,
            err_msg="plot_plotly clobbered a pre-applied transform_all",
        )

    def test_inline_transform_does_not_persist(self, hawk):
        """When plot_plotly applies its own transform via args, the animal
        must be restored to its original state afterward."""
        before = hawk.current_shape.copy()

        plot_plotly(hawk, colour="blue", bodypitch=45)

        np.testing.assert_array_almost_equal(
            hawk.current_shape, before,
            err_msg="plot_plotly with inline transform leaked state",
        )

    def test_no_args_leaves_default_untouched(self, hawk):
        """Calling plot_plotly with zero transform args must not touch the shape."""
        before = hawk.current_shape.copy()

        plot_plotly(hawk, colour="grey")

        np.testing.assert_array_almost_equal(
            hawk.current_shape, before,
            err_msg="plot_plotly with no transform args mutated current_shape",
        )


# ------------------------------------------------------------------
# plot_plotly_compare
# ------------------------------------------------------------------

class TestPlotPlotlyComparePreservesTransform:

    def test_pre_applied_transforms_preserved(self, hawk):
        hawk_a = hawk.copy()
        hawk_b = hawk.copy()
        hawk_a.transform_all(bodypitch=20)
        hawk_b.transform_display_only(bodyyaw=15)

        shape_a = hawk_a.current_shape.copy()
        shape_b = hawk_b.current_shape.copy()

        plot_plotly_compare([hawk_a, hawk_b], colours=["red", "blue"])

        np.testing.assert_array_almost_equal(hawk_a.current_shape, shape_a)
        np.testing.assert_array_almost_equal(hawk_b.current_shape, shape_b)


# ------------------------------------------------------------------
# plot_partial_plotly
# ------------------------------------------------------------------

class TestPlotPartialPreservesTransform:

    def test_section_plot_preserves_transform(self, hawk):
        hawk.transform_display_only(bodypitch=30)
        transformed = hawk.current_shape.copy()

        plot_partial_plotly(hawk, section_name="left_handwing", colour="teal")

        np.testing.assert_array_almost_equal(
            hawk.current_shape, transformed,
            err_msg="plot_partial_plotly clobbered a pre-applied transform",
        )

    def test_leg_plot_preserves_transform(self, spider):
        spider.transform_all(bodypitch=20)
        transformed = spider.current_shape.copy()

        plot_partial_plotly(spider, leg_number=3, colour="red")

        np.testing.assert_array_almost_equal(
            spider.current_shape, transformed,
            err_msg="plot_partial_plotly (leg) clobbered a pre-applied transform",
        )

    def test_inline_transform_does_not_persist(self, hawk):
        before = hawk.current_shape.copy()

        plot_partial_plotly(hawk, section_name="tail", colour="red", bodypitch=45)

        np.testing.assert_array_almost_equal(
            hawk.current_shape, before,
            err_msg="plot_partial_plotly with inline transform leaked state",
        )


# ------------------------------------------------------------------
# plot_compare_plotly
# ------------------------------------------------------------------

class TestPlotComparePlotlyPreservesTransform:

    def test_restores_state_after_keypoints_list(self, hawk):
        before = hawk.current_shape.copy()

        kp1 = hawk.markers.copy()
        kp2 = hawk.markers.copy() * 0.9
        plot_compare_plotly(hawk, [kp1, kp2], colours=["blue", "red"])

        np.testing.assert_array_almost_equal(
            hawk.current_shape, before,
            err_msg="plot_compare_plotly did not restore current_shape",
        )


# ------------------------------------------------------------------
# plot_plotly_with_trace
# ------------------------------------------------------------------

class TestPlotWithTracePreservesTransform:

    def test_preserves_pre_applied_transform(self, hawk):
        hawk.transform_all(bodypitch=15)
        transformed = hawk.current_shape.copy()

        trace_frames = np.random.rand(5, 8, 3) * 0.1
        plot_plotly_with_trace(hawk, trace_frames, colour="blue")

        np.testing.assert_array_almost_equal(
            hawk.current_shape, transformed,
            err_msg="plot_plotly_with_trace clobbered a pre-applied transform",
        )


# ------------------------------------------------------------------
# Config-driven surface rendering
# ------------------------------------------------------------------

class TestSurfaceConfig:
    """Sections with surface: false in config should render as lines only."""

    def test_is_surface_section_hawk_wing(self, hawk):
        """Hawk wing sections default to surface=true (no explicit setting)."""
        assert is_surface_section("left_handwing", hawk) is True
        assert is_surface_section("right_armwing", hawk) is True

    def test_is_surface_section_hawk_body(self, hawk):
        """Hawk body/head sections also default to surface=true."""
        assert is_surface_section("body", hawk) is True
        assert is_surface_section("head", hawk) is True

    def test_is_surface_section_spider_leg(self, spider):
        """Spider leg sections have surface: false in config."""
        assert is_surface_section("leg_1", spider) is False
        assert is_surface_section("leg_5", spider) is False

    def test_is_surface_section_spider_body(self, spider):
        """Spider body section defaults to surface=true."""
        assert is_surface_section("body", spider) is True

    def test_is_surface_section_no_instance(self):
        """Without an animal instance, default to surface=true."""
        assert is_surface_section("leg_1") is True
        assert is_surface_section("anything") is True

    def test_spider_leg_produces_no_mesh(self, spider):
        """Spider leg sections should produce no Mesh3d trace."""
        mesh, lines = get_polygon_plotly(spider, "leg_1", "blue", alpha=0.5)
        assert mesh is None
        assert lines is not None
        assert isinstance(lines, go.Scatter3d)

    def test_spider_body_produces_mesh(self, spider):
        """Spider body section should produce a Mesh3d trace."""
        mesh, lines = get_polygon_plotly(spider, "body", "blue", alpha=0.5)
        assert mesh is not None
        assert isinstance(mesh, go.Mesh3d)
        assert lines is not None

    def test_hawk_wing_produces_mesh(self, hawk):
        """Hawk wing sections should produce a Mesh3d trace."""
        mesh, lines = get_polygon_plotly(hawk, "left_handwing", "blue", alpha=0.5)
        assert mesh is not None
        assert isinstance(mesh, go.Mesh3d)
        assert lines is not None
