"""Tests for plotting output validation — figure types, trace counts,
polygon vertices, animation frames, and axis limits.
"""

import numpy as np
import plotly.graph_objs as go
import pytest

from morphing_birds import (
    Animal3D,
    animate_plotly,
    animate_plotly_partial,
    plot_compare_plotly,
    plot_partial_plotly,
    plot_plotly,
    plot_plotly_compare,
    plot_plotly_with_trace,
)
from morphing_birds.plotting.plotly_helpers import calculate_axis_limits, is_surface_section
from morphing_birds.plotting.plotly_plots import get_polygon_plotly


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def hawk():
    return Animal3D("hawk", data="data/mean_hawk_shape.csv")


@pytest.fixture
def spider():
    return Animal3D("spider", data="data/mean_spider_shape_carolina.csv")


# ------------------------------------------------------------------
# 1. Plot functions return valid Plotly Figure objects with expected
#    trace counts
# ------------------------------------------------------------------

class TestPlotReturnsValidFigures:
    """All plot functions must return ``go.Figure`` with sensible traces."""

    def test_plot_plotly_returns_figure(self, hawk):
        fig = plot_plotly(hawk, colour="steelblue")
        assert isinstance(fig, go.Figure)

    def test_plot_plotly_has_traces(self, hawk):
        """Hawk has 7 body sections. Each surface section produces a mesh
        trace + a line trace, so we expect at least 14 section traces plus
        a keypoint scatter trace."""
        fig = plot_plotly(hawk, colour="steelblue")
        # 7 sections x 2 traces (mesh + lines) + 1 keypoint scatter = 15
        assert len(fig.data) == 15

    def test_plot_plotly_spider_trace_count(self, spider):
        """Spider has 9 sections (8 legs + body). Legs are non-surface so
        they produce only line traces (1 each). Body produces mesh + lines.
        Plus a keypoint scatter trace."""
        fig = plot_plotly(spider, colour="red")
        # 8 legs x 1 (lines only) + 1 body x 2 (mesh + lines) + 1 scatter = 11
        assert len(fig.data) == 11

    def test_plot_plotly_compare_returns_figure(self, hawk):
        hawk_a = hawk.copy()
        hawk_b = hawk.copy()
        fig = plot_plotly_compare([hawk_a, hawk_b], colours=["red", "blue"])
        assert isinstance(fig, go.Figure)
        # Two animals, each contributing 14 section traces (no keypoints added)
        assert len(fig.data) == 28

    def test_plot_compare_plotly_returns_figure(self, hawk):
        kp1 = hawk.markers.copy()
        kp2 = hawk.markers.copy() * 0.9
        fig = plot_compare_plotly(hawk, [kp1, kp2], colours=["blue", "red"])
        assert isinstance(fig, go.Figure)
        # Two poses, each: 14 section traces + 1 keypoint scatter = 15, total 30
        assert len(fig.data) == 30

    def test_plot_plotly_with_trace_returns_figure(self, hawk):
        trace_frames = np.random.rand(5, 8, 3) * 0.1
        fig = plot_plotly_with_trace(hawk, trace_frames, colour="blue")
        assert isinstance(fig, go.Figure)
        # 15 base traces + 1 trace scatter = 16
        assert len(fig.data) == 16

    def test_plot_partial_section_returns_figure(self, hawk):
        fig = plot_partial_plotly(hawk, section_name="tail", colour="blue")
        assert isinstance(fig, go.Figure)
        # 1 section: mesh + lines = 2 traces
        assert len(fig.data) == 2

    def test_plot_partial_leg_returns_figure(self, spider):
        fig = plot_partial_plotly(spider, leg_number=3, colour="red")
        assert isinstance(fig, go.Figure)
        # Spider leg: non-surface, lines only = 1 trace
        assert len(fig.data) == 1


# ------------------------------------------------------------------
# 2. Section polygon vertex counts per species
# ------------------------------------------------------------------

class TestPolygonVertexCounts:
    """Polygon vertex counts must match the body_sections config."""

    def test_hawk_handwing_vertices(self, hawk):
        """Hawk handwing has 3 markers."""
        coords = hawk.get_polygon_coords("left_handwing")
        assert coords.shape == (3, 3)  # 3 markers, 3 dimensions

    def test_hawk_armwing_vertices(self, hawk):
        """Hawk armwing has 4 markers."""
        coords = hawk.get_polygon_coords("left_armwing")
        assert coords.shape == (4, 3)

    def test_hawk_body_vertices(self, hawk):
        """Hawk body has 4 markers."""
        coords = hawk.get_polygon_coords("body")
        assert coords.shape == (4, 3)

    def test_hawk_head_vertices(self, hawk):
        """Hawk head has 3 markers."""
        coords = hawk.get_polygon_coords("head")
        assert coords.shape == (3, 3)

    def test_hawk_tail_vertices(self, hawk):
        """Hawk tail has 4 markers."""
        coords = hawk.get_polygon_coords("tail")
        assert coords.shape == (4, 3)

    def test_spider_leg_vertices(self, spider):
        """Spider leg sections list 7 entries (with repeated markers for
        the line path back)."""
        coords = spider.get_polygon_coords("leg_1")
        assert coords.shape == (7, 3)

    def test_spider_body_vertices(self, spider):
        """Spider body has 10 markers (clypeus + 8 coxa + spinneret)."""
        coords = spider.get_polygon_coords("body")
        assert coords.shape == (10, 3)

    def test_all_hawk_sections_have_coords(self, hawk):
        """Every hawk section must return valid coordinates."""
        for section_name in hawk.polygons:
            coords = hawk.get_polygon_coords(section_name)
            assert coords.ndim == 2
            assert coords.shape[1] == 3
            assert not np.any(np.isnan(coords))

    def test_all_spider_sections_have_coords(self, spider):
        """Every spider section must return valid coordinates."""
        for section_name in spider.polygons:
            coords = spider.get_polygon_coords(section_name)
            assert coords.ndim == 2
            assert coords.shape[1] == 3
            assert not np.any(np.isnan(coords))


# ------------------------------------------------------------------
# 3. Non-surface sections render as lines only
# ------------------------------------------------------------------

class TestNonSurfaceSections:
    """Sections with ``surface: false`` must produce line traces only,
    never ``Mesh3d`` traces."""

    def test_spider_all_legs_produce_no_mesh(self, spider):
        """Every spider leg should produce a line trace but no mesh."""
        for leg_num in range(1, 9):
            section = f"leg_{leg_num}"
            mesh, lines = get_polygon_plotly(spider, section, "blue", alpha=0.5)
            assert mesh is None, f"{section} should not produce a mesh"
            assert lines is not None, f"{section} must produce a line trace"
            assert isinstance(lines, go.Scatter3d)
            assert lines.mode == "lines"

    def test_hawk_sections_all_produce_mesh(self, hawk):
        """All hawk sections default to surface=true and must produce meshes."""
        for section_name in hawk.polygons:
            mesh, lines = get_polygon_plotly(hawk, section_name, "blue", alpha=0.5)
            assert mesh is not None, f"{section_name} should produce a mesh"
            assert isinstance(mesh, go.Mesh3d)
            assert lines is not None, f"{section_name} must produce a line trace"

    def test_line_trace_closes_polygon(self, hawk):
        """Line traces should close the polygon by repeating the first
        vertex at the end."""
        _, lines = get_polygon_plotly(hawk, "tail", "blue", alpha=0.5)
        xs = list(lines.x)
        ys = list(lines.y)
        zs = list(lines.z)
        # First and last points should be identical (closed polygon)
        assert xs[0] == xs[-1]
        assert ys[0] == ys[-1]
        assert zs[0] == zs[-1]

    def test_line_trace_vertex_count_includes_closure(self, hawk):
        """Line traces have n+1 vertices (n polygon vertices + closure)."""
        n_tail_markers = len(hawk.polygons["tail"])
        _, lines = get_polygon_plotly(hawk, "tail", "blue", alpha=0.5)
        assert len(lines.x) == n_tail_markers + 1


# ------------------------------------------------------------------
# 4. plot_partial_plotly with invalid section name or leg number
# ------------------------------------------------------------------

class TestPlotPartialInvalidInputs:
    """plot_partial_plotly should handle invalid inputs gracefully."""

    def test_invalid_section_name_produces_empty_figure(self, hawk):
        """An unrecognised section name should produce a figure with
        no section traces (sections are silently skipped when the name
        is not found in the polygons dict)."""
        fig = plot_partial_plotly(hawk, section_name="nonexistent_section", colour="blue")
        assert isinstance(fig, go.Figure)
        # No section traces added — figure should have 0 data traces
        assert len(fig.data) == 0

    def test_invalid_leg_number_on_spider(self, spider):
        """A leg number outside the valid range (e.g. 99) should produce
        a figure with no section traces."""
        fig = plot_partial_plotly(spider, leg_number=99, colour="blue")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0

    def test_leg_number_on_hawk_falls_through(self, hawk):
        """Hawks have no leg sections, so leg_number should be ignored
        and all sections are plotted instead."""
        fig = plot_partial_plotly(hawk, leg_number=1, colour="blue")
        assert isinstance(fig, go.Figure)
        # Should fall through to plotting all sections (no "leg" keys)
        assert len(fig.data) == 14  # 7 sections x 2 traces


# ------------------------------------------------------------------
# 5. Animate functions produce correct number of frames
# ------------------------------------------------------------------

class TestAnimationFrameCounts:
    """Animation functions must produce the correct number of frames."""

    def test_animate_plotly_frame_count(self, hawk):
        n_frames = 5
        keypoints = np.tile(hawk.markers, (n_frames, 1, 1))
        # Add small random variation per frame
        keypoints += np.random.randn(*keypoints.shape) * 0.001
        fig = animate_plotly(hawk, keypoints, colour="blue")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == n_frames

    def test_animate_plotly_single_frame(self, hawk):
        keypoints = hawk.markers.reshape(1, -1, 3).copy()
        fig = animate_plotly(hawk, keypoints, colour="blue")
        assert len(fig.frames) == 1

    def test_animate_plotly_partial_frame_count(self, spider):
        n_frames = 3
        keypoints = np.tile(spider.markers, (n_frames, 1, 1))
        keypoints += np.random.randn(*keypoints.shape) * 0.001
        fig = animate_plotly_partial(spider, keypoints, leg_number=2, colour="red")
        assert isinstance(fig, go.Figure)
        assert len(fig.frames) == n_frames

    def test_animate_plotly_has_slider(self, hawk):
        """Animated figures should have a slider for frame navigation."""
        keypoints = np.tile(hawk.markers, (3, 1, 1))
        fig = animate_plotly(hawk, keypoints, colour="blue")
        assert fig.layout.sliders is not None
        assert len(fig.layout.sliders) > 0

    def test_animate_plotly_has_play_button(self, hawk):
        """Animated figures should have play/pause buttons."""
        keypoints = np.tile(hawk.markers, (3, 1, 1))
        fig = animate_plotly(hawk, keypoints, colour="blue")
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) > 0


# ------------------------------------------------------------------
# 6. Axis limits are sensible for hawk-scale and spider-scale data
# ------------------------------------------------------------------

class TestAxisLimits:
    """Axis limits should be centred, symmetric, and proportional to
    the animal's size."""

    def test_hawk_axis_limits_are_symmetric(self, hawk):
        """Each axis range should be symmetric about its centre."""
        limits = calculate_axis_limits(hawk)
        for axis_range in limits:
            low, high = axis_range
            assert high > low, "Upper limit must exceed lower limit"
            span = high - low
            assert span > 0

    def test_hawk_axis_limits_contain_all_markers(self, hawk):
        """All marker coordinates must fall within the axis limits."""
        limits = calculate_axis_limits(hawk)
        coords = hawk.current_shape[0]  # (n_markers, 3)
        for dim in range(3):
            low, high = limits[dim]
            assert np.all(coords[:, dim] >= low), (
                f"Dimension {dim}: markers below lower limit"
            )
            assert np.all(coords[:, dim] <= high), (
                f"Dimension {dim}: markers above upper limit"
            )

    def test_spider_axis_limits_contain_all_markers(self, spider):
        """All spider marker coordinates must fall within the axis limits."""
        limits = calculate_axis_limits(spider)
        coords = spider.current_shape[0]
        for dim in range(3):
            low, high = limits[dim]
            assert np.all(coords[:, dim] >= low), (
                f"Dimension {dim}: markers below lower limit"
            )
            assert np.all(coords[:, dim] <= high), (
                f"Dimension {dim}: markers above upper limit"
            )

    def test_hawk_limits_are_reasonable_scale(self, hawk):
        """Hawk data is in metres, so axis range should be under 2m."""
        limits = calculate_axis_limits(hawk)
        for axis_range in limits:
            span = axis_range[1] - axis_range[0]
            assert span < 2.0, "Hawk axis span should be under 2 metres"
            assert span > 0.01, "Hawk axis span suspiciously small"

    def test_spider_limits_differ_from_hawk(self, hawk, spider):
        """Spider and hawk should have different axis scales (different
        sized animals)."""
        hawk_limits = calculate_axis_limits(hawk)
        spider_limits = calculate_axis_limits(spider)
        hawk_span = hawk_limits[0][1] - hawk_limits[0][0]
        spider_span = spider_limits[0][1] - spider_limits[0][0]
        assert hawk_span != pytest.approx(spider_span, rel=0.01), (
            "Hawk and spider should have different axis scales"
        )

    def test_plot_plotly_applies_axis_limits(self, hawk):
        """The returned figure should have axis ranges set."""
        fig = plot_plotly(hawk, colour="blue")
        scene = fig.layout.scene
        assert scene.xaxis.range is not None
        assert scene.yaxis.range is not None
        assert scene.zaxis.range is not None
        # Ranges should be lists of two values
        assert len(scene.xaxis.range) == 2
        assert scene.xaxis.range[1] > scene.xaxis.range[0]

    def test_plot_plotly_uses_cube_aspect(self, hawk):
        """Figures should use cube aspect mode for correct proportions."""
        fig = plot_plotly(hawk, colour="blue")
        assert fig.layout.scene.aspectmode == "cube"
