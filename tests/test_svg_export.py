"""Tests for SVG export and camera conversion."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import pytest

from morphing_birds import Animal3D, camera_from_plotly, export_svg


@pytest.fixture
def hawk():
    return Animal3D("hawk", data="data/mean_hawk_shape.csv")


@pytest.fixture
def plotly_fig_with_camera():
    """Create a Plotly figure with a known camera position."""
    fig = go.Figure()
    fig.update_layout(
        scene=dict(
            camera=dict(eye=dict(x=1.0, y=1.0, z=1.0))
        )
    )
    return fig


# ------------------------------------------------------------------
# camera_from_plotly
# ------------------------------------------------------------------


class TestCameraFromPlotly:
    """camera_from_plotly must correctly convert Plotly eye to mpl angles."""

    def test_known_angles(self, plotly_fig_with_camera):
        """eye=(1,1,1) should give elev=arcsin(1/sqrt(3)), azim=45."""
        result = camera_from_plotly(plotly_fig_with_camera)

        r = math.sqrt(3)
        expected_elev = math.degrees(math.asin(1.0 / r))
        expected_azim = 45.0

        assert abs(result["elev"] - expected_elev) < 1e-6
        assert abs(result["azim"] - expected_azim) < 1e-6

    def test_top_down_view(self):
        """eye=(0,0,2) should give elev=90, azim=0."""
        fig = go.Figure()
        fig.update_layout(scene=dict(camera=dict(eye=dict(x=0, y=0, z=2))))
        result = camera_from_plotly(fig)

        assert abs(result["elev"] - 90.0) < 1e-6
        assert abs(result["azim"] - 0.0) < 1e-6

    def test_side_view(self):
        """eye=(2,0,0) should give elev=0, azim=0."""
        fig = go.Figure()
        fig.update_layout(scene=dict(camera=dict(eye=dict(x=2, y=0, z=0))))
        result = camera_from_plotly(fig)

        assert abs(result["elev"] - 0.0) < 1e-6
        assert abs(result["azim"] - 0.0) < 1e-6

    def test_returns_dict_with_correct_keys(self, plotly_fig_with_camera):
        result = camera_from_plotly(plotly_fig_with_camera)
        assert "elev" in result
        assert "azim" in result


# ------------------------------------------------------------------
# export_svg
# ------------------------------------------------------------------


class TestExportSvg:
    """export_svg must produce a valid SVG file."""

    def test_creates_svg_file(self, hawk, tmp_path):
        output = tmp_path / "test.svg"
        export_svg(hawk, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_svg_contains_svg_tag(self, hawk, tmp_path):
        output = tmp_path / "test.svg"
        export_svg(hawk, output)
        content = output.read_text()
        assert "<svg" in content

    def test_with_plotly_camera(self, hawk, plotly_fig_with_camera, tmp_path):
        output = tmp_path / "camera_test.svg"
        export_svg(hawk, output, camera_from=plotly_fig_with_camera)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_axes_visible_true(self, hawk, tmp_path):
        output = tmp_path / "axes_on.svg"
        export_svg(hawk, output, axes_visible=True)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_axes_visible_false(self, hawk, tmp_path):
        output = tmp_path / "axes_off.svg"
        export_svg(hawk, output, axes_visible=False)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_custom_colour(self, hawk, tmp_path):
        output = tmp_path / "colour_test.svg"
        export_svg(hawk, output, colour="steelblue")
        assert output.exists()
        assert output.stat().st_size > 0

    def test_does_not_mutate_shape(self, hawk, tmp_path):
        """export_svg must not change the animal's current_shape."""
        import numpy as np

        before = hawk.current_shape.copy()
        export_svg(hawk, tmp_path / "test.svg")
        np.testing.assert_array_almost_equal(hawk.current_shape, before)
