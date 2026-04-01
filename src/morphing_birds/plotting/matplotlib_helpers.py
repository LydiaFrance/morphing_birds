"""Matplotlib helper functions for plotting Animal3D."""

from __future__ import annotations

import io
import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from .plotly_helpers import get_section_style

# ....... Helper Plot Functions ........

def plot_keypoints(ax, animal3d_instance, colour='k', alpha: float = 1.0, indices=None,
                   show_display_markers=False):
    """Plot keypoints of the animal using matplotlib.

    Parameters
    ----------
    show_display_markers : bool
        If ``False`` (default), only analysis markers are plotted.
        If ``True``, all markers (including display-only) are plotted.
    """
    if indices is not None:
        plot_indices = indices
    elif show_display_markers:
        plot_indices = list(range(animal3d_instance.skeleton.n_markers))
    else:
        plot_indices = animal3d_instance.analysis_indices

    coords = animal3d_instance.current_shape[:, plot_indices, :][0]

    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        s=30, color=colour, alpha=alpha,
        edgecolor='none', antialiased=True,
    )
    return ax


def plot_sections(ax, animal3d_instance, colour, alpha: float = 1.0, section_name=None):
    """Plot body section polygons using matplotlib."""
    if section_name is not None:
        if section_name in animal3d_instance.polygons:
            polygon = get_polygon(animal3d_instance, section_name, colour, alpha)
            ax.add_collection3d(polygon)
    else:
        for section in animal3d_instance.polygons:
            polygon = get_polygon(animal3d_instance, section, colour, alpha)
            ax.add_collection3d(polygon)
    return ax


def get_polygon(animal3d_instance, section_name, colour, alpha: float = 1.0):
    """Build a Poly3DCollection for a body section."""
    if section_name not in animal3d_instance.polygons:
        msg = f"Section name {section_name} not recognised."
        raise ValueError(msg)

    # Resolve colour and alpha using the unified style system
    resolved_colour, resolved_alpha = get_section_style(
        section_name, colour, alpha, animal3d_instance
    )

    coords = animal3d_instance.get_polygon_coords(section_name)
    coords = np.vstack([coords, coords[0]])

    return Poly3DCollection(
        [coords],
        alpha=resolved_alpha,
        facecolor=resolved_colour,
        edgecolor='black',
        linewidth=1.0,
        antialiased=True,
        rasterized=False,
    )


def plot_settings(ax, origin, lims=None):
    """Configure axis appearance for a 3D matplotlib plot."""
    # Panel Shading
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Grid style
    grid_props = {
        'color': 'grey',
        'linestyle': ':',
        'linewidth': 0.3,
        'alpha': 0.5,
    }
    ax.xaxis._axinfo['grid'].update(**grid_props)
    ax.yaxis._axinfo['grid'].update(**grid_props)
    ax.zaxis._axinfo['grid'].update(**grid_props)

    # Set axis limits
    if lims is not None:
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[0], lims[1])
        ax.set_zlim(lims[0], lims[1])
        increment = round(lims[1] - lims[0], 2)
    else:
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)
        increment = 0.2

    # Set axis scale
    ax.auto_scale_xyz(
        [origin[0] - increment, origin[0] + increment],
        [origin[1] - increment, origin[1] + increment],
        [origin[2] - increment, origin[2] + increment],
    )

    # Labels and ticks
    label_props = {'fontsize': 10, 'fontweight': 'normal', 'fontfamily': 'sans-serif'}
    ax.set_xlabel('x (m)', **label_props)
    ax.set_ylabel('y (m)', **label_props)
    ax.set_zlabel('z (m)', **label_props)

    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=3)
    ax.tick_params(axis='both', which='minor', labelsize=8, width=0.5, length=2)

    n_ticks = 5
    ax.set_xticks(np.linspace(-increment, increment, n_ticks))
    ax.set_yticks(np.linspace(-increment, increment, n_ticks))
    ax.set_zticks(np.linspace(-increment, increment, n_ticks))

    ax.set_aspect('equal', 'box')
    ax.set_facecolor('white')
    ax.figure.set_dpi(150)

    return ax


def get_plot3d_view(fig=None, rows=1, cols=1, index=1):
    """Create a 3D matplotlib axis."""
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(rows, cols, index, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    return fig, ax


def save_plot_as_image(fig, cut_off=0.2):
    """Save a matplotlib figure to a cropped PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=500, bbox_inches='tight')
    buf.seek(0)

    img = Image.open(buf)

    width, height = img.size
    left = width * cut_off
    right = width * (1 - cut_off)
    cropped_img = img.crop((left, 0, right, height))

    buf.close()
    plt.close(fig)

    return cropped_img


def camera_from_plotly(fig) -> dict:
    """Extract camera position from a Plotly figure and convert to matplotlib angles.

    Reads ``layout.scene.camera.eye`` from the given Plotly figure and
    converts the (x, y, z) eye coordinates to matplotlib elevation and
    azimuth angles.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        A Plotly figure with a 3D scene.

    Returns
    -------
    dict
        Dictionary with ``'elev'`` and ``'azim'`` keys (values in degrees).
    """
    eye = fig.layout.scene.camera.eye
    # Plotly uses (1.25, 1.25, 1.25) as the default camera eye when
    # the user hasn't explicitly set or rotated the view.
    x = float(eye.x) if eye.x is not None else 1.25
    y = float(eye.y) if eye.y is not None else 1.25
    z = float(eye.z) if eye.z is not None else 1.25

    r = math.sqrt(x * x + y * y + z * z)
    elev = math.degrees(math.asin(z / r))
    azim = math.degrees(math.atan2(y, x))

    return {"elev": elev, "azim": azim}


def get_camera_angles(num_frames, rotation_type, el=20, az=60):
    """Create camera angle arrays for animation."""
    if el is None or az is None:
        return [None, None]

    def linspacer(number_array, first_value, end_value, n_frames):
        return np.append(number_array, np.linspace(first_value, end_value, n_frames))

    if "dynamic" in rotation_type:
        tenth = round(num_frames * 0.1)
        remainder = num_frames - (tenth * 9)

        az_frames = np.linspace(40, 40, tenth)
        az_frames = linspacer(az_frames, 40, 10, tenth)
        az_frames = linspacer(az_frames, 10, 10, tenth)
        az_frames = linspacer(az_frames, 10, 90, tenth)
        az_frames = linspacer(az_frames, 90, 90, tenth * 2)
        az_frames = linspacer(az_frames, 90, -90, tenth)
        az_frames = linspacer(az_frames, -90, -90, tenth)
        az_frames = linspacer(az_frames, -90, 40, tenth)
        az_frames = linspacer(az_frames, 40, 40, remainder)

        el_frames = np.linspace(20, 20, tenth)
        el_frames = linspacer(el_frames, 20, 15, tenth)
        el_frames = linspacer(el_frames, 15, 0, tenth)
        el_frames = linspacer(el_frames, 0, 0, tenth)
        el_frames = linspacer(el_frames, 0, 80, tenth)
        el_frames = linspacer(el_frames, 80, 80, tenth)
        el_frames = linspacer(el_frames, 80, 15, tenth)
        el_frames = linspacer(el_frames, 15, 15, tenth)
        el_frames = linspacer(el_frames, 15, 20, tenth)
        el_frames = linspacer(el_frames, 20, 20, remainder)

    elif "slow" in rotation_type:
        half = round(num_frames * 0.5)
        tenth = round(num_frames * 0.1)
        remainder = num_frames - (half + (tenth * 2) + tenth + tenth)

        az_frames = np.linspace(90, 90, half)
        az_frames = linspacer(az_frames, 90, -90, tenth)
        az_frames = linspacer(az_frames, -90, -90, tenth * 2)
        az_frames = linspacer(az_frames, -90, 90, tenth)
        az_frames = linspacer(az_frames, 90, 90, remainder)

        remainder2 = num_frames - (tenth * 9)
        el_frames = np.linspace(80, 80, tenth * 2)
        el_frames = linspacer(el_frames, 80, 20, tenth)
        el_frames = linspacer(el_frames, 20, 20, tenth * 2)
        el_frames = linspacer(el_frames, 20, 10, tenth)
        el_frames = linspacer(el_frames, 10, 10, tenth * 3)
        el_frames = linspacer(el_frames, 10, 80, remainder2)
    else:
        el_frames = np.linspace(el, el, num_frames)
        az_frames = np.linspace(az, az, num_frames)

    return el_frames, az_frames
