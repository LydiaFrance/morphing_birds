"""Matplotlib static plotting functions for Animal3D."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from .matplotlib_helpers import (
    camera_from_plotly,
    get_plot3d_view,
    plot_keypoints,
    plot_sections,
    plot_settings,
    save_plot_as_image,
)


def plot(animal3d_instance, ax=None, el=20, az=60, colour=None, alpha=0.5,
         axes_visible=True, show_display_markers=False):
    """Create a static 3D matplotlib plot of an animal.

    Parameters
    ----------
    axes_visible : bool
        Whether to show axes. When ``False``, calls ``ax.set_axis_off()``.
    show_display_markers : bool
        Whether to show display-only marker dots. Default ``False``.
    """

    if ax is None:
        _fig, ax = get_plot3d_view()

    ax = plot_sections(ax, animal3d_instance, colour, alpha)
    ax = plot_keypoints(ax, animal3d_instance, colour, alpha,
                        show_display_markers=show_display_markers)
    ax.view_init(elev=el, azim=az)

    if axes_visible:
        origin = animal3d_instance.origin
        ax = plot_settings(ax, origin)
    else:
        ax.set_axis_off()

    return ax


def interactive_plot(animal3d_instance, keypoints=None, fig=None, ax=None,
                     el=20, az=60, colour=None, alpha=0.3):
    """Create an interactive 3D plot with sliders."""
    if keypoints is not None:
        animal3d_instance.update_keypoints(keypoints)
        animal3d_instance.reset_transformation()

    plt.ioff()
    if ax is None:
        fig, ax = get_plot3d_view()
    plt.ion()

    az_slider = widgets.IntSlider(min=-90, max=90, step=5, value=az, description='azimuth')
    el_slider = widgets.IntSlider(min=-15, max=90, step=5, value=el, description='elevation')
    plot_output = widgets.Output()

    with plot_output:
        plot(animal3d_instance, ax=ax, el=el_slider.value, az=az_slider.value,
             colour=colour, alpha=alpha)

    def update_plot(change):
        with plot_output:
            clear_output(wait=True)
            ax.view_init(elev=el_slider.value, azim=az_slider.value)
            fig.canvas.draw_idle()
            display(fig)

    az_slider.observe(update_plot, names='value')
    el_slider.observe(update_plot, names='value')

    display(az_slider, el_slider)
    display(plot_output)
    update_plot(None)


def plot_multiple(animal3d_instance, keypoints, num_plots, spacing=(0.4, 0.7),
                  cut_off=0.2, el=20, az=0, rot=90, colour_list=None, alpha=0.5):
    """Plot multiple frames of the animal in a grid."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    grid_cols = int(np.ceil(np.sqrt(num_plots)))
    grid_rows = int(np.ceil(num_plots / grid_cols))
    middle_row = (grid_rows - 1) / 2
    middle_col = (grid_cols - 1) / 2

    for i in range(num_plots):
        animal3d_instance.restore_default()
        animal3d_instance.update_keypoints(keypoints[i])
        animal3d_instance.reset_transformation()

        row = grid_rows - 1 - (i // grid_cols)
        col = i % grid_cols

        if colour_list is None:
            colour = matplotlib.colormaps["Set3"](i)
        else:
            colour = colour_list[i]

        vertDist = (row - middle_row) * spacing[0]
        horzDist = (col - middle_col) * spacing[1]

        animal3d_instance.transform_display_only(vertDist=vertDist, horzDist=horzDist, bodyyaw=rot)
        plot(animal3d_instance, ax=ax, el=el, az=az, colour=colour, alpha=alpha, axes_visible=False)

    max_vert = (num_plots * 0.15) * spacing[0]
    max_horz = (num_plots * 0.15) * spacing[1]
    ax.set_ylim(-max_horz, max_horz)
    ax.set_zlim(-max_vert, max_vert)
    ax.set_xlim(-0.5, 0.5)

    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    return save_plot_as_image(fig, cut_off)


def export_svg(
    animal3d,
    output_path,
    camera_from=None,
    colour="cornflowerblue",
    axes_visible=False,
):
    """Export the current animal shape as a vector SVG file.

    Uses matplotlib's SVG backend to produce true vector output, unlike
    Plotly's WebGL-based rendering which rasterises 3D scenes.

    Parameters
    ----------
    animal3d : Animal3D
        Animal instance with ``current_shape`` set.
    output_path : str or Path
        Where to save the SVG file.
    camera_from : plotly.graph_objects.Figure, optional
        If provided, extract camera angle from this Plotly figure.
        If ``None``, use matplotlib's default view.
    colour : str
        Colour for the body sections.
    axes_visible : bool
        Whether to show axes in the SVG. ``False`` gives clean paper
        figures.
    """
    output_path = Path(output_path)

    fig, ax = get_plot3d_view()

    ax = plot_sections(ax, animal3d, colour, alpha=0.5)
    ax = plot_keypoints(ax, animal3d, colour, alpha=1.0)

    if camera_from is not None:
        angles = camera_from_plotly(camera_from)
        ax.view_init(elev=angles["elev"], azim=angles["azim"])

    if axes_visible:
        origin = animal3d.origin
        ax = plot_settings(ax, origin)
    else:
        ax.set_axis_off()

    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"SVG saved to {output_path}")
