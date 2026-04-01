"""Plotly static plotting functions for Animal3D."""

import numpy as np
import plotly.graph_objs as go

from .plotly_helpers import calculate_axis_limits, get_section_style, is_surface_section


def plot_plotly(animal3d_instance, colour='lightblue', alpha=0.5, axes_visible=True,
                horzDist=None, vertDist=None, bodypitch=None, bodyroll=None, bodyyaw=None,
                show_display_markers=False):
    """Create a static 3D plot of an animal using Plotly.

    Parameters
    ----------
    animal3d_instance : Animal3D
        Instance of the Animal3D class.
    colour : str
        Colour for all sections. Body/head sections get lower alpha automatically.
    alpha : float
        Transparency for analysis sections (body sections get lower alpha from config).
    axes_visible : bool
        Whether to show axes, grid, and tick labels.
    show_display_markers : bool
        Whether to show display-only marker dots. Default ``False``.
    """
    has_transform = any(p is not None for p in [bodypitch, bodyyaw, bodyroll, horzDist, vertDist])

    if has_transform:
        current_state = animal3d_instance.current_shape.copy()
        animal3d_instance.reset_transformation()
        animal3d_instance.transform_display_only(
            bodypitch=bodypitch or 0,
            horzDist=horzDist or 0,
            vertDist=vertDist or 0,
            bodyyaw=bodyyaw or 0,
            bodyroll=bodyroll or 0,
        )

    fig = go.Figure()
    fig = plot_sections_plotly(fig, animal3d_instance, colour, alpha)
    fig = plot_keypoints_plotly(fig, animal3d_instance, colour, alpha,
                                show_display_markers=show_display_markers)

    if not axes_visible:
        fig.update_layout(scene={
            'xaxis': {'visible': False, 'showgrid': False, 'showticklabels': False,
                       'showline': False, 'showbackground': False},
            'yaxis': {'visible': False, 'showgrid': False, 'showticklabels': False,
                       'showline': False, 'showbackground': False},
            'zaxis': {'visible': False, 'showgrid': False, 'showticklabels': False,
                       'showline': False, 'showbackground': False},
        })

    fig = plot_settings_plotly(fig, animal3d_instance)

    if has_transform:
        animal3d_instance.current_shape = current_state

    return fig


def plot_plotly_compare(animal3d_instances, colours=None, alpha=0.5, axes_visible=True,
                        horzDist=None, vertDist=None, bodypitch=None, bodyroll=None,
                        bodyyaw=None, show_display_markers=False):
    """Create a static 3D comparison plot of multiple animals."""
    fig = go.Figure()

    if colours is None:
        colours = ['red', None]

    has_transform = any(p is not None for p in [bodypitch, bodyyaw, bodyroll, horzDist, vertDist])

    for animal3d_instance, colour in zip(animal3d_instances, colours, strict=True):
        if has_transform:
            current_state = animal3d_instance.current_shape.copy()
            animal3d_instance.reset_transformation()
            animal3d_instance.transform_display_only(
                bodypitch=bodypitch or 0,
                horzDist=horzDist or 0,
                vertDist=vertDist or 0,
                bodyyaw=bodyyaw or 0,
                bodyroll=bodyroll or 0,
            )

        fig = plot_sections_plotly(fig, animal3d_instance, colour, alpha)

        if not axes_visible:
            fig.update_layout(scene={
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'zaxis': {'visible': False},
            })

        fig = plot_settings_plotly(fig, animal3d_instance)

        if has_transform:
            animal3d_instance.current_shape = current_state

    return fig


def plot_compare_plotly(animal3d_instance, keypoints_list, alpha=0.5, colours=None,
                        horzDist=None, bodypitch=None, vertDist=None, bodyyaw=None,
                        bodyroll=None, axes_visible=True, show_display_markers=False,
):
    """Create a static 3D comparison of multiple poses."""
    if colours is None:
        colours = [None, 'red']

    has_transform = any(p is not None for p in [bodypitch, bodyyaw, bodyroll, horzDist, vertDist])
    current_state = animal3d_instance.current_shape.copy()
    fig = go.Figure()

    for idx, keypoints in enumerate(keypoints_list):
        animal3d_instance.update_keypoints(keypoints)
        if has_transform:
            animal3d_instance.reset_transformation()
            animal3d_instance.transform_display_only(
                bodypitch=bodypitch or 0,
                horzDist=horzDist or 0,
                vertDist=vertDist or 0,
                bodyyaw=bodyyaw or 0,
                bodyroll=bodyroll or 0,
            )

        fig = plot_sections_plotly(fig, animal3d_instance, colour=colours[idx], alpha=alpha)
        fig = plot_keypoints_plotly(fig, animal3d_instance, colour=colours[idx], alpha=1,
                                    show_display_markers=show_display_markers)

    if not axes_visible:
        fig.update_layout(scene={
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'zaxis': {'visible': False},
        })

    fig = plot_settings_plotly(fig, animal3d_instance)
    animal3d_instance.current_shape = current_state

    return fig


def plot_partial_plotly(animal3d_instance, section_name=None, leg_number=None,
                        colour='blue', alpha=1, axes_visible=True,
                        horzDist=None, vertDist=None, bodypitch=None,
                        bodyroll=None, bodyyaw=None, show_display_markers=False,
):
    """Plot a specific section or leg of the animal."""
    has_transform = any(p is not None for p in [bodypitch, bodyyaw, bodyroll, horzDist, vertDist])

    if has_transform:
        current_state = animal3d_instance.current_shape.copy()
        animal3d_instance.reset_transformation()
        animal3d_instance.transform_display_only(
            bodypitch=bodypitch or 0,
            horzDist=horzDist or 0,
            vertDist=vertDist or 0,
            bodyyaw=bodyyaw or 0,
            bodyroll=bodyroll or 0,
        )

    fig = go.Figure()

    if leg_number is not None and any("leg" in key for key in animal3d_instance.polygons):
        section_name = f"leg_{leg_number}"
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, section_name=section_name)
    elif section_name is not None:
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, section_name=section_name)
    else:
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha)

    if not axes_visible:
        fig.update_layout(scene={
            'xaxis': {'visible': False},
            'yaxis': {'visible': False},
            'zaxis': {'visible': False},
        })

    fig = plot_settings_plotly(fig, animal3d_instance)

    if has_transform:
        animal3d_instance.current_shape = current_state

    return fig


def plot_plotly_with_trace(animal3d_instance, keypoints_frames, colour='lightblue',
                           alpha=0.5, trace_colour='red', trace_marker_size=2,
                           axes_visible=True, horzDist=None, vertDist=None,
                           bodypitch=None, bodyroll=None, bodyyaw=None,
                           show_display_markers=False):
    """Create a static 3D plot with a trace of keypoint frames."""
    fig = plot_plotly(animal3d_instance, colour=colour, alpha=alpha,
                      axes_visible=axes_visible, horzDist=horzDist,
                      vertDist=vertDist, bodypitch=bodypitch,
                      bodyroll=bodyroll, bodyyaw=bodyyaw,
                      show_display_markers=show_display_markers)

    trace_coords = keypoints_frames.reshape(-1, 3)
    n_points = trace_coords.shape[0]

    scatter_trace = go.Scatter3d(
        x=trace_coords[:, 0],
        y=trace_coords[:, 1],
        z=trace_coords[:, 2],
        mode='markers',
        marker={
            'size': trace_marker_size,
            'color': np.linspace(0, 1, n_points),
            'colorscale': 'plasma_r',
            'showscale': False,
        },
        hoverinfo='none',
    )
    fig.add_trace(scatter_trace)

    return fig


# ------------------------------------------------------------
#       Helper functions
# ------------------------------------------------------------

def plot_keypoints_plotly(fig, animal3d_instance, colour='black', alpha=1,
                          indices=None, show_display_markers=False):
    """Plot keypoints as a scatter on a Plotly figure.

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

    scatter = go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker={'size': 2.5, 'color': colour, 'opacity': alpha},
        hoverinfo='none',
    )
    fig.add_trace(scatter)
    return fig


def plot_sections_plotly(fig, animal3d_instance, colour, alpha=1, section_name=None):
    """Plot body section polygons on a Plotly figure."""
    if section_name is not None:
        if section_name in animal3d_instance.polygons:
            mesh, lines = get_polygon_plotly(animal3d_instance, section_name, colour, alpha)
            if mesh is not None:
                fig.add_trace(mesh)
            if lines is not None:
                fig.add_trace(lines)
    else:
        for section in animal3d_instance.polygons:
            mesh, lines = get_polygon_plotly(animal3d_instance, section, colour, alpha)
            if mesh is not None:
                fig.add_trace(mesh)
            if lines is not None:
                fig.add_trace(lines)
    return fig


def get_polygon_plotly(animal3d_instance, section_name, colour, alpha=1):
    """Build Plotly mesh and line traces for a polygon section."""
    if section_name not in animal3d_instance.polygons:
        return None, None

    resolved_colour, resolved_alpha = get_section_style(
        section_name, colour, alpha, animal3d_instance
    )

    coords = animal3d_instance.get_polygon_coords(section_name)

    if is_surface_section(section_name, animal3d_instance):
        mesh = go.Mesh3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            color=resolved_colour, opacity=resolved_alpha, hoverinfo='none',
        )
    else:
        mesh = None

    coords_closed = np.vstack([coords, coords[0]])
    lines = go.Scatter3d(
        x=coords_closed[:, 0],
        y=coords_closed[:, 1],
        z=coords_closed[:, 2],
        mode='lines',
        name=f'{section_name} {resolved_colour}',
        line={'color': 'grey', 'width': 1.5},
        hoverinfo='name',
    )

    return mesh, lines


def plot_settings_plotly(fig, animal3d_instance):
    """Apply standard layout settings to a static Plotly figure."""
    fixed_range = calculate_axis_limits(animal3d_instance)

    axes_config = {
        'gridcolor': "grey",
        'zerolinecolor': "grey",
        'showbackground': True,
        'backgroundcolor': "white",
        'gridwidth': 0.5,
        'dtick': fixed_range[0][1] / 2,
    }

    fig.update_layout(
        font={'family': "Andale Mono, Courier New, sans-serif"},
        scene={
            'xaxis': dict(range=fixed_range[0], **axes_config),
            'yaxis': dict(range=fixed_range[1], **axes_config),
            'zaxis': dict(range=fixed_range[2], **axes_config),
            'aspectmode': 'cube',
            'aspectratio': {'x': 1, 'y': 1, 'z': 1},
        },
        margin={'r': 10, 'l': 10, 'b': 10, 't': 10},
        showlegend=False,
    )
    return fig
