"""Plotly animation functions for Animal3D."""

from pathlib import Path

import imageio
import numpy as np
import plotly.graph_objs as go

from .animation_frame_helpers import check_transformation_frames, format_keypoint_frames
from .plotly_helpers import calculate_animation_limits, calculate_nice_tick_step
from .plotly_plots import plot_keypoints_plotly, plot_sections_plotly


def animate_plotly(animal3d_instance, keypoints_frames, alpha=0.3, colour=None,
                   horzDist_frames=None, bodypitch_frames=None, vertDist_frames=None,
                   bodyyaw_frames=None, bodyroll_frames=None, score_vals=None,
                   axes_visible=True, display_only_transform=False):
    """Create an animated 3D plot using Plotly.

    Parameters
    ----------
    display_only_transform : bool
        If ``False`` (default), body pitch/yaw/roll rotates all markers.
        If ``True``, only display-only (fixed) markers are rotated — use
        this for PCA shape-mode animations where analysis markers already
        encode the morphing shape.
    """

    # Format keypoints
    keypoints_frames = format_keypoint_frames(animal3d_instance, keypoints_frames)
    if keypoints_frames.shape[0] == 0:
        msg = "No frames to animate."
        raise ValueError(msg)

    num_frames = keypoints_frames.shape[0]

    # Check transformation frames
    horzDist_frames = check_transformation_frames(num_frames, horzDist_frames)
    vertDist_frames = check_transformation_frames(num_frames, vertDist_frames)
    bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)
    bodyyaw_frames = check_transformation_frames(num_frames, bodyyaw_frames)
    bodyroll_frames = check_transformation_frames(num_frames, bodyroll_frames)

    # Pre-compute fixed axis limits for ALL frames (fixes axis-jumping bug)
    fixed_range = calculate_animation_limits(animal3d_instance, keypoints_frames)

    # Create frames
    transform = (animal3d_instance.transform_display_only if display_only_transform
                 else animal3d_instance.transform_all)
    frames = []
    for frame in range(num_frames):
        animal3d_instance.reset_transformation()
        animal3d_instance.update_keypoints(keypoints_frames[frame])
        transform(
            bodypitch=bodypitch_frames[frame],
            horzDist=horzDist_frames[frame],
            vertDist=vertDist_frames[frame],
            bodyyaw=bodyyaw_frames[frame],
            bodyroll=bodyroll_frames[frame],
        )

        fig = go.Figure()
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha)
        fig = plot_keypoints_plotly(
            fig, animal3d_instance,
            colour="lightblue" if colour is None else colour, alpha=1,
        )
        fig = _apply_animation_layout(fig, fixed_range, axes_visible)

        frames.append(go.Frame(data=fig.data, layout=fig.layout, name=str(frame)))

    # Build initial figure
    initial_fig = go.Figure(data=frames[0].data)
    initial_fig.frames = frames
    initial_fig = _apply_animation_layout(initial_fig, fixed_range, axes_visible)

    if score_vals is not None:
        slider_vals = score_vals.round(2)
    else:
        slider_vals = range(num_frames)

    initial_fig.update_layout(
        updatemenus=[_create_play_button()],
        sliders=[_create_slider(num_frames, slider_vals)],
        width=800, height=700,
        margin={"l": 50, "r": 50, "t": 100, "b": 100},
        scene={
            "domain": {"x": [0, 1], "y": [0.1, 1]},
            "aspectmode": 'cube',
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}, "up": {"x": 0, "y": 0, "z": 1}},
        },
    )

    return initial_fig


def animate_plotly_compare(animal3d_instance, keypoints_frames_list, alpha=0.3,
                           colours=None, horzDist_frames_list=None,
                           bodypitch_frames_list=None, vertDist_frames_list=None,
                           bodyyaw_frames_list=None, bodyroll_frames_list=None,
                           score_vals=None, axes_visible=True,
                           display_only_transform=False):
    """Create an animated 3D comparison plot."""
    if colours is None:
        colours = [None, 'red']

    # Format keypoints
    formatted_list = []
    for kf in keypoints_frames_list:
        formatted = format_keypoint_frames(animal3d_instance, kf)
        if formatted.shape[0] == 0:
            msg = "No frames in one of the keypoint sets."
            raise ValueError(msg)
        formatted_list.append(formatted)

    num_frames = formatted_list[0].shape[0]
    if not all(kp.shape[0] == num_frames for kp in formatted_list):
        msg = "All keypoint sets must have the same number of frames."
        raise ValueError(msg)

    # Check transforms
    if horzDist_frames_list:
        horzDist_frames_list = [check_transformation_frames(num_frames, f) for f in horzDist_frames_list]
    if vertDist_frames_list:
        vertDist_frames_list = [check_transformation_frames(num_frames, f) for f in vertDist_frames_list]
    if bodypitch_frames_list:
        bodypitch_frames_list = [check_transformation_frames(num_frames, f) for f in bodypitch_frames_list]
    if bodyyaw_frames_list:
        bodyyaw_frames_list = [check_transformation_frames(num_frames, f) for f in bodyyaw_frames_list]
    if bodyroll_frames_list:
        bodyroll_frames_list = [check_transformation_frames(num_frames, f) for f in bodyroll_frames_list]

    # Pre-compute limits across all datasets
    all_kf = np.concatenate(formatted_list, axis=0)
    fixed_range = calculate_animation_limits(animal3d_instance, all_kf)

    # Create frames
    transform = (animal3d_instance.transform_display_only if display_only_transform
                 else animal3d_instance.transform_all)
    frames = []
    for frame in range(num_frames):
        fig = go.Figure()
        for idx, keypoints in enumerate(formatted_list):
            animal3d_instance.reset_transformation()
            animal3d_instance.update_keypoints(keypoints[frame])

            horz = horzDist_frames_list[idx][frame] if horzDist_frames_list else 0
            vert = vertDist_frames_list[idx][frame] if vertDist_frames_list else 0
            pitch = bodypitch_frames_list[idx][frame] if bodypitch_frames_list else 0
            yaw = bodyyaw_frames_list[idx][frame] if bodyyaw_frames_list else 0
            roll = bodyroll_frames_list[idx][frame] if bodyroll_frames_list else 0

            transform(
                bodypitch=pitch, horzDist=horz, vertDist=vert,
                bodyyaw=yaw, bodyroll=roll,
            )

            fig = plot_sections_plotly(fig, animal3d_instance, colour=colours[idx], alpha=alpha)
            fig = plot_keypoints_plotly(fig, animal3d_instance, colour=colours[idx], alpha=1)

        fig = _apply_animation_layout(fig, fixed_range, axes_visible)
        frames.append(go.Frame(data=fig.data, layout=fig.layout, name=str(frame)))

    initial_fig = go.Figure(data=frames[0].data)
    initial_fig.frames = frames
    initial_fig = _apply_animation_layout(initial_fig, fixed_range, axes_visible)

    if score_vals is not None:
        slider_vals = score_vals.round(2)
    else:
        slider_vals = range(num_frames)

    initial_fig.update_layout(
        updatemenus=[_create_play_button()],
        sliders=[_create_slider(num_frames, slider_vals)],
        width=800, height=700,
        margin={"l": 50, "r": 50, "t": 100, "b": 100},
        scene={
            "domain": {"x": [0, 1], "y": [0.1, 1]},
            "aspectmode": 'cube',
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}, "up": {"x": 0, "y": 0, "z": 1}},
        },
    )

    return initial_fig


def animate_plotly_partial(animal3d_instance, keypoints_frames, section_name=None,
                           leg_number=None, colour='blue', alpha=1,
                           horzDist_frames=None, vertDist_frames=None,
                           bodypitch_frames=None, bodyyaw_frames=None,
                           bodyroll_frames=None, score_vals=None, axes_visible=True,
                           display_only_transform=False):
    """Animate a specific section or leg."""
    keypoints_frames = format_keypoint_frames(animal3d_instance, keypoints_frames)
    num_frames = keypoints_frames.shape[0]

    horzDist_frames = check_transformation_frames(num_frames, horzDist_frames)
    vertDist_frames = check_transformation_frames(num_frames, vertDist_frames)
    bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)
    bodyyaw_frames = check_transformation_frames(num_frames, bodyyaw_frames)
    bodyroll_frames = check_transformation_frames(num_frames, bodyroll_frames)

    fixed_range = calculate_animation_limits(animal3d_instance, keypoints_frames)

    transform = (animal3d_instance.transform_display_only if display_only_transform
                 else animal3d_instance.transform_all)
    frames = []
    for frame_idx in range(num_frames):
        animal3d_instance.reset_transformation()
        animal3d_instance.update_keypoints(keypoints_frames[frame_idx])
        transform(
            bodypitch=bodypitch_frames[frame_idx],
            horzDist=horzDist_frames[frame_idx],
            vertDist=vertDist_frames[frame_idx],
            bodyyaw=bodyyaw_frames[frame_idx],
            bodyroll=bodyroll_frames[frame_idx],
        )

        fig = go.Figure()

        if leg_number is not None and any("leg" in key for key in animal3d_instance.polygons):
            sec = f"leg_{leg_number}"
            fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, section_name=sec)
        elif section_name is not None:
            fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, section_name=section_name)
        else:
            fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha)

        fig = _apply_animation_layout(fig, fixed_range, axes_visible)
        frames.append(go.Frame(data=fig.data, name=str(frame_idx)))

    initial_fig = go.Figure(data=frames[0].data)
    initial_fig.frames = frames
    initial_fig = _apply_animation_layout(initial_fig, fixed_range, axes_visible)

    if score_vals is not None:
        slider_vals = score_vals.round(2)
    else:
        slider_vals = range(num_frames)

    initial_fig.update_layout(
        updatemenus=[_create_play_button()],
        sliders=[_create_slider(num_frames, slider_vals)],
        width=800, height=700,
        margin={"l": 50, "r": 50, "t": 100, "b": 100},
        scene={
            "domain": {"x": [0, 1], "y": [0.1, 1]},
            "aspectmode": 'cube',
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.5}, "up": {"x": 0, "y": 0, "z": 1}},
        },
    )

    return initial_fig


def save_plotly_animation(fig, filename, format='gif', fps=10, width=800, height=700):
    """Save a Plotly animation as GIF or HTML."""
    if format.lower() == 'gif':
        temp_dir = Path("temp_images")
        temp_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for frame in fig.frames:
            fig.update(data=frame.data)
            image_path = temp_dir / f"frame_{frame.name}.png"
            fig.write_image(str(image_path), width=width, height=height)
            images.append(image_path)

        with imageio.get_writer(filename, mode='I', fps=fps) as writer:
            for image in images:
                writer.append_data(imageio.imread(image))

        for image in images:
            image.unlink()
        temp_dir.rmdir()

    elif format.lower() == 'html':
        fig.write_html(filename, auto_play=False, include_plotlyjs=True)
    else:
        msg = "Format must be either 'gif' or 'html'."
        raise ValueError(msg)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _apply_animation_layout(fig, fixed_range, axes_visible=True):
    """Apply consistent layout with pre-computed axis limits."""
    x_span = fixed_range[0][1] - fixed_range[0][0]
    nice_step = calculate_nice_tick_step(x_span)

    axes_config = {
        "gridcolor": "grey",
        "zerolinecolor": "grey",
        "showbackground": True,
        "backgroundcolor": "white",
        "gridwidth": 0.5,
        "dtick": nice_step,
        "tick0": 0,
    }

    if not axes_visible:
        axes_config.update(
            visible=False, showgrid=False, showticklabels=False,
            showline=False, showbackground=False,
        )

    fig.update_layout(
        scene={
            "xaxis": dict(range=fixed_range[0], **axes_config),
            "yaxis": dict(range=fixed_range[1], **axes_config),
            "zaxis": dict(range=fixed_range[2], **axes_config),
            "aspectmode": 'cube',
            "aspectratio": {"x": 1, "y": 1, "z": 1},
        },
        margin={"r": 10, "l": 10, "b": 10, "t": 10},
        showlegend=False,
    )
    return fig


def _create_slider(num_frames, slider_vals):
    """Create a Plotly animation slider."""
    if slider_vals is None:
        slider_vals = range(num_frames)

    return {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 12},
            'prefix': 'Frame:',
            'visible': True,
            'xanchor': 'right',
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {
                'args': [[ii], {'frame': {'duration': 300, 'redraw': True},
                                'mode': 'immediate',
                                'transition': {'duration': 300}}],
                'label': str(slider_vals[ii]),
                'method': 'animate',
            }
            for ii in range(num_frames)
        ],
    }


def _create_play_button():
    """Create Plotly play/pause buttons."""
    return {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Play',
                'method': 'animate',
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Pause',
                'method': 'animate',
            },
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 10},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'left',
        'y': 1.1,
        'yanchor': 'top',
    }
