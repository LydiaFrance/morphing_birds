"""Plotly animation functions for Animal3D."""

from pathlib import Path
from typing import Callable

import imageio
import numpy as np
import plotly.graph_objs as go

from .animation_frame_helpers import check_transformation_frames, format_keypoint_frames
from .plotly_helpers import (
    calculate_animation_limits,
    calculate_animation_limits_multi,
    calculate_nice_tick_step,
)
from .plotly_plots import plot_keypoints_plotly, plot_sections_plotly


def animate_mode(
    pc_number: int,
    hawk3d,
    principal_components: np.ndarray,
    scores: np.ndarray,
    mu: np.ndarray,
    colour_list: list,
    num_frames: int = 25,
    unilateral: bool = True,
    sign_fn: Callable | None = None,
) -> go.Figure:
    """Animate a single PCA morphing mode as a 3D hawk shape.

    Sweeps the selected PC score between ±2 standard deviations while
    holding all other scores at zero, then reconstructs and animates the
    resulting shape sequence.

    Parameters
    ----------
    pc_number : int
        1-based PC index to animate.
    hawk3d : Animal3D
        Hawk 3-D model used for rendering.
    principal_components : numpy.ndarray
        Component matrix, shape ``(n_components, n_features)``.
    scores : numpy.ndarray
        Score matrix, shape ``(n_frames, n_components)``, used to
        determine the sweep range (±2 std).
    mu : numpy.ndarray
        Mean shape, shape ``(1, n_markers, 3)``.
    colour_list : list of str
        Hex colour per PC (0-indexed). Must have at least ``pc_number``
        entries.
    num_frames : int, optional
        Number of animation frames (default 25).
    unilateral : bool, optional
        If True (default), mirror reconstructed frames from unilateral to
        bilateral using :func:`~morphing_birds.bilateral.mirror_to_bilateral`.
        Set False when the input data are already bilateral.
    sign_fn : callable, optional
        Function ``(score_frames) -> score_frames`` applied to the sweep
        scores before they are used as slider labels. Useful for applying
        project-specific sign conventions to display values without affecting
        reconstruction. If None (default), raw scores are used as labels.

    Returns
    -------
    plotly.graph_objects.Figure
        The animation figure (also calls ``fig.show()``).
    """
    from morphing_birds.bilateral import mirror_to_bilateral

    # --- Score sweep (triangle wave, ±2 std) ---
    n_components = scores.shape[1]
    s_mean = np.mean(scores, axis=0)
    s_std = np.std(scores, axis=0)
    s_min = s_mean - 2 * s_std
    s_max = s_mean + 2 * s_std

    half = num_frames // 2 + 1
    t = np.linspace(0, 1, half)
    t = np.concatenate([t, t[-2:0:-1]])
    score_frames = s_min + (s_max - s_min) * t[:, np.newaxis]

    # Zero out all components except the one being animated
    sweep = np.zeros_like(score_frames)
    sweep[:, pc_number - 1] = score_frames[:, pc_number - 1]

    # --- Reconstruct ---
    n_markers = mu.shape[1]
    n_dims = mu.shape[2]
    recon = np.dot(sweep, principal_components).reshape(-1, n_markers, n_dims) + mu

    if unilateral:
        recon = mirror_to_bilateral(recon)

    # --- Slider labels ---
    slider_scores = sign_fn(score_frames)[:, pc_number - 1] if sign_fn else score_frames[:, pc_number - 1]

    fig = animate_plotly(
        hawk3d,
        recon,
        colour=colour_list[pc_number - 1],
        score_vals=slider_scores,
    )
    fig.show()
    return fig


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


def animate_plotly_compare(animal3d_instances, keypoints_frames_list, alpha=0.3,
                           colours=None, horzDist_frames_list=None,
                           bodypitch_frames_list=None, vertDist_frames_list=None,
                           bodyyaw_frames_list=None, bodyroll_frames_list=None,
                           score_vals=None, axes_visible=True,
                           display_only_transform=False):
    """Create an animated 3D comparison plot.

    Flexible input handling:
    - Single instance + list of keypoints: overlay multiple animations on same skeleton
    - List of instances + single keypoints: same animation on different skeletons
    - List of instances + list of keypoints: each instance gets its own animation
    """
    if colours is None:
        colours = [None, 'red']

    animal3d_instances, keypoints_frames_list, colours = _normalise_compare_inputs(
        animal3d_instances, keypoints_frames_list, colours
    )

    # Format keypoints for each instance
    formatted_list = []
    for animal, kf in zip(animal3d_instances, keypoints_frames_list, strict=True):
        formatted = format_keypoint_frames(animal, kf)
        if formatted.shape[0] == 0:
            msg = "No frames in one of the keypoint sets."
            raise ValueError(msg)
        formatted_list.append(formatted)

    num_frames = formatted_list[0].shape[0]
    if not all(kp.shape[0] == num_frames for kp in formatted_list):
        msg = "All keypoint sets must have the same number of frames."
        raise ValueError(msg)

    transforms = _check_transform_lists(
        num_frames,
        horzDist=horzDist_frames_list,
        vertDist=vertDist_frames_list,
        bodypitch=bodypitch_frames_list,
        bodyyaw=bodyyaw_frames_list,
        bodyroll=bodyroll_frames_list,
    )

    # Pre-compute limits across all instances and keyframes
    fixed_range = calculate_animation_limits_multi(animal3d_instances, formatted_list)

    # Create frames
    frames = []
    for frame in range(num_frames):
        fig = go.Figure()
        for idx, (animal, keypoints) in enumerate(zip(animal3d_instances, formatted_list, strict=True)):
            transform = (animal.transform_display_only if display_only_transform
                         else animal.transform_all)
            animal.reset_transformation()
            animal.update_keypoints(keypoints[frame])

            transform(**_get_frame_transforms(transforms, idx, frame))

            fig = plot_sections_plotly(fig, animal, colour=colours[idx], alpha=alpha)
            fig = plot_keypoints_plotly(fig, animal, colour=colours[idx], alpha=1)

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

def _normalise_compare_inputs(animal3d_instances, keypoints_frames_list, colours):
    """Normalise and broadcast inputs for comparison animation."""
    if hasattr(animal3d_instances, 'current_shape'):
        animal3d_instances = [animal3d_instances]
    if isinstance(keypoints_frames_list, np.ndarray):
        keypoints_frames_list = [keypoints_frames_list]

    n_inst, n_kf = len(animal3d_instances), len(keypoints_frames_list)
    if n_inst == 1:
        animal3d_instances = animal3d_instances * n_kf
    elif n_kf == 1:
        keypoints_frames_list = keypoints_frames_list * n_inst
    elif n_inst != n_kf:
        msg = "Number of instances and keypoint sets must match (or one must be broadcastable)."
        raise ValueError(msg)

    n_items = len(animal3d_instances)
    if len(colours) < n_items:
        colours = colours * ((n_items // len(colours)) + 1)

    return animal3d_instances, keypoints_frames_list, colours


def _check_transform_lists(num_frames, **transform_lists):
    """Validate all transform frame lists."""
    return {
        k: [check_transformation_frames(num_frames, f) for f in v] if v else None
        for k, v in transform_lists.items()
    }


def _get_frame_transforms(transforms, idx, frame):
    """Extract transform values for a specific instance and frame."""
    return {k: v[idx][frame] if v else 0 for k, v in transforms.items()}


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
