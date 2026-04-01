"""Matplotlib animation functions for Animal3D."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .animation_frame_helpers import check_transformation_frames, format_keypoint_frames
from .matplotlib_helpers import get_camera_angles, get_plot3d_view, plot_settings
from .matplotlib_plots import plot


def animate(animal3d_instance, keypoints_frames, fig=None, ax=None,
            rotation_type="static", el=20, az=60, alpha=0.3, colour=None,
            horzDist_frames=None, bodypitch_frames=None, vertDist_frames=None,
            bodyyaw_frames=None, bodyroll_frames=None, score_vals=None,
            axes_visible=True):
    """Create an animated 3D plot using matplotlib."""
    keypoints_frames = format_keypoint_frames(animal3d_instance, keypoints_frames)
    if keypoints_frames.shape[0] == 0:
        msg = "No frames to animate."
        raise ValueError(msg)

    num_frames = keypoints_frames.shape[0]

    if ax is None or fig is None:
        fig, ax = get_plot3d_view(fig)

    el_frames, az_frames = get_camera_angles(
        num_frames=num_frames, rotation_type=rotation_type, el=el, az=az,
    )

    horzDist_frames = check_transformation_frames(num_frames, horzDist_frames)
    vertDist_frames = check_transformation_frames(num_frames, vertDist_frames)
    bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)
    bodyyaw_frames = check_transformation_frames(num_frames, bodyyaw_frames)
    bodyroll_frames = check_transformation_frames(num_frames, bodyroll_frames)

    lims = keypoints_frames.max() * 0.5
    lims = [-lims, lims]
    ax = plot_settings(ax, animal3d_instance.origin, lims)

    def update_animated_plot(frame):
        ax.clear()
        animal3d_instance.reset_transformation()
        animal3d_instance.update_keypoints(keypoints_frames[frame])
        animal3d_instance.transform_display_only(
            bodypitch=bodypitch_frames[frame],
            horzDist=horzDist_frames[frame],
            vertDist=vertDist_frames[frame],
            bodyyaw=bodyyaw_frames[frame],
            bodyroll=bodyroll_frames[frame],
        )
        plot(animal3d_instance, ax=ax, el=el_frames[frame], az=az_frames[frame],
             alpha=alpha, colour=colour, axes_visible=axes_visible)
        plot_settings(ax, animal3d_instance.origin, lims)
        return fig, ax

    animal3d_instance.restore_default()

    return FuncAnimation(
        fig, update_animated_plot, frames=num_frames, interval=20, repeat=True,
    )


def animate_compare(animal3d_instance, keypoints_frames_list, fig=None, ax=None,
                    rotation_type="static", el=20, az=60, alpha=0.3, colour=None,
                    horzDist_frames=None, bodypitch_frames=None, vertDist_frames=None,
                    axes_visible=True):
    """Create an animated 3D comparison plot using matplotlib."""
    formatted_list = []
    for kf in keypoints_frames_list:
        formatted = format_keypoint_frames(animal3d_instance, kf)
        if formatted.shape[0] == 0:
            msg = "No frames to animate."
            raise ValueError(msg)
        formatted_list.append(formatted)

    num_frames = formatted_list[0].shape[0]

    if ax is None or fig is None:
        fig, ax = get_plot3d_view(fig)

    el_frames, az_frames = get_camera_angles(
        num_frames=num_frames, rotation_type=rotation_type, el=el, az=az,
    )

    horzDist_frames = check_transformation_frames(num_frames, horzDist_frames)
    vertDist_frames = check_transformation_frames(num_frames, vertDist_frames)
    bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)

    ax = plot_settings(ax, animal3d_instance.origin)

    def update_animated_plot(frame):
        ax.clear()
        for ii, kf in enumerate(formatted_list):
            c = plt.cm.Set1(ii)
            animal3d_instance.reset_transformation()
            animal3d_instance.update_keypoints(kf[frame])
            animal3d_instance.transform_display_only(
                bodypitch=bodypitch_frames[frame],
                horzDist=horzDist_frames[frame],
                vertDist=vertDist_frames[frame],
            )
            plot(animal3d_instance, ax=ax, el=el_frames[frame], az=az_frames[frame],
                 alpha=alpha, colour=c, axes_visible=axes_visible)
        plot_settings(ax, animal3d_instance.origin)
        return fig, ax

    animal3d_instance.restore_default()

    return FuncAnimation(
        fig, update_animated_plot, frames=num_frames, interval=20, repeat=True,
    )
