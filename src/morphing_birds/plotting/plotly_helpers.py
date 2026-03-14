"""Plotly helper functions — axis limits and section styling."""

import numpy as np


def calculate_axis_limits(animal3d_instance):
    """Calculate axis limits for a single-frame plot.

    Uses bounding box + padding. No power-of-2 quantisation.
    """
    default_shape = animal3d_instance.default_shape.copy()
    current_shape = animal3d_instance.current_shape.copy()

    all_coords = np.concatenate([default_shape, current_shape], axis=0)
    overall_min = all_coords.min(axis=(0, 1))
    overall_max = all_coords.max(axis=(0, 1))

    ranges = overall_max - overall_min
    natural_size = np.max(ranges)

    # Simple padding — 40% beyond the natural size, no power-of-2
    view_radius = natural_size * 0.7

    min_coords, max_coords = animal3d_instance.get_bounding_box()
    centres = (min_coords + max_coords) / 2

    origin = np.array(animal3d_instance.origin)
    centres[0] = 0 if origin[0] == 0 else centres[0]
    centres[2] = 0 if origin[2] == 0 else centres[2]

    return [
        [centres[0] - view_radius, centres[0] + view_radius],
        [centres[1] - view_radius, centres[1] + view_radius],
        [centres[2] - view_radius, centres[2] + view_radius],
    ]


def calculate_animation_limits(animal3d_instance, keypoints_frames, transform_frames=None):
    """Pre-compute axis limits that encompass ALL frames of an animation.

    Called ONCE before the frame loop, not per-frame.  This fixes the
    axis-jumping bug caused by per-frame power-of-2 quantisation.

    Parameters
    ----------
    animal3d_instance : Animal3D
        The animal instance (used for display-only markers).
    keypoints_frames : np.ndarray
        All frames of motion data, shape ``(n_frames, n_markers, 3)``.
    transform_frames : dict, optional
        Per-frame transform parameters (not currently used for limit calc).

    Returns
    -------
    list
        ``[[x_min, x_max], [y_min, y_max], [z_min, z_max]]``
    """
    # Include both keypoint data and display-only marker positions
    all_min = keypoints_frames.min(axis=(0, 1))
    all_max = keypoints_frames.max(axis=(0, 1))

    # Also include display-only marker positions
    display_coords = animal3d_instance.default_shape[
        0, animal3d_instance.display_only_indices, :
    ]
    if len(display_coords) > 0:
        display_min = display_coords.min(axis=0)
        display_max = display_coords.max(axis=0)
        all_min = np.minimum(all_min, display_min)
        all_max = np.maximum(all_max, display_max)

    ranges = all_max - all_min
    max_range = np.max(ranges)

    # Add 20% padding
    view_radius = max_range * 0.7

    centres = (all_min + all_max) / 2
    # Centre x and z on 0 if appropriate
    centres[0] = 0
    centres[2] = 0

    return [
        [centres[0] - view_radius, centres[0] + view_radius],
        [centres[1] - view_radius, centres[1] + view_radius],
        [centres[2] - view_radius, centres[2] + view_radius],
    ]


def get_section_style(section_name, caller_colour, caller_alpha, animal3d_instance=None):
    """Resolve the final (colour, alpha) for a polygon section.

    Default behaviour: every section gets the caller's colour.  Body, head,
    and other "structural" sections get a lower alpha to appear more
    transparent.  This gives a uniform-colour animal with the body fading
    into the background.

    Parameters
    ----------
    section_name : str
        Name of the body section.
    caller_colour : str or None
        Colour passed by the calling function.
    caller_alpha : float
        Alpha passed by the calling function.
    animal3d_instance : Animal3D, optional
        Animal instance (used to read ``section_styles``).

    Returns
    -------
    tuple[str, float]
        ``(colour, alpha)``
    """
    colour = caller_colour or "lightblue"

    # Check for explicit section_styles in the config
    if animal3d_instance is not None and hasattr(animal3d_instance, "skeleton"):
        styles = animal3d_instance.skeleton.section_styles
        if styles:
            # Start with default style
            default_style = styles.get("default", {})
            default_alpha = default_style.get("alpha", 0.2)

            # Check keyword matches
            for keyword, style in styles.items():
                if keyword == "default":
                    continue
                if keyword in section_name:
                    # Matched a keyword — use its alpha, keep caller's colour
                    alpha = style.get("alpha", caller_alpha)
                    # Only override colour if explicitly set in the style
                    if "colour" in style:
                        colour = style["colour"]
                    return colour, alpha

            # No keyword match — this is a structural section (body, head, etc.)
            # Use the caller's colour but at the default (lower) alpha
            return colour, default_alpha

    # No config — simple fallback
    # Structural sections get lower alpha
    structural_keywords = ["body", "head"]
    for kw in structural_keywords:
        if kw in section_name:
            return colour, 0.2
    return colour, caller_alpha
