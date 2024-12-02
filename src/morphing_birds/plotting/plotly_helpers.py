import numpy as np




def calculate_axis_limits(animal3d_instance):
    """
    Calculate axis limits based on the animal's natural size and bounding box.
    """

    
    # Get the natural size of the animal from its default shape
    default_min, default_max = animal3d_instance.default_shape.min(axis=(0, 1)), animal3d_instance.default_shape.max(axis=(0, 1))
    natural_size = np.max(default_max - default_min) * 1.25
    # Round to nearest order of magnitude and multiply by 0.4 for typical viewing
    base_scale = 10 ** np.floor(np.log10(natural_size))
    view_radius = 0.5 * (natural_size*1.3)

    # Round to the nearest base_scale
    view_radius = np.round(view_radius / base_scale) * base_scale

    # Calculate current bounding box
    min_coords, max_coords = animal3d_instance.get_bounding_box()
    ranges = max_coords - min_coords
    
    # Calculate buffer for each axis
    buffers = np.maximum(ranges * 0.1, view_radius)
    
    # Get centers based on origin
    centers = (min_coords + max_coords) / 2
    origin = np.array(animal3d_instance.origin)
    
    # Use 0 as center for x and z only if their origin is 0
    centers[0] = 0 if origin[0] == 0 else centers[0]  # x-axis
    centers[2] = 0 if origin[2] == 0 else centers[2]  # z-axis

    # Set ranges with consistent scale for all axes
    fixed_range = [
        [centers[0] - buffers[0], centers[0] + buffers[0]],  # x-axis
        [centers[1] - buffers[1], centers[1] + buffers[1]],  # y-axis
        [centers[2] - buffers[2], centers[2] + buffers[2]]   # z-axis
    ]

    return fixed_range
