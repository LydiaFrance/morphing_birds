import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from PIL import Image
import io


# ....... Helper Plot Functions ........

def plot_keypoints(ax, animal3d_instance, colour='k', alpha=1, indices=None):
    """
    Plots keypoints of the animal using matplotlib.
    
    Parameters:
    - ax : matplotlib.axes.Axes
        The axes to plot on
    - animal3d_instance : Animal3D
        Instance of the Animal3D class
    - colour : str, optional
        Color for the keypoints
    - alpha : float, optional
        Transparency of the keypoints
    - indices : list, optional
        Specific indices of keypoints to plot. If None, plot all keypoints
    """
    # Use provided indices or default to marker_index
    if indices is None:
        indices = animal3d_instance.marker_index

    # Extract keypoint coordinates
    coords = animal3d_instance.current_shape[:, indices, :][0]

    # Plot the keypoints with improved appearance
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              s=30,  # Slightly larger markers
              color=colour,
              alpha=alpha,
              edgecolor='none',  # Remove marker edges for cleaner look
              antialiased=True)  # Enable antialiasing

    return ax

def plot_sections(ax, animal3d_instance, colour, alpha=1, section_name=None):
    """
    Plots sections of the animal using matplotlib.
    
    Parameters:
    - ax : matplotlib.axes.Axes
        The axes to plot on
    - animal3d_instance : Animal3D
        Instance of the Animal3D class
    - colour : str
        Color for the plot
    - alpha : float, optional
        Transparency of the plot
    - section_name : str, optional
        The name of the body section to plot. If None, plot all sections
    """
    if section_name is not None:
        # Skip if section doesn't exist in polygons
        if section_name in animal3d_instance.polygons:
            polygon = get_polygon(animal3d_instance, section_name, colour, alpha)
            ax.add_collection3d(polygon)
    else:
        # Plot all available sections
        for section in animal3d_instance.polygons.keys():
            polygon = get_polygon(animal3d_instance, section, colour, alpha)
            ax.add_collection3d(polygon)

    return ax

def get_polygon(animal3d_instance, section_name, colour, alpha=1):
    """
    Returns the coordinates of the polygon representing the given section.
    
    Parameters:
    - animal3d_instance : Animal3D
        Instance of the Animal3D class
    - section_name : str
        Name of the section to get polygon for
    - colour : str
        Color for the polygon
    - alpha : float, optional
        Transparency of the polygon
    """
    if section_name not in animal3d_instance.polygons:
        raise ValueError(f"Section name {section_name} not recognised.")

    # Get color and alpha based on section type
    colour = colour_polygon(section_name, colour, animal3d_instance)
    alpha = alpha_polygon(section_name, alpha, animal3d_instance)

    # Get polygon coordinates
    coords = animal3d_instance.get_polygon_coords(section_name)
    
    # Add first point to end to close polygon
    coords = np.vstack([coords, coords[0]])

    # Create polygon collection with improved line rendering
    polygon = Poly3DCollection([coords],
                             alpha=alpha,
                             facecolor=colour,
                             edgecolor='black',  # Darker edge color for better contrast
                             linewidth=1.0,  # Thinner lines for crispness
                             antialiased=True,  # Enable antialiasing
                             rasterized=False)  # Vector rendering for crisp lines
    return polygon

def alpha_polygon(section_name, alpha, animal3d_instance=None):
    """
    Determines the transparency of a polygon section.
    """
    if animal3d_instance is None or not hasattr(animal3d_instance, 'colour_sections'):
        if "handwing" in section_name or "tail" in section_name:
            alpha = 0.5
        else:
            alpha = 0.3
    else:
        # Check if any of the colored sections are in the section name
        for keyword in animal3d_instance.colour_sections:
            if keyword in section_name:
                alpha = 0.5
            else:
                alpha = 0.3
    return alpha

def colour_polygon(section_name, colour, animal3d_instance=None):
    """
    Determines the color of a polygon section.
    """
    if colour is None:
        colour = 'lightblue'

    if animal3d_instance is None or not hasattr(animal3d_instance, 'colour_sections'):
        if 'handwing' in section_name or 'tail' in section_name:
            return colour
        else:
            return 'grey'
    
    # Check if any of the colored sections are in the section name
    for keyword in animal3d_instance.colour_sections:
        if keyword in section_name:  # This will match "wing" within "right_handwing", etc.
            return colour
    # If no coloured sections are found, return grey
    return 'grey'

def plot_settings(ax, origin, lims=None):
    """
    Plot settings & set the azimuth and elev. for camera view of 3D axis.
    
    Parameters:
    - ax : matplotlib.axes.Axes
        The axes to configure
    - origin : array-like
        Origin point for the plot
    - lims : tuple, optional
        Axis limits as (min, max). If None, uses default limits
    """
    # Panel Shading
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Grid color and style - make lines thinner and crisper
    grid_props = {
        'color': 'grey',
        'linestyle': ':',
        'linewidth': 0.3,  # Thinner grid lines
        'alpha': 0.5  # Slightly transparent
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
    ax.auto_scale_xyz([origin[0]-increment, origin[0]+increment],
                     [origin[1]-increment, origin[1]+increment],
                     [origin[2]-increment, origin[2]+increment])

    # Axis labels and ticks with improved font rendering
    label_props = {'fontsize': 10, 'fontweight': 'normal', 'fontfamily': 'sans-serif'}
    ax.set_xlabel('x (m)', **label_props)
    ax.set_ylabel('y (m)', **label_props)
    ax.set_zlabel('z (m)', **label_props)
    
    # Configure tick properties
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=3)
    ax.tick_params(axis='both', which='minor', labelsize=8, width=0.5, length=2)

    # Set tick locations
    n_ticks = 5  # Match plotly's default
    ax.set_xticks(np.linspace(-increment, increment, n_ticks))
    ax.set_yticks(np.linspace(-increment, increment, n_ticks))
    ax.set_zticks(np.linspace(-increment, increment, n_ticks))

    # Set aspect ratio to be equal
    ax.set_aspect('equal', 'box')

    # Set background color to match plotly
    ax.set_facecolor('white')
    
    # Enable antialiasing for the entire axis
    ax.figure.set_dpi(150)  # Higher DPI for sharper rendering
    
    return ax

    
def get_plot3d_view(fig=None, rows=1, cols=1, index=1):
        """
        From HumanPose by Kevin Schegel

        Convenience function to create 3d matplotlib axis object.
        Wraps figure creation if need be and add_subplot.
        Parameters
        ----------
        fig : matplotlib.figure object, optional
            For re-use of an existing figure object. A new one is created if
            not given.
        rows : int
            Number of subplot rows. Like fig.add_subplot
        cols : int
            Number of subplot cols. Like fig.add_subplot
        index : int
            Index of subplot to use. Like fig.add_subplot
        Returns
        -------
        (fig, ax)
            fig : matplotlib.figure object
            ax : matplotlib.Axes object
        """
        if fig is None:
            fig = plt.figure(figsize=(6,6))

        
        ax = fig.add_subplot(rows, cols, index, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        return fig, ax

def save_plot_as_image(fig, cut_off=0.2):
     
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=500, bbox_inches='tight')
    buf.seek(0)  # Rewind the buffer

    # Load the image from the buffer using PIL
    img = Image.open(buf)


    # Crop the image: Define the left, top, right, and bottom pixel coordinates
    width, height = img.size
    left = width * cut_off
    right = width * (1-cut_off)
    top = 0
    bottom = height
    cropped_img = img.crop((left, top, right, bottom))

    # Close the buffer and the plot
    buf.close()
    plt.close(fig)

    return cropped_img




def get_camera_angles(num_frames, rotation_type, el=20, az=60):
        """
        Creates two arrays of camera angles for the length of the animation.

        "static" -- the angles do not change in the animation, set using el, az.

        "dynamic" -- fast rotations

        "slow" -- slow rotations

        Used in: Creates inputs for plot_hawk3D_frame
        """

        if el is None or az is None:
            return [None, None]

        def linspacer(number_array, firstValue, endValue, nFrames):
            """
            Function to make linearly spaced numbers between two values of a
            given length.
            """

            number_array = np.append(
                number_array, np.linspace(firstValue, endValue, nFrames))
            return number_array

        if "dynamic" in rotation_type:

            tenthFrames = round(num_frames * 0.1)
            remainderFrames = num_frames - (tenthFrames * 9)

            az_frames = np.linspace(40, 40, tenthFrames)
            az_frames = linspacer(az_frames, 40, 10, tenthFrames)
            az_frames = linspacer(az_frames, 10, 10, tenthFrames)
            az_frames = linspacer(az_frames, 10, 90, tenthFrames)
            az_frames = linspacer(az_frames, 90, 90, tenthFrames * 2)
            az_frames = linspacer(az_frames, 90, -90, tenthFrames)
            az_frames = linspacer(az_frames, -90, -90, tenthFrames)
            az_frames = linspacer(az_frames, -90, 40, tenthFrames)
            az_frames = linspacer(az_frames, 40, 40, remainderFrames)

            el_frames = np.linspace(20, 20, tenthFrames)
            el_frames = linspacer(el_frames, 20, 15, tenthFrames)
            el_frames = linspacer(el_frames, 15, 0, tenthFrames)
            el_frames = linspacer(el_frames, 0, 0, tenthFrames)
            el_frames = linspacer(el_frames, 0, 80, tenthFrames)
            el_frames = linspacer(el_frames, 80, 80, tenthFrames)
            el_frames = linspacer(el_frames, 80, 15, tenthFrames)
            el_frames = linspacer(el_frames, 15, 15, tenthFrames)
            el_frames = linspacer(el_frames, 15, 20, tenthFrames)
            el_frames = linspacer(el_frames, 20, 20, remainderFrames)

        elif "slow" in rotation_type:

            halfFrames = round(num_frames * 0.5)
            tenthFrames = round(num_frames * 0.1)
            remainderFrames = num_frames - (halfFrames + (tenthFrames * 2) +
                                            tenthFrames + tenthFrames)

            az_frames = np.linspace(90, 90, halfFrames)
            az_frames = linspacer(az_frames, 90, -90,
                                  tenthFrames)  # Switch to back
            az_frames = linspacer(az_frames, -90, -90, tenthFrames * 2)
            az_frames = linspacer(az_frames, -90, 90,
                                  tenthFrames)  # Switch to front
            az_frames = linspacer(az_frames, 90, 90, remainderFrames)

            remainderFrames = num_frames - (tenthFrames * 9)

            el_frames = np.linspace(80, 80, tenthFrames * 2)
            el_frames = linspacer(el_frames, 80, 20,
                                  tenthFrames)  # Transition to lower
            el_frames = linspacer(el_frames, 20, 20, tenthFrames * 2)
            el_frames = linspacer(el_frames, 20, 10,
                                  tenthFrames)  # Switch to back
            el_frames = linspacer(el_frames, 10, 10, tenthFrames * 3)
            el_frames = linspacer(el_frames, 10, 80,
                                  remainderFrames)  # Switch to front

        else:
            el_frames = np.linspace(el, el, num_frames)
            az_frames = np.linspace(az, az, num_frames)

        return el_frames, az_frames


