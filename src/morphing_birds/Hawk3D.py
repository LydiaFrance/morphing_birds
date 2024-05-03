import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ipywidgets as widgets
from IPython.display import display, clear_output
from matplotlib.animation import FuncAnimation
from PIL import Image
import io


class Hawk3D:
    # Class attributes
    right_marker_names = [
        "right_wingtip", 
        "right_primary", 
        "right_secondary",
        "right_tailtip"]
    
    left_marker_names = [
        "left_wingtip", 
        "left_primary", 
        "left_secondary",
        "left_tailtip"]
    
    marker_names = [
        "left_wingtip",   "right_wingtip", 
        "left_primary",   "right_primary", 
        "left_secondary", "right_secondary", 
        "left_tailtip",   "right_tailtip"]
    
    fixed_marker_names = [
        "left_shoulder", 
        "left_tailbase", 
        "right_tailbase", 
        "right_shoulder",
        "hood", 
        "tailpack"
    ]
    
    body_sections = {
        "left_handwing": [
            "left_wingtip", 
            "left_primary", 
            "left_secondary",
        ],
        "right_handwing": [
            "right_wingtip", 
            "right_primary", 
            "right_secondary",
        ],
        "left_armwing": [
            "left_primary", 
            "left_secondary", 
            "left_tailbase", 
            "left_shoulder",
        ],
        "right_armwing": [
            "right_primary", 
            "right_secondary", 
            "right_tailbase",
            "right_shoulder",
        ],
        "body": [
            "right_shoulder", 
            "left_shoulder", 
            "left_tailbase", 
            "right_tailbase",
        ],
        "head": [
            "right_shoulder", 
            "hood", 
            "left_shoulder",
        ],
        "tail": [
            "right_tailtip", 
            "left_tailtip", 
            "left_tailbase", 
            "right_tailbase",
        ],
    }

    def __init__(self, csv_path):
        """
        Initialise the Hawk3D class.
        Loads default keypoint shape from external csv file. 
        Initialises polygon shapes drawn from the keypoints. 
        """
        # Create default shape from loaded csv file. 
        self.load_and_initialise_keypoints(csv_path)

        self.init_polygons()

    def load_and_initialise_keypoints(self, csv_path):
        data = self.load_csv_data(csv_path)
        self.csv_marker_names = self.get_csv_marker_names(data)
        keypoints = self.get_csv_keypoints(data)

        # Define the indices of the markers
        self.define_indices()

        # The default shape is loaded from the csv file
        self.default_shape = self.validate_keypoints(keypoints)

        # The user may update the keypoints. For now, the current shape
        # is the default shape
        self.current_shape = np.copy(self.default_shape)
        
        # For now, set up the transformation matrix as the identity matrix
        # This will be updated when the user applies transformations.
        self.transformation_matrix = np.eye(4)
        self.origin = np.array([0.0,0.0,0.0])

        # Store an untransformed copy useful for resetting.
        self.untransformed_shape = np.copy(self.current_shape)

    def load_csv_data(self, csv_path):

        # load the data
        with open(csv_path, 'r') as file:
            return np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')

    def get_csv_keypoints(self,data):
        # Load marker coordinates and reshape to [n,3] matrix where n is the
        # number of markers
        keypoints = data[1].astype(float)
        keypoints = keypoints.reshape(-1, 3) # [n,3]

        # Save the default shape as keypoints. 
        return keypoints

    def get_csv_marker_names(self,data):
            """
            Get the marker names from the first row of the csv file, 
            get every 3rd name and remove the '_x' from the names
            """
            csv_marker_names = data[0].reshape(-1, 3)
            csv_marker_names = list(np.char.strip(csv_marker_names[:, 0], '_x'))

            return csv_marker_names

    def define_indices(self):
        self.right_marker_index = self.get_keypoint_indices(self.right_marker_names)
        self.left_marker_index  = self.get_keypoint_indices(self.left_marker_names)
        self.marker_index       = self.get_keypoint_indices(self.marker_names)
        self.fixed_marker_index = self.get_keypoint_indices(self.fixed_marker_names)

    def get_keypoint_indices(self,names_to_find=None):
        """
        Returns the indices of the keypoints with the given names.
        """
        csv_marker_names = self.csv_marker_names

        # If no names are given, use all the marker names
        if names_to_find is None:
            names_to_find = csv_marker_names

        indices = [csv_marker_names.index(name) for name in names_to_find]

        return indices

    def init_polygons(self):
        """
        Initialise the polygons for plotting. 
        Gets the indices of the keypoints for each section.
        """
        self.polygons = {}
        for name, marker_names in self.body_sections.items():
            self.polygons[name] = self.get_keypoint_indices(marker_names)

    def get_polygon_coords(self, section_name):

        if section_name not in self.body_sections.keys():
            raise ValueError(f"Section name {section_name} not recognised.")

        indices = self.polygons[section_name]
        coords = self.current_shape[0, indices, :]

        return coords

    def validate_keypoints(self, keypoints):
        """
        Validates the keypoints, ensuring they are three-dimensional and reshapes/mirrors them if necessary.

        Parameters:
        - keypoints (numpy.ndarray): The keypoints array to validate.

        Returns:
        - numpy.ndarray: The validated and potentially reshaped and mirrored keypoints.
        """

        if keypoints.size == 0:
            raise ValueError("No keypoints provided.")
        
        # Check they are in 3D
        if keypoints.shape[-1] != 3:
            raise ValueError("Keypoints must be in 3D space.")
        
        # If [4,3] or [8,3] is given, reshape to [1,4,3] or [1,8,3]
        if len(np.shape(keypoints)) == 2:
            keypoints = keypoints.reshape(1, -1, 3)

        # If [1,4,3] is given, mirror it to make [1,8,3]
        if keypoints.shape[1] == len(self.right_marker_names):
            keypoints = self.mirror_keypoints(keypoints)

        return keypoints

    def mirror_keypoints(self,keypoints):
        """
        Mirrors keypoints across the y-axis to create a symmetrical set.

        Parameters:
        - keypoints (numpy.ndarray): Keypoints in shape [1, n, 3].

        Returns:
        - numpy.ndarray: Mirrored keypoints array in shape [1, 2*n, 3].
        """
        mirrored = np.copy(keypoints)
        mirrored[:, :, 0] *= -1

        nFrames, nMarkers, nCoords = np.shape(keypoints)

        # Create [n,8,3] array
        new_keypoints = np.empty((nFrames, nMarkers * 2, nCoords),
                                 dtype=keypoints.dtype)
        
        new_keypoints[:, 0::2, :] = mirrored
        new_keypoints[:, 1::2, :] = keypoints

        return new_keypoints

    def update_keypoints(self,user_keypoints):
        """
        Updates the keypoints based on user-provided data. Resets to default if no keypoints are provided.
        Validates and mirrors the keypoints if necessary, and applies transformations.

        Parameters:
        - user_keypoints (numpy.ndarray or None): Array of keypoints provided by the user. 
        If None, resets to the default keypoints setup.
        """

        # Make sure the current_shape starts as default to reset it
        if user_keypoints is None:
            # Reset to default shape if no user keypoints are provided
            self.restore_keypoints_to_average()
            return

        # First validate the keypoints. This will mirror them 
        # if only the right side is given. Also checks they are in 3D and 
        # will return [n,8,3]. 

        # Validate and possibly mirror the user keypoints -- returns [n,8,3] 
        # If only the right side is given, the left side is created by mirroring.
        # Also checks in 3D space. 
        validated_keypoints = self.validate_keypoints(user_keypoints)


        # Update the keypoints with the user marker info. 
        # Only non fixed markers are updated.
        self.current_shape[:,self.marker_index,:] = validated_keypoints      

        # Save the untransformed shape
        self.untransformed_shape = self.current_shape.copy()

        # Apply transformations
        self.apply_transformation()

    def transform_keypoints(self, bodypitch=0, horzDist=0, vertDist=0, yaw=0):
        """
        Transforms the keypoints by rotating them around the body pitch, 
        and translating them by the horizontal and vertical distances.
        """

        # Reset the transformation matrix
        self.reset_transformation()

        # Apply any translations
        self.update_translation(horzDist, vertDist)

        # Apply any rotations
        self.update_rotation(bodypitch)

        self.update_rotation(yaw, which='z')

        # Apply the transformation
        self.apply_transformation()

    def update_rotation(self, degrees=0, which='x'):
        """
        Updates the transformation matrix with a rotation around the x-axis.
        """
        radians = np.deg2rad(degrees)
        if which == "x":
            rotation_matrix = np.array([
                [1,0,0,0],
                [0, np.cos(radians), -np.sin(radians), 0],
                [0, np.sin(radians),  np.cos(radians), 0],
                [0,0,0,1]
            ])
        elif which == "y":
            rotation_matrix = np.array([
                [np.cos(radians), 0, np.sin(radians), 0],
                [0,1,0,0],
                [-np.sin(radians), 0, np.cos(radians), 0],
                [0,0,0,1]
            ])
        elif which == "z":
            rotation_matrix = np.array([
                [np.cos(radians), -np.sin(radians), 0, 0],
                [np.sin(radians),  np.cos(radians), 0, 0],
                [0,0,1,0],
                [0,0,0,1]
            ])

        # rotation_matrix = np.array([
        #     [1,0,0,0],
        #     [0, np.cos(radians), -np.sin(radians), 0],
        #     [0, np.sin(radians),  np.cos(radians), 0],
        #     [0,0,0,1]
        # ])

        self.transformation_matrix = self.transformation_matrix @ rotation_matrix



    def update_translation(self,horzDist=0, vertDist=0):
        """
        Updates the transformation matrix with horizontal and vertical translations.
        """
        translation_matrix = np.array([
            [1,0,0,0],
            [0,1,0,horzDist],
            [0,0,1,vertDist],
            [0,0,0,1]
        ])
        self.transformation_matrix = self.transformation_matrix @ translation_matrix

        # Update origin
        self.origin = [0, horzDist, vertDist]

    def apply_transformation(self):

        """
        Applies the current transformation matrix to the keypoints.
        """
        # Adding a homogeneous coordinate directly to the current_shape
        homogeneous_keypoints = np.hstack((self.current_shape.reshape(-1, 3), np.ones((self.current_shape.shape[1], 1))))
        
        transformed_keypoints = np.dot(homogeneous_keypoints, self.transformation_matrix.T)
        
        self.current_shape = transformed_keypoints[:, :3].reshape(1, -1, 3)

    def reset_transformation(self):
        self.transformation_matrix = np.eye(4)
        self.current_shape = self.untransformed_shape

        # Also reset the origin
        self.origin = np.array([0,0,0])

    def restore_keypoints_to_average(self):
        """
        Restores the keypoints and origin to the default shape.
        """
        self.current_shape = self.default_shape.copy()
        self.untransformed_shape = self.current_shape.copy()

        # Also update the origin
        self.origin = np.array([0,0,0])

    @property
    def markers(self):
        """
        Returns the non-fixed markers.
        """
        
        marker_index = self.marker_index
        
        return self.current_shape[:,marker_index,:]

    @property
    def right_markers(self):
        """
        Returns the right side markers.
        """
        
        right_marker_index = self.right_marker_index
        
        return self.current_shape[:,right_marker_index,:]
    
    @property
    def default_markers(self):
        """
        Returns the default markers.
        """
        
        marker_index = self.marker_index
        
        return self.default_shape[:,marker_index,:]

    @property
    def default_right_markers(self):
        """
        Returns the right side markers.
        """
        
        right_marker_index = self.right_marker_index
        
        return self.default_shape[:,right_marker_index,:]



     

# ----- Plot Functions -----
 
def interactive_plot(Hawk3D_instance, ax=None, el=20, az=60, colour=None, alpha=0.3):


        plt.ioff()  # Turn off interactive mode
        
        if ax is None:
            fig, ax = get_plot3d_view()
            
        plt.ion()  # Turn on interactive mode
        
        az_slider = widgets.IntSlider(min=-90, max=90, step=5, value=az, description='azimuth')
        el_slider = widgets.IntSlider(min=-15, max=90, step=5, value=el, description='elevation')

        plot_output = widgets.Output()

        # Initial plot
        with plot_output:
            plot(Hawk3D_instance, 
                    ax=ax,
                    el=el_slider.value,
                    az=az_slider.value,
                    colour=colour,
                    alpha=alpha) 

        def update_plot(change):
            with plot_output:
                clear_output(wait=True)
            
                ax.view_init(elev=el_slider.value, azim=az_slider.value)
                
                fig.canvas.draw_idle()  # Redraw the figure
                    
                display(fig)


        # Update the slider
        az_slider.observe(update_plot, names='value')
        el_slider.observe(update_plot, names='value')

        # Display the sliders
        display(az_slider, el_slider)
        display(plot_output)
    
        # Initial plot
        update_plot(None)

def plot(Hawk3D_instance, ax=None, el=20, az=60, colour=None, alpha=0.3, axisOn=True):

    if ax is None:
        fig, ax = get_plot3d_view()
        print("No axes given, creating new figure inside plot.")

    # Plot the polygons
    ax = plot_sections(ax, Hawk3D_instance, colour, alpha)

    # Plot the keypoints (only the measured markers)
    ax = plot_keypoints(ax, Hawk3D_instance, colour, alpha)

    # Set the azimuth and elev. for camera view of 3D axis.
    ax.view_init(elev=el, azim=az)

    # Set the plot settings
    if axisOn:
        origin = Hawk3D_instance.origin
        ax = plot_settings(ax,origin)

    return ax

def plot_multiple(Hawk3D_instance, keypoints, num_plots, spacing = (0.4, 0.7), cut_off=0.2, el=20, az=0, rot=90, colour_list=None, alpha=0.5):
    """
    Plots multiple frames of the hawk video.
    """

    # Create the figure and axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')


    grid_cols = int(np.ceil(np.sqrt(num_plots)))  # Set number of columns to square root of num_plots, rounded up
    grid_rows = int(np.ceil(num_plots / grid_cols))  # Calculate number of rows needed

    # Calculate middle indices for centering
    middle_row = (grid_rows - 1) / 2
    middle_col = (grid_cols - 1) / 2


    for i in range(num_plots):
        Hawk3D_instance.restore_keypoints_to_average()
        Hawk3D_instance.update_keypoints(keypoints[i])
        Hawk3D_instance.reset_transformation()

        # Calculate grid position
        row = grid_rows - 1 - (i // grid_cols)  # Inverts the order
        col = i % grid_cols

        # Get the colour from the Set3 colormap
        if colour_list is None:
            colour = plt.cm.Set3(i)
        else:
            colour = colour_list[i]

        # Calculate displacements centered around the origin
        vertDist = (row - middle_row) * spacing[0]
        horzDist = (col - middle_col) * spacing[1]


        Hawk3D_instance.transform_keypoints(vertDist=vertDist, horzDist=horzDist, yaw=rot)
        plot(Hawk3D_instance, ax=ax, el=el, az=az, colour=colour, alpha=alpha, axisOn=False)


    # Max vertical displacement
    max_vert_axis = (num_plots*0.15)*spacing[0]
    max_horz_axis = (num_plots*0.15)*spacing[1]
    ax.set_ylim(-max_horz_axis,max_horz_axis)
    ax.set_zlim(-max_vert_axis,max_vert_axis)
    ax.set_xlim(-0.5,0.5)

    # Set axis equal
    ax.set_aspect('equal', 'box')
    # Remove axes entirely and just leave polygons
    # ax.axis('off')
    # Make grid area white
            # --- Panel Shading
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('k')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)

    cropped_img = save_plot_as_image(fig, cut_off)

    return cropped_img


# ....... Helper Plot Functions ........

def plot_keypoints(ax,Hawk3D_instance, colour='k', alpha=1):

    # Only plot the markers. 
    coords = Hawk3D_instance.current_shape[:,Hawk3D_instance.marker_index,:][0]

    # Plot the keypoints
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                s = 5, c=colour, alpha=alpha)

    return ax

def plot_sections(ax, Hawk3D_instance, colour, alpha=1):

    # Plot each section
    for section in Hawk3D_instance.body_sections.keys():
        polygon = get_polygon(Hawk3D_instance, section, colour, alpha)
        ax.add_collection3d(polygon)

    return ax

def get_polygon(Hawk3D_instance, section_name, colour, alpha=1):
    """
    Returns the coordinates of the polygon representing the given section.
    """

    if section_name not in Hawk3D_instance.body_sections.keys():
        raise ValueError(f"Section name {section_name} not recognised.")

    colour = colour_polygon(section_name, colour)

    alpha = alpha_polygon(section_name, alpha)

    coords = Hawk3D_instance.get_polygon_coords(section_name)

    polygon = Poly3DCollection([coords],
                               alpha=alpha,
                               facecolor=colour,
                               edgecolor='k',
                               linewidths=0.5)
    return polygon

def alpha_polygon(section_name, alpha):

        # The alpha of the polygon is determined by whether the landmarks are
        # estimated or measured.
        if "handwing" in section_name or "tail" in section_name:
            alpha = alpha
        else:
            alpha = 0.3

        return alpha

def colour_polygon(section_name, colour):
    
        # The colour of the polygon is determined by whether the landmarks are
        # estimated or measured.
        if "handwing" in section_name or "tail" in section_name:
            colour = colour
        else:
            colour = np.array((0.5, 0.5, 0.5, 0.5))
    
        return colour

def plot_settings(ax,origin):
        """
        Plot settings & set the azimuth and elev. for camera view of 3D axis.
        """

        # --- Panel Shading
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # Put a line in the back corner of the plot
        corner = 0.32
        # ax.plot([-corner,-corner], [-corner,-corner], [-corner,corner], color='grey', linestyle=':', linewidth=0.5, zorder = -10)

        # Grid colour
        ax.xaxis._axinfo['grid'].update(color = 'grey', linestyle = ':', linewidth = 0.5)
        ax.yaxis._axinfo['grid'].update(color = 'grey', linestyle = ':',linewidth = 0.5)
        ax.zaxis._axinfo['grid'].update(color = 'grey', linestyle = ':',linewidth = 0.5)

        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)

        # --- Axis Limits
        increment = 0.28

        ax.auto_scale_xyz(  [origin[0]-increment, origin[0]+increment], 
                            [origin[1]-increment, origin[1]+increment],
                            [origin[2]-increment, origin[2]+increment])

        # --- Axis labels and Ticks
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_zlabel('z (m)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        ax.set_xticks(np.linspace(-0.2, 0.2, 3))
        ax.set_yticks(np.linspace(-0.2, 0.2, 3))
        ax.set_zticks(np.linspace(-0.2, 0.2, 3))

        # --- Axis Equal
        ax.set_aspect('equal', 'box')
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

# ----- Animation Functions -----

def animate(Hawk3D_instance, 
                keypoints_frames, 
                fig=None, 
                ax=None, 
                rotation_type="static", 
                el=20, 
                az=60, 
                alpha=0.3, 
                colour=None, 
                horzDist_frames=None, 
                bodypitch_frames=None, 
                vertDist_frames=None):
        """
        Create an animated 3D plot of a hawk video.
        """
        
        # Check dimensions and mirror the keypoints if only the right is given.
        keypoints_frames = format_keypoint_frames(Hawk3D_instance,keypoints_frames)

        if keypoints_frames.shape[0] == 0:
            raise ValueError("No frames to animate. Check the keypoints_frames input.")

        # Find the number of frames 
        num_frames = keypoints_frames.shape[0]

        # Initialize figure and axes
        if ax is None or fig is None:
            fig, ax = get_plot3d_view(fig)
        print("Figure and axes initialized.")


        # Prepare camera angles
        el_frames, az_frames = get_camera_angles(num_frames=num_frames, 
                                                      rotation_type=rotation_type, 
                                                      el=el, 
                                                      az=az)
        
        # Check if the horzDist_frames is given, if so check it is the correct length
        # If none given, return a zero array of the correct length.
        horzDist_frames  = check_transformation_frames(num_frames, horzDist_frames)
        vertDist_frames  = check_transformation_frames(num_frames, vertDist_frames)
        bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)

        # # Plot settings
        ax = plot_settings(ax, Hawk3D_instance.origin)

        
        
        def update_animated_plot(frame):
            """
            Function to update the animated plot.
            """
            
            ax.clear()

            # Update the keypoints for the current frame

            # Make sure the keypoints are restored to the default shape to remove any transformations
            Hawk3D_instance.reset_transformation()

            # Update the keypoints for the current frame
            Hawk3D_instance.update_keypoints(keypoints_frames[frame])

            
            # Transform the keypoints
            # If none provided, uses 0 to transform the keypoints
            Hawk3D_instance.transform_keypoints(bodypitch = bodypitch_frames[frame],
                                                horzDist  = horzDist_frames[frame],
                                                vertDist  = vertDist_frames[frame])
            
            # Then plot the current frame
            plot(Hawk3D_instance, 
                    ax=ax, 
                    el=el_frames[frame], 
                    az=az_frames[frame], 
                    alpha=alpha, 
                    colour=colour)
            
            # ax.set_title(f"Frame {frame+1}/{num_frames}")
            # ax.set_title(Hawk3D_instance.origin)
            plot_settings(ax, Hawk3D_instance.origin)

            return fig, ax
        # Make sure the keypoints are restored to the default shape to remove any transformations
        Hawk3D_instance.restore_keypoints_to_average()


        # Creating the animation
        animation = FuncAnimation(fig, update_animated_plot, 
                                  frames=num_frames, 
                                  interval=20, repeat=True)
        
        return animation


def animate_compare(Hawk3D_instance, 
                keypoints_frames_list, 
                fig=None, 
                ax=None, 
                rotation_type="static", 
                el=20, 
                az=60, 
                alpha=0.3, 
                colour=None, 
                horzDist_frames=None, 
                bodypitch_frames=None, 
                vertDist_frames=None):
     

        """
        Create an animated 3D plot of a hawk video.
        """
        formatted_keypoints_frames_list = []
        for keypoints_frames in keypoints_frames_list:
            # Check dimensions and mirror the keypoints if only the right is given.
            keypoints_frames = format_keypoint_frames(Hawk3D_instance,keypoints_frames)

            if keypoints_frames.shape[0] == 0:
                raise ValueError("No frames to animate. Check the keypoints_frames input.")

            # Find the number of frames 
            num_frames = keypoints_frames.shape[0]

            formatted_keypoints_frames_list.append(keypoints_frames)


        # Initialize figure and axes
        if ax is None or fig is None:
            fig, ax = get_plot3d_view(fig)
        print("Figure and axes initialized.")


        # Prepare camera angles
        el_frames, az_frames = get_camera_angles(num_frames=num_frames, 
                                                      rotation_type=rotation_type, 
                                                      el=el, 
                                                      az=az)
        
        # Check if the horzDist_frames is given, if so check it is the correct length
        # If none given, return a zero array of the correct length.
        horzDist_frames  = check_transformation_frames(num_frames, horzDist_frames)
        vertDist_frames  = check_transformation_frames(num_frames, vertDist_frames)
        bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)

        # # Plot settings
        ax = plot_settings(ax, Hawk3D_instance.origin)

        
        
        def update_animated_plot(frame):
            """
            Function to update the animated plot.
            """
            
            ax.clear()

            # Update the keypoints for the current frame

            # If keypoints_frame is a list
            for ii, keypoints_frames in enumerate(formatted_keypoints_frames_list):
                
                # Get a colour from matplotlib Set 2 using ii
                colour = plt.cm.Set2(ii)

                # Make sure the keypoints are restored to the default shape to remove any transformations
                Hawk3D_instance.reset_transformation()

                # Update the keypoints for the current frame
                Hawk3D_instance.update_keypoints(keypoints_frames[frame])

                
                # Transform the keypoints
                # If none provided, uses 0 to transform the keypoints
                Hawk3D_instance.transform_keypoints(bodypitch = bodypitch_frames[frame],
                                                    horzDist  = horzDist_frames[frame],
                                                    vertDist  = vertDist_frames[frame])
                
                # Then plot the current frame
                plot(Hawk3D_instance, 
                        ax=ax, 
                        el=el_frames[frame], 
                        az=az_frames[frame], 
                        alpha=alpha, 
                        colour=colour)
                
            # ax.set_title(f"Frame {frame+1}/{num_frames}")
            # ax.set_title(Hawk3D_instance.origin)
            plot_settings(ax, Hawk3D_instance.origin)

            return fig, ax
        # Make sure the keypoints are restored to the default shape to remove any transformations
        Hawk3D_instance.restore_keypoints_to_average()


        # Creating the animation
        animation = FuncAnimation(fig, update_animated_plot, 
                                  frames=num_frames, 
                                  interval=20, repeat=True)
        
        return animation


        # """
        # Create an animated 3D plot of a hawk video.
        # """
        
        # # Check dimensions and mirror the keypoints if only the right is given.
        # keypoints_frames = format_keypoint_frames(Hawk3D_instance,keypoints_frames)
        # compare_keypoints_frames = format_keypoint_frames(Hawk3D_instance,compare_keypoints_frames)

        # if keypoints_frames.shape[0] == 0:
        #     raise ValueError("No frames to animate. Check the keypoints_frames input.")

        # # Find the number of frames 
        # num_frames = keypoints_frames.shape[0]

        # # Initialize figure and axes
        # if ax is None or fig is None:
        #     fig, ax = get_plot3d_view(fig)
        # print("Figure and axes initialized.")


        # # Prepare camera angles
        # el_frames, az_frames = get_camera_angles(num_frames=num_frames, 
        #                                               rotation_type=rotation_type, 
        #                                               el=el, 
        #                                               az=az)
        
        # # Check if the horzDist_frames is given, if so check it is the correct length
        # # If none given, return a zero array of the correct length.
        # horzDist_frames  = check_transformation_frames(num_frames, horzDist_frames)
        # vertDist_frames  = check_transformation_frames(num_frames, vertDist_frames)
        # bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)

        # # # Plot settings
        # ax = plot_settings(ax, Hawk3D_instance.origin)

        
        
        # def update_animated_plot(frame):
        #     """
        #     Function to update the animated plot.
        #     """
            
        #     ax.clear()

        #     # Update the keypoints for the current frame

        #     # Make sure the keypoints are restored to the default shape to remove any transformations
        #     Hawk3D_instance.reset_transformation()

        #     # Update the keypoints for the current frame
        #     Hawk3D_instance.update_keypoints(keypoints_frames[frame])

            
        #     # Transform the keypoints
        #     # If none provided, uses 0 to transform the keypoints
        #     Hawk3D_instance.transform_keypoints(bodypitch = bodypitch_frames[frame],
        #                                         horzDist  = horzDist_frames[frame],
        #                                         vertDist  = vertDist_frames[frame])
            
        #     # Then plot the current frame
        #     plot(Hawk3D_instance, 
        #             ax=ax, 
        #             el=el_frames[frame], 
        #             az=az_frames[frame], 
        #             alpha=alpha, 
        #             colour="firebrick")
            
        #     Hawk3D_instance.reset_transformation()

        #     # Update the keypoints for the current frame
        #     Hawk3D_instance.update_keypoints(compare_keypoints_frames[frame])

        #     Hawk3D_instance.transform_keypoints(bodypitch = bodypitch_frames[frame],
        #                                         horzDist  = horzDist_frames[frame],
        #                                         vertDist  = vertDist_frames[frame])
            
        #     # Then plot the current frame
        #     plot(Hawk3D_instance, 
        #             ax=ax, 
        #             el=el_frames[frame], 
        #             az=az_frames[frame], 
        #             alpha=alpha, 
        #             colour="skyblue")
            
            
            
        #     # ax.set_title(f"Frame {frame+1}/{num_frames}")
        #     # ax.set_title(Hawk3D_instance.origin)
        #     plot_settings(ax, Hawk3D_instance.origin)

        #     return fig, ax
        # # Make sure the keypoints are restored to the default shape to remove any transformations
        # Hawk3D_instance.restore_keypoints_to_average()


        # # Creating the animation
        # animation = FuncAnimation(fig, update_animated_plot, 
        #                           frames=num_frames, 
        #                           interval=20, repeat=True)
        
        # return animation
    

# ....... Helper Animation Functions ........

def format_keypoint_frames(Hawk3D_instance, keypoints_frames):

        """
        Formats the keypoints_frames to be [n,8,3] where n is the number of frames 
        and 8 is the number of keypoints and 3 is the number of dimensions.
        If just 4 keypoints are given, the function will mirror the keypoints to make the left side.
        """

        if len(np.shape(keypoints_frames)) == 2:
            keypoints_frames = keypoints_frames.reshape(1, -1, 3)
            print("Warning: Only one frame given.")

    # Mirror the keypoints_frames if only the right is given. 
        if keypoints_frames.shape[1] == len(Hawk3D_instance.right_marker_names):
            keypoints_frames = Hawk3D_instance.mirror_keypoints(keypoints_frames)

        return keypoints_frames

def check_transformation_frames(num_frames, transformation_frames):

        """
        Checks that the transformation frames are the same length as the keypoints frames.
        If passed None, create an array of zeros for the transformation.
        """

        if transformation_frames is None:
            transformation_frames = np.zeros(num_frames)
        
        if len(transformation_frames) != num_frames:
            raise ValueError("Transformation frames must be the same length as keypoints_frames.")
        
        
        return transformation_frames

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



