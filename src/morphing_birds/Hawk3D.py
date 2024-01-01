import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA



class KeypointManager:
    
    names_fixed_keypoints = [
        "left_shoulder", 
        "left_tailbase", 
        "right_tailbase", 
        "right_shoulder",
        "hood", 
        "tailpack"
    ]

    names_right_keypoints = [
        "right_wingtip", 
        "right_primary", 
        "right_secondary",
        "right_tailtip",
    ]

    names_keypoints = [
        "left_wingtip",   "right_wingtip", 
        "left_primary",   "right_primary", 
        "left_secondary", "right_secondary", 
        "left_tailtip",   "right_tailtip"]
    
    

    def __init__(self, filename):
        self.names_all_keypoints, self.all_keypoints = self.load_data(filename)
        self.fixed_keypoints = self.get_keypoints_by_names(self.names_fixed_keypoints)
        self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)
        
        # Expecting the user to provide these. However, if they are not provided,
        # the average keypoints will be used as loaded from the file. 
        self.avg_keypoints = self.get_keypoints_by_names(self.names_keypoints)
        self.keypoints = self.avg_keypoints
        

    def load_data(self,filename):
        with open(filename, 'r') as file:
            data = np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')
        
        # Get the marker names from the first row of the csv file, get every 3rd name
        # and remove the '_x' from the names
        names_all_keypoints = data[0].reshape(-1, 3)
        names_all_keypoints = list(np.char.strip(names_all_keypoints[:, 0], '_x'))

        # Load marker coordinates and reshape to [n,3] matrix where n is the
        # number of markers
        all_keypoints = data[1].astype(float)
        all_keypoints = all_keypoints.reshape(-1, 3) # [n,3]

        return names_all_keypoints, all_keypoints

    def get_keypoint_indices(self,names_to_find=None):
        """
        Returns the indices of the keypoints with the given names.
        """

        marker_names = self.names_all_keypoints

        if names_to_find is None:
            names_to_find = marker_names

        indices = [marker_names.index(name) for name in names_to_find]

        return indices
    

    def get_keypoints_by_names(self,names_to_find=None):
        """
        Returns the keypoints with the given names.
        """
        
        indices = self.get_keypoint_indices(names_to_find)

        return self.all_keypoints[indices]
    

    def is_empty_keypoints(self, keypoints):
        if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
            return False
        else:
            return True

    def update_keypoints(self, user_keypoints):

        """
        Updates the keypoints with the given user_keypoints.
        ASSUMES THAT THE USER KEYPOINTS ARE GIVEN IN THE SAME ORDER AS THE
        NAMES OF THE KEYPOINTS.
        Either just the right or both keypoints can be given. 
        [expected shape: [1, 4, 3] or [1, 8, 3]
        """

        if len(np.shape(user_keypoints)) == 2:
            user_keypoints = user_keypoints.reshape(1, -1, 3)

        if self.is_empty_keypoints(user_keypoints):
            # Throw error
            raise ValueError("No keypoints given.")
        

        # If only the right keypoints are given, mirror them
        if user_keypoints.shape[1] == len(self.names_right_keypoints):
            user_keypoints = self._mirror_keypoints(user_keypoints)

        # If the left and right keypoints are given. 
        if user_keypoints.shape[1] == len(self.names_keypoints):
            
            # Check the keypoints are valid
            user_keypoints = self._validate_keypoints(user_keypoints)

            # Update all_keypoints 
            for ii, name in enumerate(self.names_keypoints):
                index = self.names_all_keypoints.index(name)
                self.all_keypoints[index] = user_keypoints[0,ii]

            # Update the keypoints too. 
            self.keypoints = self.get_keypoints_by_names(self.names_keypoints)

            # Update the right keypoints too.
            self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)
        else:
            raise ValueError("Wrong number of keypoints given.")

    def _validate_keypoints(self, keypoints):

        if keypoints.shape[-1] != 3:
            raise ValueError("Keypoints not in 3D.")

        if len(np.shape(keypoints)) == 2:
            keypoints = keypoints.reshape(1, -1, 3)
                
        if len(keypoints.shape) != 3:
            keypoints = keypoints.reshape(-1, keypoints.shape[1], 3)

        # If the keypoints are only for the right side, mirror them
        if keypoints.shape[1] == len(self.names_right_keypoints):
            keypoints = self._mirror_keypoints(keypoints)

        if keypoints.shape[1] != len(self.names_keypoints):
            print(keypoints.shape)
            raise ValueError("Keypoints missing.")
        

        return keypoints

    def _mirror_keypoints(self, keypoints):
        """
        Mirrors keypoints across the y-axis.
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
    
    def add_horzDist(self,horzDist):
        """
        Updates the keypoints (all) with a given horzDist. 
        Just changes the y.
        """
        if horzDist is None:
            horzDist = 0

        self.all_keypoints[:,1] += horzDist

        # Update the keypoints too. 
        self.keypoints = self.get_keypoints_by_names(self.names_keypoints)

        # Update the right keypoints too.
        self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)


    def add_vertDist(self,vertDist):
        """
        Updates the keypoints (all) with a given vertDist. 
        Just changes the z.
        """
        if vertDist is None:
            vertDist = 0
        
        self.all_keypoints[:,2] += vertDist

        # Update the keypoints too. 
        self.keypoints = self.get_keypoints_by_names(self.names_keypoints)

        # Update the right keypoints too.
        self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)


    def add_pitchRotation(self,bodypitch):
        """
        Updates the keypoints (all) with a given bodypitch.
        """

        if bodypitch is None:
            return
    
        rotmat = R.from_euler('x', bodypitch, degrees=True)
        self.all_keypoints = rotmat.apply(self.all_keypoints)

        # Update the keypoints too.
        self.keypoints = self.get_keypoints_by_names(self.names_keypoints)

        # Update the right keypoints too.
        self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)


    def validate_frames(self, keypoints_frames):
        """
        Validates the keypoints_frames and returns a list of keypoints for each frame.
        """
        if not isinstance(keypoints_frames, list):
            keypoints_frames = [keypoints_frames]
        for ii, keypoints in enumerate(keypoints_frames):

            # The if statement is to ensure that all keypoint sequences have the same length
            if ii > 0 and keypoints.shape[0] != keypoints_frames[ii - 1].shape[0]:
                raise ValueError("All keypoint sequences must have the same length")
            
            keypoints_frames[ii] = self._validate_keypoints(keypoints)
        return keypoints_frames
    
    def format_keypoints(self, keypoints):
        """
        Apply transforms on keypoints to make sure they are the correct shape.

        - Reshaping of flat keypoint arrays into (T, L, 3) arrays, where T is
            the number of frames and L the number of landmarks
        """
        nMarkers = len(self.names_keypoints)

        if len(keypoints.shape) == 1:
            keypoints = keypoints.reshape((-1, len(BirdData.landmarks), 3))
        if keypoints.shape[0] == 1:
            keypoints = keypoints.reshape((len(BirdData.landmarks), 3))


        return keypoints

class HawkData:

    def __init__(self, csv_path):
        self.markers = self.load_marker_frames(csv_path)
        
        self.load_frame_data(csv_path)

        self.check_data()

    def load_marker_frames(self, csv_path):
        """Load the unilateral markers dataset.

        Returns
        -------
        data : pandas.DataFrame
            The data frame containing the unilateral markers dataset.
        """
        # Load the data
        markers_csv = pd.read_csv(csv_path)

        # Rename the columns
        markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_1", "_x")
        markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_2", "_y")
        markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_3", "_z")

        # Get the index of the columns that contain the markers
        marker_index = markers_csv.columns[markers_csv.columns.str.contains('_x|_y|_z')]
        markers_csv = markers_csv[marker_index]

        # Make a numpy array with the markers
        unilateral_markers = markers_csv.to_numpy()

        # Reshape to 3D
        unilateral_markers = unilateral_markers.reshape(-1,4,3)
        
        return unilateral_markers

    def load_frame_data(self, csv_path):
        """Load the frame info from the dataset.

        Returns
        -------
        data : pandas.DataFrame
            The data frame containing the unilateral markers dataset.
        """
        # Load the data
        markers_csv = pd.read_csv(csv_path)

        self.horzDist = markers_csv['HorzDistance'].to_numpy()
        self.frameID = markers_csv['frameID']
        self.leftBool = markers_csv['Left'].to_numpy()
        self.body_pitch = markers_csv['body_pitch'].to_numpy()
        self.obstacleBool = markers_csv['Obstacle'].to_numpy()
        self.IMUBool = markers_csv['IMU'].to_numpy()
        self.time = markers_csv['time'].to_numpy()

    def check_data(self):
        """
        Check that the data are the same length.
        """

        num_frames = self.markers.shape[0]

        if self.horzDist.shape[0] != num_frames:
            raise ValueError("horzDist must be the same length as keypoints_frames.")
        
        if len(self.frameID) != num_frames:
            raise ValueError("frameID must be the same length as keypoints_frames.")
        
        if self.leftBool.shape[0] != num_frames:
            raise ValueError("leftBool must be the same length as keypoints_frames.")
        
        if self.body_pitch.shape[0] != num_frames:
            raise ValueError("body_pitch must be the same length as keypoints_frames.")
        
        if self.obstacleBool.shape[0] != num_frames:
            raise ValueError("obstacleBool must be the same length as keypoints_frames.")
        
        if self.IMUBool.shape[0] != num_frames:
            raise ValueError("IMUBool must be the same length as keypoints_frames.")
        
    def filter_by(self,
                  hawk=None,
                  perchDist=None, 
                  obstacle=False, 
                  year=None, 
                  Left=None,
                  IMU=False):
        """
        Returns boolean array of indices to filter the data.
        """

        def filter_by_bool(variable, bool_value):

            if bool_value is None:
                # Simply return the full array bool mask if passed None
                is_selected = np.ones(variable.shape, dtype=bool)
                return is_selected
            
            is_selected = variable == bool_value
            return is_selected

        # Initialise the filter
        filter = np.ones(len(self.frameID), dtype=bool)

        # Filter by hawk_ID
        if hawk is not None:
            filter = np.logical_and(filter, self.filter_by_hawk_ID(hawk))

        # Filter by perchDist
        if perchDist is not None:
            filter = np.logical_and(filter, self.filter_by_perchDist(perchDist))

        # Filter by obstacleToggle
        # if obstacle is not None:
        filter = np.logical_and(filter, filter_by_bool(self.obstacleBool, obstacle))

        # Filter by IMUToggle
        # if IMU is not None:
        filter = np.logical_and(filter, filter_by_bool(self.IMUBool, IMU))

        # Filter by Left
        # if Left is not None:
        filter = np.logical_and(filter, filter_by_bool(self.leftBool, Left))

        # Filter by year
        # if year is not None:
        filter = np.logical_and(filter, self.filter_by_year(year))
        
        return filter


            
    def filter_by_hawk_ID(self, hawk: str):

        def get_hawkID(hawk_name):

            if hawk_name.isdigit():
                # Transform the hawk_ID into a string with the correct format
                hawk_ID = str(hawk_name).zfill(2) + "_"
                
            # The user may have provided the full name of the hawk, or just the first few letters
            # And so returns the matching ID.

            if "dr" in hawk_name.lower():
                hawk_ID = "01_"  
            if "rh" in hawk_name.lower():
                hawk_ID = "02_"
            if "ru" in hawk_name.lower():
                hawk_ID = "03_"  
            if "to" in hawk_name.lower():
                hawk_ID = "04_"  
            if "ch" in hawk_name.lower():
                hawk_ID = "05_"
            
            return hawk_ID

        frameID = self.frameID

        if hawk is None:
            is_selected = np.ones(len(frameID), dtype=bool)
            return is_selected
        else:
            hawk_ID = get_hawkID(hawk)

        is_selected = frameID.str.startswith(hawk_ID)

        return is_selected
    
    def filter_by_perchDist(self, perchDist):

        # If perchDist is None, return the full array bool mask
        if perchDist is None:
            is_selected = np.ones(self.horzDist.shape, dtype=bool)
            return is_selected

        # Get any number from the perchDist string. The user may have given 
        # "12m" or "12 m" or "12"
        if perchDist.isdigit():
            perchDist = int(perchDist)
        else:
            perchDist = int(''.join(filter(str.isdigit, perchDist)))

        # Build back up the string to search for
        # Make sure the integer is padded such that it is 2 digits in length
        perchDist_str = "_" + str(perchDist).zfill(2) + "_"
        
        # Now looks for _05_ or _12_ etc in the frameID. Note, 05_09_ would be 
        # charmander flying 9m so we need to make sure we don't select that by leading 
        # and trailing _ . HawkID should always be found with "startswith". 

        is_selected = self.frameID.str.contains(perchDist_str)
        
        return is_selected

    def filter_by_year(self, year):
        frameID = self.frameID

        if year is None:
            is_selected = np.ones(len(frameID), dtype=bool)
            return is_selected

        # Data from 2017 and 2020 have different frameID formats, 
        # there's an extra _ in the frameID for 2020
        if year == 2017:
            is_selected = frameID.str.count('_') == 3
        elif year == 2020:
            is_selected = frameID.str.count('_') == 4
        else:
            raise ValueError("Year must be 2017 or 2020.")
        
        return is_selected
    
class HawkPlotter:

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

    def __init__(self, keypoint_manager):
        self.keypoint_manager = keypoint_manager
        self._init_polygons()

    def _init_polygons(self):
        """
        Initialise the polygons for plotting.
        """
        self._polygons = {}
        for name, marker_names in self.body_sections.items():
            self._polygons[name] = self.keypoint_manager.get_keypoint_indices(marker_names)

    def get_polygon(self, section_name, colour, alpha=1):
        """
        Returns the coordinates of the polygon representing the given section.
        """
        
        if section_name not in self.body_sections.keys():
            raise ValueError(f"Section name {section_name} not recognised.")
        
        colour = self._colour_polygon(section_name, colour)

        keypoint_indices = self._polygons[section_name]
        coords = self.keypoint_manager.all_keypoints[keypoint_indices]

        polygon = Poly3DCollection([coords],
                                   alpha=alpha,
                                   facecolor=colour,
                                   edgecolor='k',
                                   linewidths=0.5)
        return polygon

    def _colour_polygon(self, section_name, colour):
        
        # The colour of the polygon is determined by whether the landmarks are
        # estimated or measured.
        if "handwing" in section_name or "tail" in section_name:
            colour = colour
        else:
            colour = np.array((0.5, 0.5, 0.5, 0.5))

        return colour

    def plot_keypoints(self, ax, colour='k', alpha=1):
        """
        Plots the keypoints of the hawk.
        """
        
        keypoints = self.keypoint_manager.keypoints

        # Plot the keypoints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2],
                   s = 2, c=colour, alpha=alpha)
        
        return ax
    
    def plot_sections(self, ax, colour, alpha=1):
        """
        Plots the polygons representing the different sections of the hawk.
        """

        # Plot each section
        for section in self.body_sections.keys():
            polygon = self.get_polygon(section, colour, alpha)
            ax.add_collection3d(polygon)

        return ax
    
    def plot(self,
             keypoints=None,
             fig = None,
             ax=None,
             el=20,
             az=60,
             colour=None,
             alpha=0.3,
             horzDist=None,
             bodypitch=None,
             vertDist=None):
        """
        Plots the hawk.
        """

        # Check if keypoints are given, otherwise use the average. 
        # Updates the state of the object. 
        if keypoints is not None and not self.keypoint_manager.is_empty_keypoints(keypoints):
            self.keypoint_manager.update_keypoints(keypoints)

        # Apply transformations.
        # (Default: nothing happens)
        self.keypoint_manager.add_horzDist(horzDist)
        self.keypoint_manager.add_vertDist(vertDist)
        self.keypoint_manager.add_pitchRotation(bodypitch)

        # Initialise the figure and axes if not given
        if ax is None:
            fig, ax = self.get_plot3d_view(fig)


        # Plot the polygons
        ax = self.plot_sections(ax, colour, alpha)

        # Plot the keypoints (only the measured markers)
        ax = self.plot_keypoints(ax, colour, alpha)

        # Set the azimuth and elev. for camera view of 3D axis.
        ax.view_init(elev=el, azim=az)

        # Set the plot settings
        ax = self._plot_settings(ax,horzDist)
    
        return ax
    

    def interactive_plot(self, 
                         keypoints=None, 
                         fig=None,
                         ax=None,
                         el=20,
                         az=60,
                         colour=None,
                         alpha=0.3,
                         horzDist=None,
                         bodypitch=None,
                         vertDist=None):
        """
        Interactive plot of the hawk, 
        sliders to change the azimuth and elevation.
        """
        # Initialise the figure and axes if not given
        plt.ioff()  # Turn off interactive mode
        
        if ax is None:
            fig, ax = self.get_plot3d_view(fig)

        plt.ion()  # Turn on interactive mode
        
        az_slider = widgets.IntSlider(min=-90, max=90, step=5, value=az, description='azimuth')
        el_slider = widgets.IntSlider(min=-15, max=90, step=5, value=el, description='elevation')

        plot_output = widgets.Output()

        def update_plot(change):
            with plot_output:
                clear_output(wait=True)
                # ax.plot([0, 1], [0, 1], [0, 1])  # Initial drawing
                # ax.view_init(elev=el_slider.value, azim=az_slider.value)  # Update view
                ax.clear()
                self.plot(keypoints=keypoints,
                        fig=fig,
                        ax=ax,
                        el=el_slider.value,
                        az=az_slider.value,
                        colour=colour,
                        alpha=alpha,
                        horzDist=horzDist,
                        bodypitch=bodypitch,
                        vertDist=vertDist)  
    
                display(fig)


        # Update the slider
        az_slider.observe(update_plot, names='value')
        el_slider.observe(update_plot, names='value')

        

        # Display the sliders
        display(az_slider, el_slider)
        display(plot_output)

        # Initial plot
        update_plot(None)
        

    def get_plot3d_view(self,fig=None, rows=1, cols=1, index=1):
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
    
    def _plot_settings(self,ax,horzDist=None):
        """
        Plot settings & set the azimuth and elev. for camera view of 3D axis.
        """

        if horzDist is None:
            horzDist = 0

        # --- Panel Shading
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # --- Axis Limits
        minbound = -0.28
        maxbound = 0.28
        ax.auto_scale_xyz(  [minbound, maxbound], 
                            [horzDist-minbound, horzDist+maxbound],
                            [minbound, maxbound])

        # --- Axis labels and Ticks
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_zlabel('z (m)', fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        ax.set_xticks(np.linspace(-0.25, 0.25, 3))
        ax.set_yticks(np.linspace(horzDist-0.25,horzDist+0.25, 3))
        ax.set_zticks(np.linspace(-0.25, 0.25, 3))

        return ax

class HawkAnimator:
    def __init__(self, plotter):
        self.plotter = plotter

    def _format_keypoint_frames(self, keypoints_frames):

        if len(np.shape(keypoints_frames)) == 2:
            keypoints_frames = keypoints_frames.reshape(1, -1, 3)
            print("Warning: Only one frame given.")

    # Mirror the keypoints_frames if only the right is given. 
        if keypoints_frames.shape[1] == len(self.plotter.keypoint_manager.names_right_keypoints):
            keypoints_frames = self.plotter.keypoint_manager._mirror_keypoints(keypoints_frames)

        return keypoints_frames
        
    def animate(self, 
                keypoints_frames, 
                fig=None, 
                ax=None, 
                rotation_type="static", 
                el=20, 
                az=60, 
                alpha=0.3, 
                colour=None, 
                horzDist_frames=None, 
                bodypitch_frames=None):
        """
        Create an animated 3D plot of a hawk video.
        """

        # Mirror the keypoints if only the right is given.
        keypoints_frames = self._format_keypoint_frames(keypoints_frames)

        # Find the number of frames 
        num_frames = keypoints_frames.shape[0]


        # Initialize figure and axes
        if ax is None:
            fig, ax = self.plotter.get_plot3d_view(fig)

        # Prepare camera angles
        el_frames, az_frames = self.get_camera_angles(num_frames=num_frames, 
                                                      rotation_type=rotation_type, 
                                                      el=el, 
                                                      az=az)
        
        # Check if the horzDist_frames is given, if so check it is the correct length
        if horzDist_frames is not None:
            if len(horzDist_frames) != num_frames:
                raise ValueError("horzDist_frames must be the same length as keypoints_frames.")
            
        # Check if the bodypitch_frames is given, if so check it is the correct length
        if bodypitch_frames is not None:
            if len(bodypitch_frames) != num_frames:
                raise ValueError("bodypitch_frames must be the same length as keypoints_frames.")
            
        # Plot settings
        self.plotter._plot_settings(ax, horzDist=0 if horzDist_frames is None else horzDist_frames)

        # Update function for animation
        def update_animated_plot(frame, *fargs):
            fig, ax, keypoints, el_frames, az_frames, alpha, colour, horzDist_frames, bodypitch_frames = fargs
            ax.clear()
            # Here, you need to adjust how keypoints for the current frame are passed to plot
            self.plotter.plot(keypoints=keypoints[frame], 
                              fig=fig, 
                              ax=ax, 
                              el=el_frames[frame], 
                              az=az_frames[frame], 
                              alpha=alpha, 
                              colour=colour, 
                              horzDist=horzDist_frames[frame] if horzDist_frames else None, 
                              bodypitch=bodypitch_frames[frame] if bodypitch_frames else None)
            return fig, ax

        # Creating the animation
        animation = FuncAnimation(fig, update_animated_plot, 
                                  frames=num_frames, 
                                  fargs=(fig, ax, keypoints_frames, el_frames, az_frames, alpha, colour, horzDist_frames, bodypitch_frames), 
                                  interval=20, repeat=True)
        
        return animation

    def get_camera_angles(self,num_frames, rotation_type, el=20, az=60):
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

class HawkPCA:
    
    def __init__(self, HawkData, KeypointManager):
        self.data = HawkData
        self.mu = KeypointManager.right_keypoints
        
        # Make the dimensions fit for PCA
        self.mu = self.mu.reshape(1,12)

    def get_input(self, data=None):

        if data is None:
            data = self.data.markers

        # The data is in the shape (n_frames, n_markers*n_dimensions)
        pca_input = data.reshape(-1,12)

        return pca_input
    
    def run_PCA(self, data=None):

        pca_input = self.get_input(data)
        num_components = pca_input.shape[1]

        pca = PCA()
        pca_output = pca.fit(pca_input)


        # Another word for eigenvectors is components.
        self.principal_components = pca_output.components_

        # Another word for scores is projections.
        self.scores = pca_output.transform(pca_input)


    def get_score_range(self, num_frames=30):

        num_components = self.scores.shape[1]

        min_score = np.mean(self.scores, axis=0) - (2 * np.std(self.scores, axis=0))
        max_score = np.mean(self.scores, axis=0) + (2 * np.std(self.scores, axis=0))

        half_length = num_frames // 2 + 1

        # Initialize score_frames with the shape [n, 12]
        self.score_frames = np.zeros([num_frames, num_components])


        for ii in range(num_components):
        
            # Create forward and backward ranges for each component using np.linspace
            forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
            
            # # If num_frames is odd, we add an extra element to 'forward'
            # if num_frames % 2 != 0:
            #     forward = np.append(forward, max_score[ii])
            
            backward = forward[::-1]  # Reverse the forward range

            # Combine forward and backward, and assign to the i-th column
            self.score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

        
    def select_components(self, components_list):

        selected_components = self.principal_components[:,components_list]
        
        return selected_components

    def reconstruct(self,components_list=None):

        if components_list is None:
            components_list = range(12)
        
        selected_PCs = self.principal_components[components_list]
        selected_scores = self.score_frames[:,components_list]

        
        num_frames = self.score_frames.shape[0]
        reconstructed_frames = np.empty((0,4,3))

        # for ii in range(num_frames):
        #     score = selected_scores[ii]
            
            
        #     selected_PCs = selected_PCs.reshape(1,4,3)
        #     mu = self.mu.reshape(1,4,3)

        #     frame = mu + score * selected_PCs
        #     frame = frame.reshape(-1,4,3)

        #     reconstructed_frames = np.append(reconstructed_frames,frame,axis=0)
            
        selected_PCs = selected_PCs.reshape(1, -1, 4, 3)  # [1, nPCs, 4, 3]
        selected_scores = selected_scores.reshape(num_frames, -1, 1, 1)  # [n, nPCs, 1, 1]
        mu = self.mu.reshape(1, 4, 3)  # [1, 4, 3]

        reconstructed_frames = mu + np.sum(selected_scores * selected_PCs, axis=1)
        

        # reconstructed_frames = reconstructed.reshape(-1,4,3)

        return reconstructed_frames

    



# ----- Hawk3D Class -----

class Hawk3D:
    def __init__(self, filename):

        # As an explanation, anything within KeypointManager can be found by calling
        # Hawk3D.keypoint_manager. For example, Hawk3D.keypoint_manager.keypoints
        # will return the keypoints. 
        
        self.keypoint_manager = KeypointManager(filename)
        self.plotter = HawkPlotter(self.keypoint_manager)
        self.animator = HawkAnimator(self.plotter)

        

    def get_data(self, csv_path):
        
        self.frames = HawkData(csv_path)
        self.markers = self.frames.markers
        self.horzDist = self.frames.horzDist

        self.PCA = HawkPCA(self.frames, self.keypoint_manager)

    def display_hawk(self, user_keypoints=None, el = 20, az = 60):
        
        """
        Displays the hawk using either the default keypoints or user-provided keypoints.

        Parameters:
        user_keypoints (numpy.ndarray, optional): An array of keypoints provided by the user.
                                                    Expected shape is [1, 4, 3].
        """

        # Update the keypoints if the user has provided them. otherwise use the average points. 
        if user_keypoints is None:
            self.keypoint_manager.update_keypoints(self.keypoint_manager.avg_keypoints)
        else:
            self.keypoint_manager.update_keypoints(user_keypoints)

        # Use the plotter to display the hawk with the current keypoints. Average from file by default.
        self.plotter.interactive_plot()


    def animate_hawk(self, keypoint_sequence,
                rotation_type="static", 
                el=20, 
                az=60, 
                alpha=0.3, 
                colour=None, 
                horzDist_frames=None, 
                bodypitch_frames=None):
        """
        Animates the hawk using a sequence of keypoints.

        Parameters:
        keypoint_sequence (numpy.ndarray): An array of keypoints for each frame.
                                           Expected shape is [n, 4, 3], where n is the number of frames.
        """
            
        # Use the animator to animate the hawk with the given keypoints.
        self.animation = self.animator.animate(keypoint_sequence,
                                               rotation_type=rotation_type, 
                                               el=el, 
                                               az=az, 
                                               alpha=alpha, 
                                               colour=colour, 
                                               horzDist_frames=horzDist_frames, 
                                               bodypitch_frames=bodypitch_frames)

       

    def quick_PCA(self, selected_PCs=0,num_frames=30):
        
        self.PCA.run_PCA()
        self.PCA.get_score_range(num_frames)
        frames = self.PCA.reconstruct(selected_PCs)

        self.animate_hawk(frames)

# ----- Main Function/Script -----
# def main():
#     hawk3d = Hawk3D("data/mean_hawk_shape.csv")
#     hawk3d.display_hawk()
#     user_keypoints = np.array([...])  # Shape [1, 4, 3]
#     hawk3d.display_hawk(user_keypoints)
#     hawk3d.animate_hawk()

# if __name__ == "__main__":
#     main()