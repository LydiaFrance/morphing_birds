import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
from matplotlib.animation import FuncAnimation


# ----- KeypointManager Class -----

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
                                  interval=20, repeat=False)
        
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



# ----- Hawk3D Class -----

class Hawk3D:
    def __init__(self, filename):

        # As an explanation, anything within KeypointManager can be found by calling
        # Hawk3D.keypoint_manager. For example, Hawk3D.keypoint_manager.keypoints
        # will return the keypoints. 
        
        self.keypoint_manager = KeypointManager(filename)
        self.plotter = HawkPlotter(self.keypoint_manager)
        self.animator = HawkAnimator(self.plotter)

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



# ----- Main Function/Script -----
# def main():
#     hawk3d = Hawk3D("data/mean_hawk_shape.csv")
#     hawk3d.display_hawk()
#     user_keypoints = np.array([...])  # Shape [1, 4, 3]
#     hawk3d.display_hawk(user_keypoints)
#     hawk3d.animate_hawk()

# if __name__ == "__main__":
#     main()