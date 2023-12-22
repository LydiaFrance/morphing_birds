import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

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

    def animate(self):
        self.plotter.plot()

        pass

# ----- Hawk3D Class -----

class Hawk3D:
    def __init__(self, filename):

        # As an explanation, anything within KeypointManager can be found by calling
        # Hawk3D.keypoint_manager. For example, Hawk3D.keypoint_manager.keypoints
        # will return the keypoints. 
        
        self.keypoint_manager = KeypointManager(filename)
        self.plotter = HawkPlotter(self.keypoint_manager)
        self.animator = HawkAnimator(self.plotter)

    def display_hawk(self, user_keypoints=None):
        
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
        self.plotter.plot()


    def animate_hawk(self, keypoint_sequence):
        """
        Animates the hawk using a sequence of keypoints.

        Parameters:
        keypoint_sequence (numpy.ndarray): An array of keypoints for each frame.
                                           Expected shape is [n, 4, 3], where n is the number of frames.
        """
        for frame_keypoints in keypoint_sequence:
            # Update the keypoints in the plotter for the current frame
            self.plotter.update_keypoints(frame_keypoints)
            
            # Render the plot for the current frame
            self.plotter.plot()



# ----- Main Function/Script -----
# def main():
#     hawk3d = Hawk3D("data/mean_hawk_shape.csv")
#     hawk3d.display_hawk()
#     user_keypoints = np.array([...])  # Shape [1, 4, 3]
#     hawk3d.display_hawk(user_keypoints)
#     hawk3d.animate_hawk()

# if __name__ == "__main__":
#     main()