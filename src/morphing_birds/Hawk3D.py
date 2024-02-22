from .Keypoints import KeypointManager
from .HawkData import HawkData
from .HawkPCA import HawkPCA
from .HawkPlotter import HawkPlotter
from .Animator import HawkAnimator



# ----- Hawk3D Class -----

class Hawk3D:
    def __init__(self, filename):

        # As an explanation, anything within KeypointManager can be found by calling
        # Hawk3D.keypoint_manager. For example, Hawk3D.keypoint_manager.keypoints
        # will return the keypoints. 
        
        self.keypoint_manager = KeypointManager(filename)
        self.plotter = HawkPlotter(self.keypoint_manager)
        self.animator = HawkAnimator(self.plotter)

        self.keypoint_manager.keypoints_no_translation = None

    def get_data(self, csv_path):
        
        """
        Loads the data from the csv file.
        """

        self.frames = HawkData(csv_path)
        self.markers = self.frames.markers
        self.horzDist = self.frames.horzDist

        self.PCA = HawkPCA(self.frames, self.keypoint_manager)

    def display_hawk(self, 
                     user_keypoints=None, 
                     el = 20, 
                     az = 60, 
                     alpha = 0.3, 
                     colour = None,
                     horzDist = None, 
                     bodypitch = None, 
                     vertDist = None):
        
        """
        Displays the hawk using either the default keypoints or user-provided keypoints.

        Parameters:
        user_keypoints (numpy.ndarray, optional): An array of keypoints provided by the user.
                                                    Expected shape is [1, 4, 3].
        """

        if user_keypoints is not None:
            self.keypoint_manager.update_keypoints(user_keypoints)

        # Transforms the keypoints if horzdist or bodypitch is given
        # Saves a copy of the untransformed keypoints to restore later
        self.keypoint_manager.keypoints = self.keypoint_manager.transform_keypoints(horzDist=horzDist, 
                                                    bodypitch=bodypitch, 
                                                    vertDist=vertDist)
        
        # Use the plotter to display the hawk with the current keypoints. Average from file by default.
        self.plotter.interactive_plot(el=el, 
                                      az=az, 
                                      alpha=alpha, 
                                      colour=colour,
                                      horzDist=horzDist)
        # 2024-01-04 22:49 SERIOUSLY CAN'T UNDERSTAND WHY THIS WON'T WORK
        # The keypoints are not updating when I call the function again.
        # And the keypoints won't rotate.

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
        
        """
        Quick PCA reconstruction of the hawk using the given number of principal components.
        Animates the hawk using the reconstructed frames. Default first PCs. 
        """

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