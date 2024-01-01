import numpy as np
from matplotlib.animation import FuncAnimation



class HawkAnimator:
    def __init__(self, plotter):

        """ 
        Uses the HawkPlotter class to create an animated 3D plot of a hawk video.
        
        """
        self.plotter = plotter

    def _format_keypoint_frames(self, keypoints_frames):

        """
        Formats the keypoints_frames to be [n,8,3] where n is the number of frames 
        and 8 is the number of keypoints and 3 is the number of dimensions.
        If just 4 keypoints are given, the function will mirror the keypoints to make the left side.
        """

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

        
        def update_animated_plot(frame, *fargs):
            """
            Function to update the animated plot.
            """

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

