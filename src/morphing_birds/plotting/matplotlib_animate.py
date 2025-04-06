import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .animation_frame_helpers import format_keypoint_frames, check_transformation_frames
from .matplotlib_helpers import get_plot3d_view, get_camera_angles
from .matplotlib_plots import plot, plot_settings


def animate(animal3d_instance, 
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
                vertDist_frames=None,
                bodyyaw_frames=None,
                bodyroll_frames=None,
                score_vals=None):
        """
        Create an animated 3D plot of a hawk video.
        
        Parameters:
        -----------
        animal3d_instance : Animal3D
            Instance of the Animal3D class
        keypoints_frames : array-like
            Array of keypoint frames to animate
        fig : matplotlib.figure.Figure, optional
            Figure to plot on
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        rotation_type : str, optional
            Type of camera rotation ("static", "dynamic", or "slow")
        el : float, optional
            Elevation angle for camera
        az : float, optional
            Azimuth angle for camera
        alpha : float, optional
            Transparency of the plot
        colour : str, optional
            Color for the plot
        horzDist_frames : array-like, optional
            Horizontal distance transformation for each frame
        bodypitch_frames : array-like, optional
            Body pitch transformation for each frame
        vertDist_frames : array-like, optional
            Vertical distance transformation for each frame
        bodyyaw_frames : array-like, optional
            Body yaw transformation for each frame
        bodyroll_frames : array-like, optional
            Body roll transformation for each frame
        score_vals : array-like, optional
            Values to show in the slider
        """
        
        # Check dimensions and mirror the keypoints if only the right is given.
        keypoints_frames = format_keypoint_frames(animal3d_instance,keypoints_frames)

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
        
        # Check if the transformation frames are given, if so check they are the correct length
        # If none given, return a zero array of the correct length
        horzDist_frames = check_transformation_frames(num_frames, horzDist_frames)
        vertDist_frames = check_transformation_frames(num_frames, vertDist_frames)
        bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)
        bodyyaw_frames = check_transformation_frames(num_frames, bodyyaw_frames)
        bodyroll_frames = check_transformation_frames(num_frames, bodyroll_frames)

        # Plot settings
        # Define the axis limits based on the keypoints scale 
        lims = keypoints_frames.max()*0.5
        lims = [-lims, lims]
        ax = plot_settings(ax, animal3d_instance.origin, lims)
        
        def update_animated_plot(frame):
            """
            Function to update the animated plot.
            """
            ax.clear()

            # Make sure the keypoints are restored to the default shape to remove any transformations
            animal3d_instance.reset_transformation()

            # Update the keypoints for the current frame
            animal3d_instance.update_keypoints(keypoints_frames[frame])
            
            # Transform the keypoints
            animal3d_instance.transform_keypoints(bodypitch=bodypitch_frames[frame],
                                                horzDist=horzDist_frames[frame],
                                                vertDist=vertDist_frames[frame],
                                                bodyyaw=bodyyaw_frames[frame],
                                                bodyroll=bodyroll_frames[frame])
            
            # Then plot the current frame
            plot(animal3d_instance, 
                    ax=ax, 
                    el=el_frames[frame], 
                    az=az_frames[frame], 
                    alpha=alpha, 
                    colour=colour)
            
            plot_settings(ax, animal3d_instance.origin, lims)

            return fig, ax

        # Make sure the keypoints are restored to the default shape to remove any transformations
        animal3d_instance.restore_keypoints_to_average()

        # Creating the animation
        animation = FuncAnimation(fig, update_animated_plot, 
                                  frames=num_frames, 
                                  interval=20, repeat=True)
        
        return animation


def animate_compare(animal3d_instance, 
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
            keypoints_frames = format_keypoint_frames(animal3d_instance,keypoints_frames)

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
        ax = plot_settings(ax, animal3d_instance.origin)

        
        
        def update_animated_plot(frame):
            """
            Function to update the animated plot.
            """
            
            ax.clear()

            # Update the keypoints for the current frame

            # If keypoints_frame is a list
            for ii, keypoints_frames in enumerate(formatted_keypoints_frames_list):
                
                # Get a colour from matplotlib Set 2 using ii
                colour = plt.cm.Set1(ii)

                # Make sure the keypoints are restored to the default shape to remove any transformations
                animal3d_instance.reset_transformation()

                # Update the keypoints for the current frame
                animal3d_instance.update_keypoints(keypoints_frames[frame])

                
                # Transform the keypoints
                # If none provided, uses 0 to transform the keypoints
                animal3d_instance.transform_keypoints(bodypitch = bodypitch_frames[frame],
                                                    horzDist  = horzDist_frames[frame],
                                                    vertDist  = vertDist_frames[frame])
                
                # Then plot the current frame
                plot(animal3d_instance, 
                        ax=ax, 
                        el=el_frames[frame], 
                        az=az_frames[frame], 
                        alpha=alpha, 
                        colour=colour)
                
            # ax.set_title(f"Frame {frame+1}/{num_frames}")
            # ax.set_title(animal3d_instance.origin)
            plot_settings(ax, animal3d_instance.origin)

            return fig, ax
        # Make sure the keypoints are restored to the default shape to remove any transformations
        animal3d_instance.restore_keypoints_to_average()


        # Creating the animation
        animation = FuncAnimation(fig, update_animated_plot, 
                                  frames=num_frames, 
                                  interval=20, repeat=True)
        
        return animation





