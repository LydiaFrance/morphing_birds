
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from .matplotlib_helpers import get_plot3d_view, plot_sections, plot_keypoints, plot_settings, save_plot_as_image

def plot(animal3d_instance, ax=None, el=20, az=60, colour=None, alpha=0.3, axisOn=True):

    if ax is None:
        fig, ax = get_plot3d_view()
        print("No axes given, creating new figure inside plot.")

    # Plot the polygons
    ax = plot_sections(ax, animal3d_instance, colour, alpha)

    # Plot the keypoints (only the measured markers)
    ax = plot_keypoints(ax, animal3d_instance, colour, alpha)

    # Set the azimuth and elev. for camera view of 3D axis.
    ax.view_init(elev=el, azim=az)

    # Set the plot settings
    if axisOn:
        origin = animal3d_instance.origin
        ax = plot_settings(ax,origin)

    return ax


def interactive_plot(animal3d_instance, keypoints = None, fig = None, ax=None, el=20, az=60, colour=None, alpha=0.3):

        if keypoints is not None:
            animal3d_instance.update_keypoints(keypoints)
            animal3d_instance.reset_transformation()


        plt.ioff()  # Turn off interactive mode
        
        if ax is None:
            fig, ax = get_plot3d_view()
            
        plt.ion()  # Turn on interactive mode
        
        az_slider = widgets.IntSlider(min=-90, max=90, step=5, value=az, description='azimuth')
        el_slider = widgets.IntSlider(min=-15, max=90, step=5, value=el, description='elevation')

        plot_output = widgets.Output()

        # Initial plot
        with plot_output:
            plot(animal3d_instance, 
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


def plot_multiple(animal3d_instance, keypoints, num_plots, spacing = (0.4, 0.7), cut_off=0.2, el=20, az=0, rot=90, colour_list=None, alpha=0.5):
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
        animal3d_instance.restore_keypoints_to_average()
        animal3d_instance.update_keypoints(keypoints[i])
        animal3d_instance.reset_transformation()

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


        animal3d_instance.transform_keypoints(vertDist=vertDist, horzDist=horzDist, yaw=rot)
        plot(animal3d_instance, ax=ax, el=el, az=az, colour=colour, alpha=alpha, axisOn=False)


    # Max vertical displacement
    max_vert_axis = (num_plots*0.15)*spacing[0]
    max_horz_axis = (num_plots*0.15)*spacing[1]
    ax.set_ylim(-max_horz_axis,max_horz_axis)
    ax.set_zlim(-max_vert_axis,max_vert_axis)
    ax.set_xlim(-0.5,0.5)

    # Set axis equal
    ax.set_aspect('equal', 'box')
    # Remove axes entirely and just leave polygons
    ax.axis('off')
    # Make grid area white
            # --- Panel Shading
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.xaxis.pane.set_edgecolor('w')
    # ax.yaxis.pane.set_edgecolor('w')
    # ax.zaxis.pane.set_edgecolor('k')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.grid(False)

    cropped_img = save_plot_as_image(fig, cut_off)

    return cropped_img
