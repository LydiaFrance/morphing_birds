import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import ipywidgets as widgets
from IPython.display import display, clear_output


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

        """
        Class for plotting the hawk. Uses Keypoints.py to get and 
        manage the keypoints.
        """
        
        self.keypoint_manager = keypoint_manager
        self.keypoints = keypoint_manager.keypoints
        self._init_polygons()

    def _init_polygons(self):
        """
        Initialise the polygons for plotting. 
        Gets the indices of the keypoints for each section.
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

        polygon_keypoint_indices = self._polygons[section_name]
        coords = self.keypoints[polygon_keypoint_indices]

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

    def plot_keypoints(self, 
                       ax, 
                       colour='k', 
                       alpha=1):
        """
        Plots the keypoints of the hawk.
        """
        
        # Only plot the markers. 
        marker_index = self.keypoint_manager.marker_index
        markers = self.keypoints[marker_index]

        # Plot the keypoints
        ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2],
                   s = 2, c=colour, alpha=alpha)
                
        return ax
    
    def plot_sections(self, 
                      ax, 
                      colour, 
                      alpha=1):
        """
        Plots the polygons representing the different sections of the hawk.
        """

        # Plot each section
        for section in self.body_sections.keys():
            polygon = self.get_polygon(section, colour, alpha)
            ax.add_collection3d(polygon)

        return ax
    
    def plot(self,
             fig = None,
             ax=None,
             el=20,
             az=60,
             colour=None,
             alpha=0.3,
             horzDist=None):
        """
        Plots the hawk.
        """

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
                         fig=None,
                         ax=None,
                         el=20,
                         az=60,
                         colour=None,
                         alpha=0.3,
                         horzDist=None):
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

        # Initial plot
        with plot_output:
            self.plot(fig=fig,
                    ax=ax,
                    el=el_slider.value,
                    az=az_slider.value,
                    colour=colour,
                    alpha=alpha,
                    horzDist=horzDist) 

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

