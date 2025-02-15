import numpy as np
import plotly.graph_objs as go

from .plotly_helpers import calculate_axis_limits

def plot_plotly(animal3d_instance, colour='lightblue', alpha=0.3, axisOn=True,
                horzDist=None, vertDist=None, bodypitch=None, bodyroll=None, bodyyaw=None):
    """
    Create a static 3D plot of a hawk using Plotly.
    
    Parameters:
    -----------
    animal3d_instance : Animal3D
        Instance of the Animal3D class
    colour : str, optional
        Colour for the plot
    alpha : float, optional
        Transparency of the plot
    axisOn : bool, optional
        Whether to show axes
    horzDist : float, optional
        Horizontal distance transformation
    vertDist : float, optional
        Vertical distance transformation
    bodypitch : float, optional
        Body pitch transformation
    """
    # Store the current state
    current_state = animal3d_instance.current_shape.copy()
    
    # Apply transformations if provided
    animal3d_instance.reset_transformation()
    animal3d_instance.transform_keypoints(bodypitch=bodypitch,
                                      horzDist=horzDist,
                                      vertDist=vertDist, 
                                      bodyyaw=bodyyaw,
                                      bodyroll=bodyroll)

    # Create plot
    fig = go.Figure()
    fig = plot_sections_plotly(fig, animal3d_instance, colour, alpha)
    fig = plot_keypoints_plotly(fig, animal3d_instance, colour, alpha)

    # Set axis visibility
    if not axisOn:
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ))
    
    fig = plot_settings_plotly(fig, animal3d_instance)

    # Restore the original state
    animal3d_instance.current_shape = current_state
    
    return fig

def plot_plotly_compare(animal3d_instances,
                        colours=None,
                        alpha=0.3,
                        axisOn=True,
                        horzDist=None,
                        vertDist=None,
                        bodypitch=None,
                        bodyroll=None,
                        bodyyaw=None):
    
    # Create plot
    fig = go.Figure()

    if colours is None:
        colours = ['red', None]  # First shape red, second shape default
    
    for animal3d_instance, colour in zip(animal3d_instances, colours):
        # Store the current state
        current_state = animal3d_instance.current_shape.copy()
        
        # Apply transformations if provided
        animal3d_instance.reset_transformation()
        animal3d_instance.transform_keypoints(bodypitch=bodypitch,
                                        horzDist=horzDist,
                                        vertDist=vertDist, 
                                        bodyyaw=bodyyaw,
                                        bodyroll=bodyroll)

        fig = plot_sections_plotly(fig, animal3d_instance, colour, alpha)
    
        # Set axis visibility
        if not axisOn:
            fig.update_layout(scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            ))
        
        fig = plot_settings_plotly(fig, animal3d_instance)

        # Restore the original state
        animal3d_instance.current_shape = current_state
    
    return fig


def plot_compare_plotly(animal3d_instance, 
                       keypoints_list,
                       alpha=0.3, 
                       colours=None, 
                       horzDist=None, 
                       bodypitch=None, 
                       vertDist=None,
                       bodyyaw=None,
                       bodyroll=None,
                       axisOn=True):
    """
    Create a static 3D plot comparing multiple hawk poses using Plotly.
    
    Parameters:
    -----------
    animal3d_instance : Animal3D
        Instance of the Animal3D class
    keypoints_list : list
        List of keypoint arrays to compare
    alpha : float, optional
        Transparency of the plots
    colours : list, optional
        List of colours for each set of keypoints. If None, uses default colour scheme
    horzDist : float, optional
        Horizontal distance transformation
    bodypitch : float, optional
        Body pitch transformation
    vertDist : float, optional
        Vertical distance transformation
    bodyyaw : float, optional
        Body yaw transformation
    bodyroll : float, optional
        Body roll transformation
    axisOn : bool, optional
        Whether to show axes
    """
    # If colours not provided, use default colour scheme
    if colours is None:
        colours = [None, 'red']  # First shape default colour, second shape red
    
    # Store the current state
    current_state = animal3d_instance.current_shape.copy()
    
    # Create figure
    fig = go.Figure()
    
    # Plot each set of keypoints
    for idx, keypoints in enumerate(keypoints_list):
        animal3d_instance.reset_transformation()
        animal3d_instance.update_keypoints(keypoints)
        animal3d_instance.transform_keypoints(
            bodypitch=bodypitch,
            horzDist=horzDist,
            vertDist=vertDist, 
            bodyyaw=bodyyaw,
            bodyroll=bodyroll
        )
        
        # Plot sections and keypoints for this set
        fig = plot_sections_plotly(fig, animal3d_instance, 
                                 colour=colours[idx], alpha=alpha)
        fig = plot_keypoints_plotly(fig, animal3d_instance, 
                                  colour=colours[idx], alpha=1)
    
    # Set axis visibility
    if not axisOn:
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ))
    
    # Apply plot settings
    fig = plot_settings_plotly(fig, animal3d_instance)
    
    # Restore the original state
    animal3d_instance.current_shape = current_state
    
    return fig

def plot_partial_plotly(animal3d_instance, section_name=None, leg_number=None, colour='blue', alpha=1, 
                        axisOn=True, horzDist=None, vertDist=None, bodypitch=None, bodyroll=None, bodyyaw=None):
    """
    Plots a specific section or leg of the animal using Plotly, with full transformation and plot settings capabilities.
    
    Parameters:
    - fig : plotly.graph_objs.Figure
        The Plotly figure to add the section plot to.
    - animal3d_instance : Animal3D
        Instance of the Animal3D class.
    - section_name : str, optional
        The name of the body section to plot. If None, plots the entire animal.
    - leg_number : int, optional
        The leg number (1 to 8) to plot for spiders. If None, plots the specified section or entire animal.
    - colour : str, optional
        Colour for the plot.
    - alpha : float, optional
        Transparency of the plot.
    - axisOn : bool, optional
        Whether to show axes.
    - horzDist : float, optional
        Horizontal distance transformation.
    - vertDist : float, optional
        Vertical distance transformation.
    - bodypitch : float, optional
        Body pitch transformation.
    - bodyroll : float, optional
        Body roll transformation.
    - bodyyaw : float, optional
        Body yaw transformation.
    """
    # Store the current state
    current_state = animal3d_instance.current_shape.copy()
    
    # Apply transformations if provided
    animal3d_instance.reset_transformation()
    animal3d_instance.transform_keypoints(bodypitch=bodypitch,
                                          horzDist=horzDist,
                                          vertDist=vertDist, 
                                          bodyyaw=bodyyaw,
                                          bodyroll=bodyroll)


    # Create figure
    fig = go.Figure()

    # Plot the specified section or leg
    if leg_number is not None and any("leg" in key for key in animal3d_instance.skeleton_definition.body_sections):
        section_name = f"leg_{leg_number}"
        leg_marker_names = animal3d_instance.skeleton_definition.get_leg_marker_names(leg_number)
        leg_indices = animal3d_instance.skeleton_definition.get_marker_indices(leg_marker_names, animal3d_instance.csv_marker_names)
        fig = plot_keypoints_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, indices=leg_indices)
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, section_name=section_name)

    elif section_name is not None:
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha, section_name=section_name)
    else:
        fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha)

    # Set axis visibility
    if not axisOn:
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ))
    
    # Apply plot settings
    fig = plot_settings_plotly(fig, animal3d_instance)

    # Restore the original state
    animal3d_instance.current_shape = current_state

    return fig


def plot_keypoints_plotly(fig, animal3d_instance, colour='black', alpha=1, indices=None):
    """
    Plots keypoints of the animal using Plotly.
    
    Parameters:
    - fig : plotly.graph_objs.Figure
        The Plotly figure to add the keypoints to.
    - animal3d_instance : Animal3D
        Instance of the Animal3D class.
    - colour : str, optional
        Colour for the keypoints.
    - alpha : float, optional
        Transparency of the keypoints.
    - indices : list, optional
        Specific indices of keypoints to plot. If None, plot all keypoints.
    """
    if indices is None:
        indices = animal3d_instance.marker_index

    # Extract keypoint coordinates
    coords = animal3d_instance.current_shape[:, indices, :][0]

    # Add keypoints scatter plot to the figure
    scatter = go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=2.5, color=colour, opacity=alpha), 
        hoverinfo='none'
    )
    fig.add_trace(scatter)

    return fig

def plot_sections_plotly(fig, animal3d_instance, colour, alpha=1, section_name=None):
    """
    Plots sections of the animal using Plotly.
    
    Parameters:
    - fig : plotly.graph_objs.Figure
        The Plotly figure to add the sections to.
    - animal3d_instance : Animal3D
        Instance of the Animal3D class.
    - colour : str
        Colour for the plot.
    - alpha : float, optional
        Transparency of the plot.
    - section_name : str, optional
        The name of the body section to plot. If None, plot all sections.
    """
    if section_name is not None:
        # Plot a specific body section
        if section_name not in animal3d_instance.skeleton_definition.body_sections:
            raise ValueError(f"Section name {section_name} not recognized.")
        
        mesh, lines = get_polygon_plotly(animal3d_instance, section_name, colour, alpha)
        if mesh is not None:
            fig.add_trace(mesh)
        fig.add_trace(lines)
    else:
        # Plot all sections
        for section in animal3d_instance.skeleton_definition.body_sections.keys():
            mesh, lines = get_polygon_plotly(animal3d_instance, section, colour, alpha)
            if mesh is not None:
                fig.add_trace(mesh)
            fig.add_trace(lines)

    return fig

def get_polygon_plotly(animal3d_instance, section_name, colour, alpha=1):
    if section_name not in animal3d_instance.skeleton_definition.body_sections.keys():
        raise ValueError(f"Section name {section_name} not recognised.")

    # You can modify these functions to customize colour and alpha
    colour = colour_polygon_plotly(section_name, colour)
    alpha = alpha_polygon_plotly(section_name, alpha)

    # Extract polygon coordinates
    coords = animal3d_instance.get_polygon_coords(section_name)

    # Construct a mesh plotly object
    if "leg" in section_name:
        mesh = None
    else:
        mesh = go.Mesh3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        color=colour, opacity=alpha, hoverinfo='none'
    )

    # Add the first point to the end to close the polygon
    coords = np.vstack([coords, coords[0]])

    # Draw the lines too
    lines = go.Scatter3d(
                   x=coords[:, 0],
                   y=coords[:, 1],
                   z=coords[:, 2],
                   mode='lines',
                   name='',
                   line=dict(color= 'grey', width=1.5), hoverinfo='none'
                )  

    return mesh, lines

def alpha_polygon_plotly(section_name, alpha):

        # The alpha of the polygon is determined by whether the landmarks are
        # estimated or measured.
        if "handwing" in section_name or "tail" in section_name:
            alpha = 0.5
        else:
            alpha = 0.3

        return alpha

def colour_polygon_plotly(section_name, colour):
        
        if colour is None:
            colour = 'lightblue'
    
        # The colour of the polygon is determined by whether the landmarks are
        # estimated or measured.
        if "handwing" in section_name or "tail" in section_name:
            colour = colour
        else:
            colour = 'grey'
    
        return colour

def plot_settings_plotly(fig, animal3d_instance):
    """
    Updates the plot settings for a static plot using the animal's natural size
    to determine appropriate axis scaling, handling all axes consistently.
    """
    # Calculate axis limits
    fixed_range = calculate_axis_limits(animal3d_instance)

    # Update axes
    axes_config = dict(
        gridcolor="grey",
        zerolinecolor="grey",
        showbackground=True,
        backgroundcolor="white",
        gridwidth=0.5,
        dtick=fixed_range[0][1] / 2  # Set grid lines to be consistent with view scale
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=fixed_range[0], **axes_config),
            yaxis=dict(range=fixed_range[1], **axes_config),
            zaxis=dict(range=fixed_range[2], **axes_config),
            aspectmode='cube',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=False
    )

    return fig
