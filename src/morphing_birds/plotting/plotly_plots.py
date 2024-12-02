
import numpy as np
import plotly.graph_objs as go

from .plotly_helpers import calculate_axis_limits

def plot_plotly(Hawk3D_instance, colour='lightblue', alpha=0.3, axisOn=True):
    # Create a new 3D Plotly figure
    fig = go.Figure()

    # Plot sections and keypoints
    fig = plot_sections_plotly(fig, Hawk3D_instance, colour, alpha)
    fig = plot_keypoints_plotly(fig, Hawk3D_instance, colour, alpha)

    # Update camera view
    # fig.update_layout(scene_camera=dict(eye=dict(x=az, y=el, z=20)))

    # Set axis visibility
    if not axisOn:
        fig.update_layout(scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ))
    
    origin = Hawk3D_instance.origin
    fig = plot_settings_plotly(fig,Hawk3D_instance)


    return fig

def plot_keypoints_plotly(fig, Hawk3D_instance, colour='black', alpha=1):
    # Extract keypoint coordinates
    coords = Hawk3D_instance.current_shape[:, Hawk3D_instance.marker_index, :][0]

    # Add keypoints scatter plot to the figure
    scatter = go.Scatter3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        mode='markers',
        marker=dict(size=2.5, color=colour, opacity=1), 
        hoverinfo='none'
    )
    fig.add_trace(scatter)

    return fig

def plot_sections_plotly(fig, Hawk3D_instance, colour, alpha=1):
    for section in Hawk3D_instance.skeleton_definition.body_sections.keys():
        mesh, lines = get_polygon_plotly(Hawk3D_instance, section, colour, alpha)
        if mesh is not None:
            fig.add_trace(mesh)
        fig.add_trace(lines)

    return fig

def get_polygon_plotly(Hawk3D_instance, section_name, colour, alpha=1):
    if section_name not in Hawk3D_instance.skeleton_definition.body_sections.keys():
        raise ValueError(f"Section name {section_name} not recognised.")

    # You can modify these functions to customize colour and alpha
    colour = colour_polygon_plotly(section_name, colour)
    alpha = alpha_polygon_plotly(section_name, alpha)

    # Extract polygon coordinates
    coords = Hawk3D_instance.get_polygon_coords(section_name)

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
