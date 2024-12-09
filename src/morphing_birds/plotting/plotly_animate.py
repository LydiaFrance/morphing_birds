import numpy as np
import plotly.graph_objs as go
import imageio
import os
import plotly.io as pio


from .animation_frame_helpers import format_keypoint_frames, check_transformation_frames
from .plotly_helpers import calculate_axis_limits
from .plotly_plots import plot_sections_plotly, plot_keypoints_plotly


def animate_plotly(animal3d_instance, 
                   keypoints_frames,
                   alpha=0.3, 
                   colour=None, 
                   horzDist_frames=None, 
                   bodypitch_frames=None, 
                   vertDist_frames=None, 
                   bodyyaw_frames=None,
                   bodyroll_frames=None,
                   score_vals=None):
    """
    Create an animated 3D plot of a hawk video using Plotly.
    """
    
    def create_frames(animal3d_instance, keypoints_frames, horzDist_frames, vertDist_frames, bodypitch_frames, colour, alpha):
        frames = []

        # Store the current scaling factors
        # current_scaling = animal3d_instance.fixed_marker_scaling.copy()
    
        for frame in range(len(keypoints_frames)):
            animal3d_instance.reset_transformation()
            animal3d_instance.update_keypoints(keypoints_frames[frame])
            # animal3d_instance.set_fixed_marker_scaling(current_scaling)
            animal3d_instance.transform_keypoints(bodypitch=bodypitch_frames[frame],
                                                horzDist=horzDist_frames[frame],
                                                vertDist=vertDist_frames[frame],
                                                bodyyaw=bodyyaw_frames[frame],
                                                bodyroll=bodyroll_frames[frame])

            fig = go.Figure()
            fig = plot_sections_plotly(fig, animal3d_instance, colour=colour, alpha=alpha)
            fig = plot_keypoints_plotly(fig, animal3d_instance, colour = "lightblue" if colour is None else colour, alpha=1)

            # Apply the updated plot settings
            fig = plot_settings_animateplotly(fig, animal3d_instance)

            frames.append(go.Frame(
                data=fig.data,
                layout=fig.layout,  # Include layout in frame
                name=str(frame)
            ))
        return frames
    

    
    # Check dimensions and mirror the keypoints if only the right is given.
    keypoints_frames = format_keypoint_frames(animal3d_instance, keypoints_frames)

    if keypoints_frames.shape[0] == 0:
        raise ValueError("No frames to animate. Check the keypoints_frames input.")

    # Find the number of frames 
    num_frames = keypoints_frames.shape[0]


    # Check transformation frames
    horzDist_frames = check_transformation_frames(num_frames, horzDist_frames)
    vertDist_frames = check_transformation_frames(num_frames, vertDist_frames)
    bodypitch_frames = check_transformation_frames(num_frames, bodypitch_frames)
    bodyyaw_frames = check_transformation_frames(num_frames, bodyyaw_frames)
    bodyroll_frames = check_transformation_frames(num_frames, bodyroll_frames)

    limit = keypoints_frames.max().round(2)
    fixed_range = [-limit, limit]  # Adjust these limits based on your data


    # Create frames for the animation
    frames = create_frames(animal3d_instance, keypoints_frames, horzDist_frames, vertDist_frames, bodypitch_frames, colour, alpha)

    # Create the initial figure
    initial_fig = go.Figure(data=frames[0].data)
    initial_fig.frames = frames

    # Apply plot settings to the initial figure
    initial_fig = plot_settings_animateplotly(initial_fig, animal3d_instance)

    if score_vals is not None:
        slider_vals = score_vals.round(2)
    else:
        slider_vals = range(num_frames)

    initial_fig.update_layout(
        updatemenus=[create_play_button()],
        sliders=[create_slider(num_frames, slider_vals)],
        width=800,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.1, 1]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
        )
    )
    # Set dynamic axis limits based on the current shape
    initial_fig = plot_settings_animateplotly(initial_fig, animal3d_instance)
    

    return initial_fig

def animate_plotly_compare(animal3d_instance, 
                         keypoints_frames_list,
                         alpha=0.3, 
                         colours=None, 
                         horzDist_frames_list=None, 
                         bodypitch_frames_list=None, 
                         vertDist_frames_list=None, 
                         bodyyaw_frames_list=None,
                         bodyroll_frames_list=None,
                         score_vals=None):
    """
    Create an animated 3D plot comparing multiple hawk animations using Plotly.
    
    Parameters:
    -----------
    animal3d_instance : Animal3D
        Instance of the Animal3D class
    keypoints_frames_list : list
        List of keypoint frame arrays to compare
    alpha : float, optional
        Transparency of the plots
    colours : list, optional
        List of colours for each set of keypoints
    horzDist_frames_list : list of array-like, optional
        List of horizontal distance transformations for each keypoint set
    bodypitch_frames_list : list of array-like, optional
        List of body pitch transformations for each keypoint set
    vertDist_frames_list : list of array-like, optional
        List of vertical distance transformations for each keypoint set
    bodyyaw_frames_list : list of array-like, optional
        List of body yaw transformations for each keypoint set
    bodyroll_frames_list : list of array-like, optional
        List of body roll transformations for each keypoint set
    score_vals : array-like, optional
        Values to show in the slider
    """
    
    def create_comparison_frames(animal3d_instance, keypoints_frames_list, horzDist_frames_list, 
                               vertDist_frames_list, bodypitch_frames_list, bodyyaw_frames_list, bodyroll_frames_list, colours, alpha):
        frames = []
        
        # If colours not provided, use default colour scheme
        if colours is None:
            colours = [None, 'red']  # First shape default colour, second shape red
        
        # Format each set of keypoints
        formatted_keypoints_list = []
        for keypoints_frames in keypoints_frames_list:
            keypoints = format_keypoint_frames(animal3d_instance, keypoints_frames)
            if keypoints.shape[0] == 0:
                raise ValueError("No frames to animate in one of the keypoint sets.")
            formatted_keypoints_list.append(keypoints)
        
        # Verify all keypoint sets have same number of frames
        num_frames = formatted_keypoints_list[0].shape[0]
        if not all(kp.shape[0] == num_frames for kp in formatted_keypoints_list):
            raise ValueError("All keypoint sets must have the same number of frames.")
        # Check transformation frames
        if horzDist_frames_list:
            horzDist_frames_list = [check_transformation_frames(num_frames, frames) for frames in horzDist_frames_list]
        if vertDist_frames_list:
            vertDist_frames_list = [check_transformation_frames(num_frames, frames) for frames in vertDist_frames_list]
        if bodypitch_frames_list:
            bodypitch_frames_list = [check_transformation_frames(num_frames, frames) for frames in bodypitch_frames_list]
        if bodyyaw_frames_list:
            bodyyaw_frames_list = [check_transformation_frames(num_frames, frames) for frames in bodyyaw_frames_list]
        if bodyroll_frames_list:
            bodyroll_frames_list = [check_transformation_frames(num_frames, frames) for frames in bodyroll_frames_list]
           

        # Create frames
        for frame in range(num_frames):
            fig = go.Figure()
            
            # Plot each set of keypoints
            for idx, keypoints in enumerate(formatted_keypoints_list):
                animal3d_instance.reset_transformation()
                animal3d_instance.update_keypoints(keypoints[frame])
                
                # Get transformations for this keypoint set if available
                horz = horzDist_frames_list[idx][frame] if horzDist_frames_list else 0
                vert = vertDist_frames_list[idx][frame] if vertDist_frames_list else 0
                pitch = bodypitch_frames_list[idx][frame] if bodypitch_frames_list else 0
                bodyyaw = bodyyaw_frames_list[idx][frame] if bodyyaw_frames_list else 0
                bodyroll = bodyroll_frames_list[idx][frame] if bodyroll_frames_list else 0
                animal3d_instance.transform_keypoints(
                    bodypitch=pitch,
                    horzDist=horz,
                    vertDist=vert,
                    bodyyaw=bodyyaw,
                    bodyroll=bodyroll
                )
                
                # Plot sections and keypoints for this set
                fig = plot_sections_plotly(fig, animal3d_instance, 
                                         colour=colours[idx], alpha=alpha)
                fig = plot_keypoints_plotly(fig, animal3d_instance, 
                                          colour=colours[idx], alpha=1)
            
            # Apply plot settings
            fig = plot_settings_animateplotly(fig, animal3d_instance)
            
            frames.append(go.Frame(
                data=fig.data,
                layout=fig.layout,
                name=str(frame)
            ))
        return frames

    # Create frames for animation
    frames = create_comparison_frames(
        animal3d_instance, keypoints_frames_list, 
        horzDist_frames_list, vertDist_frames_list, bodypitch_frames_list, 
        bodyyaw_frames_list, bodyroll_frames_list, colours, alpha
    )

    # Create initial figure
    initial_fig = go.Figure(data=frames[0].data)
    initial_fig.frames = frames

    # Apply plot settings
    initial_fig = plot_settings_animateplotly(initial_fig, animal3d_instance)

    num_frames = len(frames)

    # Set up slider values
    if score_vals is not None:
        slider_vals = score_vals.round(2)
    else:
        slider_vals = range(num_frames)

    # Add animation controls
    initial_fig.update_layout(
        updatemenus=[create_play_button()],
        sliders=[create_slider(num_frames, slider_vals)],
        width=800,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.1, 1]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
        )
    )

    return initial_fig


def animate_plotly_partial(animal3d_instance, keypoints_frames, section_name=None, leg_number=None, colour='blue', alpha=1, 
                           horzDist_frames=None, vertDist_frames=None, bodypitch_frames=None, bodyyaw_frames=None, bodyroll_frames=None, 
                           score_vals=None):
    """
    Animates a specific section or leg of the animal using Plotly, with full transformation and plot settings capabilities.
    
    Parameters:
    - animal3d_instance : Animal3D
        Instance of the Animal3D class.
    - keypoints_frames : list
        List of keypoint frames for animation.
    - section_name : str, optional
        The name of the body section to animate. If None, animates the entire animal.
    - leg_number : int, optional
        The leg number (1 to 8) to animate for animals with legs. If None, animates the specified section or entire animal.
    - colour : str, optional
        Colour for the animation.
    - alpha : float, optional
        Transparency of the animation.
    - horzDist_frames : list, optional
        List of horizontal distance transformations for each frame.
    - vertDist_frames : list, optional
        List of vertical distance transformations for each frame.
    - bodypitch_frames : list, optional
        List of body pitch transformations for each frame.
    - bodyyaw_frames : list, optional
        List of body yaw transformations for each frame.
    - bodyroll_frames : list, optional
        List of body roll transformations for each frame.
    - score_vals : array-like, optional
        Values to show in the slider.
    """
    frames = []

    for frame_idx, keypoints in enumerate(keypoints_frames):
        animal3d_instance.reset_transformation()
        animal3d_instance.update_keypoints(keypoints)

        # Apply transformations for the current frame
        animal3d_instance.transform_keypoints(
            bodypitch=bodypitch_frames[frame_idx] if bodypitch_frames else None,
            horzDist=horzDist_frames[frame_idx] if horzDist_frames else None,
            vertDist=vertDist_frames[frame_idx] if vertDist_frames else None,
            bodyyaw=bodyyaw_frames[frame_idx] if bodyyaw_frames else None,
            bodyroll=bodyroll_frames[frame_idx] if bodyroll_frames else None
        )

        fig = go.Figure()

        # Check if the instance has sections containing "leg" and animate a specific leg if requested
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

        frames.append(go.Frame(data=fig.data, name=str(frame_idx)))

    # Create the initial figure
    initial_fig = go.Figure(data=frames[0].data)
    initial_fig.frames = frames

    # Apply plot settings
    initial_fig = plot_settings_animateplotly(initial_fig, animal3d_instance)

    # Set up slider values
    num_frames = len(frames)
    if score_vals is not None:
        slider_vals = score_vals.round(2)
    else:
        slider_vals = range(num_frames)

    # Add animation controls
    initial_fig.update_layout(
        updatemenus=[create_play_button()],
        sliders=[create_slider(num_frames, slider_vals)],
        width=800,
        height=700,
        margin=dict(l=50, r=50, t=100, b=100),
        scene=dict(
            domain=dict(x=[0, 1], y=[0.1, 1]),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
        )
    )

    return initial_fig



def save_plotly_animation(fig, filename, format='gif', fps=10, width=800, height=700):
    """
    Save a plotly animation as either GIF or HTML.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The plotly figure containing the animation
    filename : str
        Path to save the file (include extension .gif or .html)
    format : str
        'gif' or 'html'
    fps : int
        Frames per second for GIF output
    width : int
        Width of the output in pixels
    height : int
        Height of the output in pixels
    """
    if format.lower() == 'gif':
        # Create a temporary directory to store images
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Export each frame as a static image
        images = []
        for frame in fig.frames:
            fig.update(data=frame.data)
            image_path = os.path.join(temp_dir, f"frame_{frame.name}.png")
            fig.write_image(image_path, width=width, height=height)
            images.append(image_path)
        
        # Create a GIF from the images
        with imageio.get_writer(filename, mode='I', fps=fps) as writer:
            for image in images:
                writer.append_data(imageio.imread(image))
        
        # Clean up temporary images
        for image in images:
            os.remove(image)
        os.rmdir(temp_dir)
        
        print(f"Animation saved as GIF: {filename}")
        
    elif format.lower() == 'html':
        # Save as HTML
        fig.write_html(
            filename,
            auto_play=False,
            include_plotlyjs=True
        )
        print(f"Animation saved as HTML: {filename}")
        
    else:
        raise ValueError("Format must be either 'gif' or 'html'")
        

def plot_settings_animateplotly(fig, animal3d_instance):
    """
    Updates the plot settings for each frame of the animation using the animal's natural size
    to determine appropriate axis scaling, handling all axes consistently.
    Preserves user's camera position between frames.
    """
    # Get current camera position if it exists
    current_camera = None
    if fig.layout.scene and fig.layout.scene.camera:
        current_camera = fig.layout.scene.camera

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

    # Prepare the scene dictionary
    scene_dict = dict(
        xaxis=dict(range=fixed_range[0], **axes_config),
        yaxis=dict(range=fixed_range[1], **axes_config),
        zaxis=dict(range=fixed_range[2], **axes_config),
        aspectmode='cube',
        aspectratio=dict(x=1, y=1, z=1)
    )

    # Only set the camera if it hasn't been modified by the user
    if current_camera is None:
        scene_dict['camera'] = dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            up=dict(x=0, y=0, z=1)
        )
    else:
        scene_dict['camera'] = current_camera

    fig.update_layout(
        scene=scene_dict,
        margin=dict(r=10, l=10, b=10, t=10),
        showlegend=False
    )

    return fig




def create_slider(num_frames,slider_vals):
    # Check if slider_vals is the same length as num_frames
    if slider_vals is None:
        slider_vals = range(num_frames)
        if len(slider_vals) != num_frames:
            raise ValueError("slider_vals must be the same length as num_frames.")
    
    return {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 12},
            'prefix': 'Frame:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {'args': [[ii], {'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}}],
            'label': str(slider_vals[ii]),
            'method': 'animate'} for ii in range(num_frames)
        ]
    }
    
def create_play_button():
    return {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 10},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'left',
        'y': 1.1,
        'yanchor': 'top'
    }



