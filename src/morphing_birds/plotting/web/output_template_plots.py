import pathlib

import numpy as np
import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from morphing_birds import (
    Hawk3D,
    plot_keypoints_plotly,
    plot_sections_plotly,
    plot_settings_animateplotly,
)

# get directory of file using pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()

hawk3d = Hawk3D(SCRIPT_DIR.parents[3] / "data/mean_hawk_shape.csv")

# read in principal components, scores and mu
principal_components = np.load(SCRIPT_DIR.parents[3] / "data/website_principal_components.npy")
score_frames = np.load(SCRIPT_DIR.parents[3] / "data/website_score_10frames.npy")
mu = np.load(SCRIPT_DIR.parents[3] / "data/website_mu.npy")
# copy the score frames data into a matrix with 20 frames
score_frames = np.tile(score_frames, (20, 1))


# def create_fake_pca_data(hawk3d, n_samples=20, n_components=12, n_markers=4, n_dims=3):
#     # Simulate the mean shape of the hawk (mu)
#     mu = hawk3d.left_markers.copy()

#     # Generate an orthogonal matrix for principal components
#     principal_components = ortho_group.rvs(dim=n_markers * n_dims)

#     # Simulate decreasing variances for principal components
#     explained_variance = np.linspace(2, 0.1, n_components)

#     # Adjust principal components by the square root of variances
#     principal_components = principal_components[:n_components] * np.sqrt(
#         explained_variance[:, np.newaxis]
#     )

#     # Simulate score frames with temporal dynamics
#     time = np.linspace(0, 4 * np.pi, n_samples)
#     score_frames = np.zeros((n_samples, n_components))

#     for i in range(n_components):
#         amplitude = np.exp(
#             -i / n_components
#         )  # Decreasing amplitude for higher components
#         frequency = i + 1
#         score_frames[:, i] = amplitude * np.sin(frequency * time)

#     # Reconstruct frames using the simulated principal components and scores
#     components_list = list(range(n_components))  # Use all components or a subset
#     selected_PCs = principal_components[components_list, :]
#     selected_scores = score_frames[:, components_list]

#     reconstruction = np.dot(selected_scores, selected_PCs)
#     reconstruction = reconstruction.reshape(-1, n_markers, n_dims)
#     reconstructed_frames = mu + reconstruction

#     return reconstructed_frames, principal_components, score_frames


# # Call the function to create fake PCA data
# reconstructed_frames, principal_components, score_frames = create_fake_pca_data(hawk3d)


def reconstruct_frames(
    selected_components,
    n_frames,
    mu,
    principal_components,
    score_frames,
):
    if not selected_components:
        # If no components are selected, use the mean shape for all frames
        reconstructed = np.repeat(mu, n_frames, axis=0)
    else:
        selected_PCs = principal_components[selected_components, :]
        selected_scores = score_frames[:, selected_components]
        reconstruction = np.dot(selected_scores, selected_PCs)
        reconstruction = reconstruction.reshape(-1, n_markers, n_dims)
        reconstructed = mu + reconstruction
    return reconstructed


# Create a plot that allows selecting principal components
# Parameters
alpha = 0.3
colour = "lightblue"
n_frames = 20
n_markers = 4  # Define the number of markers
n_dims = 3  # Define the number of dimensions
# Define the predefined combinations
predefined_combinations = [
    {"label": "PC 1", "components": [0]},
    {"label": "PC 2", "components": [1]},
    {"label": "PC 3", "components": [2]},
    {"label": "PC 4", "components": [3]},
    {"label": "PC 5", "components": [4]},
    {"label": "PC 6", "components": [5]},
    {"label": "PC 7", "components": [6]},
    {"label": "PC 8", "components": [7]},
    {"label": "PC 9", "components": [8]},
    {"label": "PC 10", "components": [9]},
    {"label": "PC 11", "components": [10]},
    {"label": "PC 12", "components": [11]},
]


def create_create_components_plot(
    hawk3d,
    predefined_combinations,
    principal_components,
    score_frames,
    n_frames=20,
    alpha=0.3,
    colour="lightblue",
):
    component_buttons = []
    play_pause_buttons = []
    all_frames = []
    # Initialize the figure with subplots
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
    )

    # Define the domains for the main plot and the inset plot
    fig.update_layout(
        scene={
            "domain": {"x": [0.1, 0.9], "y": [0.1, 0.8]},
            "aspectmode": "cube",
            "xaxis": {"range": [-0.5, 0.5], "autorange": False},
            "yaxis": {"range": [-0.5, 0.5], "autorange": False},
            "zaxis": {"range": [-0.5, 0.5], "autorange": False},
        },
        xaxis2={
            "domain": [0.8, 1],
            "anchor": "y2",
        },
        yaxis2={
            "domain": [0.8, 1],
            "anchor": "x2",
        },
        showlegend=False,
    )

    initial_combo_name = predefined_combinations[0]["label"]
    for combo in predefined_combinations:
        components_list = combo["components"]
        reconstructed_frames = reconstruct_frames(
            components_list,
            n_frames,
            mu,
            principal_components,
            score_frames,
        )

        # Prepare data for the line plot (component value vs. time)
        component_scores = score_frames[
            :, components_list[0]
        ]  # Assuming one component at a time

        # Center the scores around zero
        component_scores_centered = component_scores - np.mean(component_scores)

        # Calculate global min and max for y-axis of the line plot
        y_min = component_scores_centered.min()
        y_max = component_scores_centered.max()

        frames = []
        for i in range(n_frames):
            hawk3d.reset_transformation()
            hawk3d.restore_keypoints_to_average()
            hawk3d.update_keypoints(reconstructed_frames[i])

            # Create the main 3D plot data
            scatter3d = go.Figure()
            scatter3d = plot_sections_plotly(
                scatter3d, hawk3d, colour=colour, alpha=alpha
            )
            scatter3d = plot_keypoints_plotly(scatter3d, hawk3d, colour=colour, alpha=1)
            scatter3d = plot_settings_animateplotly(scatter3d, hawk3d)
            scatter3d_traces = scatter3d.data

            # Create the line plot data
            line_plot = go.Scatter(
                x=np.arange(n_frames),
                y=component_scores_centered,
                mode="lines",
                xaxis="x2",
                yaxis="y2",
                showlegend=False,
                line={"color": "blue"},
            )

            # Add a marker to indicate the current frame
            current_frame_marker = go.Scatter(
                x=[i],
                y=[component_scores_centered[i]],
                mode="markers",
                xaxis="x2",
                yaxis="y2",
                marker={"color": "red", "size": 10},
                showlegend=False,
            )

            # Set frame-specific layout updates including title and y-axis label
            frame_layout = go.Layout(
                title={
                    "text": f"Selected Component - {combo['label']}",
                    "xanchor": "center",
                    "yanchor": "top",
                    "x": 0.5,
                    "y": 0.9,
                },
                xaxis2={"title": "Frame", "domain": [0.8, 0.95], "anchor": "y2"},
                yaxis2={
                    "title": f"{combo['label']} value",
                    "domain": [0.8, 0.95],
                    "anchor": "x2",
                },
                scene=scatter3d.layout.scene,
            )

            frame_data = [*list(scatter3d_traces), line_plot, current_frame_marker]
            # Create the frame
            frame = go.Frame(
                data=frame_data,
                name=f"{combo['label']}_frame_{i}",
                layout=frame_layout,
            )
            frames.append(frame)

        all_frames.extend(frames)

        # Save initial data for the first frame
        if combo["label"] == initial_combo_name:
            initial_data = [*frames[0].data]
            initial_y_min = y_min
            initial_y_max = y_max

    for i_data in initial_data:
        fig.add_trace(i_data)

    fig.frames = all_frames

    # Create buttons for component selection
    for combo in predefined_combinations:
        frame_names = [f"{combo['label']}_frame_{i}" for i in range(n_frames)]
        button = {
            "label": combo["label"],
            "method": "animate",
            "args": [
                frame_names,
                {
                    "frame": {"duration": 100, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                    "fromcurrent": True,
                },
            ],
        }
        component_buttons.append(button)

    play_pause_buttons = [
        {
            "args": [
                None,
                {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"},
            ],
            "label": "Play All",
            "method": "animate",
        },
        {
            "args": [
                [None],
                {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
            ],
            "label": "Pause",
            "method": "animate",
        },
    ]

    # Update layout with buttons and sliders
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": component_buttons,
                "x": 0,
                "y": 0.9,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": True,
            },
            {
                "type": "buttons",
                "buttons": play_pause_buttons,
                "x": 0,
                "y": 0.05,
                "xanchor": "left",
                "direction": "left",
                "yanchor": "bottom",
                "showactive": True,
            },
        ],
        width=800,
        height=700,
        margin={"l": 50, "r": 100, "t": 100, "b": 50},
    )

    # Adjust axes for the inset plot
    fig.update_layout(
        xaxis2={
            "domain": [0.8, 0.95],
            "anchor": "y2",
            "showgrid": False,
            "zeroline": True,
            "showticklabels": True,
            "title": "Frame",
        },
        yaxis2={
            "domain": [0.8, 0.95],
            "anchor": "x2",
            "showgrid": False,
            "zeroline": True,
            "showticklabels": True,
            "title": "Component Value",
            "range": [initial_y_min, initial_y_max],
        },
    )

    return fig


# Call the function to create the animated figure
components_plot = create_create_components_plot(
    hawk3d, predefined_combinations, principal_components, score_frames
)

plotly_jinja_data = {
    "components_plot": components_plot.to_html(full_html=False, include_plotlyjs=False),
}
# Save the figure as an HTML file
with (SCRIPT_DIR / "index.html").open("w", encoding="utf-8") as output_file, (
    SCRIPT_DIR / "template.html"
).open() as template_file:
    j2_template = Template(template_file.read())
    output_file.write(j2_template.render(plotly_jinja_data))
