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

SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()

hawk3d = Hawk3D(SCRIPT_DIR.parents[3] / "data/mean_hawk_shape.csv")

principal_components = np.load(
    SCRIPT_DIR.parents[3] / "data/website_principal_components.npy"
)
print(f"principal_components.shape {principal_components.shape}")
left_score_frames = np.load(SCRIPT_DIR.parents[3] / "data/Left_scores_RightTurn.npy")
right_score_frames = np.load(SCRIPT_DIR.parents[3] / "data/Right_scores_RightTurn.npy")
mu = np.load(SCRIPT_DIR.parents[3] / "data/website_mu.npy")
# copy the score frames data into a matrix with 20 frames
# score_frames = np.tile(score_frames, (20, 1))
alpha = 0.3
colour_list = [
    "#B5E675",
    "#6ED8A9",
    "#51B3D4",
    "#4579AA",
    "#F19EBA",
    "#BC96C9",
    "#917AC2",
    "#BE607F",
    "#624E8B",
    "#888888",
    "#888888",
    "#888888",
]
n_frames = 149
n_markers = 4
n_dims = 3
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
        print(f"reconstructed.shape {reconstructed.shape}")
    return reconstructed


def initialize_figure():
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
    )
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
    return fig


def generate_frames(
    hawk3d,
    predefined_combinations,
    principal_components,
    colour_list,
    n_frames,
    alpha,
):
    all_frames = []
    initial_combo_name = predefined_combinations[0]["label"]
    initial_data = []
    initial_y_min = None
    initial_y_max = None

    for combo in predefined_combinations:
        components_list = combo["components"]
        reconstructed_frames_left = reconstruct_frames(
            components_list,
            n_frames,
            mu,
            principal_components,
            left_score_frames,
        )
        reconstructed_frames_left[:, :, 0] *= -1

        reconstructed_frames_right = reconstruct_frames(
            components_list,
            n_frames,
            mu,
            principal_components,
            right_score_frames,
        )
        reconstructed_full = np.zeros((reconstructed_frames_left.shape[0], 8, 3))
        # Fill alternating indices
        reconstructed_full[:,::2,:] = reconstructed_frames_left
        reconstructed_full[:,1::2,:] = reconstructed_frames_right

        component_scores_centered = center_scores(left_score_frames[:, components_list[0]])
        y_min, y_max = component_scores_centered.min(), component_scores_centered.max()
        frames = create_frames_for_combo(
            hawk3d,
            combo,
            reconstructed_full,
            component_scores_centered,
            colour_list,
            n_frames,
            alpha,
        )
        all_frames.extend(frames)
        if combo["label"] == initial_combo_name:
            initial_data = [*frames[0].data]
            initial_y_min = y_min
            initial_y_max = y_max

    return all_frames, initial_data, initial_y_min, initial_y_max


def center_scores(scores):
    return scores - np.mean(scores)


def create_frames_for_combo(
    hawk3d,
    combo,
    reconstructed_frames,
    component_scores_centered,
    colour_list,
    n_frames,
    alpha,
):
    frames = []
    for i in range(n_frames):
        hawk3d.reset_transformation()
        hawk3d.restore_keypoints_to_average()
        hawk3d.update_keypoints(reconstructed_frames[i])
        selected_colour = colour_list[combo["components"][0]]
        scatter3d_traces = create_scatter3d_traces(hawk3d, selected_colour, alpha)
        line_plot = create_line_plot(
            component_scores_centered, selected_colour, n_frames
        )
        current_frame_marker = create_current_frame_marker(i, component_scores_centered)
        frame_layout = create_frame_layout(combo, scatter3d_traces)
        frame_data = [*list(scatter3d_traces), line_plot, current_frame_marker]
        frame = go.Frame(
            data=frame_data,
            name=f"{combo['label']}_frame_{i}",
            layout=frame_layout,
        )
        frames.append(frame)
        update_frame_layout(frames)
    return frames


def create_scatter3d_traces(hawk3d, selected_colour, alpha):
    scatter3d = go.Figure()
    scatter3d = plot_sections_plotly(
        scatter3d, hawk3d, colour=selected_colour, alpha=alpha
    )
    scatter3d = plot_keypoints_plotly(
        scatter3d, hawk3d, colour=selected_colour, alpha=1
    )
    scatter3d = plot_settings_animateplotly(scatter3d, hawk3d)
    return scatter3d.data


def create_line_plot(component_scores_centered, selected_colour, n_frames):
    return go.Scatter(
        x=np.arange(n_frames),
        y=component_scores_centered,
        mode="lines",
        xaxis="x2",
        yaxis="y2",
        showlegend=False,
        line={"color": selected_colour},
    )


def create_current_frame_marker(i, component_scores_centered):
    return go.Scatter(
        x=[i],
        y=[component_scores_centered[i]],
        mode="markers",
        xaxis="x2",
        yaxis="y2",
        marker={"color": "red", "size": 10},
        showlegend=False,
    )


def create_frame_layout(combo, scatter3d_traces):
    return go.Layout(
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
        scene=scatter3d_traces[0].scene,
    )


def update_frame_layout(frames):
    for frame in frames:
        frame.layout.update(
            scene={
                "xaxis": {"range": [-0.6, 0.6], "autorange": False},
                "yaxis": {"range": [-0.6, 0.6], "autorange": False},
                "zaxis": {"range": [-0.6, 0.6], "autorange": False},
            }
        )


def create_component_buttons(predefined_combinations, n_frames):
    component_buttons = []
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
    return component_buttons


def create_play_pause_buttons():
    return [
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


def update_layout(
    fig, component_buttons, play_pause_buttons, initial_y_min, initial_y_max
):
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


def create_create_components_plot(
    hawk3d,
    predefined_combinations,
    principal_components,
    colour_list,
    n_frames,
    alpha=0.3,
):
    fig = initialize_figure()
    all_frames, initial_data, initial_y_min, initial_y_max = generate_frames(
        hawk3d,
        predefined_combinations,
        principal_components,
        colour_list,
        n_frames,
        alpha,
    )
    fig.frames = all_frames
    for i_data in initial_data:
        fig.add_trace(i_data)
    component_buttons = create_component_buttons(predefined_combinations, n_frames)
    play_pause_buttons = create_play_pause_buttons()
    update_layout(
        fig, component_buttons, play_pause_buttons, initial_y_min, initial_y_max
    )
    return fig


def main():
    components_plot = create_create_components_plot(
        hawk3d,
        predefined_combinations,
        principal_components,
        colour_list,
        n_frames,
    )
    plotly_jinja_data = {
        "components_plot": components_plot.to_html(
            full_html=False, include_plotlyjs=False
        ),
    }
    # Save the figure as an HTML file
    with (SCRIPT_DIR / "straight.html").open("w", encoding="utf-8") as output_file, (
        SCRIPT_DIR / "template.html"
    ).open() as template_file:
        j2_template = Template(template_file.read())
        output_file.write(j2_template.render(plotly_jinja_data))


if __name__ == "__main__":
    main()
