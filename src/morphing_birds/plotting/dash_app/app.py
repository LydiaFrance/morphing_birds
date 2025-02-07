import pathlib

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from scipy.stats import ortho_group

from morphing_birds import (
    Hawk3D,
    plot_keypoints_plotly,
    plot_sections_plotly,
    plot_settings_animateplotly,
)

# get directory of file using pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()

hawk3d = Hawk3D(SCRIPT_DIR.parents[3] / "data/mean_hawk_shape.csv")


def create_fake_pca_data(hawk3d, n_samples=20, n_components=12, n_markers=4, n_dims=3):
    mu = hawk3d.left_markers.copy()
    principal_components = ortho_group.rvs(dim=n_markers * n_dims)
    explained_variance = np.linspace(2, 0.1, n_components)
    principal_components = principal_components[:n_components] * np.sqrt(
        explained_variance[:, np.newaxis]
    )

    time = np.linspace(0, 4 * np.pi, n_samples)
    score_frames = np.zeros((n_samples, n_components))
    for i in range(n_components):
        amplitude = np.exp(-i / n_components)  # decreasing amplitude
        frequency = i + 1
        score_frames[:, i] = amplitude * np.sin(frequency * time)

    reconstruction = np.dot(score_frames, principal_components)
    reconstruction = reconstruction.reshape(-1, n_markers, n_dims)
    reconstructed_frames = mu + reconstruction
    return reconstructed_frames, principal_components, score_frames, mu


n_frames = 20
n_components = 12
n_markers = 4
n_dims = 3
alpha = 0.3
colour = "lightblue"

reconstructed_frames, principal_components, score_frames, mu = create_fake_pca_data(
    hawk3d,
    n_samples=n_frames,
    n_components=n_components,
    n_markers=n_markers,
    n_dims=n_dims,
)


def reconstruct_frames(
    selected_components,
    n_frames,
    mu,
    principal_components,
    score_frames,
    n_markers=4,
    n_dims=3,
):
    if not selected_components:
        # No components selected, return mean shape repeated
        return np.repeat(mu[np.newaxis, :, :], n_frames, axis=0), np.zeros(n_frames)

    # Extract the scores for selected PCs and sum them (or combine as needed)
    selected_scores = score_frames[:, selected_components]
    combined_scores = np.sum(selected_scores, axis=1)  # sum over selected PCs

    # Extract only the selected PCs from principal_components
    selected_PCs = principal_components[selected_components, :]

    # Reconstruct from selected PCs
    reconstruction = np.dot(selected_scores, selected_PCs)
    reconstruction = reconstruction.reshape(-1, n_markers, n_dims)
    frames = mu + reconstruction
    return frames, combined_scores


def create_figure(selected_components):
    # Recompute reconstructed frames and line plot data
    frames_data, combined_scores = reconstruct_frames(
        selected_components,
        n_frames,
        mu,
        principal_components,
        score_frames,
        n_markers,
        n_dims,
    )

    # Center the combined scores for plotting
    combined_scores_centered = combined_scores - np.mean(combined_scores)
    y_min = combined_scores_centered.min()
    y_max = combined_scores_centered.max()

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
        showlegend=False,
    )

    # Create frames for animation
    frames_list = []
    for i in range(n_frames):
        hawk3d.reset_transformation()
        hawk3d.update_keypoints(frames_data[i])

        scatter3d = go.Figure()
        scatter3d = plot_sections_plotly(scatter3d, hawk3d, colour=colour, alpha=alpha)
        scatter3d = plot_keypoints_plotly(scatter3d, hawk3d, colour=colour, alpha=1)
        scatter3d = plot_settings_animateplotly(scatter3d, hawk3d)
        scatter3d_traces = scatter3d.data

        line_plot = go.Scatter(
            x=np.arange(n_frames),
            y=combined_scores_centered,
            mode="lines",
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            line={"color": "blue"},
        )

        current_frame_marker = go.Scatter(
            x=[i],
            y=[combined_scores_centered[i]],
            mode="markers",
            xaxis="x2",
            yaxis="y2",
            marker={"color": "red", "size": 10},
            showlegend=False,
        )

        frame_layout = go.Layout(
            title={
                "text": f"Selected Components: {', '.join([str(s+1) for s in sorted(selected_components)]) if selected_components else 'None'}",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "y": 0.92,
            },
            xaxis2={"domain": [0.8, 0.95]},
            yaxis2={"domain": [0.8, 0.95]},
            scene=scatter3d.layout.scene,
        )

        frame_data = [*list(scatter3d_traces), line_plot, current_frame_marker]

        frame = go.Frame(
            data=frame_data,
            name=f"frame_{i}",
            layout=frame_layout,
        )
        frames_list.append(frame)

    # Use the first frame as initial data
    initial_data = [*frames_list[0].data]

    for i_data in initial_data:
        fig.add_trace(i_data)

    fig.frames = frames_list

    # Play/Pause buttons
    play_pause_buttons = [
        {
            "args": [
                None,
                {
                    "frame": {"duration": 100, "redraw": True},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                },
            ],
            "label": "Play",
            "method": "animate",
        },
        {
            "args": [
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                },
            ],
            "label": "Pause",
            "method": "animate",
        },
    ]

    # -- Add slider --
    # Create one slider with a step for each frame
    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }
    ]
    for i in range(n_frames):
        slider_step = {
            "label": str(i),
            "method": "animate",
            "args": [
                [f"frame_{i}"],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
        }
        sliders[0]["steps"].append(slider_step)

    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "buttons": play_pause_buttons,
                "x": 0,
                "y": 0,
                "xanchor": "left",
                "direction": "left",
                "yanchor": "bottom",
                "showactive": True,
            },
        ],
        sliders=sliders,
        xaxis2={
            "domain": [0.8, 0.95],
            "anchor": "y2",
            "title": "Frame",
        },
        yaxis2={
            "domain": [0.8, 0.95],
            "anchor": "x2",
            "title": "Combined PC Value",
            "range": [y_min, y_max],
        },
        width=800,
        height=700,
        margin={"l": 50, "r": 100, "t": 125, "b": 25},
    )

    return fig


app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Hawk Wing PCA", style={"textAlign": "center", "font-family": "Arial"}),
        html.Label(
            "Select Principal Components:",
            style={
                "font-family": "Arial",
                "display": "block",
                "margin": "auto",
                "width": "75%",
                "textAlign": "left",
            },
        ),
        dcc.Dropdown(
            id="pc-dropdown",
            options=[{"label": f"PC {i+1}", "value": i} for i in range(n_components)],
            value=[],  # start with no PCs selected
            multi=True,
            style={"width": "75%", "font-family": "Arial", "margin": "auto"},
        ),
        dcc.Graph(id="graph", style={"width": "75%", "margin": "auto"}),
    ]
)


@app.callback(Output("graph", "figure"), Input("pc-dropdown", "value"))
def update_plot(selected_components):
    fig = create_figure(selected_components)
    return fig


server = app.server  # For deployment

if __name__ == "__main__":
    app.run_server(debug=True)
