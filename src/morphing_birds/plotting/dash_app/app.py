import pathlib

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from morphing_birds import (
    Hawk3D,
    plot_keypoints_plotly,
    plot_sections_plotly,
)

# get directory of file using pathlib
SCRIPT_DIR = pathlib.Path(__file__).parent.absolute()
HAWK3D = Hawk3D(SCRIPT_DIR.parents[3] / "data/mean_hawk_shape.csv")
PRINCIPAL_COMPONENTS = np.load(
    SCRIPT_DIR.parents[3] / "data/website_principal_components.npy"
)
LEFT_SCORE_FRAMES = np.load(SCRIPT_DIR.parents[3] / "data/Left_scores_RightTurn.npy")
RIGHT_SCORE_FRAMES = np.load(SCRIPT_DIR.parents[3] / "data/Right_scores_RightTurn.npy")
MU = np.load(SCRIPT_DIR.parents[3] / "data/website_mu.npy")
ALPHA = 0.3
N_FRAMES = 149
N_MARKERS = 4
N_DIMS = 3
N_COMPONENTS = 12
COLOUR = "lightblue"


def reconstruct_frames(
    selected_components: list[int], score_frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct frames from selected principal components.

    Args:
        selected_components (list[int]): list of selected principal components.
        score_frames (np.ndarray): Score frames array.

    Returns:
        tuple[np.ndarray, np.ndarray]: Reconstructed frames and combined scores.
    """
    if not selected_components:
        # No components selected, return mean shape repeated
        return np.repeat(MU, N_FRAMES, axis=0), np.zeros(N_FRAMES)

    # Extract the scores for selected PCs and sum them (or combine as needed)
    selected_scores = score_frames[:, selected_components]
    combined_scores = np.sum(selected_scores, axis=1)  # sum over selected PCs

    # Extract only the selected PCs from principal_components
    selected_PCs = PRINCIPAL_COMPONENTS[selected_components, :]

    # Reconstruct from selected PCs
    reconstruction = np.dot(selected_scores, selected_PCs)
    reconstruction = reconstruction.reshape(-1, N_MARKERS, N_DIMS)
    frames = MU + reconstruction
    return frames, combined_scores


def create_figure(selected_components: list[int]) -> go.Figure:
    """Create the figure for the animation.

    Args:
        selected_components (list[int]): list of selected principal components.

    Returns:
        go.Figure: Plotly figure object.
    """
    left_frames_data, left_combined_scores = reconstruct_frames(
        selected_components, LEFT_SCORE_FRAMES
    )
    left_frames_data[:, :, 0] *= -1

    right_frames_data, right_combined_scores = reconstruct_frames(
        selected_components, RIGHT_SCORE_FRAMES
    )
    full_frames_data = combine_frames(left_frames_data, right_frames_data)
    combined_scores_centered, y_min, y_max = center_combined_scores(
        left_combined_scores, right_combined_scores
    )

    fig = initialize_figure()

    frames_list = create_animation_frames(
        selected_components, full_frames_data, combined_scores_centered
    )

    add_initial_data(fig, frames_list)

    fig.frames = frames_list

    add_buttons_and_sliders(fig, y_min, y_max)
    return fig


def combine_frames(
    left_frames_data: np.ndarray, right_frames_data: np.ndarray
) -> np.ndarray:
    """Combine left and right frames data.

    Args:
        left_frames_data (np.ndarray): Left frames data.
        right_frames_data (np.ndarray): Right frames data.

    Returns:
        np.ndarray: Combined frames data.
    """
    full_frames_data = np.zeros((left_frames_data.shape[0], 8, 3))
    full_frames_data[:, ::2, :] = left_frames_data.squeeze()
    full_frames_data[:, 1::2, :] = right_frames_data.squeeze()
    return full_frames_data


def center_combined_scores(
    left_combined_scores: np.ndarray, right_combined_scores: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Center combined scores.

    Args:
        left_combined_scores (np.ndarray): Left combined scores.
        right_combined_scores (np.ndarray): Right combined scores.

    Returns:
        tuple[np.ndarray, float, float]: Centered combined scores, minimum and maximum values.
    """
    combined_scores = np.mean([left_combined_scores, right_combined_scores], axis=0)
    combined_scores_centered = combined_scores - np.mean(combined_scores)
    y_min = combined_scores_centered.min()
    y_max = combined_scores_centered.max()
    return combined_scores_centered, y_min, y_max


def initialize_figure() -> go.Figure:
    """Initialize the figure layout.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"type": "scene"}]],
    )

    fig.update_layout(
        scene={
            "domain": {"x": [0.1, 0.9], "y": [0.1, 0.8]},
            "aspectmode": "cube",
            "xaxis": {"range": [-0.6, 0.6], "autorange": False, "dtick": 0.25},
            "yaxis": {"range": [-0.6, 0.6], "autorange": False, "dtick": 0.25},
            "zaxis": {"range": [-0.6, 0.6], "autorange": False, "dtick": 0.25},
        },
        showlegend=False,
    )
    return fig


def create_animation_frames(
    selected_components: list[int],
    full_frames_data: np.ndarray,
    combined_scores_centered: np.ndarray,
) -> list[go.Frame]:
    """Create animation frames.

    Args:
        selected_components (list[int]): list of selected principal components.
        full_frames_data (np.ndarray): Full frames data.
        combined_scores_centered (np.ndarray): Centered combined scores.

    Returns:
        list[go.Frame]: list of Plotly frames.
    """
    frames_list = []

    line_plot = go.Scatter(
            x=np.arange(N_FRAMES),
            y=combined_scores_centered,
            mode="lines",
            xaxis="x2",
            yaxis="y2",
            showlegend=False,
            line={"color": "blue"},
        )

    for i in range(N_FRAMES):
        HAWK3D.reset_transformation()
        HAWK3D.update_keypoints(full_frames_data[i])

        scatter3d = go.Figure()
        scatter3d = plot_sections_plotly(scatter3d, HAWK3D, colour=COLOUR, alpha=ALPHA)
        scatter3d = plot_keypoints_plotly(scatter3d, HAWK3D, colour=COLOUR, alpha=1)
        scatter3d_traces = scatter3d.data


        current_frame_marker = go.Scatter(
            x=[i],
            y=[combined_scores_centered[i]],
            mode="markers",
            xaxis="x2",
            yaxis="y2",
            marker={"color": "red", "size": 10},
            showlegend=False,
        )
        axis_dict = {"range": [-0.6, 0.6],
                          "autorange": False,
                          "dtick": 0.25,
                    }
        frame_layout = go.Layout(
            title={
                "text": f"Selected Components: {', '.join([str(s + 1) for s in sorted(selected_components)]) if selected_components else 'None'}",
                "xanchor": "center",
                "yanchor": "top",
                "x": 0.5,
                "y": 0.92,
            },
            xaxis2={"domain": [0.8, 0.95]},
            yaxis2={"domain": [0.8, 0.95]},
            scene={
                "xaxis": axis_dict,
                "yaxis": axis_dict,
                "zaxis": axis_dict,
            },
        )

        frame_data = [*list(scatter3d_traces), line_plot, current_frame_marker]

        frame = go.Frame(
            data=frame_data,
            name=f"frame_{i}",
            layout=frame_layout,
        )
        frames_list.append(frame)
    return frames_list


def add_initial_data(fig: go.Figure, frames_list: list[go.Frame]) -> None:
    """Add initial data to the figure.

    Args:
        fig (go.Figure): Plotly figure object.
        frames_list (list[go.Frame]): list of Plotly frames.
    """
    initial_data = [*frames_list[0].data]
    for i_data in initial_data:
        fig.add_trace(i_data)


def add_buttons_and_sliders(
    fig: go.Figure, y_min: float, y_max: float
) -> None:
    """Add buttons and sliders to the figure.

    Args:
        fig (go.Figure): Plotly figure object.
        y_min (float): Minimum value of the centered combined scores.
        y_max (float): Maximum value of the centered combined scores.
    """
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
    for i in range(N_FRAMES):
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
        uirevision=True,
    )


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
            options=[{"label": f"PC {i + 1}", "value": i} for i in range(N_COMPONENTS)],
            value=[],  # start with no PCs selected
            multi=True,
            style={"width": "75%", "font-family": "Arial", "margin": "auto"},
        ),
        dcc.Loading(
            dcc.Graph(id="graph", style={"width": "75%", "margin": "auto"}),
        ),
    ]
)


@app.callback(Output("graph", "figure"), Input("pc-dropdown", "value"))
def update_plot(selected_components: list[int]) -> go.Figure:
    """Update the plot based on selected principal components.

    Args:
        selected_components (list[int]): list of selected principal components.

    Returns:
        go.Figure: Updated Plotly figure object.
    """
    fig = create_figure(selected_components)
    return fig


server = app.server  # For deployment

if __name__ == "__main__":
    app.run_server(debug=True)
