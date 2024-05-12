# Import necessary libraries
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from timeit import default_timer as timer
from dash import callback_context

from morphing_birds import Hawk3D

# Initialize Hawk3D
hawk3d = Hawk3D("../data/mean_hawk_shape.csv")
hawk3d.restore_keypoints_to_average()


def create_dash_app(new_keypoints=None, mean_scores=None, binned_horzDist=None):

    # Generate fake keypoints and mean scores data for testing purposes
    if new_keypoints is None:
        # Generate fake keypoints data for testing purposes
        np.random.seed(0)
        hawk3d.reset_transformation()
        hawk3d.restore_keypoints_to_average()
        new_keypoints = np.random.normal(0, 0.01, (100, 8, 3)) + hawk3d.markers

    if mean_scores is None:
        # Generate fake mean scores data for testing purposes
        mean_scores = np.random.normal(0, 1, (100, 8))
    
    if binned_horzDist is None:
        # Generate fake binned horizontal distance data for testing purposes
        binned_horzDist = np.random.randint(0, 10, 100)

    # Initialize figures
    app = dash.Dash(__name__)

    # Function to create a 2D subplot figure with 8x8 scatter plots
    def create_2d_subplots(mean_scores, binned_horzDist):
        nScores = 5

        # The specific colours for each score
        colour_list = ['#B5E675', '#6ED8A9', '#51B3D4', 
                       '#4579AA', '#BC96C9', '#917AC2', 
                       '#5A488B', '#888888', '#888888', 
                       '#888888', '#888888', '#888888']
        
        # Define an 8x9 grid to accommodate the new column on the left for time
        specs = [
            [{'type': 'scatter'} for _ in range(nScores+1)]  # Initialize all cells as scatter plot cells
            for _ in range(nScores)
        ]


        # Create the subplots with the specified layout
        fig = make_subplots(
            rows=nScores,
            cols=nScores+1,  # Include the new column on the left
            specs=specs
            )

        # Plotting variable against time in the first column
        for ii in range(nScores):
            fig.add_trace(
                go.Scattergl(
                    x= binned_horzDist,
                    y= mean_scores[:, ii],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colour_list[ii], 
                        line=dict(
                        width=0  # No border around the markers
                    )),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row = ii + 1,
                col = 1  # First column for time plots
            )

        # Plotting each variable against itself on the diagonal starting from column 3 (second position)
        for ii in range(nScores):
            fig.add_trace(
                go.Scattergl(
                    x= mean_scores[:, ii],
                    y= mean_scores[:, ii],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colour_list[ii], 
                        line=dict(
                        width=0  # No border around the markers
                    )),
                    showlegend=False,
                    hoverinfo='none'
                ),
                row = ii + 1,
                col = ii + 2  # Shift the self-comparison plot to the right
            )

        # Filling the lower triangle for variable vs variable plots, adjust as per new column arrangement
        for ii in range(nScores):
            for jj in range(ii):
                fig.add_trace(
                    go.Scattergl(
                        x= mean_scores[:, ii],
                        y= mean_scores[:, jj],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=colour_list[ii], 
                            line=dict(
                            width=0  # No border around the markers
                        )),
                        showlegend=False, 
                        hoverinfo='none'
                    ),
                    row = ii + 1,
                    col = jj + 2  # Adjust for the new first column
                )

        # Update axes and layout settings to clean up the plot appearance
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(height=800, width=900, hovermode='closest')

        fig.update_layout(
        plot_bgcolor='white',  # Background color within the plot area
        paper_bgcolor='white',  # Background color around the plot area
        xaxis=dict(
            showgrid=False,  # No gridlines
            zeroline=True  # No zero line for the X-axis
        ),
        yaxis=dict(
            showgrid=False,  # No gridlines
            zeroline=True  # No zero line for the Y-axis
        )
    )


        return fig

    # Function to create a 3D scatter plot figure
    def create_3d_scatter_plot():
        hawk3d.restore_keypoints_to_average()
        fig = plot_plotly(hawk3d)
        fig.update_layout(height=400, width=400,
                        scene=dict(camera=dict(eye=dict(x=1.25, y=1.25, z=1.25))))
        return fig


    # Initialize figures
    fig_2d = create_2d_subplots(mean_scores, binned_horzDist)
    fig_3d = create_3d_scatter_plot()

    # Set up the layout for the Dash application
    app.layout = html.Div([
        html.Div([
            dcc.Graph(id='2d-subplots', figure=fig_2d)
        ], style={
            'flex': '1',  # Flex grow factor for the 3D plot
            'min-width': '0',  # Ensure the div can shrink below its content size if needed
            'padding-right': '-100px'  # Space between the figures
        }),
        html.Div([
            dcc.Graph(id='3d-scatter-plot', figure=fig_3d)
        ], style={
            'flex': '1',
            'min-width': '0',  # Similar to the 3D plot div
            'padding-left': '-100px'  # Space between the figures
        })
    ], style={
        'display': 'flex',  # This ensures that child divs are laid out as flex items
        'flex-wrap': 'nowrap',  # Prevents the flex items from wrapping onto multiple lines
        'align-items': 'stretch',  # Stretches the items to fill the container vertically
        'justify-content': 'space-between',  # Distributes space between and around content items
        'width': '100%'  # Ensures the container takes full width of its parent
})


    # Callback to update plots based on hover interactions and manage 3D plot camera settings
    @app.callback(
    [Output('2d-subplots', 'figure'), Output('3d-scatter-plot', 'figure')],
    [Input('2d-subplots', 'hoverData'), Input('3d-scatter-plot', 'relayoutData')],
    [State('2d-subplots', 'figure'), State('3d-scatter-plot', 'figure')]
    )


    def update_plots(hoverData, relayoutData, current_2d_fig, current_3d_fig):
        # Determine if the update was triggered by a hover event
        ctx = dash.callback_context

        if not ctx.triggered:
            # No input triggered yet, return the existing figures
            return [current_2d_fig, current_3d_fig]

        input_id = ctx.triggered[0]['prop_id'].split('.')[0]

        # Update 3D plot based on hover and apply the last known camera settings
        if input_id == '2d-subplots' and hoverData:
            start_3d = timer()
            point_index = hoverData['points'][0]['pointIndex']
            hawk3d.update_keypoints(new_keypoints[point_index, :, :])
            updated_3d_fig = plot_plotly(hawk3d)
            ctx.record_timing('update_3d_plot', timer() - start_3d, 'Time to update 3D plot')

        else:
            updated_3d_fig = current_3d_fig

        # Retrieve and apply the latest camera position from relayoutData
        if relayoutData and 'scene.camera' in relayoutData:
            camera = relayoutData['scene.camera']
        else:
            camera = current_3d_fig['layout']['scene']['camera']  # Use the last known camera


        start_camera = timer()
        updated_3d_fig['layout']['scene']['camera'] = camera
        ctx.record_timing('update_camera_settings', timer() - start_camera, 'Time to update camera settings')

        # Update the 2D subplot marker sizes based on hover
        nPoints = new_keypoints.shape[0]
        if input_id == '2d-subplots' and hoverData:
            start_2d = timer()
            updated_2d_fig = go.Figure(current_2d_fig)
            for ii in range(len(updated_2d_fig.data)):
                sizes = [8 if idx == point_index else 3 for idx in range(nPoints)]
                updated_2d_fig.data[ii].marker.size = sizes
            ctx.record_timing('update_2d_plots', timer() - start_2d, 'Time to update 2D plots')

            return [updated_2d_fig, updated_3d_fig]

        return [current_2d_fig, updated_3d_fig]


    return app


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
    fig = plot_settings_plotly(fig,origin)


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
    for section in Hawk3D_instance.body_sections.keys():
        mesh, lines = get_polygon_plotly(Hawk3D_instance, section, colour, alpha)
        fig.add_trace(mesh)
        fig.add_trace(lines)

    return fig

def get_polygon_plotly(Hawk3D_instance, section_name, colour, alpha=1):
    if section_name not in Hawk3D_instance.body_sections.keys():
        raise ValueError(f"Section name {section_name} not recognised.")

    # You can modify these functions to customize colour and alpha
    colour = colour_polygon_plotly(section_name, colour)
    alpha = alpha_polygon_plotly(section_name, alpha)

    # Extract polygon coordinates
    coords = Hawk3D_instance.get_polygon_coords(section_name)

    # Construct a mesh plotly object
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
    
        # The colour of the polygon is determined by whether the landmarks are
        # estimated or measured.
        if "handwing" in section_name or "tail" in section_name:
            colour = 'lightblue'
        else:
            colour = 'grey'
    
        return colour

def plot_settings_plotly(fig, origin):
     

    # Axis Ranges and Grid Customization
    axis_range = [-0.32, 0.32]
    tickvals = np.linspace(-0.2, 0.2, 3)
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor='white',
                gridcolor="grey",
                showbackground=True,
                zerolinecolor="grey", 
                range=axis_range,
                dtick=0.1,
                tickvals=tickvals, 
                gridwidth=0.5
            ),
            yaxis=dict(
                backgroundcolor="white",
                gridcolor="grey",
                showbackground=True,
                zerolinecolor="grey",
                range=axis_range,
                dtick=0.1,
                tickvals=tickvals, 
                gridwidth=0.5
            ),
            zaxis=dict(
                backgroundcolor="rgba(10, 10, 10, 0.1)",
                gridcolor="grey",
                showbackground=True,
                zerolinecolor="grey",
                range=axis_range,
                dtick=0.1,
                tickvals=tickvals, 
                gridwidth=0.5
            )
        ),
        margin=dict(r=20, l=10, b=10, t=10)
    )

    # Updating axis limits based on custom 'origin' and 'increment'
    increment = 0.32

    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(scene=dict(
        xaxis=dict(range=[origin[0]-increment, origin[0]+increment]),
        yaxis=dict(range=[origin[1]-increment, origin[1]+increment]),
        zaxis=dict(range=[origin[2]-increment, origin[2]+increment])
    ))
    fig.update_layout(showlegend=False)
    return fig

# Create the Dash app

if __name__ == '__main__':
    app = create_dash_app()
    app.run_server(debug=True)