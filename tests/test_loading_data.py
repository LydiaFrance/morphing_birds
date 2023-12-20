import pytest
import numpy as np

from morphing_birds import Hawk3D

@pytest.fixture
def hawk3d_fixture():
    filename = 'data/mean_hawk_shape.csv'
    return Hawk3D(filename)

def test_load_data(hawk3d_fixture):
    assert hawk3d_fixture.names_all_keypoints is not None
    assert hawk3d_fixture.all_keypoints is not None

def test_marker_name_extraction(hawk3d_fixture):
    expected_first_marker_name = "left_secondary"
    assert hawk3d_fixture.names_all_keypoints[0] == expected_first_marker_name

def test_keypoint_initialisation(hawk3d_fixture):
    assert hawk3d_fixture.all_keypoints.shape == (14, 3)
    assert hawk3d_fixture.fixed_keypoints.shape == (6, 3)
    assert hawk3d_fixture.right_keypoints.shape == (4, 3)
    assert hawk3d_fixture.left_keypoints.shape == (4, 3)

def test_polygon(hawk3d_fixture):

    # Get coords from head directly
    expected_keypoints = ["right_shoulder", "hood", "left_shoulder"]
    expected_coords = hawk3d_fixture.get_keypoints_by_names(expected_keypoints)

    # Retrieve the coordinates used in the "head" polygon
    head_polygon_coords = hawk3d_fixture._polygons["head"]

    assert np.allclose(head_polygon_coords, expected_coords)
    
    # Get coords from left handwing directly
    expected_keypoints = ["left_wingtip", "left_primary", "left_secondary"]
    expected_coords = hawk3d_fixture.get_keypoints_by_names(expected_keypoints)

    # Retrieve the coordinates used in the "left_handwing" polygon
    left_handwing_polygon_coords = hawk3d_fixture._polygons["left_handwing"]

    assert np.allclose(left_handwing_polygon_coords, expected_coords)



