import pytest
import numpy as np

from morphing_birds import Hawk3D


@pytest.fixture()
def hawk3d():
    filename = 'data/mean_hawk_shape.csv'
    return Hawk3D(filename)

def test_load_data(hawk3d):
    assert hawk3d.keypoint_manager.names_all_keypoints is not None
    assert hawk3d.keypoint_manager.keypoints is not None

def test_marker_name_extraction(hawk3d):
    expected_first_marker_name = "left_secondary"
    assert hawk3d.keypoint_manager.names_all_keypoints[0] == expected_first_marker_name

def test_keypoint_initialisation(hawk3d):
    assert hawk3d.keypoint_manager.all_keypoints.shape == (14, 3)
    assert hawk3d.keypoint_manager.fixed_keypoints.shape == (6, 3)
    assert hawk3d.keypoint_manager.keypoints.shape == (8, 3)
    assert hawk3d.keypoint_manager.right_keypoints.shape == (4, 3)

def test_polygon(hawk3d):

    # Get coords from head directly
    expected_keypoints = ["right_shoulder", "hood", "left_shoulder"]
    expected_coords = hawk3d.keypoint_manager.get_keypoints_by_names(expected_keypoints)

    # Retrieve the coordinates used in the "head" polygon
    head_polygon_coords = hawk3d.keypoint_manager.all_keypoints[hawk3d.plotter._polygons["head"]]

    assert np.allclose(head_polygon_coords, expected_coords)
    
    # Get coords from right handwing directly
    expected_keypoints = ["right_wingtip", "right_primary", "right_secondary"]
    expected_coords = hawk3d.keypoint_manager.get_keypoints_by_names(expected_keypoints)

    # Retrieve the coordinates used in the "right handwing" polygon
    handwing_polygon_coords = hawk3d.keypoint_manager.all_keypoints[hawk3d.plotter._polygons["right_handwing"]]

    assert np.allclose(handwing_polygon_coords, expected_coords)
    

def test_validate_keypoints(hawk3d):

    # Check works with right markers only
    test_keypoints = np.ones((4,3))
    result = hawk3d.keypoint_manager._validate_keypoints(test_keypoints) 
    assert result.shape == (1,8,3)

    # Check works with left & right markers
    test_keypoints = np.ones((1,8,3))
    result = hawk3d.keypoint_manager._validate_keypoints(test_keypoints) 
    assert result.shape == (1,8,3)

    # Check the mirroring
    test_keypoints = np.ones((100,4,3))
    result = hawk3d.keypoint_manager._validate_keypoints(test_keypoints) 
    assert result.shape == (100,8,3)
    assert test_keypoints[0,0,0] == -result[0,0,0]


def test_update_keypoints(hawk3d):
    test_keypoints = np.ones((1,4,3))

    hawk3d.keypoint_manager.update_keypoints(test_keypoints)
    keypoints = hawk3d.keypoint_manager.right_keypoints

    assert np.allclose(keypoints, test_keypoints[0])


