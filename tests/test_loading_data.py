import pytest

from morphing_birds import Hawk3D

@pytest.fixture
def hawk3d_fixture():
    filename = 'data/mean_hawk_shape.csv'
    return Hawk3D(filename)

def test_load_data(hawk3d_fixture):
    assert hawk3d_fixture.marker_names is not None
    assert hawk3d_fixture.keypoints is not None

def test_marker_name_extraction(hawk3d_fixture):
    expected_first_marker_name = "left_secondary"
    assert hawk3d_fixture.marker_names[0] == expected_first_marker_name

def test_keypoint_initialisation(hawk3d_fixture):
    assert hawk3d_fixture.keypoints.shape == (14, 3)
    
    
