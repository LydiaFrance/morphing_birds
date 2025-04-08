import unittest
import numpy as np
from morphing_birds import Hawk3D

class TestHawk3D(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Set up the Hawk3D object for testing.
        """
        cls.csv_path = "tests/test_mean_hawk_shape.csv"
        cls.hawk3d = Hawk3D(cls.csv_path)

    def test_csv_loading(self):
        """
        Test that the CSV file is loaded correctly. 
        - Check that the default shape is not None
        - Check that the default shape has the correct number of markers
        - Check that the csv_marker_names list is not empty
        """
        self.assertIsNotNone(self.hawk3d.default_shape, msg="Default shape is None")
        expected_marker_count = len(self.hawk3d.skeleton_definition.get_all_marker_names())
        self.assertEqual(self.hawk3d.default_shape.shape[1], expected_marker_count, "Default shape does not have the correct number of markers")
        self.assertGreater(len(self.hawk3d.csv_marker_names), 0, "No markers found in CSV")

    def test_marker_indexing(self):
        """
        Test that the marker indices are defined correctly. 
        - Check that the total number of markers matches the sum of right, left and fixed markers
        - Check that the number of indices for each marker type matches the expected number of markers for that type
        """
        total_markers = len(self.hawk3d.right_marker_index) + len(self.hawk3d.left_marker_index) + len(self.hawk3d.fixed_marker_index)
        self.assertEqual(total_markers, len(self.hawk3d.skeleton_definition.get_all_marker_names()), "Total markers do not match")
        self.assertEqual(len(self.hawk3d.right_marker_index), len(self.hawk3d.skeleton_definition.get_right_marker_names()), "Right marker indices are not defined correctly")
        self.assertEqual(len(self.hawk3d.left_marker_index), len(self.hawk3d.skeleton_definition.get_left_marker_names()), "Left marker indices are not defined correctly")
        self.assertEqual(len(self.hawk3d.fixed_marker_index), len(self.hawk3d.skeleton_definition.fixed_marker_names), "Fixed marker indices are not defined correctly")


    def test_marker_indices(self):
        """
        Test that the marker indices are defined correctly. 
        - Check that the indices match the expected order from the test CSV file.
        - Check the number of markers matches 
        """
        expected_indices = {
            'left_wingtip': 1,
            'right_wingtip': 12,
            'left_primary': 2,
            'right_primary': 11,
            'left_secondary': 0,
            'right_secondary': 13,
            'left_tailtip': 5,
            'right_tailtip': 6,
            # Add fixed markers if they're included in csv_marker_names
            'left_shoulder': 3,
            'left_tailbase': 4,
            'right_tailbase': 7,
            'right_shoulder': 10,
            'hood': 9,
            'tailpack': 8
        }


        # Check if the indices match the expected values
        for marker, expected_index in expected_indices.items():
            self.assertIn(marker, self.hawk3d.csv_marker_names, f"{marker} not found in csv_marker_names")
            actual_index = self.hawk3d.csv_marker_names.index(marker)
            self.assertEqual(actual_index, expected_index, 
                             f"Index mismatch for {marker}. Expected {expected_index}, got {actual_index}")

        # Verify that the number of markers matches
        self.assertEqual(len(self.hawk3d.csv_marker_names), len(expected_indices),
                         "Number of markers in csv_marker_names doesn't match expected count")

    def test_right_left_marker_order(self):
        """
        Test that the left and right marker names are in the correct order. 
        - Check that the markers alternate between left and right
        """
        marker_names = self.hawk3d.marker_names
        for ii in range(0, len(marker_names), 2):
            self.assertTrue(marker_names[ii].startswith('left_'), f"Expected left marker at index {ii}, got {marker_names[ii]}")
            if ii + 1 < len(marker_names):
                self.assertTrue(marker_names[ii+1].startswith('right_'), f"Expected right marker at index {ii+1}, got {marker_names[ii+1]}")

        # Test that right and left marker counts match
        right_markers = [name for name in marker_names if name.startswith('right_')]
        left_markers = [name for name in marker_names if name.startswith('left_')]
        self.assertEqual(len(right_markers), len(left_markers), "Number of right and left markers should be equal")

    def test_2d_coords(self):
        """
        Test that 2D coordinates raise an error. 
        """
        invalid_keypoints = np.array([[1, 2], [3, 4]])  # 2D coords instead of 3D
        with self.assertRaises(ValueError):
            self.hawk3d.validate_keypoints(invalid_keypoints)


    def test_keypoint_reshape(self):
        """
        Test that the keypoints are reshaped correctly. 
        - Check it changes [n,3] to [1,n,3]
        """
        # Check it changes [n,3] to [1,n,3]
        keypoints_2d_array = np.array([[1,2,3],[4,5,6]])
        new_keypoints = self.hawk3d.validate_keypoints(keypoints_2d_array)
        self.assertEqual(len(new_keypoints.shape), 3, "Keypoints should be 3D array [1,n,3]")


    def test_polygons_index(self):
        """
        Test if polygons are initialized correctly
        - Check that the polygon indices match the expected indices for each body section
        """
        for section, markers in self.hawk3d.skeleton_definition.body_sections.items():
            indices = self.hawk3d.polygons[section]
            expected_indices = self.hawk3d.skeleton_definition.get_marker_indices(markers, self.hawk3d.csv_marker_names)
            self.assertEqual(indices, expected_indices)

    def test_polygon_coords(self):
        """
        Test that the polygon coordinates are correct. 
        - Check that the coordinates are 3D
        - Test getting coords for all sections
        - Test that an error is raised for a non-existent section
        """
        for section in self.hawk3d.skeleton_definition.body_sections.keys():
            coords = self.hawk3d.get_polygon_coords(section)
            self.assertEqual(coords.shape[1], 3, "Coords should be 3D coordinates") 
            indices = self.hawk3d.polygons[section]
            expected_coords = self.hawk3d.current_shape[0, indices, :]
            np.testing.assert_array_equal(coords, expected_coords)

        for section in self.hawk3d.skeleton_definition.body_sections:
            coords = self.hawk3d.get_polygon_coords(section)
        with self.assertRaises(ValueError):
            self.hawk3d.get_polygon_coords("non_existent_section")


    def test_polygon_shape(self):
        """
        Test that the polygon shape validation step. 
        """
        try:
            self.hawk3d.validate_polygon_shape()
        except AssertionError:
            self.fail("validate_polygon_shape raised an AssertionError unexpectedly!")

    def test_update_keypoints(self):
        """
        Test that the keypoints are updated correctly. 
        - Test updating with new keypoints
        - Test resetting to default shape
        - Test updating with only right side keypoints
        - Test updating with invalid data
        """
        self.hawk3d.restore_keypoints_to_average()
        original_keypoints = self.hawk3d.current_shape.copy()

        # Test updating with new keypoints
        new_keypoints = np.random.rand(1, 8, 3)  # Assuming 8 total markers
        self.hawk3d.update_keypoints(new_keypoints)
        self.assertFalse(np.array_equal(self.hawk3d.current_shape, original_keypoints),
                                             "Current shape should not match original keypoints")
        self.assertFalse(np.array_equal(self.hawk3d.markers, self.hawk3d.default_markers),
                                             "Current shape should not match originally loaded keypoints")
        np.testing.assert_array_equal(self.hawk3d.markers, new_keypoints, "When given new markers, keypoints changed unexpectedly")


        # Test restore to average
        self.hawk3d.restore_keypoints_to_average()
        self.assertTrue(np.array_equal(self.hawk3d.current_shape, original_keypoints),
                        "Shape should be restored to average after restore")
        self.assertTrue(np.array_equal(self.hawk3d.markers, self.hawk3d.default_markers),
                                             "Markers should be restored to average after restore")

        # Test updating with only right side keypoints
        right_keypoints = np.random.rand(1, 4, 3)  # Assuming 4 markers on each side
        self.hawk3d.update_keypoints(right_keypoints)
        np.testing.assert_array_equal(self.hawk3d.right_markers, right_keypoints, 
                                      "When given new right markers, right keypoints should be updated")
        self.assertFalse(np.array_equal(self.hawk3d.left_markers, self.hawk3d.default_left_markers),
                         "When given new right markers, left markers should have changed")


        # # Test resetting to default shape
        self.hawk3d.update_keypoints(None)
        np.testing.assert_array_equal(self.hawk3d.markers, self.hawk3d.default_markers,
                                      "Shape should reset to default when None is passed")

        # # Test updating with invalid data
        with self.assertRaises(ValueError):
            self.hawk3d.update_keypoints(np.random.rand(1, 8, 2))  # Invalid number of dimensions
        
        # # Restore original keypoints
        self.hawk3d.restore_keypoints_to_average()



    def test_transformations(self):
        """
        Test that the transformations are applied correctly. 
        - Test pitch rotation
        - Test reset transformation
        - Test yaw rotation
        - Test horizontal translation
        - Test restore to average
        """
        original_shape = self.hawk3d.current_shape.copy()

        # Body Pitch Test
        self.hawk3d.transform_keypoints(bodypitch=25)
        self.assertFalse(np.allclose(self.hawk3d.current_shape, original_shape), "Shape should change after pitch rotation")
        self.assertTrue(np.allclose(self.hawk3d.untransformed_shape, original_shape), "Untransformed shape should be unchanged")

        # Reset transformation test
        self.hawk3d.reset_transformation()
        self.assertTrue(np.allclose(self.hawk3d.current_shape, original_shape), "Shape should be unchanged after reset")

        # Yaw Test
        self.hawk3d.transform_keypoints(yaw=25)
        self.assertFalse(np.allclose(self.hawk3d.current_shape, original_shape), "Shape should change after yaw rotation")
        self.assertTrue(np.allclose(self.hawk3d.untransformed_shape, original_shape), "Untransformed shape should be unchanged")

        self.hawk3d.reset_transformation()

        # Horizontal Translation Test
        self.hawk3d.transform_keypoints(horzDist=10)
        self.assertFalse(np.allclose(self.hawk3d.current_shape, original_shape), "Shape should change after horizontal translation")
        self.assertTrue(np.allclose(self.hawk3d.untransformed_shape, original_shape), "Untransformed shape should be unchanged")

        # Restore to average test
        self.hawk3d.reset_transformation()
        self.hawk3d.restore_keypoints_to_average()
        self.assertTrue(np.allclose(self.hawk3d.current_shape, self.hawk3d.default_shape), "Shape should be restored to average after restore")

if __name__ == '__main__':
    unittest.main()

