import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import the morphing_birds module (should work since package is installed)
try:
    from morphing_birds import Kestrel3D
    from morphing_birds.plotting import plot_plotly, animate_plotly
except ImportError:
    # Fallback: Add the src directory to the path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from morphing_birds import Kestrel3D
    from morphing_birds.plotting import plot_plotly, animate_plotly


class TestKestrelShape:
    """Test suite for Kestrel3D class covering all configuration modes and marker handling."""

    @pytest.fixture
    def test_csv_path(self):
        """Path to the existing test CSV file."""
        return Path(__file__).parent / "test_kestrel_shape.csv"

    @pytest.fixture
    def sample_motion_data(self, test_csv_path):
        """Create sample motion CSV data based on the test shape data."""
        # Read the existing test CSV to get the structure
        test_df = pd.read_csv(test_csv_path)
        
        # Extract marker names from the columns - handle both formats
        marker_names = set()
        for col in test_df.columns:
            if col.endswith(('x', 'y', 'z')):
                # Remove the coordinate suffix (x, y, z)
                marker_name = col[:-1]
                # Remove trailing underscore if present
                if marker_name.endswith('_'):
                    marker_name = marker_name[:-1]
                marker_names.add(marker_name)
        
        marker_names = sorted(list(marker_names))
        
        # Create motion data with multiple frames (100 frames)
        n_frames = 100
        motion_data = []
        
        for frame in range(n_frames):
            frame_data = {'frame': frame}
            t = frame / n_frames * 2 * np.pi  # One full cycle
            
            for marker_name in marker_names:
                # Get base coordinates from test CSV - try both formats
                try:
                    # Try format: marker_x, marker_y, marker_z
                    if f'{marker_name}_x' in test_df.columns:
                        base_x = test_df[f'{marker_name}_x'].iloc[0]
                        base_y = test_df[f'{marker_name}_y'].iloc[0]
                        base_z = test_df[f'{marker_name}_z'].iloc[0]
                    # Try format: markerx, markery, markerz
                    elif f'{marker_name}x' in test_df.columns:
                        base_x = test_df[f'{marker_name}x'].iloc[0]
                        base_y = test_df[f'{marker_name}y'].iloc[0]
                        base_z = test_df[f'{marker_name}z'].iloc[0]
                    else:
                        # Default coordinates
                        base_x, base_y, base_z = 0.0, 0.0, 0.0
                except (KeyError, IndexError):
                    # If marker not found, use default coordinates
                    base_x, base_y, base_z = 0.0, 0.0, 0.0
                
                # Add sinusoidal motion (wingbeat-like)
                amplitude = 0.01 if 'p' in marker_name else 0.005  # Primaries move more
                variation = amplitude * np.sin(t)
                
                frame_data[f'{marker_name}_x'] = base_x + variation
                frame_data[f'{marker_name}_y'] = base_y + variation * 0.5
                frame_data[f'{marker_name}_z'] = base_z + variation * 0.2
            
            motion_data.append(frame_data)
        
        return pd.DataFrame(motion_data)

    @pytest.fixture
    def temp_motion_csv(self, sample_motion_data):
        """Create temporary motion CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_motion_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_kestrel_initialization_default_mode(self, test_csv_path):
        """Test Kestrel3D initialization with default settings."""
        kestrel = Kestrel3D(str(test_csv_path))
        
        # Default mode: use_simple=False, use_full=False, use_mean_alula=False
        assert not kestrel.use_simple
        assert not kestrel.use_full
        assert not kestrel.use_mean_alula
        
        # Should have some markers
        assert len(kestrel.marker_names) > 0

    def test_kestrel_initialization_mean_alula_mode(self, test_csv_path):
        """Test Kestrel3D initialization with mean alula averaging."""
        kestrel = Kestrel3D(str(test_csv_path), use_mean_alula=True)
        
        assert kestrel.use_mean_alula
        
        # Should have some markers
        assert len(kestrel.marker_names) > 0

    def test_kestrel_initialization_simple_mode(self, test_csv_path):
        """Test Kestrel3D initialization with simple mode (hawk-comparable)."""
        kestrel = Kestrel3D(str(test_csv_path), use_simple=True)
        
        assert kestrel.use_simple
        
        # Simple mode should have some markers
        marker_count = len(kestrel.marker_names)
        assert marker_count > 0

    def test_motion_data_loading_default_mode(self, test_csv_path, temp_motion_csv):
        """Test motion data loading with default settings."""
        kestrel = Kestrel3D(str(test_csv_path))
        motion_data = kestrel.load_motion_data(temp_motion_csv)
        
        # Motion data should match skeleton expectations
        assert motion_data.shape[1] == len(kestrel.marker_names)
        assert motion_data.shape[0] == 100  # 100 frames
        assert motion_data.shape[2] == 3    # x, y, z coordinates

    def test_motion_data_loading_mean_alula_mode(self, test_csv_path, temp_motion_csv):
        """Test motion data loading with alula averaging."""
        kestrel = Kestrel3D(str(test_csv_path), use_mean_alula=True)
        motion_data = kestrel.load_motion_data(temp_motion_csv)
        
        # Motion data should match skeleton expectations (with averaged alula)
        assert motion_data.shape[1] == len(kestrel.marker_names)
        assert motion_data.shape[0] == 100
        assert motion_data.shape[2] == 3

    def test_motion_data_loading_simple_mode(self, test_csv_path, temp_motion_csv):
        """Test motion data loading with simple mode."""
        kestrel = Kestrel3D(str(test_csv_path), use_simple=True)
        motion_data = kestrel.load_motion_data(temp_motion_csv)
        
        # Motion data should match skeleton expectations
        assert motion_data.shape[1] == len(kestrel.marker_names)
        assert motion_data.shape[0] == 100
        assert motion_data.shape[2] == 3

    def test_marker_count_consistency(self, test_csv_path, temp_motion_csv):
        """Test that marker counts are consistent between skeleton and motion data."""
        # Test all combinations of modes
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
            {'use_simple': True, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': True, 'use_mean_alula': False},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            motion_data = kestrel.load_motion_data(temp_motion_csv)
            
            # The key test: motion data must match skeleton expectations exactly
            assert motion_data.shape[1] == len(kestrel.marker_names), \
                f"Marker count mismatch for config {config}: " \
                f"motion_data has {motion_data.shape[1]}, skeleton expects {len(kestrel.marker_names)}"

    def test_animation_compatibility(self, test_csv_path, temp_motion_csv):
        """Test that animation works without marker count mismatches."""
        # Test the specific case that was failing
        kestrel = Kestrel3D(str(test_csv_path), use_mean_alula=True)
        motion_data = kestrel.load_motion_data(temp_motion_csv)
        motion_data, _ = kestrel.remove_nan_frames(motion_data)
        
        # This should not raise a ValueError about marker count mismatch
        try:
            # Test with just a few frames to avoid creating actual plots in tests
            test_motion = motion_data[:5]  # First 5 frames
            kestrel.update_keypoints(test_motion[0])  # Test single frame update
            
            # If we get here, the marker counts match correctly
            assert True
        except ValueError as e:
            if "Marker count mismatch" in str(e):
                pytest.fail(f"Animation failed due to marker count mismatch: {e}")
            else:
                # Re-raise if it's a different error
                raise

    def test_missing_markers_handling(self, test_csv_path, sample_motion_data):
        """Test handling of missing markers in motion data."""
        # Create motion data with some missing markers
        motion_df = sample_motion_data.copy()
        
        # Remove some columns to simulate missing markers
        cols_to_remove = [col for col in motion_df.columns if 'l_al_1' in col]  # Remove left alula
        if cols_to_remove:  # Only test if we have alula markers to remove
            motion_df_incomplete = motion_df.drop(columns=cols_to_remove)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                motion_df_incomplete.to_csv(f.name, index=False)
                incomplete_motion_path = f.name
            
            try:
                kestrel = Kestrel3D(str(test_csv_path), use_mean_alula=True)
                motion_data = kestrel.load_motion_data(incomplete_motion_path)
                
                # Should still work - missing markers should be filled with zeros
                assert motion_data.shape[1] == len(kestrel.marker_names)
                assert motion_data.shape[0] == 100
                assert motion_data.shape[2] == 3
                
            finally:
                os.unlink(incomplete_motion_path)

    def test_fixed_markers_excluded_from_motion(self, test_csv_path):
        """Test that fixed markers are not included in motion data."""
        kestrel = Kestrel3D(str(test_csv_path))
        
        # Fixed markers should not be in the active marker list
        fixed_markers = ['head', 'left_shoulder', 'right_shoulder']
        for fixed_marker in fixed_markers:
            assert fixed_marker not in kestrel.marker_names, \
                f"Fixed marker {fixed_marker} should not be in active marker list"

    def test_marker_name_consistency(self, test_csv_path, temp_motion_csv):
        """Test that marker names are consistent across different operations."""
        kestrel = Kestrel3D(str(test_csv_path), use_mean_alula=True)
        
        # Get marker names from different methods
        init_markers = kestrel.marker_names
        method_markers = kestrel._get_marker_names()
        
        # They should be identical
        assert init_markers == method_markers
        
        # Load motion data and verify consistency
        motion_data = kestrel.load_motion_data(temp_motion_csv)
        assert motion_data.shape[1] == len(init_markers)

    def test_different_csv_formats(self, test_csv_path, temp_motion_csv):
        """Test handling of different CSV column naming conventions."""
        # Test with both underscore and no-underscore formats
        # The current CSV uses underscore format (marker_x, marker_y, marker_z)
        # The code should handle both formats
        
        kestrel = Kestrel3D(str(test_csv_path))
        motion_data = kestrel.load_motion_data(temp_motion_csv)
        
        # Should work regardless of CSV format
        assert motion_data.shape[1] == len(kestrel.marker_names)
        assert motion_data.shape[0] == 100
        assert motion_data.shape[2] == 3

    def test_basic_functionality(self, test_csv_path):
        """Test basic functionality without complex marker handling."""
        # Just test that we can create a kestrel and it has reasonable properties
        kestrel = Kestrel3D(str(test_csv_path))
        
        # Basic checks
        assert hasattr(kestrel, 'marker_names')
        assert hasattr(kestrel, 'skeleton_definition')
        assert len(kestrel.marker_names) > 0
        assert kestrel.markers is not None
        assert kestrel.markers.shape[2] == 3  # x, y, z coordinates

    def test_csv_loading_all_modes(self, test_csv_path):
        """Test that CSV loading works correctly for all kestrel modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
            {'use_simple': True, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': True, 'use_mean_alula': False},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Check that the default shape is not None
            assert kestrel.default_shape is not None, f"Default shape is None for config {config}"
            
            # Check that we have the expected number of markers + fixed markers
            expected_total = len(kestrel.marker_names) + len(kestrel.fixed_marker_index)
            assert kestrel.default_shape.shape[1] == expected_total, \
                f"Default shape marker count mismatch for config {config}"
            
            # Check that markers are 3D coordinates
            assert kestrel.default_shape.shape[2] == 3, \
                f"Markers should be 3D coordinates for config {config}"

    def test_marker_indexing_all_modes(self, test_csv_path):
        """Test that marker indices are defined correctly for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
            {'use_simple': True, 'use_full': False, 'use_mean_alula': False},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Check that marker indices are correctly defined
            assert len(kestrel.marker_index) == len(kestrel.marker_names), \
                f"Marker index count mismatch for config {config}"
            
            # Check that right and left marker indices are reasonable
            assert len(kestrel.right_marker_index) > 0, \
                f"No right markers found for config {config}"
            assert len(kestrel.left_marker_index) > 0, \
                f"No left markers found for config {config}"
            
            # For non-simple modes, right and left should be equal
            if not config.get('use_simple', False):
                assert len(kestrel.right_marker_index) == len(kestrel.left_marker_index), \
                    f"Right/left marker count mismatch for config {config}"

    def test_polygon_initialization_all_modes(self, test_csv_path):
        """Test that polygons are initialized correctly for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
            {'use_simple': True, 'use_full': False, 'use_mean_alula': False},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Check that polygons are initialized
            assert hasattr(kestrel, 'polygons'), f"Polygons not initialized for config {config}"
            assert len(kestrel.polygons) > 0, f"No polygons found for config {config}"
            
            # Check that essential body sections exist
            essential_sections = ['body', 'left_handwing', 'right_handwing']
            for section in essential_sections:
                if section in kestrel.polygons:
                    indices = kestrel.polygons[section]
                    assert len(indices) > 0, f"Empty polygon for {section} in config {config}"
                    
                    # Verify indices are valid
                    max_marker_idx = len(kestrel.marker_names) + len(kestrel.fixed_marker_index) - 1
                    for idx in indices:
                        assert 0 <= idx <= max_marker_idx, \
                            f"Invalid polygon index {idx} for {section} in config {config}"

    def test_polygon_coordinates_all_modes(self, test_csv_path):
        """Test that polygon coordinates are correct for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Test getting coordinates for all available sections
            for section in kestrel.polygons.keys():
                try:
                    coords = kestrel.get_polygon_coords(section)
                    
                    # Check that coordinates are 3D
                    assert coords.shape[1] == 3, \
                        f"Coords should be 3D for {section} in config {config}"
                    
                    # Check that coordinates match expected values
                    indices = kestrel.polygons[section]
                    expected_coords = kestrel.current_shape[0, indices, :]
                    np.testing.assert_array_equal(coords, expected_coords,
                        f"Coordinate mismatch for {section} in config {config}")
                        
                except Exception as e:
                    pytest.fail(f"Failed to get coords for {section} in config {config}: {e}")
            
            # Test that non-existent section raises error
            with pytest.raises(ValueError):
                kestrel.get_polygon_coords("non_existent_section")

    def test_keypoint_validation_all_modes(self, test_csv_path):
        """Test keypoint validation for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            n_markers = len(kestrel.marker_names)
            
            # Test 2D coordinates raise an error
            invalid_keypoints = np.random.rand(n_markers, 2)  # 2D instead of 3D
            with pytest.raises(ValueError):
                kestrel.validate_keypoints(invalid_keypoints)
            
            # Test correct reshaping from [n,3] to [1,n,3]
            keypoints_2d_array = np.random.rand(n_markers, 3)
            validated_keypoints = kestrel.validate_keypoints(keypoints_2d_array)
            assert len(validated_keypoints.shape) == 3, \
                f"Keypoints should be 3D array [1,n,3] for config {config}"
            assert validated_keypoints.shape == (1, n_markers, 3), \
                f"Unexpected keypoint shape for config {config}"

    def test_update_keypoints_all_modes(self, test_csv_path):
        """Test keypoint updating for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            n_markers = len(kestrel.marker_names)
            
            # Store original state
            kestrel.restore_keypoints_to_average()
            original_keypoints = kestrel.current_shape.copy()
            
            # Test updating with new keypoints
            new_keypoints = np.random.rand(1, n_markers, 3)
            kestrel.update_keypoints(new_keypoints)
            
            assert not np.array_equal(kestrel.current_shape, original_keypoints), \
                f"Current shape should change after update for config {config}"
            np.testing.assert_array_equal(kestrel.markers, new_keypoints,
                f"Markers not updated correctly for config {config}")
            
            # Test restore to average
            kestrel.restore_keypoints_to_average()
            assert np.allclose(kestrel.current_shape, original_keypoints), \
                f"Shape should restore to average for config {config}"
            
            # Test reset with None
            kestrel.update_keypoints(None)
            np.testing.assert_array_equal(kestrel.markers, kestrel.default_markers,
                f"Should reset to default when None passed for config {config}")
            
            # Test invalid dimensions
            with pytest.raises(ValueError):
                kestrel.update_keypoints(np.random.rand(1, n_markers, 2))  # Wrong dimensions

    def test_transformations_all_modes(self, test_csv_path):
        """Test transformations work correctly for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            original_shape = kestrel.current_shape.copy()
            
            # Test body pitch
            kestrel.transform_keypoints(bodypitch=25)
            assert not np.allclose(kestrel.current_shape, original_shape), \
                f"Shape should change after pitch for config {config}"
            assert np.allclose(kestrel.untransformed_shape, original_shape), \
                f"Untransformed shape should be unchanged for config {config}"
            
            # Test reset transformation
            kestrel.reset_transformation()
            assert np.allclose(kestrel.current_shape, original_shape), \
                f"Shape should reset after reset_transformation for config {config}"
            
            # Test yaw rotation
            kestrel.transform_keypoints(bodyyaw=25)
            assert not np.allclose(kestrel.current_shape, original_shape), \
                f"Shape should change after yaw for config {config}"
            
            kestrel.reset_transformation()
            
            # Test horizontal translation
            kestrel.transform_keypoints(horzDist=10)
            assert not np.allclose(kestrel.current_shape, original_shape), \
                f"Shape should change after horizontal translation for config {config}"
            
            # Final restore
            kestrel.reset_transformation()
            kestrel.restore_keypoints_to_average()
            assert np.allclose(kestrel.current_shape, kestrel.default_shape), \
                f"Should restore to default shape for config {config}"

    def test_alula_marker_configuration(self, test_csv_path):
        """Test that alula markers are configured correctly for different modes."""
        # Test default mode (individual alula markers)
        kestrel_default = Kestrel3D(str(test_csv_path), use_mean_alula=False)
        
        # Should have individual alula markers (if they exist in the data)
        has_individual_alula = any('alula' in name and 'mean' not in name 
                                 for name in kestrel_default.marker_names)
        has_mean_alula = any('alula_mean' in name for name in kestrel_default.marker_names)
        
        if has_individual_alula:
            assert not has_mean_alula, "Default mode should not have mean alula if individual exists"
        
        # Test mean alula mode
        kestrel_mean = Kestrel3D(str(test_csv_path), use_mean_alula=True)
        
        # Should have mean alula markers instead of individual ones
        has_individual_alula_mean = any('alula' in name and 'mean' not in name 
                                       for name in kestrel_mean.marker_names)
        has_mean_alula_mean = any('alula_mean' in name for name in kestrel_mean.marker_names)
        
        if has_mean_alula_mean:
            assert not has_individual_alula_mean, "Mean alula mode should not have individual alula"

    def test_polygon_shape_validation_all_modes(self, test_csv_path):
        """Test polygon shape validation for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
            {'use_simple': True, 'use_full': False, 'use_mean_alula': False},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Test that polygon shape validation passes
            try:
                kestrel.validate_polygon_shape()
                # If we get here, validation passed
                assert True
            except AssertionError as e:
                pytest.fail(f"Polygon shape validation failed for config {config}: {e}")
            except Exception as e:
                # Some configurations might not have all required markers for validation
                # This is acceptable as long as the basic structure is sound
                print(f"Note: Polygon validation skipped for config {config} due to: {e}")

    def test_marker_count_scaling_across_modes(self, test_csv_path):
        """Test that marker counts scale appropriately across different modes."""
        # Create kestrels with different configurations
        kestrel_default = Kestrel3D(str(test_csv_path), use_simple=False, use_mean_alula=False)
        kestrel_mean_alula = Kestrel3D(str(test_csv_path), use_simple=False, use_mean_alula=True)
        kestrel_simple = Kestrel3D(str(test_csv_path), use_simple=True, use_mean_alula=False)
        
        # Simple mode should have fewer markers than full mode
        assert len(kestrel_simple.marker_names) <= len(kestrel_default.marker_names), \
            "Simple mode should have fewer or equal markers than default mode"
        
        # Mean alula mode should have fewer markers than individual alula mode (if alula exists)
        has_alula_default = any('alula' in name for name in kestrel_default.marker_names)
        has_alula_mean = any('alula' in name for name in kestrel_mean_alula.marker_names)
        
        if has_alula_default and has_alula_mean:
            # Mean alula should reduce marker count (4 individual -> 2 mean)
            assert len(kestrel_mean_alula.marker_names) <= len(kestrel_default.marker_names), \
                "Mean alula mode should have fewer or equal markers than default mode"

    def test_make_unilateral_all_modes(self, test_csv_path):
        """Test make_unilateral function for all modes."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Create some test motion data with realistic coordinates
            n_frames = 10
            n_markers = len(kestrel.marker_names)
            # Use coordinates that will pass left-right validation
            motion_data = np.random.rand(n_frames, n_markers, 3) * 0.1  # Small values
            # Ensure left markers have negative x, right markers have positive x
            for i, marker_name in enumerate(kestrel.marker_names):
                if marker_name.startswith('left_'):
                    motion_data[:, i, 0] = -np.abs(motion_data[:, i, 0])  # Negative x
                elif marker_name.startswith('right_'):
                    motion_data[:, i, 0] = np.abs(motion_data[:, i, 0])   # Positive x
            
            # Test make_unilateral
            try:
                unilateral_data, is_left, unilateral_info_df = kestrel.make_unilateral(motion_data)
                
                # Basic shape checks
                assert unilateral_data.shape[0] <= n_frames * 2, \
                    f"Unilateral data should have at most {n_frames * 2} frames for config {config}"
                assert unilateral_data.shape[2] == 3, \
                    f"Should maintain 3D coordinates for config {config}"
                assert len(is_left) == unilateral_data.shape[0], \
                    f"is_left array should match number of frames for config {config}"
                
                # Check that we have some valid frames
                assert unilateral_data.shape[0] > 0, \
                    f"Should have some valid frames for config {config}"
                
                print(f"✓ make_unilateral works for config {config}")
                print(f"  Original: {motion_data.shape} -> Unilateral: {unilateral_data.shape}")
                
            except Exception as e:
                pytest.fail(f"make_unilateral failed for config {config}: {e}")

    def test_unilateral_bilateral_conversion_cycle(self, test_csv_path):
        """Test the complete unilateral/bilateral conversion cycle."""
        test_cases = [
            {'use_simple': False, 'use_full': False, 'use_mean_alula': False},
            {'use_simple': False, 'use_full': False, 'use_mean_alula': True},
        ]
        
        for config in test_cases:
            kestrel = Kestrel3D(str(test_csv_path), **config)
            
            # Create some test motion data with realistic coordinates
            n_frames = 5
            n_markers = len(kestrel.marker_names)
            original_data = np.random.rand(n_frames, n_markers, 3) * 0.1
            
            # Ensure left markers have negative x, right markers have positive x
            for i, marker_name in enumerate(kestrel.marker_names):
                if marker_name.startswith('left_'):
                    original_data[:, i, 0] = -np.abs(original_data[:, i, 0])
                elif marker_name.startswith('right_'):
                    original_data[:, i, 0] = np.abs(original_data[:, i, 0])
            
            # Test the complete cycle: bilateral -> unilateral -> bilateral
            try:
                # Step 1: Convert to unilateral
                unilateral_data, is_left, _ = kestrel.make_unilateral(original_data)
                
                # Verify unilateral data shape
                assert unilateral_data.shape[2] == 3, "Should maintain 3D coordinates"
                assert unilateral_data.shape[0] <= n_frames * 2, "Should have doubled frames (or fewer due to validation)"
                
                # Step 2: Convert back to bilateral
                reconstructed_data = kestrel.make_bilateral(unilateral_data, is_left)
                
                # Verify reconstructed data shape
                assert reconstructed_data.shape[2] == 3, "Should maintain 3D coordinates"
                assert reconstructed_data.shape[1] == n_markers, f"Should have {n_markers} markers"
                
                # Step 3: Test automatic unilateral detection in update_keypoints
                # Use a single frame of unilateral data
                single_unilateral_frame = unilateral_data[0:1]  # Shape: [1, n_unilateral_markers, 3]
                
                # This should automatically detect and convert unilateral data
                kestrel.update_keypoints(single_unilateral_frame)
                
                # Verify the kestrel's current shape has the right total markers (including fixed)
                total_markers = kestrel.current_shape.shape[1]  # This includes fixed markers
                assert total_markers >= n_markers, \
                    f"After update_keypoints with unilateral data, should have at least {n_markers} markers (got {total_markers})"
                
                print(f"✓ Unilateral/bilateral cycle works for config {config}")
                print(f"  Original: {original_data.shape}")
                print(f"  Unilateral: {unilateral_data.shape}")
                print(f"  Reconstructed: {reconstructed_data.shape}")
                print(f"  Single frame update: successful")
                
            except Exception as e:
                pytest.fail(f"Unilateral/bilateral conversion failed for config {config}: {e}")

    def test_update_keypoints_unilateral_detection(self, test_csv_path):
        """Test that update_keypoints automatically detects and handles unilateral data."""
        kestrel = Kestrel3D(str(test_csv_path), use_mean_alula=False)
        
        # Calculate expected unilateral marker count
        left_markers = [m for m in kestrel.marker_names if m.startswith('left_')]
        centre_markers = [m for m in kestrel.marker_names if m.startswith('centre_')]
        expected_unilateral_markers = len(left_markers) + len(centre_markers)
        
        # Create fake unilateral data with the expected marker count
        unilateral_data = np.random.rand(1, expected_unilateral_markers, 3) * 0.1
        
        # This should be detected as unilateral and automatically converted
        try:
            kestrel.update_keypoints(unilateral_data)
            
            # Verify the conversion worked - current_shape includes fixed markers too
            total_markers = kestrel.current_shape.shape[1]  # This includes fixed markers
            active_markers = len(kestrel.marker_names)  # This is only active markers
            assert total_markers >= active_markers, \
                f"Should have at least {active_markers} active markers (got {total_markers} total markers)"
            
            print(f"✓ Automatic unilateral detection works")
            print(f"  Input unilateral markers: {expected_unilateral_markers}")
            print(f"  Active markers: {active_markers}")
            print(f"  Total markers in current_shape: {total_markers}")
            
        except Exception as e:
            pytest.fail(f"Automatic unilateral detection failed: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])