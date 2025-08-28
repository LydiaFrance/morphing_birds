import numpy as np
from morphing_birds import Animal3D, PigeonSkeletonDefinition
import pandas as pd
import copy

class Pigeon3D(Animal3D):
    """
    A class representing a 3D model of a Pigeon.

    Inherits from Animal3D and provides specific functions for 
    loading CSV data and validating polygon shapes of a pigeon. 
    """
    def __init__(self, csv_path: str, use_simple: bool = False, verbose=False):
        """
        Initialize the Pigeon3D class.
        
        Args:
            csv_path (str): Path to the CSV file containing marker data
            use_simple (bool): If True, uses simplified body sections for polygons.
                             If False, uses detailed body sections. Default is False.
        """
        skeleton_definition = PigeonSkeletonDefinition()
        super().__init__(skeleton_definition)
        
        self.use_simple = use_simple
        
        # Load the CSV data
        self.data = pd.read_csv(csv_path)
        
        # Get marker names in canonical order
        if self.use_simple:
            self.marker_names = self.skeleton_definition.get_marker_names_simple()
            fixed_markers = self.skeleton_definition.fixed_marker_names
        else:
            self.marker_names = self.skeleton_definition.get_marker_names_full(verbose=verbose)
            fixed_markers = self.skeleton_definition.fixed_marker_names
        
        # First, create a mapping of all available markers in the CSV
        marker_data = {}  # Store marker data temporarily
        for marker_name in self.skeleton_definition.marker_name_change.keys():
            if marker_name in self.skeleton_definition.ignored_marker_names:
                continue
                
            csv_name = self.skeleton_definition.marker_name_change[marker_name]
            # Try both underscore and no underscore versions
            x_col = f"{csv_name}_x" if f"{csv_name}_x" in self.data.columns else f"{csv_name}x"
            y_col = f"{csv_name}_y" if f"{csv_name}_y" in self.data.columns else f"{csv_name}y"
            z_col = f"{csv_name}_z" if f"{csv_name}_z" in self.data.columns else f"{csv_name}z"
            
            if x_col in self.data.columns:
                x = self.data[x_col].values
                y = self.data[y_col].values
                z = self.data[z_col].values
                marker_data[marker_name] = np.stack([x, y, z], axis=1)
                # print(f"Loaded marker {marker_name} from CSV columns: {x_col}, {y_col}, {z_col}")
        
        if not marker_data:
            raise ValueError("No markers were found in the CSV file. Check the column names.")
            
        # Now create all_markers array in the correct order
        all_markers = []
        marker_to_idx = {}  # Map marker names to their position in all_markers
        idx = 0
        
        # First add active markers in canonical order
        for marker_name in self.marker_names:
            if marker_name in marker_data:
                all_markers.append(marker_data[marker_name])
                marker_to_idx[marker_name] = idx
                idx += 1
            else:
                raise ValueError(f"Required marker {marker_name} not found in CSV data")
                
        # Then add fixed markers
        fixed_marker_indices = []  # Keep track of fixed marker indices
        for marker_name in fixed_markers:
            if marker_name in marker_data:
                all_markers.append(marker_data[marker_name])
                marker_to_idx[marker_name] = idx
                fixed_marker_indices.append(idx)
                idx += 1
                
        # Stack all markers into a single array (n_frames, n_markers, 3)
        all_markers = np.stack(all_markers, axis=1)
        
        # Set up the shapes required by Animal3D
        self.default_shape = all_markers.copy()
        self.current_shape = all_markers.copy()
        self.untransformed_shape = all_markers.copy()
        
        # Store the fixed marker indices
        self.fixed_marker_index = fixed_marker_indices
        
        # Extract active markers (they're already in the right order at the start of all_markers)
        self._markers = all_markers[:, :len(self.marker_names)]
        
        # Initialize polygons after markers are set up
        self.init_polygons()
        
        # Define indices for active markers
        self.define_indices()
        
        # Specify which sections should be colored (rest will be grey)
        self.colour_sections = ["handwing", "tail"]  # Only handwing and tail gets the color
        if verbose:
            print(f"Input keypoints shape: {self._markers.shape}")
            print(f"Number of markers in self.marker_names: {len(self.marker_names)}")
            print(f"Number of markers in self.marker_index: {len(self.marker_index)}")
            print(f"Marker names: {self.marker_names}")
            print(f"Marker indices: {self.marker_index}")
            print(f"Fixed marker indices: {self.fixed_marker_index}")

    @property
    def markers(self):
        """Get the marker positions."""
        return self._markers

    @markers.setter
    def markers(self, value):
        """Set the marker positions."""
        self._markers = value

    @property
    def right_marker_names(self):
        """Get the list of right side marker names."""
        return [name for name in self.marker_names if name.startswith("right_")]

    @property
    def right_markers(self):
        """
        Returns the right side markers.
        """
        # Get indices of right markers
        right_marker_names = self.right_marker_names
        right_indices = [self.marker_names.index(name) for name in right_marker_names]
        
        # Return markers at those indices
        return self.markers[:, right_indices, :]
    
    def init_polygons(self):
        """Initialize the polygons for visualization."""
        # Get the appropriate sections based on mode
        if self.use_simple:
            sections = [s for s in self.skeleton_definition.body_sections if s.endswith("_simple")]
            # Remove '_simple' suffix for the polygon dictionary
            sections_dict = {s.replace('_simple', ''): self.skeleton_definition.body_sections[s] 
                           for s in sections}
            fixed_markers = self.skeleton_definition.fixed_marker_names
        else:
            sections = [s for s in self.skeleton_definition.body_sections if not s.endswith("_simple")]
            sections_dict = {s: self.skeleton_definition.body_sections[s] for s in sections}
            fixed_markers = self.skeleton_definition.fixed_marker_names

        # Convert marker names to indices
        self.polygons = {}
        for section, markers in sections_dict.items():
            # Get indices for each marker in the section
            indices = []
            missing_markers = []
            
            for marker in markers:
                if marker in self.marker_names:
                    idx = self.marker_names.index(marker)
                    indices.append(self.marker_index[idx])
                elif marker in fixed_markers:
                    # For fixed markers, find their position in the fixed marker list
                    fixed_idx = fixed_markers.index(marker)
                    indices.append(self.fixed_marker_index[fixed_idx])
                else:
                    missing_markers.append(marker)
            
            # Only add the polygon if we have at least 3 markers (minimum for a polygon)
            if len(indices) >= 3:
                self.polygons[section] = indices
            elif missing_markers:
                # Silently skip sections with missing markers in simple mode
                if not self.use_simple:  # Only warn in full mode
                    print(f"Warning: Skipping polygon section '{section}' - missing markers: {missing_markers}")

        # Print debug info about polygons (only if any were created)
        if self.polygons:
            print(f"\nInitialized {len(self.polygons)} polygon sections:")
            for section, indices in self.polygons.items():
                print(f"  {section}: {len(indices)} vertices")
        else:
            print("\nNo polygon sections initialized")

    def define_indices(self):
        """Define indices for active markers in canonical order."""
        self.marker_index = list(range(len(self.marker_names)))

    def load_csv(self, csv_path: str):
        """
        Loads CSV data specific to the pigeon skeleton.

        Parameters:
        - csv_path (str): Path to the CSV file.

        Raises:
        - IOError: If there's an error loading the CSV file.
        - ValueError: If keypoints data is invalid.
        """
        try:
            # Load the data
            data = self.load_csv_data(csv_path)
            
            csv_headers = data[0]
            self.csv_marker_names = self.get_csv_marker_names(csv_headers)
            
            # Load and validate keypoints before defining indices
            keypoints = self.get_csv_keypoints(data)
            validated_keypoints = self.validate_keypoints(keypoints)
            
            self.default_shape = validated_keypoints
            self.current_shape = self.default_shape.copy()
            self.untransformed_shape = self.default_shape.copy()
            
            # Define marker indices after setting up the shapes
            self.define_indices(self.csv_marker_names)
        
        except IOError as e:
            raise IOError(f"Error loading CSV file: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid keypoints data: {e}")
    
    def load_csv_data(self, csv_path):
        """
        Loads data from the CSV file.

        Parameters:
        - csv_path (str): Path to the CSV file.

        Returns:
        - numpy.ndarray: Loaded data as a NumPy array.
        """
        with open(csv_path, 'r') as file:
            return np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')

    def copy(self) -> 'Pigeon3D':
        """
        Create a deep copy of this Pigeon3D instance.
        
        Returns:
            A new Pigeon3D instance that is a deep copy of this one
        """
        return copy.deepcopy(self)

    def overwrite_keypoints(self, keypoints: np.ndarray) -> None:
        """
        Set the given keypoints as the default, current, and untransformed shape.
        This overwrites the existing shapes with the new keypoints.
        
        Args:
            keypoints: Array of shape (1, n_markers, 3) containing the new keypoints
        """
        # Validate shape
        if keypoints.shape[0] != 1:
            raise ValueError("Keypoints must have shape (1, n_markers, 3)")
        if keypoints.shape[2] != 3:
            raise ValueError("Keypoints must be 3D coordinates")
            
        # Create new shape array that includes both moving and fixed markers
        new_shape = np.zeros_like(self.default_shape)
        
        # Update the moving markers (non-fixed) with the new keypoints
        new_shape[:, self.marker_index, :] = keypoints
        
        # Keep the fixed markers in their original positions
        new_shape[:, self.fixed_marker_index, :] = self.default_shape[:, self.fixed_marker_index, :]
        
        # Set all shapes to the new combined shape
        self.default_shape = new_shape.copy()
        self.current_shape = new_shape.copy()
        self.untransformed_shape = new_shape.copy()

    def get_csv_marker_names(self, header_row: list) -> list:
        """
        Extracts unique marker names from the header row by removing coordinate suffixes.

        Parameters:
        - header_row (list): First row of the CSV containing marker names with suffixes.

        Returns:
        - list: Unique marker names in the order they appear in the CSV.
        """
        # Remove coordinate suffixes (_x, _y, _z)
        cleaned_names = []
        for name in header_row:
            if name.endswith('x'):
                cleaned_names.append(name[:-1])
            elif name.endswith('y'):
                cleaned_names.append(name[:-1])
            elif name.endswith('z'):
                cleaned_names.append(name[:-1])
            else:
                cleaned_names.append(name)
        
        # Preserve order and ensure uniqueness
        unique_names = []
        seen = set()
        for name in cleaned_names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        return unique_names

    def get_csv_keypoints(self, data) -> np.ndarray:
        """
        Loads and reshapes keypoint coordinates from the CSV data.

        Parameters:
        - data (numpy.ndarray): Loaded CSV data.

        Returns:
        - numpy.ndarray: Reshaped keypoints as [n, 3] matrix.
        """
        # Convert string data to float and reshape
        keypoints = data[1:].astype(float)  # Skip the first row (header)
        keypoints = keypoints.reshape(-1, 3)  # [n, 3]
        return keypoints

    def validate_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Validates and processes keypoints, including mirroring if only unilateral data is provided.
        
        Parameters:
        - keypoints (np.ndarray): Input keypoints of shape [n_frames, n_markers, 3]
        
        Returns:
        - np.ndarray: Validated and potentially mirrored keypoints
        """
        # First, perform basic validation from Animal3D
        keypoints = super().validate_keypoints(keypoints)
        
        # Get marker organization
        left_markers, right_markers, center_markers = self.skeleton_definition.get_marker_pairs_and_centers(self.use_simple)
        n_pairs = len(left_markers)
        n_centers = len(center_markers)
        expected_bilateral = n_pairs * 2 + n_centers
        
        # If we have unilateral data (half the expected markers), mirror it
        if keypoints.shape[1] == n_pairs + n_centers:
            # Create array for full bilateral data
            n_frames = keypoints.shape[0]
            bilateral_data = np.zeros((n_frames, expected_bilateral, 3))
            
            # Get indices for each type
            unilateral_indices = []  # Indices in the input data
            left_indices = []        # Where to put left markers in output
            right_indices = []       # Where to put right markers in output
            
            # Build indices for paired markers
            for i, (left, right) in enumerate(zip(left_markers, right_markers)):
                left_indices.append(self.marker_names.index(left))
                right_indices.append(self.marker_names.index(right))
                unilateral_indices.append(i)
            
            # Extract unilateral data (excluding centers)
            unilateral = keypoints[:, unilateral_indices, :]
            
            # Place original data as right side
            bilateral_data[:, right_indices, :] = unilateral
            
            # Mirror and place as left side
            mirrored = unilateral.copy()
            mirrored[..., 0] *= -1  # Mirror x-coordinate
            bilateral_data[:, left_indices, :] = mirrored
            
            # Handle center markers if any
            if n_centers > 0:
                center_indices_in = range(n_pairs, n_pairs + n_centers)  # Indices in input
                center_indices_out = [self.marker_names.index(m) for m in center_markers]  # Indices in output
                bilateral_data[:, center_indices_out, :] = keypoints[:, center_indices_in, :]
            
            return bilateral_data
            
        return keypoints

    def mirror_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Creates a full set of keypoints by mirroring to create left-side markers.
        Used for visualization when only right-side markers are provided.
        
        Assumes markers are ordered left_1, right_1, left_2, right_2, etc.

        Parameters:
        - keypoints (np.ndarray): Array of shape [n_frames, n_unilateral_markers, 3] containing only right-side markers

        Returns:
        - np.ndarray: Array of shape [n_frames, n_bilateral_markers, 3] with newly created left-side markers
        """
        n_frames = keypoints.shape[0]
        n_markers = keypoints.shape[1]

        # Create array for full set of markers
        full_keypoints = np.zeros((n_frames, n_markers * 2, 3))
        
        # Place right markers in their correct positions (odd indices: 1,3,5,7)
        full_keypoints[:, 1::2, :] = keypoints
        
        # Create left markers by mirroring right markers
        left_markers = keypoints.copy()
        left_markers[:,:, 0] *= -1  # Mirror x-coordinate
        
        # Place left markers in their correct positions (even indices: 0,2,4,6)
        full_keypoints[:, 0::2, :] = left_markers
        
        return full_keypoints
    
    def validate_polygon_shape(self):
        """
        Validates the shape of the polygons to ensure they are as expected.
        """
        assert self.get_polygon_coords('left_handwing')[0][0] < self.get_polygon_coords('right_handwing')[0][0], \
            "Left wing should be to the left of the right wing"
        
        assert self.get_polygon_coords('tail')[0][1] < self.get_polygon_coords('left_handwing')[0][1], \
            "Tail should be behind the wings"
        
        assert self.get_polygon_coords('tail')[0][1] < self.get_polygon_coords('right_handwing')[0][1], \
            "Tail should be behind the wings"

        # Tell the user that the polygon shape is valid
        print("Polygon shape is valid")
        
    def load_motion_data(self, csv_path: str, use_simple: bool = None, verbose: bool = False) -> np.ndarray:
        """
        Load motion data from CSV file.
        Uses the same reliable approach as the Pigeon3D constructor.
        
        Parameters:
        - csv_path (str): Path to the CSV file containing motion data
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses current setting.
        
        Returns:
        - np.ndarray: Motion data in shape [nFrames, nMarkers, 3]
        
        Example:
            motion_data = pigeon.load_motion_data("data/2025-08-28-FullPigeons.csv")
            clean_data, valid_frames = pigeon.remove_nan_frames(motion_data)
        """
        import pandas as pd
        
        # Determine which marker set to use
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        
        # Load the CSV data using pandas (more reliable than the original method)
        data = pd.read_csv(csv_path)
        
        print(f"Loading motion data from: {csv_path}")
        print(f"CSV shape: {data.shape}")
        print(f"Using {'simple' if use_simple_markers else 'full'} marker set")
        
        # Get target marker names
        if use_simple_markers:
            target_markers = self.skeleton_definition.get_marker_names_simple()
        else:
            target_markers = self.skeleton_definition.get_marker_names_full(verbose=verbose)
            
        print(f"Target markers ({len(target_markers)}): {target_markers}")
        
        # Load marker data using the reliable approach from constructor
        marker_data = {}
        loaded_count = 0
        
        for marker_name in self.skeleton_definition.marker_name_change.keys():
            if marker_name in self.skeleton_definition.ignored_marker_names:
                continue
                
            csv_name = self.skeleton_definition.marker_name_change[marker_name]
            
            # Try both underscore and no underscore versions
            x_col = f"{csv_name}_x" if f"{csv_name}_x" in data.columns else f"{csv_name}x"
            y_col = f"{csv_name}_y" if f"{csv_name}_y" in data.columns else f"{csv_name}y"
            z_col = f"{csv_name}_z" if f"{csv_name}_z" in data.columns else f"{csv_name}z"
            
            if x_col in data.columns:
                x = data[x_col].values
                y = data[y_col].values
                z = data[z_col].values
                marker_data[marker_name] = np.stack([x, y, z], axis=1)
                loaded_count += 1
                
        print(f"Successfully loaded {loaded_count} markers from CSV")
        
        # Extract motion data for target markers in correct order
        motion_list = []
        found_markers = []
        missing_markers = []
        
        for marker_name in target_markers:
            if marker_name in marker_data:
                motion_list.append(marker_data[marker_name])
                found_markers.append(marker_name)
            else:
                missing_markers.append(marker_name)
                
        if missing_markers:
            print(f"Warning: Missing markers: {missing_markers}")
            
        if not motion_list:
            raise ValueError("No target markers found in CSV data")
            
        # Stack into final motion data array
        motion_data = np.stack(motion_list, axis=1)
        
        print(f"✓ Created motion data: {motion_data.shape}")
        print(f"✓ Found markers ({len(found_markers)}): {found_markers}")
        
        return motion_data
    
    def load_data_complete(self, csv_path: str, info_path: str = None, use_simple: bool = None, verbose: bool = False) -> tuple:
        """
        Complete workflow to load motion data and info data, with NaN removal.
        
        Parameters:
        - csv_path (str): Path to the CSV file containing motion data (e.g., "FullPigeons.csv")
        - info_path (str, optional): Path to info CSV. If None, auto-generates from csv_path
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses current setting.
        
        Returns:
        - tuple: (clean_motion_data, clean_info_df, valid_frames)
            - clean_motion_data: np.ndarray of shape [nCleanFrames, nMarkers, 3]
            - clean_info_df: pd.DataFrame with Frame, Time, GustPosition, FileName info
            - valid_frames: np.ndarray boolean mask of valid frames
            
        Example:
            motion_data, info_df, valid_frames = pigeon.load_data_complete(
                "data/2025-08-28-FullPigeons.csv",
                "data/2025-08-28-PigeonInfo.csv"
            )
        """
        import pandas as pd
        import os
        if verbose:
            print("=== Complete Data Loading Workflow ===")
        
        # Load motion data
        motion_data = self.load_motion_data(csv_path, use_simple)
        
        # Auto-generate info path if not provided
        if info_path is None:
            base_name = os.path.splitext(csv_path)[0]
            if "Full" in base_name:
                info_path = base_name.replace("Full", "") + "Info.csv"
            else:
                info_path = base_name + "Info.csv"
        
        # Load info data
        if os.path.exists(info_path):
            info_df = pd.read_csv(info_path)
            if verbose:
                print(f"✓ Loaded info data: {info_df.shape}")
        else:
            print(f"Warning: Info file not found at {info_path}")
            
        
        # Remove NaN frames
        if verbose:
            print("\\nRemoving NaN frames...")
        clean_motion_data, valid_frames = self.remove_nan_frames(motion_data)
        
        # Filter info data to match clean frames
        clean_info_df = info_df.iloc[valid_frames].reset_index(drop=True)
        
        if verbose:
            print(f"\\n=== Summary ===")
            print(f"✓ Original frames: {motion_data.shape[0]}")
            print(f"✓ Clean frames: {clean_motion_data.shape[0]}")
            print(f"✓ Removed: {motion_data.shape[0] - clean_motion_data.shape[0]} frames")
            print(f"✓ Markers: {clean_motion_data.shape[1]}")
            print(f"✓ Info data: {clean_info_df.shape}")
        
        return clean_motion_data, clean_info_df, valid_frames
    
    def update_to_frame(self, motion_data: np.ndarray, frame_idx: int):
        """
        Convenience method to update pigeon shape to a specific frame from motion data.
        
        Parameters:
        - motion_data (np.ndarray): Clean motion data from load_data_complete
        - frame_idx (int): Frame index to update to
        
        Example:
            motion_data, info_df, _ = pigeon.load_data_complete("data/FullPigeons.csv")
            pigeon.update_to_frame(motion_data, 1000)  # Update to frame 1000
        """
        if frame_idx >= motion_data.shape[0]:
            raise ValueError(f"Frame index {frame_idx} is out of range. Max frame is {motion_data.shape[0]-1}")
        
        # Update current shape with the specified frame
        frame_data = motion_data[frame_idx:frame_idx+1]  # Keep frame dimension
        self.current_shape[:, self.marker_index, :] = frame_data
        self.untransformed_shape = self.current_shape.copy()
        
    
    def remove_nan_frames(self, motion_data: np.ndarray):
        """
        Removes frames with NaN values from the motion data.
        """
        # Remove rows with NaN values
        valid_frames = ~np.isnan(motion_data).any(axis=(1, 2))
        if np.sum(~valid_frames) > 0:
            print(f"Removed {np.sum(~valid_frames)} frames containing NaN values.")
            motion_data = motion_data[valid_frames]
        
        return motion_data, valid_frames
    
    def update_keypoints_from_motion(self, motion_data: np.ndarray, frame_idx: int = 0):
        """
        Updates the keypoints using data from a specific frame of motion data.
        
        Parameters:
        - motion_data (np.ndarray): Motion data in shape [nFrames, nMarkers, 3]
        - frame_idx (int): Index of the frame to use (default: 0)
        """
        if frame_idx >= motion_data.shape[0]:
            raise ValueError(f"Frame index {frame_idx} is out of range. Max frame is {motion_data.shape[0]-1}")
        
        # Get the marker names that should be updated
        if self.use_simple:
            motion_marker_names = self.skeleton_definition.get_marker_names_simple()
        else:
            motion_marker_names = self.skeleton_definition.marker_names
            
        # Remove fixed markers from motion markers
        fixed_markers = self.skeleton_definition.fixed_marker_names
        motion_marker_names = [name for name in motion_marker_names if name not in fixed_markers]
        
        # Build mapping from motion data indices to marker indices
        marker_indices = []
        motion_idx = 0
        
        for name in self.marker_names:
            if name in motion_marker_names:
                try:
                    original_name = self.skeleton_definition.marker_name_change[name]
                    if original_name in self.csv_marker_names:
                        marker_idx = self.csv_marker_names.index(original_name)
                        marker_indices.append(marker_idx)
                        motion_idx += 1
                except KeyError:
                    pass
        
        # Verify shape compatibility
        if motion_data.shape[1] != len(motion_marker_names):
            print(f"WARNING: Motion data has {motion_data.shape[1]} markers but motion_marker_names has {len(motion_marker_names)}. This may cause issues.")
        
        # Get the frame data
        frame_data = motion_data[frame_idx:frame_idx+1]  # Keep the frame dimension
        
        # Update keypoints only for non-fixed markers
        moving_indices = [i for i, name in enumerate(self.marker_names) if name in motion_marker_names]
        
        # Create a new mapping from motion data indices to marker positions in self.current_shape
        for i, motion_idx in enumerate(moving_indices):
            marker_name = self.marker_names[motion_idx]
            if i < frame_data.shape[1]:  # Ensure we don't go out of bounds
                self.current_shape[0, self.marker_index[motion_idx], :] = frame_data[0, i, :]
                
        self.untransformed_shape = self.current_shape.copy()

    def get_motion_data_marker_names(self, use_simple: bool = None) -> list:
        """
        Returns the list of marker names in the order they appear in motion data.
        
        Parameters:
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        
        Returns:
        - list: Marker names in the order they appear in motion data
        """
        # Determine which markers to include
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        
        # Get the appropriate marker list based on simple/detailed mode
        if use_simple_markers:
            motion_marker_names = self.skeleton_definition.get_marker_names_simple()
        else:
            motion_marker_names = self.skeleton_definition.marker_names
            
        # Remove fixed markers from motion markers
        fixed_markers = self.skeleton_definition.fixed_marker_names
        motion_marker_names = [name for name in motion_marker_names if name not in fixed_markers]
        
        return motion_marker_names

    def print_motion_data_info(self, motion_data: np.ndarray, use_simple: bool = None):
        """
        Prints information about the motion data including marker names and shape.
        
        Parameters:
        - motion_data (np.ndarray): The motion data array
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        """
        marker_names = self.get_motion_data_marker_names(use_simple)
        print(f"Motion data shape: {motion_data.shape}")
        print("\nMarker names in order:")
        for i, name in enumerate(marker_names):
            print(f"{i}: {name}")
            
    def print_fixed_vs_moving_markers(self, use_simple: bool = None, verbose: bool = True):
        """
        Prints information about which markers are fixed vs. moving.
        
        Parameters:
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        """
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        
        if use_simple_markers:
            # In simple mode: active markers + fixed markers
            active_markers = self.skeleton_definition.get_marker_names_simple()
            fixed_markers = self.skeleton_definition.fixed_marker_names
            print("\nUsing SIMPLE marker set:")
        else:
            # In full mode: active markers + fixed markers
            active_markers = self.skeleton_definition.get_marker_names_full(verbose=verbose)
            fixed_markers = self.skeleton_definition.fixed_marker_names
            print("\nUsing DETAILED marker set:")
        
        print(f"\nFixed markers ({len(fixed_markers)}):")
        for i, name in enumerate(sorted(fixed_markers)):
            print(f"  {i}: {name}")
        
        print(f"\nMoving markers ({len(active_markers)}):")
        for i, name in enumerate(sorted(active_markers)):
            print(f"  {i}: {name}")
        
        print(f"\nTotal markers: {len(active_markers) + len(fixed_markers)}")
        print(f"  - Fixed: {len(fixed_markers)}")
        print(f"  - Moving: {len(active_markers)}")
        
        # Print which ones will be included in motion data
        print(f"\nMarkers included in motion data: {len(active_markers)}")
        print("Motion data marker order:")
        for i, name in enumerate(active_markers):  # Using original order, not sorted
            print(f"  {i}: {name}")

    def get_marker_pairs_and_centers(self, use_simple: bool = None) -> tuple:
        """
        Returns lists of marker pairs and center markers based on the mode.
        
        Parameters:
        - use_simple (bool, optional): Whether to use simple mode. If None, uses current setting.
        
        Returns:
        - tuple: (left_markers, right_markers, center_markers)
        """
        return self.skeleton_definition.get_marker_pairs_and_centers(use_simple if use_simple is not None else self.use_simple)

    def make_unilateral(self, motion_data: np.ndarray, info_df: pd.DataFrame = None) -> tuple:
        """
        Takes bilateral motion data and creates unilateral data by mirroring left markers
        to match right markers and stacking them. Handles both simple and full modes,
        including center markers in full mode.

        Parameters:
        - motion_data (np.ndarray): Motion data in shape [nFrames, nMarkers, 3]
        - info_df (pd.DataFrame, optional): DataFrame with same number of rows as motion_data frames

        Returns:
        - np.ndarray: Motion data with doubled frames and halved markers (plus centers)
                     Shape will be [nValidFrames*2, nMarkers//2 + nCenters, 3]
        - np.ndarray: Boolean array indicating which frames were originally left
        - pd.DataFrame: Optional unilateral info_df with doubled rows corresponding to motion data
        """
        # Make a hard copy of the motion data   
        motion_data_copy = np.copy(motion_data)
        info_df_copy = info_df.copy() if info_df is not None else None
        
        # Get marker organization
        left_markers, right_markers, center_markers = self.skeleton_definition.get_marker_pairs_and_centers(self.use_simple)
        
        # Get indices for each marker type
        left_indices = [self.marker_names.index(m) for m in left_markers]
        right_indices = [self.marker_names.index(m) for m in right_markers]
        center_indices = [self.marker_names.index(m) for m in center_markers] if center_markers else []
        
        # Validate left-right positions and get valid frame mask
        valid_frames = self.validate_left_right_positions(motion_data_copy, left_indices, right_indices)
        
        # Keep only valid frames
        motion_data_copy = motion_data_copy[valid_frames]
        
        if len(motion_data_copy) == 0:
            raise ValueError("No valid frames found after left-right position validation")
            
        # If info_df is provided, filter it to match valid frames
        if info_df_copy is not None:
            if len(info_df_copy) != len(motion_data):
                raise ValueError("info_df must have same number of rows as motion_data frames")
            info_df_copy = info_df_copy.iloc[valid_frames]
        
        # Extract data for each type
        left_data = motion_data_copy[:, left_indices, :]
        right_data = motion_data_copy[:, right_indices, :]
        center_data = motion_data_copy[:, center_indices, :] if center_indices else None
        
        # Mirror the left side in x to match the right
        left_mirrored = np.copy(left_data)
        left_mirrored[..., 0] *= -1  # Mirror x-coordinate
        
        # Stack mirrored left data and right data
        paired_data = np.concatenate((left_mirrored, right_data), axis=0)
        
        if center_data is not None:
            # Duplicate center data for both halves
            center_data_doubled = np.concatenate((center_data, center_data), axis=0)
            # Combine paired and center data
            unilateral_data = np.concatenate((paired_data, center_data_doubled), axis=1)
        else:
            unilateral_data = paired_data
            
        # If info_df is provided, duplicate it to match the doubled frames
        if info_df_copy is not None:
            unilateral_info_df = pd.concat([info_df_copy, info_df_copy], axis=0, ignore_index=True)
        else:
            unilateral_info_df = None
        
        # Create boolean array marking which frames were originally left
        # Note: now only includes valid frames
        n_valid_frames = len(motion_data_copy)
        is_left = np.zeros(n_valid_frames * 2, dtype=bool)
        is_left[:n_valid_frames] = True  # First half are from left
        
        print(f"\nCreated unilateral data:")
        print(f"  Original frames: {len(motion_data)}")
        print(f"  Valid frames: {n_valid_frames}")
        print(f"  Final shape: {unilateral_data.shape}")
        
        return unilateral_data, is_left, unilateral_info_df

    def make_bilateral(self, unilateral_data: np.ndarray, is_left: np.ndarray) -> np.ndarray:
        """
        Takes unilateral motion data and reconstructs bilateral data by un-mirroring
        the appropriate frames and interleaving left and right markers. Handles both
        simple and full modes, including center markers in full mode.

        Parameters:
        - unilateral_data (np.ndarray): Motion data in shape [nFrames*2, nMarkers//2 + nCenters, 3]
        - is_left (np.ndarray): Boolean or integer array (0/1) indicating which frames were originally left

        Returns:
        - np.ndarray: Reconstructed bilateral motion data in shape [nFrames, nMarkers, 3]
        """
        # Convert is_left to boolean if it's integer
        if is_left.dtype.kind in 'iu':  # integer type
            is_left = is_left.astype(bool)
            
        # Get marker organisation
        left_markers, right_markers, center_markers = self.skeleton_definition.get_marker_pairs_and_centers(self.use_simple)
        n_pairs = len(left_markers)
        n_centers = len(center_markers)
        
        # Make a hard copy of the unilateral data
        unilateral_data_copy = np.copy(unilateral_data)
        n_frames = len(is_left) // 2  # Number of original frames (half of total frames)
        
        if n_centers > 0:
            # Split into paired and center data
            paired_data = unilateral_data_copy[:, :n_pairs, :]
            center_data = unilateral_data_copy[:, n_pairs:, :]
            
            # Split paired data into left and right portions based on is_left
            left_data = paired_data[is_left]
            right_data = paired_data[~is_left]
            
            # Take only first half of center data (they're duplicated)
            center_data = center_data[:n_frames]
        else:
            # All data is paired
            left_data = unilateral_data_copy[is_left]
            right_data = unilateral_data_copy[~is_left]
            center_data = None
        
        # Un-mirror the left data
        left_data[:,:, 0] *= -1
        
        # Create output array with correct number of frames
        total_markers = n_pairs * 2 + n_centers
        bilateral_data = np.zeros((n_frames, total_markers, 3))
        
        # Get indices for each marker type in the final array
        left_indices = [self.marker_names.index(m) for m in left_markers]
        right_indices = [self.marker_names.index(m) for m in right_markers]
        center_indices = [self.marker_names.index(m) for m in center_markers] if center_markers else []
        
        # Place data in correct positions
        bilateral_data[:, left_indices, :] = left_data
        bilateral_data[:, right_indices, :] = right_data
        if center_data is not None:
            bilateral_data[:, center_indices, :] = center_data
        
        return bilateral_data

    def validate_left_right_positions(self, motion_data: np.ndarray, left_indices: list, right_indices: list) -> np.ndarray:
        """
        Validates that left markers are always to the left of their corresponding right markers.
        Returns a boolean mask indicating which frames are valid.
        
        Parameters:
        - motion_data (np.ndarray): Motion data in shape [nFrames, nMarkers, 3]
        - left_indices (list): Indices of left markers
        - right_indices (list): Indices of right markers (in same order as left_indices)
        
        Returns:
        - np.ndarray: Boolean mask of valid frames
        """
        # Get x-coordinates for all frames
        left_x = motion_data[:, left_indices, 0]  # [nFrames, nLeftMarkers]
        right_x = motion_data[:, right_indices, 0]  # [nFrames, nRightMarkers]
        
        # Check if any left marker is to the right of its corresponding right marker
        x_differences = right_x - left_x
        min_difference = 0.001  # 1mm minimum difference
        
        # Frame is valid if all marker pairs have sufficient separation
        valid_frames = np.all(x_differences >= min_difference, axis=1)
        
        # Print information about invalid frames
        n_invalid = np.sum(~valid_frames)
        if n_invalid > 0:
            print(f"\nFound {n_invalid} invalid frames where left-right markers are too close or incorrectly positioned:")
            # Find the first few invalid frames and their issues
            invalid_frame_indices = np.where(~valid_frames)[0]
            for frame_idx in invalid_frame_indices[:5]:  # Show up to 5 examples
                problems = np.where(x_differences[frame_idx] < min_difference)[0]
                for marker_idx in problems:
                    left_marker = self.marker_names[left_indices[marker_idx]]
                    right_marker = self.marker_names[right_indices[marker_idx]]
                    diff = x_differences[frame_idx, marker_idx]
                    print(f"  Frame {frame_idx}: {left_marker} - {right_marker} separation = {diff:.3f}m")
            if len(invalid_frame_indices) > 5:
                print(f"  ... and {len(invalid_frame_indices) - 5} more invalid frames")
        
        return valid_frames
