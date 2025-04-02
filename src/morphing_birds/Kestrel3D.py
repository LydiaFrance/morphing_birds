import numpy as np
from morphing_birds import Animal3D, KestrelSkeletonDefinition

class Kestrel3D(Animal3D):
    """
    A class representing a 3D model of a Kestrel.

    Inherits from Animal3D and provides specific functions for 
    loading CSV data and validating polygon shapes of a kestrel. 
    """
    def __init__(self, csv_path: str, use_simple: bool = False):
        """
        Initializes the Kestrel3D class with the given CSV file path.

        Parameters:
        - csv_path (str): Path to the CSV file containing the kestrel's keypoints.
        - use_simple (bool): If True, uses simplified body sections for polygons.
                           If False, uses detailed body sections. Default is False.
        """
        skeleton_definition = KestrelSkeletonDefinition()
        super().__init__(skeleton_definition)

        self.use_simple = use_simple
        
        # If using simple mode, get only the markers used in simple sections
        if self.use_simple:
            self.marker_names = self.skeleton_definition.get_marker_names_simple()
            # Update fixed markers to use the simple version
            self.skeleton_definition.fixed_marker_names = self.skeleton_definition.fixed_marker_names_simple
        else:
            self.marker_names = self.skeleton_definition.marker_names
        
        # Remove additional fixed markers from moving markers list
        self.marker_names = [name for name in self.marker_names 
                            if name not in self.skeleton_definition.additional_fixed_markers]
        
        self.right_marker_names = [name for name in self.marker_names if name.startswith("right_")]
        self.left_marker_names = [name for name in self.marker_names if name.startswith("left_")]
        
        self.load_csv(csv_path)
        self.init_polygons(self.csv_marker_names)

        # Specify which sections should be colored (rest will be grey)
        self.colour_sections = ["wing", "tail", "alula"]  # Only wing, tail and alula gets the color
        
        
    def load_csv(self, csv_path: str):
        """
        Loads CSV data specific to the kestrel skeleton.

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
            
            # Define marker indices based on CSV marker names
            self.define_indices(self.csv_marker_names)
            
            # Load and validate keypoints
            keypoints = self.get_csv_keypoints(data)
            validated_keypoints = self.validate_keypoints(keypoints)
            
            self.default_shape = validated_keypoints
            self.current_shape = self.default_shape.copy()
            self.untransformed_shape = self.default_shape.copy()
        
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

    def validate_keypoints(self, keypoints):
        """
        Validates the keypoints, ensuring they are three-dimensional, reshapes them if necessary,
        and mirrors them if only one side is provided.

        Parameters:
        - keypoints (numpy.ndarray): The keypoints array to validate.

        Returns:
        - numpy.ndarray: The validated, potentially reshaped, and mirrored keypoints.
        """
        # First, perform basic validation from Animal3D
        keypoints = super().validate_keypoints(keypoints)
        return keypoints
    

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
        
    def define_indices(self, csv_marker_names):
        """
        Defines marker indices for the kestrel skeleton.
        
        This method maps the readable marker names to the original CSV marker names
        and finds their indices in the CSV data.
        
        Parameters:
        - csv_marker_names (list): List of marker names from the CSV in order.
        """
        # Define marker indices only for markers we care about based on use_simple
        self.marker_index = []
        for name in self.marker_names:  # Using filtered marker_names from __init__
            try:
                original_name = self.skeleton_definition.marker_name_change[name]
                if original_name in csv_marker_names:
                    self.marker_index.append(csv_marker_names.index(original_name))
                else:
                    print(f"Original name '{original_name}' not found in CSV markers for '{name}'")
            except KeyError:
                print(f"Warning: Missing translation for marker: {name}")
        
        # Define fixed marker indices
        self.fixed_marker_index = []
        fixed_markers = (self.skeleton_definition.fixed_marker_names_simple 
                        if self.use_simple 
                        else self.skeleton_definition.fixed_marker_names)
        
        for name in fixed_markers:
            try:
                original_name = self.skeleton_definition.marker_name_change[name]
                if original_name in csv_marker_names:
                    self.fixed_marker_index.append(csv_marker_names.index(original_name))
                else:
                    print(f"Original name '{original_name}' not found in CSV markers for fixed '{name}'")
            except KeyError:
                print(f"Warning: Missing translation for fixed marker: {name}")
                
        # Define right and left marker indices only for markers we're using
        self.right_marker_index = []
        for name in self.right_marker_names:
            try:
                original_name = self.skeleton_definition.marker_name_change[name]
                if original_name in csv_marker_names:
                    self.right_marker_index.append(csv_marker_names.index(original_name))
                else:
                    print(f"Original name '{original_name}' not found in CSV markers for right '{name}'")
            except KeyError:
                print(f"Warning: Missing translation for right marker: {name}")
                
        self.left_marker_index = []
        for name in self.left_marker_names:
            try:
                original_name = self.skeleton_definition.marker_name_change[name]
                if original_name in csv_marker_names:
                    self.left_marker_index.append(csv_marker_names.index(original_name))
                else:
                    print(f"Original name '{original_name}' not found in CSV markers for left '{name}'")
            except KeyError:
                print(f"Warning: Missing translation for left marker: {name}")

    def init_polygons(self, csv_marker_names):
        """
        Initializes polygon definitions for kestrels, handling the translation between
        readable marker names and original CSV marker names.
        """
        # Create a list of marker names that the Animal3D class can use
        # This maps the readable names in body_sections to their original CSV names
        translated_csv_marker_names = []
        
        # Create a translation dictionary from original to readable
        original_to_readable = {}
        for readable, original in self.skeleton_definition.marker_name_change.items():
            original_to_readable[original] = readable
        
        # For each CSV marker name, find its readable equivalent if it exists
        for csv_name in csv_marker_names:
            if csv_name in original_to_readable:
                translated_csv_marker_names.append(original_to_readable[csv_name])
            else:
                # Keep the original name if no translation exists
                translated_csv_marker_names.append(csv_name)
        
        # Now use the translated marker names to initialize polygons
        self.body_section_indices = {}
        
        # Filter body sections based on use_simple parameter
        for section, markers in self.skeleton_definition.body_sections.items():
            # Skip sections that don't match our simple/detailed preference
            if self.use_simple:
                if not section.endswith('_simple'):
                    continue
            else:
                if section.endswith('_simple'):
                    continue
                    
            indices = []
            for marker in markers:
                # Find the index of this marker in the translated names
                if marker in translated_csv_marker_names:
                    idx = translated_csv_marker_names.index(marker)
                    indices.append(idx)
                else:
                    print(f"Warning: Marker '{marker}' from section '{section}' not found in translated CSV marker names")
            
            # Only add the section if we found at least one marker
            if indices:
                # Remove '_simple' suffix from section name if using simple mode
                section_name = section.replace('_simple', '') if self.use_simple else section
                self.body_section_indices[section_name] = indices
        
        # Set up polygons dictionary
        self.polygons = {}
        for section, indices in self.body_section_indices.items():
            self.polygons[section] = indices

    def load_motion_data(self, csv_path: str, use_simple: bool = None) -> np.ndarray:
        """
        Loads motion data from a CSV file and returns it in the correct format for update_keypoints.
        
        Parameters:
        - csv_path (str): Path to the CSV file containing motion data
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        
        Returns:
        - np.ndarray: Motion data in shape [nFrames, nMarkers, 3]
        """
        # Load the CSV data
        data = self.load_csv_data(csv_path)
        csv_headers = data[0]
        csv_marker_names = self.get_csv_marker_names(csv_headers)
        
        # Determine which markers to include
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        
        # Get the appropriate marker list based on simple/detailed mode
        if use_simple_markers:
            motion_marker_names = self.skeleton_definition.get_marker_names_simple()
        else:
            motion_marker_names = self.skeleton_definition.marker_names
            
        # Remove fixed markers from motion markers
        motion_marker_names = [name for name in motion_marker_names 
                              if name not in self.skeleton_definition.additional_fixed_markers]
        
        # Build the CSV column indices for all the markers we need
        marker_indices = []
        not_found = []
        
        for name in motion_marker_names:
            # Find the original name in the CSV
            try:
                original_name = self.skeleton_definition.marker_name_change[name]
                if original_name in csv_marker_names:
                    marker_indices.append(csv_marker_names.index(original_name))
                else:
                    not_found.append(f"{name} (original: {original_name})")
            except KeyError:
                not_found.append(name)
        
        if not_found:
            print(f"Warning: Could not find {len(not_found)} markers in CSV: {', '.join(not_found)}")
            
        if not marker_indices:
            raise ValueError("No valid markers found in CSV for selected marker set")
        
        # Convert data to float and reshape
        motion_data = data[1:].astype(float)  # Skip header row
        n_frames = motion_data.shape[0]
        
        # Reshape to [nFrames, nTotal, 3]
        total_markers = len(csv_marker_names)
        motion_data = motion_data.reshape(n_frames, total_markers, 3)
        
        # Extract only the columns we want
        motion_data = motion_data[:, marker_indices, :]

        return motion_data
    
    def remove_nan_frames(self, motion_data: np.ndarray):
        """
        Removes frames with NaN values from the motion data.
        """
        # Remove rows with NaN values
        valid_frames = ~np.isnan(motion_data).any(axis=(1, 2))
        if np.sum(~valid_frames) > 0:
            print(f"Removed {np.sum(~valid_frames)} frames containing NaN values.")
            motion_data = motion_data[valid_frames]
        
        return motion_data
    
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
        motion_marker_names = [name for name in motion_marker_names 
                              if name not in self.skeleton_definition.additional_fixed_markers]
        
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
        motion_marker_names = [name for name in motion_marker_names 
                              if name not in self.skeleton_definition.additional_fixed_markers]
        
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
            
    def print_fixed_vs_moving_markers(self, use_simple: bool = None):
        """
        Prints information about which markers are fixed vs. moving.
        
        Parameters:
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        """
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        
        if use_simple_markers:
            all_markers = self.skeleton_definition.get_marker_names_simple()
            built_in_fixed = self.skeleton_definition.fixed_marker_names_simple
            print("\nUsing SIMPLE marker set:")
        else:
            all_markers = self.skeleton_definition.marker_names
            built_in_fixed = self.skeleton_definition.fixed_marker_names
            print("\nUsing DETAILED marker set:")
        
        # Combine built-in fixed with additional fixed markers
        all_fixed_markers = list(set(built_in_fixed + self.skeleton_definition.additional_fixed_markers))
        
        # Get moving markers (not fixed)
        moving_markers = [name for name in all_markers if name not in all_fixed_markers]
        
        print(f"\nFixed markers ({len(all_fixed_markers)}):")
        for i, name in enumerate(sorted(all_fixed_markers)):
            source = " (additional)" if name in self.skeleton_definition.additional_fixed_markers else " (built-in)"
            print(f"  {i}: {name}{source}")
        
        print(f"\nMoving markers ({len(moving_markers)}):")
        for i, name in enumerate(sorted(moving_markers)):
            print(f"  {i}: {name}")
        
        print(f"\nTotal markers: {len(all_markers)}")
        print(f"  - Fixed: {len(all_fixed_markers)}")
        print(f"  - Moving: {len(moving_markers)}")
        
        # Print which ones will actually be included in motion data
        motion_marker_names = self.get_motion_data_marker_names(use_simple)
        print(f"\nMarkers included in motion data: {len(motion_marker_names)}")

    def print_debug_info(self):
        """
        Prints debugging information about the current state of the Kestrel3D object.
        """
        print(f"Number of marker names: {len(self.marker_names)}")
        print(f"Number of marker indices: {len(self.marker_index)}")
        
        if len(self.marker_names) != len(self.marker_index):
            print("WARNING: Mismatch between marker_names and marker_index lengths!")
            
        # Check for marker indices that might be out of bounds for current_shape
        if hasattr(self, 'current_shape'):
            total_markers = self.current_shape.shape[1]
            out_of_bounds = [i for i in self.marker_index if i >= total_markers]
            if out_of_bounds:
                print(f"WARNING: The following marker indices are out of bounds: {out_of_bounds}")
        
        # Print the first few marker names and indices
        print("\nFirst 10 marker names and indices:")
        for i, name in enumerate(self.marker_names[:10]):
            if i < len(self.marker_index):
                print(f"{i}: {name} -> index {self.marker_index[i]}")
            else:
                print(f"{i}: {name} -> NO INDEX")
        
        print("\nPolygon definitions:")
        for section, indices in self.polygons.items():
            print(f"{section}: {len(indices)} points")