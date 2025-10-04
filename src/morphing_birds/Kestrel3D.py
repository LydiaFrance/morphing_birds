import numpy as np
from morphing_birds import Animal3D, KestrelSkeletonDefinition
import pandas as pd
import copy

class Kestrel3D(Animal3D):
    """
    A class representing a 3D model of a Kestrel.

    Inherits from Animal3D and provides specific functions for 
    loading CSV data and validating polygon shapes of a kestrel. 
    """
    def __init__(self, csv_path: str, use_simple: bool = False, use_full: bool = False, use_mean_alula: bool = False):
        """
        Initialise the Kestrel3D class.
        
        Args:
            csv_path (str): Path to the CSV file containing marker data
            use_simple (bool): If True, uses simplified body sections (hawk-comparable).
                             Overrides use_full. Default is False.
            use_full (bool): If True, uses full detail sections including elbow.
                           If False, uses standard sections without elbow. 
                           Ignored if use_simple=True. Default is False.
            use_mean_alula (bool): If True, replaces individual alula markers with their mean positions
                                 to reduce noise from varying definitions. 
                                 Only relevant when use_simple=False. Default is False.
        
        Mode hierarchy (in order of precedence):
        1. use_simple=True → simple sections (no alula, no elbow)
        2. use_full=True → full sections (with elbow, with alula)
        3. Default → standard sections (no elbow, with alula)
        """
        skeleton_definition = KestrelSkeletonDefinition()
        super().__init__(skeleton_definition)
        
        self.use_simple = use_simple
        self.use_full = use_full
        self.use_mean_alula = use_mean_alula
        
        # Load the CSV data
        self.data = pd.read_csv(csv_path)
        
        # Get marker names in canonical order
        self.marker_names = self._get_marker_names()
        fixed_markers = self._get_fixed_marker_names()
        
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
        
        # Apply alula averaging if requested - this affects the skeleton structure
        if self.use_mean_alula:
            marker_data = self._apply_alula_averaging(marker_data)
            # Update marker names to reflect the averaged markers
            self.marker_names = self._update_marker_names_for_averaging(self.marker_names)
            # Update body sections to use averaged marker names
            self._update_body_sections_for_averaging()
            
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
        
        # Initialise polygons after markers are set up
        self.init_polygons()
        
        # Define indices for active markers
        self.define_indices()
        
        # Specify which sections should be coloured (rest will be grey)
        self.colour_sections = ["wing", "tail", "alula"]  # Only wing, tail and alula gets the colour

    # Remove the markers property override - use parent class implementation
    # The parent Animal3D.markers property correctly returns current_shape[:,marker_index,:]
    # which gets updated by update_keypoints()

    
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
        """Initialise the polygons for visualisation."""
        # Get the appropriate sections based on mode
        if self.use_simple:
            # Simple mode (overrides use_full)
            sections = [s for s in self.skeleton_definition.body_sections if s.endswith("_simple")]
            # Remove '_simple' suffix for the polygon dictionary
            sections_dict = {s.replace('_simple', ''): self.skeleton_definition.body_sections[s] 
                           for s in sections}
        elif self.use_full:
            # Full mode (with elbow)
            sections = [s for s in self.skeleton_definition.body_sections if s.endswith("_full")]
            # Remove '_full' suffix for the polygon dictionary
            sections_dict = {s.replace('_full', ''): self.skeleton_definition.body_sections[s] 
                           for s in sections}
        else:
            # Standard mode (no elbow, default)
            sections = [s for s in self.skeleton_definition.body_sections 
                       if not s.endswith("_simple") and not s.endswith("_full")]
            sections_dict = {s: self.skeleton_definition.body_sections[s] for s in sections}
        
        fixed_markers = self._get_fixed_marker_names()

        # Convert marker names to indices
        self.polygons = {}
        for section, markers in sections_dict.items():
            # Get indices for each marker in the section
            try:
                # First try to find the marker in marker_names (for active markers)
                indices = []
                for marker in markers:
                    if marker in self.marker_names:
                        idx = self.marker_names.index(marker)
                        indices.append(self.marker_index[idx])
                    elif marker in fixed_markers:
                        # For fixed markers, find their position in the fixed marker list
                        fixed_idx = fixed_markers.index(marker)
                        indices.append(self.fixed_marker_index[fixed_idx])
                    else:
                        raise ValueError(f"Marker {marker} not found in either active or fixed markers")
                self.polygons[section] = indices
            except ValueError as e:
                print(f"Warning: Could not initialise polygon for section {section}: {str(e)}")
                continue

        # Print debug info about polygons
        print("\nInitialised polygons:")
        for section, indices in self.polygons.items():
            print(f"{section}: {indices}")

    def define_indices(self):
        """Define indices for active markers in canonical order."""
        self.marker_index = list(range(len(self.marker_names)))

    def _get_marker_names(self, use_simple: bool = None, use_full: bool = None) -> list:
        """
        Get marker names based on mode settings.
        
        Args:
            use_simple: Override for use_simple setting. If None, uses self.use_simple
            use_full: Override for use_full setting. If None, uses self.use_full
            
        Returns:
            List of marker names for the specified mode
        """
        use_simple = self.use_simple if use_simple is None else use_simple
        use_full = self.use_full if use_full is None else use_full
        
        if use_simple:
            return self.skeleton_definition.get_marker_names_simple()
        else:
            marker_names = self.skeleton_definition.get_marker_names_full()
            if not use_full:
                marker_names = [name for name in marker_names if 'elbow' not in name]
            return marker_names

    def _get_fixed_marker_names(self, use_simple: bool = None) -> list:
        """
        Get fixed marker names based on mode.
        
        Args:
            use_simple: Override for use_simple setting. If None, uses self.use_simple
            
        Returns:
            List of fixed marker names for the specified mode
        """
        use_simple = self.use_simple if use_simple is None else use_simple
        return (self.skeleton_definition.fixed_marker_names_simple 
                if use_simple else self.skeleton_definition.fixed_marker_names)

    def systematic_debug_check(self):
        """Comprehensive debug check to understand the data flow."""
        print("=== SYSTEMATIC DEBUG CHECK ===")
        
        print("\n1. SKELETON CONFIGURATION:")
        print(f"   use_simple: {self.use_simple}")
        print(f"   use_full: {self.use_full}")
        print(f"   use_mean_alula: {self.use_mean_alula}")
        
        print("\n2. SKELETON MARKER COUNTS:")
        full_markers = self.skeleton_definition.get_marker_names_full()
        print(f"   Full skeleton markers: {len(full_markers)}")
        filtered_markers = self._get_marker_names()
        print(f"   Filtered markers (current): {len(filtered_markers)}")
        print(f"   self.marker_names count: {len(self.marker_names)}")
        
        print("\n3. SKELETON MARKER DETAILS:")
        print(f"   Has left_alula_mean: {'left_alula_mean' in self.marker_names}")
        print(f"   Has right_alula_mean: {'right_alula_mean' in self.marker_names}")
        print(f"   Has left_alula: {'left_alula' in self.marker_names}")
        print(f"   Has right_alula: {'right_alula' in self.marker_names}")
        
        print("\n4. ARRAY SIZES:")
        print(f"   current_shape: {self.current_shape.shape}")
        print(f"   marker_index length: {len(self.marker_index)}")
        print(f"   marker_index: {self.marker_index}")
        
        print("\n5. MARKER NAME LISTS:")
        print(f"   First 5 skeleton markers: {self.marker_names[:5]}")
        print(f"   Last 5 skeleton markers: {self.marker_names[-5:]}")
        
    def debug_motion_data(self, motion_data, motion_csv_path=None):
        """Debug motion data structure."""
        print(f"\n=== MOTION DATA DEBUG ===")
        print(f"Motion data shape: {motion_data.shape}")
        print(f"Number of frames: {motion_data.shape[0]}")
        print(f"Number of markers: {motion_data.shape[1]}")
        print(f"Coordinates per marker: {motion_data.shape[2]}")
        
        # Try to guess what markers these might be
        motion_marker_count = motion_data.shape[1]
        
        if motion_marker_count == 24:
            print("Likely contains: averaged alula markers (no individual alula)")
        elif motion_marker_count == 28:
            print("Likely contains: individual alula markers (left_alula, left_alula_lower, etc.)")
        elif motion_marker_count == 26:
            print("Likely contains: skeleton's expected markers with averaged alula")
        else:
            print(f"Unknown marker configuration for count: {motion_marker_count}")
            
        # If we have the CSV path, let's see what markers were actually found
        if motion_csv_path:
            print(f"\n=== MOTION CSV ANALYSIS ===")
            try:
                temp_data = pd.read_csv(motion_csv_path)
                csv_columns = temp_data.columns.tolist()
                print(f"Total CSV columns: {len(csv_columns)}")
                
                # Try to identify which markers were found using the skeleton definition
                found_markers = []
                for readable_name, csv_name in self.skeleton_definition.marker_name_change.items():
                    x_col = f"{csv_name}x"
                    if x_col in csv_columns:
                        found_markers.append(readable_name)
                        
                print(f"Markers found in CSV: {len(found_markers)}")
                print(f"Found markers: {found_markers}")
                
                # Check for missing expected markers
                expected_markers = set(self.marker_names)
                found_markers_set = set(found_markers)
                missing = expected_markers - found_markers_set
                extra = found_markers_set - expected_markers
                
                if missing:
                    print(f"Missing from CSV: {list(missing)}")
                if extra:
                    print(f"Extra in CSV: {list(extra)}")
                    
            except Exception as e:
                print(f"Could not analyze CSV: {e}")
        
    def update_keypoints(self, user_keypoints):
        """
        Override the parent's update_keypoints to handle motion data conversion automatically.
        
        This handles the case where motion data has individual alula markers but 
        skeleton expects averaged alula markers.
        """
        # Handle None case - pass to parent to reset to default
        if user_keypoints is None:
            from .Animal3D import Animal3D
            Animal3D.update_keypoints(self, user_keypoints)
            return
            
        # Get the shape of incoming keypoints
        if hasattr(user_keypoints, 'shape'):
            input_shape = user_keypoints.shape
        else:
            user_keypoints = np.array(user_keypoints)
            input_shape = user_keypoints.shape
            
        # If we have a 2D array, reshape to 3D for consistency
        if len(input_shape) == 2:
            user_keypoints = user_keypoints.reshape(1, input_shape[0], input_shape[1])
            input_shape = user_keypoints.shape
            
        expected_markers = len(self.marker_names)
        provided_markers = input_shape[1]
        
        # Check for marker count mismatch
        if provided_markers != expected_markers:
            print(f"Marker count mismatch:")
            print(f"  Motion data: {provided_markers} markers")
            print(f"  Skeleton expects: {expected_markers} markers")
            print(f"  use_mean_alula: {self.use_mean_alula}")
            
            # This shouldn't happen now that load_motion_data is fixed
            raise ValueError(f"Marker count mismatch: provided {provided_markers}, expected {expected_markers}. "
                           f"This suggests load_motion_data didn't return the correct marker count.")
            
        # Now call the parent's update_keypoints method
        from .Animal3D import Animal3D
        Animal3D.update_keypoints(self, user_keypoints)

    def _debug_print_marker_info(self, context: str = ""):
        """Consolidated debug printing for marker information."""
        if context:
            print(f"\n=== {context} ===")
        print(f"Input keypoints shape: {self._markers.shape}")
        print(f"Number of markers in self.marker_names: {len(self.marker_names)}")
        print(f"Number of markers in self.marker_index: {len(self.marker_index)}")
        print(f"Marker names: {self.marker_names}")
        print(f"Marker indices: {self.marker_index}")
        print(f"Fixed marker indices: {self.fixed_marker_index}")

    def _print_mode_info(self, use_simple: bool = None):
        """Print information about current marker mode."""
        use_simple = self.use_simple if use_simple is None else use_simple
        mode = "SIMPLE" if use_simple else "DETAILED"
        print(f"\nUsing {mode} marker set:")

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

    def copy(self) -> 'Kestrel3D':
        """
        Create a deep copy of this Kestrel3D instance.
        
        Returns:
            A new Kestrel3D instance that is a deep copy of this one
        """
        return copy.deepcopy(self)
    
    def _apply_alula_averaging(self, marker_data: dict) -> dict:
        """
        Apply alula averaging to the marker data, replacing individual alula markers
        with their mean positions to reduce noise.
        
        Args:
            marker_data (dict): Dictionary of marker positions
            
        Returns:
            dict: Updated marker data with averaged alula positions
        """
        try:
            # Use the skeleton definition method to get averaged alula data
            processed_data = self.skeleton_definition.get_marker_data_with_mean_alula(marker_data)
            
            print(f"Applied alula averaging:")
            print(f"  - Replaced left_alula + left_alula_lower with left_alula_mean")
            print(f"  - Replaced right_alula + right_alula_lower with right_alula_mean")
            
            return processed_data
            
        except KeyError as e:
            print(f"Warning: Could not apply alula averaging - {e}")
            print("Continuing with original marker data...")
            return marker_data
        except Exception as e:
            print(f"Warning: Error during alula averaging - {e}")
            print("Continuing with original marker data...")
            return marker_data
    
    def _update_marker_names_for_averaging(self, marker_names: list) -> list:
        """
        Update marker names list to use averaged alula names instead of individual ones.
        
        Args:
            marker_names (list): Original list of marker names
            
        Returns:
            list: Updated marker names with alula averaging applied
        """
        updated_names = []
        
        for name in marker_names:
            if name == 'left_alula':
                # Replace with mean
                updated_names.append('left_alula_mean')
            elif name == 'right_alula':
                # Replace with mean
                updated_names.append('right_alula_mean')
            elif name in ['left_alula_lower', 'right_alula_lower']:
                # Skip these as they're now averaged into the mean
                continue
            else:
                # Keep the original marker name
                updated_names.append(name)
        
        print(f"Updated marker names for alula averaging:")
        print(f"  Original count: {len(marker_names)} -> Updated count: {len(updated_names)}")
        
        return updated_names
    
    def _update_body_sections_for_averaging(self):
        """
        Update body sections to use averaged alula marker names instead of individual ones.
        This modifies the skeleton_definition's body_sections dictionary.
        """
        updated_sections = {}
        
        for section_name, markers in self.skeleton_definition.body_sections.items():
            updated_markers = []
            for marker in markers:
                if marker == 'left_alula':
                    updated_markers.append('left_alula_mean')
                elif marker == 'right_alula':
                    updated_markers.append('right_alula_mean')
                elif marker in ['left_alula_lower', 'right_alula_lower']:
                    # Skip these as they're now part of the mean
                    continue
                else:
                    updated_markers.append(marker)
            updated_sections[section_name] = updated_markers
        
        # Update the skeleton definition
        self.skeleton_definition.body_sections = updated_sections
        
        print(f"Updated body sections for alula averaging:")
        original_alula_sections = [s for s in self.skeleton_definition.body_sections.keys() if 'alula' in s]
        print(f"  Alula sections updated: {original_alula_sections}")
    
    def _apply_alula_averaging_to_motion_data(self, motion_data: np.ndarray, marker_names: list) -> np.ndarray:
        """
        Apply alula averaging to motion data array.
        
        Args:
            motion_data (np.ndarray): Motion data in shape [nFrames, nMarkers, 3]
            marker_names (list): List of marker names corresponding to motion_data columns
            
        Returns:
            np.ndarray: Motion data with averaged alula positions
        """
        try:
            # Find alula marker indices
            alula_indices = {}
            for marker in ['left_alula', 'left_alula_lower', 'right_alula', 'right_alula_lower']:
                if marker in marker_names:
                    alula_indices[marker] = marker_names.index(marker)
            
            # Check if we have all required markers
            required_markers = ['left_alula', 'left_alula_lower', 'right_alula', 'right_alula_lower']
            missing_markers = [m for m in required_markers if m not in alula_indices]
            
            if missing_markers:
                print(f"Warning: Cannot apply alula averaging to motion data - missing markers: {missing_markers}")
                return motion_data
            
            # Create new motion data array with fewer markers
            n_frames = motion_data.shape[0]
            
            # Calculate mean positions
            left_alula_mean = (motion_data[:, alula_indices['left_alula'], :] + 
                              motion_data[:, alula_indices['left_alula_lower'], :]) / 2.0
            right_alula_mean = (motion_data[:, alula_indices['right_alula'], :] + 
                               motion_data[:, alula_indices['right_alula_lower'], :]) / 2.0
            
            # Create new marker names list
            new_marker_names = []
            new_motion_data_list = []
            
            for i, marker in enumerate(marker_names):
                if marker == 'left_alula':
                    # Replace with mean
                    new_marker_names.append('left_alula_mean')
                    new_motion_data_list.append(left_alula_mean)
                elif marker == 'right_alula':
                    # Replace with mean  
                    new_marker_names.append('right_alula_mean')
                    new_motion_data_list.append(right_alula_mean)
                elif marker in ['left_alula_lower', 'right_alula_lower']:
                    # Skip these as they're now averaged
                    continue
                else:
                    # Keep original marker
                    new_marker_names.append(marker)
                    new_motion_data_list.append(motion_data[:, i, :])
            
            # Stack the new motion data
            new_motion_data = np.stack(new_motion_data_list, axis=1)
            
            return new_motion_data
            
        except Exception as e:
            print(f"Warning: Error during motion data alula averaging - {e}")
            print("Continuing with original motion data...")
            return motion_data
    
    def _update_marker_names_for_averaging(self, marker_names: list) -> list:
        """Update marker names list to use averaged alula names instead of individual ones."""
        updated_names = []
        for name in marker_names:
            if name == 'left_alula':
                updated_names.append('left_alula_mean')
            elif name == 'right_alula':
                updated_names.append('right_alula_mean')
            elif name in ['left_alula_lower', 'right_alula_lower']:
                continue  # Skip these as they're now averaged
            else:
                updated_names.append(name)
        
        
        return updated_names

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
        
        # Get marker organisation
        left_markers, right_markers, centre_markers = self.skeleton_definition.get_marker_pairs_and_centres(self.use_simple)
        n_pairs = len(left_markers)
        n_centres = len(centre_markers)
        expected_bilateral = n_pairs * 2 + n_centres
        
        # If we have unilateral data (half the expected markers), mirror it
        if keypoints.shape[1] == n_pairs + n_centres:
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
            
            # Handle centre markers if any
            if n_centres > 0:
                centre_indices_in = range(n_pairs, n_pairs + n_centres)  # Indices in input
                centre_indices_out = [self.marker_names.index(m) for m in centre_markers]  # Indices in output
                bilateral_data[:, centre_indices_out, :] = keypoints[:, centre_indices_in, :]
            
            return bilateral_data
            
        return keypoints

    def mirror_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Creates a full set of keypoints by mirroring to create left-side markers.
        Used for visualisation when only right-side markers are provided.
        
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
        
    def load_motion_data(self, csv_path: str, use_simple: bool = None, use_mean_alula: bool = None) -> np.ndarray:
        """
        Loads motion data from a CSV file and returns it in the correct format for update_keypoints.
        
        The motion data will match the skeleton's configuration exactly. If the skeleton was initialized
        with use_mean_alula=True, the motion data will also have averaged alula markers.
        
        Parameters:
        - csv_path (str): Path to the CSV file containing motion data
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        - use_mean_alula (bool, optional): Whether to apply alula averaging. If None, uses the current setting.
        
        Returns:
        - np.ndarray: Motion data in shape [nFrames, nMarkers, 3] matching skeleton expectations
        """
        # Load the CSV data
        data = self.load_csv_data(csv_path)
        csv_headers = data[0]
        csv_marker_names = self.get_csv_marker_names(csv_headers)
        
        # Determine which markers to include based on mode
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        use_mean_alula_markers = self.use_mean_alula if use_mean_alula is None else use_mean_alula
        use_full_markers = self.use_full if use_simple_markers else self.use_full
        
        # The target is to match exactly what the skeleton expects
        target_marker_names = self.marker_names  # This is what the skeleton expects
        
        # Strategy: Load available markers from CSV, then process to match target
        available_markers = {}
        not_found = []
        
        # First, try to load all markers that exist in the CSV
        for marker_name in self.skeleton_definition.marker_name_change.keys():
            if marker_name in self.skeleton_definition.ignored_marker_names:
                continue
                
            try:
                original_name = self.skeleton_definition.marker_name_change[marker_name]
                if original_name in csv_marker_names:
                    idx = csv_marker_names.index(original_name)
                    available_markers[marker_name] = idx
                else:
                    not_found.append(f"{marker_name} (original: {original_name})")
            except KeyError:
                not_found.append(marker_name)
        
        # Convert data to float and reshape
        motion_data = data[1:].astype(float)  # Skip header row
        n_frames = motion_data.shape[0]
        n_columns = motion_data.shape[1]
        
        # The CSV might have a 'frame' column or other non-marker columns
        # We need to extract only the marker coordinate columns
        coordinate_columns = []
        for i, col_name in enumerate(csv_headers):
            if col_name.endswith(('x', 'y', 'z')):
                coordinate_columns.append(i)
        
        # Extract only coordinate data
        coordinate_data = motion_data[:, coordinate_columns]
        n_coordinate_columns = coordinate_data.shape[1]
        
        # Verify we have a multiple of 3 columns (x, y, z for each marker)
        if n_coordinate_columns % 3 != 0:
            raise ValueError(f"Number of coordinate columns ({n_coordinate_columns}) is not divisible by 3")
        
        total_markers = n_coordinate_columns // 3
        motion_data = coordinate_data.reshape(n_frames, total_markers, 3)
        
        # Now build motion data to match target_marker_names exactly
        # First, create a mapping from CSV marker names to motion data indices
        csv_marker_to_index = {}
        marker_idx = 0
        for i, col_name in enumerate(csv_headers):
            if col_name.endswith('x'):
                # This is the start of a marker (x coordinate)
                marker_name = col_name[:-1]  # Remove 'x'
                if marker_name.endswith('_'):
                    marker_name = marker_name[:-1]  # Remove trailing underscore
                csv_marker_to_index[marker_name] = marker_idx
                marker_idx += 1
        
        result_data = []
        result_marker_names = []
        
        for target_marker in target_marker_names:
            marker_found = False
            
            # Try to find the target marker in available_markers (direct mapping)
            if target_marker in available_markers:
                # Find the corresponding CSV marker name
                original_name = None
                try:
                    original_name = self.skeleton_definition.marker_name_change[target_marker]
                except KeyError:
                    pass
                
                # Look for this marker in our CSV mapping
                if original_name and original_name in csv_marker_to_index:
                    idx = csv_marker_to_index[original_name]
                    result_data.append(motion_data[:, idx, :])
                    result_marker_names.append(target_marker)
                    marker_found = True
            
            # Special handling for mean alula markers
            if not marker_found and target_marker.endswith('_mean') and use_mean_alula_markers:
                if target_marker == 'left_alula_mean':
                    # Try to compute from individual markers
                    left_alula_orig = self.skeleton_definition.marker_name_change.get('left_alula', 'l_al_1')
                    left_alula_lower_orig = self.skeleton_definition.marker_name_change.get('left_alula_lower', 'l_al_3')
                    
                    if left_alula_orig in csv_marker_to_index and left_alula_lower_orig in csv_marker_to_index:
                        alula_idx = csv_marker_to_index[left_alula_orig]
                        alula_lower_idx = csv_marker_to_index[left_alula_lower_orig]
                        mean_data = (motion_data[:, alula_idx, :] + motion_data[:, alula_lower_idx, :]) / 2.0
                        result_data.append(mean_data)
                        result_marker_names.append(target_marker)
                        marker_found = True
                        
                elif target_marker == 'right_alula_mean':
                    # Try to compute from individual markers
                    right_alula_orig = self.skeleton_definition.marker_name_change.get('right_alula', 'r_al_1')
                    right_alula_lower_orig = self.skeleton_definition.marker_name_change.get('right_alula_lower', 'r_al_3')
                    
                    if right_alula_orig in csv_marker_to_index and right_alula_lower_orig in csv_marker_to_index:
                        alula_idx = csv_marker_to_index[right_alula_orig]
                        alula_lower_idx = csv_marker_to_index[right_alula_lower_orig]
                        mean_data = (motion_data[:, alula_idx, :] + motion_data[:, alula_lower_idx, :]) / 2.0
                        result_data.append(mean_data)
                        result_marker_names.append(target_marker)
                        marker_found = True
            
            # If marker still not found, use zeros
            if not marker_found:
                zero_data = np.zeros((n_frames, 3))
                result_data.append(zero_data)
                result_marker_names.append(target_marker)
        
        # Stack the result data
        if result_data:
            result_motion_data = np.stack(result_data, axis=1)
        else:
            raise ValueError("No valid markers found for motion data")
        
        # Verify the shape matches expectations
        if result_motion_data.shape[1] != len(target_marker_names):
            raise ValueError(f"Motion data shape mismatch: got {result_motion_data.shape[1]} markers, expected {len(target_marker_names)}")
        
        return result_motion_data
    
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
    
    def update_keypoints_from_motion(self, motion_data: np.ndarray, frame_idx: int = 0, 
                                    motion_marker_names: list = None):
        """
        Updates the keypoints using data from a specific frame of motion data.
        
        Parameters:
        - motion_data (np.ndarray): Motion data in shape [nFrames, nMarkers, 3]
        - frame_idx (int): Index of the frame to use (default: 0)
        - motion_marker_names (list, optional): Names of markers in motion_data. If None, uses current marker names.
        """
        if frame_idx >= motion_data.shape[0]:
            raise ValueError(f"Frame index {frame_idx} is out of range. Max frame is {motion_data.shape[0]-1}")
        
        # Smart alula averaging: if motion data has individual alula markers but skeleton expects averaged ones
        if motion_marker_names is not None:
            expected_markers = set(self.marker_names)
            provided_markers = set(motion_marker_names)
            
            # Check if we need to apply alula averaging
            skeleton_has_averaged_alula = 'left_alula_mean' in expected_markers and 'right_alula_mean' in expected_markers
            motion_has_individual_alula = all(marker in provided_markers for marker in 
                                            ['left_alula', 'left_alula_lower', 'right_alula', 'right_alula_lower'])
            
            if skeleton_has_averaged_alula and motion_has_individual_alula:
                print("Converting individual alula markers to averaged ones...")
                motion_data = self._apply_alula_averaging_to_motion_data(motion_data, motion_marker_names)
                # Update the marker names to match what the skeleton expects
                updated_names = []
                for name in motion_marker_names:
                    if name == 'left_alula':
                        updated_names.append('left_alula_mean')
                    elif name == 'right_alula':
                        updated_names.append('right_alula_mean')
                    elif name in ['left_alula_lower', 'right_alula_lower']:
                        continue  # Skip these
                    else:
                        updated_names.append(name)
                motion_marker_names = updated_names
        
        # Get the marker names that should be updated (if not provided)
        if motion_marker_names is None:
            motion_marker_names = self._get_marker_names()
            
        # Remove fixed markers from motion markers
        fixed_markers = self._get_fixed_marker_names()
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
        use_full_markers = self.use_full if use_simple_markers else self.use_full
        
        # Get the appropriate marker list based on mode
        motion_marker_names = self._get_marker_names(use_simple_markers, use_full_markers)
            
        # Remove fixed markers from motion markers
        fixed_markers = self._get_fixed_marker_names(use_simple_markers)
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
            
    def print_fixed_vs_moving_markers(self, use_simple: bool = None):
        """
        Prints information about which markers are fixed vs. moving.
        
        Parameters:
        - use_simple (bool, optional): Whether to use simple marker set. If None, uses the current setting.
        """
        use_simple_markers = self.use_simple if use_simple is None else use_simple
        use_full_markers = self.use_full if use_simple_markers else self.use_full
        
        # Get marker lists using helper methods
        active_markers = self._get_marker_names(use_simple_markers, use_full_markers)
        fixed_markers = self._get_fixed_marker_names(use_simple_markers)
        
        # Print mode info using helper method
        self._print_mode_info(use_simple_markers)
        
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

    def make_unilateral(self, motion_data: np.ndarray, info_df: pd.DataFrame = None) -> tuple:
        """
        Takes bilateral motion data and creates unilateral data by mirroring left markers
        to match right markers and stacking them. Handles both simple and full modes,
        including centre markers in full mode.

        Parameters:
        - motion_data (np.ndarray): Motion data in shape [nFrames, nMarkers, 3]
        - info_df (pd.DataFrame, optional): DataFrame with same number of rows as motion_data frames

        Returns:
        - np.ndarray: Motion data with doubled frames and halved markers (plus centres)
                                     Shape will be [nValidFrames*2, nMarkers//2 + nCentres, 3]
        - np.ndarray: Boolean array indicating which frames were originally left
        - pd.DataFrame: Optional unilateral info_df with doubled rows corresponding to motion data
        """
        # Make a hard copy of the motion data   
        motion_data_copy = np.copy(motion_data)
        info_df_copy = info_df.copy() if info_df is not None else None
        
        # Get marker organisation
        left_markers, right_markers, centre_markers = self.skeleton_definition.get_marker_pairs_and_centres(self.use_simple)
        
        # Get indices for each marker type
        left_indices = [self.marker_names.index(m) for m in left_markers]
        right_indices = [self.marker_names.index(m) for m in right_markers]
        centre_indices = [self.marker_names.index(m) for m in centre_markers] if centre_markers else []
        
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
        centre_data = motion_data_copy[:, centre_indices, :] if centre_indices else None
        
        # Mirror the left side in x to match the right
        left_mirrored = np.copy(left_data)
        left_mirrored[..., 0] *= -1  # Mirror x-coordinate
        
        # Stack mirrored left data and right data
        paired_data = np.concatenate((left_mirrored, right_data), axis=0)
        
        if centre_data is not None:
            # Duplicate centre data for both halves
            centre_data_doubled = np.concatenate((centre_data, centre_data), axis=0)
            # Combine paired and centre data
            unilateral_data = np.concatenate((paired_data, centre_data_doubled), axis=1)
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
        simple and full modes, including centre markers in full mode.

        Parameters:
        - unilateral_data (np.ndarray): Motion data in shape [nFrames*2, nMarkers//2 + nCentres, 3]
        - is_left (np.ndarray): Boolean or integer array (0/1) indicating which frames were originally left

        Returns:
        - np.ndarray: Reconstructed bilateral motion data in shape [nFrames, nMarkers, 3]
        """
        # Convert is_left to boolean if it's integer
        if is_left.dtype.kind in 'iu':  # integer type
            is_left = is_left.astype(bool)
            
        # Get marker organisation
        left_markers, right_markers, centre_markers = self.skeleton_definition.get_marker_pairs_and_centres(self.use_simple)
        n_pairs = len(left_markers)
        n_centres = len(centre_markers)
        
        # Make a hard copy of the unilateral data
        unilateral_data_copy = np.copy(unilateral_data)
        n_frames = len(is_left) // 2  # Number of original frames (half of total frames)
        
        if n_centres > 0:
            # Split into paired and centre data
            paired_data = unilateral_data_copy[:, :n_pairs, :]
            centre_data = unilateral_data_copy[:, n_pairs:, :]
            
            # Split paired data into left and right portions based on is_left
            left_data = paired_data[is_left]
            right_data = paired_data[~is_left]
            
            # Take only first half of centre data (they're duplicated)
            centre_data = centre_data[:n_frames]
        else:
            # All data is paired
            left_data = unilateral_data_copy[is_left]
            right_data = unilateral_data_copy[~is_left]
            centre_data = None
        
        # Un-mirror the left data
        left_data[:,:, 0] *= -1
        
        # Create output array with correct number of frames
        total_markers = n_pairs * 2 + n_centres
        bilateral_data = np.zeros((n_frames, total_markers, 3))
        
        # Get indices for each marker type in the final array
        left_indices = [self.marker_names.index(m) for m in left_markers]
        right_indices = [self.marker_names.index(m) for m in right_markers]
        centre_indices = [self.marker_names.index(m) for m in centre_markers] if centre_markers else []
        
        # Place data in correct positions
        bilateral_data[:, left_indices, :] = left_data
        bilateral_data[:, right_indices, :] = right_data
        if centre_data is not None:
            bilateral_data[:, centre_indices, :] = centre_data
        
        return bilateral_data