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