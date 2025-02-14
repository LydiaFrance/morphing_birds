import numpy as np

from morphing_birds import Animal3D, HawkSkeletonDefinition

class Hawk3D(Animal3D):
    """
    A class representing a 3D model of a Hawk.

    Inherits from Animal3D and provides specific functions for 
    loading CSV data and validating polygon shapes of a hawk. 
    """
    def __init__(self, csv_path: str):
        """
        Initializes the Hawk3D class with the given CSV file path.

        Parameters:
        - csv_path (str): Path to the CSV file containing the hawk's keypoints.
        """
        skeleton_definition = HawkSkeletonDefinition()
        super().__init__(skeleton_definition)
        self.marker_names = self.skeleton_definition.marker_names
        self.right_marker_names = self.skeleton_definition.get_right_marker_names()
        self.left_marker_names = self.skeleton_definition.get_left_marker_names()
        self.load_csv(csv_path)
        self.init_polygons(self.csv_marker_names)

    def load_csv(self, csv_path: str):
        """
        Loads CSV data specific to the hawk skeleton.

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
        # can't use split('_')[0] because some markers have more than one underscore instead remove _x _y _z explicitly
        cleaned_names = [name.replace('_x', '').replace('_y', '').replace('_z', '') for name in header_row]
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

        # Check if only one side is given and mirror if necessary
        if keypoints.shape[1] == len(self.skeleton_definition.get_right_marker_names()):
            keypoints = self.mirror_keypoints(keypoints)

        return keypoints

    def validate_polygon_shape(self):
        """
        Validates the shape of the polygons to ensure they are as expected.
        """
        assert self.get_polygon_coords('left_handwing')[0][0] < self.get_polygon_coords('right_handwing')[0][0], \
            "Left wing should be to the left of the right wing"
        
        assert self.get_polygon_coords('left_handwing')[0][0] < self.get_polygon_coords('left_armwing')[0][0], \
            "Left handwing should be to the left of the left armwing"
        
        assert self.get_polygon_coords('tail')[0][1] < self.get_polygon_coords('left_handwing')[0][1], \
            "Tail should be behind the wings"
        
        assert self.get_polygon_coords('tail')[0][1] < self.get_polygon_coords('right_handwing')[0][1], \
            "Tail should be behind the wings"

        # Tell the user that the polygon shape is valid
        print("Polygon shape is valid")