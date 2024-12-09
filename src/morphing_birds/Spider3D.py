import numpy as np

from morphing_birds import Animal3D, SpiderSkeletonDefinition

class Spider3D(Animal3D):
    """
    A class representing a 3D model of a Spider.
    """
    def __init__(self, csv_path: str):
        """
        Initialises the Spider3D class with the given CSV file path.
        """
        skeleton_definition = SpiderSkeletonDefinition()
        super().__init__(skeleton_definition)

        self.marker_names = self.skeleton_definition.marker_names

        self.right_marker_names = self.skeleton_definition.get_right_marker_names()
        self.left_marker_names = self.skeleton_definition.get_left_marker_names()

        self.body_marker_names = self.skeleton_definition.body_sections["body"]
        self.frontlegs_marker_names = self.skeleton_definition.get_frontlegs_marker_names()
        self.backlegs_marker_names = self.skeleton_definition.get_backlegs_marker_names()   
        
        self.load_csv(csv_path)
        self.init_polygons(self.csv_marker_names)
        
    def load_csv(self, csv_path: str):
        """
        Loads CSV data specific to the spider skeleton.
        """
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

    def load_csv_data(self, csv_path: str):
        """
        Loads CSV data specific to the spider skeleton.
        """
        with open(csv_path, 'r') as file:
            return np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')
        
    def get_csv_marker_names(self, header_row: list) -> list:
        """
        Extracts unique marker names from the header row by removing coordinate suffixes.
        """
        cleaned_names = [name.replace('_x', '').replace('_y', '').replace('_z', '') for name in header_row]
        
        unique_names = []
        seen = set()
        for name in cleaned_names:
            if name not in seen:
                unique_names.append(name)
                seen.add(name)
        return unique_names
    
    def get_csv_keypoints(self, data) -> np.ndarray:
        # Convert string data to float and reshape
        keypoints = data[1:].astype(float)  # Skip the first row (header)
        keypoints = keypoints.reshape(-1, 3)  # [n, 3]
        return keypoints
    
    
        
