import numpy as np

class SkeletonDefinition:
    """
    The SkeletonDefinition class serves as a blueprint for defining the shape ("skeleton") of an animal.

    It encapsulates the concept of markers, which are moving points of reference on the animal, and body sections, 
    which group these markers into meaningful anatomical parts. Fixed markers are those that do not move during 
    animation but might help with visualisation. 

    Attributes:
    - marker_names (list): A list of all marker names that represent key points on the skeleton.
    - fixed_marker_names (list): A list of marker names that are fixed and do not change position.
    - body_sections (dict): A dictionary that maps body section names to their corresponding marker names.

    Methods:
    - get_marker_indices(marker_subset, csv_marker_names): Retrieves the indices of specified markers from a 
      given list of marker names.
    - get_body_section_indices(csv_marker_names): Returns a mapping of body sections to their respective 
      marker indices, ensuring that all markers are accounted for.

    This class is designed to be extended by specific animal skeleton definitions, allowing for tailored 
    implementations while maintaining a consistent interface.
    """

    def __init__(self, marker_names: list, fixed_marker_names: list, body_sections: dict):
        """
        Initializes the SkeletonDefinition with marker names and body sections.

        Parameters:
        - csv_marker_names (list): List of all marker names from the CSV in correct order.
        - fixed_marker_names (list): List of fixed marker names.
        - body_sections (dict): Dictionary defining body sections with their respective marker names.
        """
        self.marker_names = marker_names
        self.fixed_marker_names = fixed_marker_names
        self.body_sections = body_sections

    def get_marker_indices(self, marker_subset: list, csv_marker_names: list = None) -> list:
        """
        Retrieves indices for a subset of markers based on csv_marker_names.

        Parameters:
        - marker_subset (list): Specific markers to retrieve indices for.
        - csv_marker_names (list): List of all marker names from the CSV in the correct order.

        Returns:
        - list: List of indices corresponding to the marker_subset.

        Raises:
        - ValueError: If any marker in the subset is not found in csv_marker_names.
        """
        indices = []

        if csv_marker_names is None:
            csv_marker_names = self.marker_names

        # Get the indices of the markers in the marker_subset
        for name in marker_subset:
            if name in csv_marker_names:
                index = csv_marker_names.index(name)
                indices.append(index)
            else:
                raise ValueError(f"Marker '{name}' not found in CSV marker names.")
        return indices
    
    def get_body_section_indices(self, csv_marker_names: list) -> dict:
        """
        Returns a dictionary mapping each body section to its respective marker indices.

        Parameters:
        - csv_marker_names (list): List of all marker names from the CSV in the correct order.

        Returns:
        - dict: Dictionary with body section names as keys and lists of marker indices as values.

        Raises:
        - ValueError: If any marker in a body section is not found in csv_marker_names.
        """
        section_indices = {}
        for section, markers in self.body_sections.items():
            section_indices[section] = self.get_marker_indices(markers, csv_marker_names)
        return section_indices
    
    def get_right_marker_names(self) -> list:
        """
        Returns a list of names for all right-side markers.
        """
        return [name for name in self.marker_names if name not in self.fixed_marker_names]

    def get_left_marker_names(self) -> list:
        """
        Returns a list of names for all left-side markers.
        """
        return [name for name in self.marker_names if name not in self.fixed_marker_names]

    def get_right_marker_indices(self) -> list:
        """
        Returns a list of indices for all right-side markers.
        """
        return self.get_marker_indices(self.get_right_marker_names())

    def get_left_marker_indices(self) -> list:
        """
        Returns a list of indices for all left-side markers.
        """
        return self.get_marker_indices(self.get_left_marker_names())
    

    def validate_marker_positions(self):
        """
        Validates the anatomical relationships between markers to ensure the skeleton
        maintains correct biological proportions and orientations.
        
        The validation checks:
        1. Left/Right symmetry (left markers are left of right markers)
        2. Wing structure (primary and secondary feather positions relative to wingtips)
        3. Tail position (tail markers behind body markers)
        4. Body integrity (shoulder and tailbase relationships)
        
        Raises:
        - ValueError: If marker positions violate anatomical constraints
        """
        
        # Left/Right symmetry checks
        if self.marker_positions["left_wingtip"][0] >= self.marker_positions["right_wingtip"][0]:
            raise ValueError("Left wingtip must be to the left of right wingtip")
        
        if self.fixed_marker_positions["left_shoulder"][0] >= self.fixed_marker_positions["right_shoulder"][0]:
            raise ValueError("Left shoulder must be to the left of right shoulder")
        
        if self.fixed_marker_positions["left_tailbase"][0] >= self.fixed_marker_positions["right_tailbase"][0]:
            raise ValueError("Left tailbase must be to the left of right tailbase")

        # Wing structure checks
        for side in ["left", "right"]:
            wingtip = self.marker_positions[f"{side}_wingtip"]
            primary = self.marker_positions[f"{side}_primary"]
            secondary = self.marker_positions[f"{side}_secondary"]
            shoulder = self.fixed_marker_positions[f"{side}_shoulder"]

            # Check wing length progression
            if side == "left":
                if not (wingtip[0] <= primary[0] <= secondary[0] <= shoulder[0]):
                    raise ValueError(f"{side.capitalize()} wing markers must progress inward from wingtip to shoulder")
            else:  # right side
                if not (wingtip[0] >= primary[0] >= secondary[0] >= shoulder[0]):
                    raise ValueError(f"{side.capitalize()} wing markers must progress inward from wingtip to shoulder")

        # Tail position checks
        hood = self.fixed_marker_positions["hood"]
        left_tailbase = self.fixed_marker_positions["left_tailbase"]
        right_tailbase = self.fixed_marker_positions["right_tailbase"]
        
        # Check that tail is behind the body
        if not (left_tailbase[1] > hood[1] and right_tailbase[1] > hood[1]):
            raise ValueError("Tail base must be behind the hood")

        # Check tail tips
        left_tailtip = self.marker_positions["left_tailtip"]
        right_tailtip = self.marker_positions["right_tailtip"]
        
        if not (left_tailtip[1] > left_tailbase[1] and right_tailtip[1] > right_tailbase[1]):
            raise ValueError("Tail tips must be behind tail base")

        # Body integrity checks
        # Verify hood is between shoulders
        left_shoulder = self.fixed_marker_positions["left_shoulder"]
        right_shoulder = self.fixed_marker_positions["right_shoulder"]
        
        if not (left_shoulder[0] <= hood[0] <= right_shoulder[0]):
            raise ValueError("Hood must be between left and right shoulders")

        # Verify shoulders are above tailbase
        if not (left_shoulder[2] > left_tailbase[2] and right_shoulder[2] > right_tailbase[2]):
            raise ValueError("Shoulders must be above tailbase")

        # Verify tailpack position (if used for tracking)
        tailpack = self.fixed_marker_positions["tailpack"]
        if not (left_tailbase[2] >= tailpack[2] and right_tailbase[2] >= tailpack[2]):
            raise ValueError("Tailpack must be below tailbase markers")