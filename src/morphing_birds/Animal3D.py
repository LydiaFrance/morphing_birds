import numpy as np

"""
The `Animal3D` class is designed to represent a 3D model of an animal's skeleton, 
and it relies on the `SkeletonDefinition` class to define the structure and 
characteristics of that skeleton. 

By using `SkeletonDefinition`, `Animal3D` can access essential information about 
the markers that represent key anatomical points on the animal, as well as the 
relationships between these markers. This allows for a more organised and 
systematic approach to handling the animal's shape.

The `SkeletonDefinition` class serves as a blueprint, providing the necessary 
methods and attributes to manage the markers and body sections. This design 
promotes code reusability and ensures that the `Animal3D` class can easily 
adapt to different animal types by simply changing the skeleton definition 
used during initialisation.

In summary, `Animal3D` is a higher-level class that utilises the foundational 
capabilities of `SkeletonDefinition` to create a comprehensive representation 
of an animal's 3D structure, enabling various functionalities such as 
loading keypoints, transforming shapes, and visualising the model.
"""



class Animal3D:
    """
    A class representing a 3D model of an animal.

    This class provides functionality to load, manipulate, and visualise
    3D keypoints representing a biomechanical shape ("skeleton"). 
    
    The Animal3D class allows users to:

    - Make sure all the markers are initialised with the correct index from the data source.
    - Define the polygons that make up the body sections of the animal
    - Retrieve the coordinates of these polygons for plotting.
    - Allow the user to update the keypoints with new values and change the shape of the animal.
    - Allow the user to translate the keypoints in a horizontal or vertical direction.
    - Allow the user to rotate the keypoints in the x, y and z directions (whole body pitch, yaw and roll)
    - Reset any transformations
    - Restore the keypoints to the original loaded shape
    - Mirror the keypoints across the y-axis, so the right side defines the left side.
    - Make the keypoints array 3D if they are not already, so [n,3] becomes [m,n,3] where n is the number of markers.
        - This is necessary for animations, where m is the number of frames.


    Attributes:
        skeleton_definition (SkeletonDefinition): Defines the animal's shape (skeleton) structure
        csv_marker_names (list): Names of markers as they appear in the CSV file, the order that define the indices
        marker_index (list): Index for moving markers
        fixed_marker_index (list): Index for fixed markers
        right_marker_index (list): Index for moving markers on the right side of the animal
        left_marker_index (list): Index for moving markers on the left side of the animal
        polygons (dict): Dictionary of which marker names that make up the polyons for body sections of the animal
        body_section_indices (dict): Dictionary of which markers (using an index) make up the polygons of the animal
        default_shape (numpy.ndarray): The original keypoint positions loaded from the CSV
        current_shape (numpy.ndarray): The current (possibly transformed) keypoint positions
        transformation_matrix (numpy.ndarray): 4x4 matrix representing current transformations including rotations
        origin (numpy.ndarray): The origin point for transformations
        untransformed_shape (numpy.ndarray): A copy of the original keypoint positions
    """
    def __init__(self, skeleton_definition):

        self.skeleton_definition = skeleton_definition

        # Initialize marker indices
        self.marker_index = []
        self.fixed_marker_index = []
        self.right_marker_index = []
        self.left_marker_index = []
        self.body_section_indices = {}
        self.polygons = {}

        # Initialize keypoints
        self.default_shape = None
        self.current_shape = None
        self.untransformed_shape = None

        # Transformation attributes
        self.transformation_matrix = np.identity(4)
        self.origin = np.zeros(3)

        # Add scaling factor attribute with default of no scaling
        self.fixed_marker_scaling = np.ones(3)



    def define_indices(self, csv_marker_names):
        """
        Defines marker indices for moving, fixed, right, and left markers based on CSV marker names.

        Parameters:
        - csv_marker_names (list): List of marker names from the CSV in order.
        """
        # Define moving markers
        self.marker_index = self.skeleton_definition.get_marker_indices(
            self.skeleton_definition.marker_names, csv_marker_names
        )

        # Define fixed markers
        self.fixed_marker_index = self.skeleton_definition.get_marker_indices(
            self.skeleton_definition.fixed_marker_names, csv_marker_names
        )

        # Define right markers
        self.right_marker_index = self.skeleton_definition.get_marker_indices(
            self.skeleton_definition.get_right_marker_names(), csv_marker_names
        )

        # Define left markers
        self.left_marker_index = self.skeleton_definition.get_marker_indices(
            self.skeleton_definition.get_left_marker_names(), csv_marker_names
        )

    def init_polygons(self, csv_marker_names):
        """
        Initializes polygon definitions based on body sections in the skeleton definition and their corresponding marker indices.
        """
        self.body_section_indices = self.skeleton_definition.get_body_section_indices(csv_marker_names)
        self.polygons = {}
        for section, indices in self.body_section_indices.items():
            self.polygons[section] = indices


    def get_polygon_coords(self, section_name: str) -> np.ndarray:
        """
        Retrieves the coordinates for a specific body section.

        Parameters:
        - section_name (str): Name of the body section.

        Returns:
        - numpy.ndarray: Coordinates of the polygon's keypoints.

        Raises:
        - ValueError: If the section name is not recognized.
        """
        if section_name not in self.polygons:
            raise ValueError(f"Section name '{section_name}' not recognized.")

        indices = self.polygons[section_name]
        coords = self.current_shape[0, indices, :]
        return coords
        

    def reshape_keypoints_to_3d(self, keypoints):
        """
        Reshapes the keypoints to 3D if they are not already.
        """
        if len(np.shape(keypoints)) == 2:
            keypoints = keypoints.reshape(1, -1, 3)
        return keypoints

    def validate_keypoints(self, keypoints):
        """
        Validates the keypoints, ensuring they are three-dimensional and reshapes them if necessary.

        Parameters:
        - keypoints (numpy.ndarray): The keypoints array to validate.

        Returns:
        - numpy.ndarray: The validated and potentially reshaped keypoints.
        """
        if keypoints.size == 0:
            raise ValueError("No keypoints provided.")
        
        # Check they are in 3D
        if keypoints.shape[-1] != 3:
            raise ValueError("Keypoints must be in 3D space.")
        
        # If the array is in 2d, make it 3d, e.g. [8,3] becomes [1,8,3]
        if len(keypoints.shape) == 2:
            keypoints = keypoints.reshape(1, -1, 3)

        return keypoints

    def mirror_keypoints(self, keypoints):
        """
        Mirrors keypoints across the y-axis to create a symmetrical set.

        Parameters:
        - keypoints (numpy.ndarray): Keypoints in shape [1, n, 3].

        Returns:
        - numpy.ndarray: Mirrored keypoints array in shape [1, 2*n, 3].
        """
        mirrored = np.copy(keypoints)
        mirrored[:, :, 0] *= -1

        nFrames, nMarkers, nCoords = np.shape(keypoints)

        # Create [n,8,3] array
        new_keypoints = np.empty((nFrames, nMarkers * 2, nCoords),
                                 dtype=keypoints.dtype)
        
        new_keypoints[:, 0::2, :] = mirrored
        new_keypoints[:, 1::2, :] = keypoints

        return new_keypoints

    def update_keypoints(self,user_keypoints):
        """
        Updates the keypoints based on user-provided data. Resets to default if no keypoints are provided.
        Validates and mirrors the keypoints if necessary, and applies transformations.

        Parameters:
        - user_keypoints (numpy.ndarray or None): Array of keypoints provided by the user. 
        If None, resets to the default keypoints setup.
        """

        # Make sure the current_shape starts as default to reset it
        if user_keypoints is None:
            # Reset to default shape if no user keypoints are provided
            self.restore_keypoints_to_average()
            return

        # First validate the keypoints. This will mirror them 
        # if only the right side is given. Also checks they are in 3D and 
        # will return [n,8,3]. 

        # Validate and possibly mirror the user keypoints -- returns [n,8,3] 
        # If only the right side is given, the left side is created by mirroring.
        # Also checks in 3D space. 
        validated_keypoints = self.validate_keypoints(user_keypoints)


        # Update the keypoints with the user marker info. 
        # Only non fixed markers are updated.
        self.current_shape[:,self.marker_index,:] = validated_keypoints      

        # Save the untransformed shape
        self.untransformed_shape = self.current_shape.copy()

        # Apply transformations
        self.apply_transformation()

    def transform_keypoints(self, bodypitch=0, horzDist=0, vertDist=0, yaw=0):
        """
        Transforms the keypoints by rotating them around the body pitch, 
        and translating them by the horizontal and vertical distances.
        """

        # Reset the transformation matrix
        self.reset_transformation()

        # Rescale the fixed markers
        self.apply_fixed_marker_scaling()

        # Ensure horzDist and vertDist are scalar values
        horzDist = float(horzDist) if np.isscalar(horzDist) else float(horzDist[0])
        vertDist = float(vertDist) if np.isscalar(vertDist) else float(vertDist[0])

        # Apply any translations
        self.update_translation(horzDist, vertDist)

        # Apply any rotations
        self.update_rotation(bodypitch)
        self.update_rotation(yaw, which='z')

        # Apply the transformation
        self.apply_transformation()

    def update_rotation(self, degrees=0, which='x'):
        """
        Updates the transformation matrix with a rotation around the specified axis.
        
        Parameters:
        - degrees: float or array-like, rotation angle in degrees
        - which: str, axis of rotation ('x', 'y', or 'z')
        """
        # Ensure degrees is a scalar value
        degrees = float(degrees) if np.isscalar(degrees) else float(degrees[0])
        radians = np.deg2rad(degrees)
        
        if which == "x":
            rotation_matrix = np.array([
                [1,0,0,0],
                [0, np.cos(radians), -np.sin(radians), 0],
                [0, np.sin(radians),  np.cos(radians), 0],
                [0,0,0,1]
            ])
        elif which == "y":
            rotation_matrix = np.array([
                [np.cos(radians), 0, np.sin(radians), 0],
                [0,1,0,0],
                [-np.sin(radians), 0, np.cos(radians), 0],
                [0,0,0,1]
            ])
        elif which == "z":
            rotation_matrix = np.array([
                [np.cos(radians), -np.sin(radians), 0, 0],
                [np.sin(radians),  np.cos(radians), 0, 0],
                [0,0,1,0],
                [0,0,0,1]
            ])

        self.transformation_matrix = self.transformation_matrix @ rotation_matrix



    def update_translation(self, horzDist=0, vertDist=0):
        """
        Updates the transformation matrix with horizontal and vertical translations.
        """
        # Ensure horzDist and vertDist are scalar values
        horzDist = float(horzDist) if np.isscalar(horzDist) else float(horzDist[0])
        vertDist = float(vertDist) if np.isscalar(vertDist) else float(vertDist[0])

        translation_matrix = np.array([
            [1,0,0,0],
            [0,1,0,horzDist],
            [0,0,1,vertDist],
            [0,0,0,1]
        ])
        self.transformation_matrix = self.transformation_matrix @ translation_matrix

        # Update origin
        # self.origin = [0, horzDist, vertDist]
        self.origin = self.origin + np.array([0, horzDist, vertDist])
        

    def apply_transformation(self):

        """
        Applies the current transformation matrix to the keypoints.
        """
        # Adding a homogeneous coordinate directly to the current_shape
        homogeneous_keypoints = np.hstack((self.current_shape.reshape(-1, 3), np.ones((self.current_shape.shape[1], 1))))
        
        transformed_keypoints = np.dot(homogeneous_keypoints, self.transformation_matrix.T)
        
        self.current_shape = transformed_keypoints[:, :3].reshape(1, -1, 3)

    def reset_transformation(self):
        self.transformation_matrix = np.eye(4)
        self.current_shape = self.untransformed_shape

        # Also reset the origin
        self.origin = np.array([0,0,0])

    def restore_keypoints_to_average(self):
        """
        Restores the keypoints and origin to the default shape.
        """
        self.current_shape = self.default_shape.copy()
        self.untransformed_shape = self.current_shape.copy()

        # Also update the origin
        self.origin = np.array([0,0,0])

    
    def get_bounding_box(self):
        """
        Calculates the bounding box of the current shape.

        Returns:
        - tuple: (min_coords, max_coords) representing the bounding box.
        """
        min_coords = self.current_shape.min(axis=(0, 1))
        max_coords = self.current_shape.max(axis=(0, 1))
        return min_coords, max_coords

    def set_fixed_marker_scaling(self, scaling_factors):
        """
        Sets the scaling factors for fixed markers.

        Parameters:
        - scaling_factors (numpy.ndarray): Array of scaling factors [x, y, z]
            Must be a 3-element array corresponding to x, y, z dimensions.
        """
        if not isinstance(scaling_factors, np.ndarray) or scaling_factors.shape != (3,):
            raise ValueError("scaling_factors must be a numpy array with shape (3,)")
        
        self.fixed_marker_scaling = scaling_factors

      
    def apply_fixed_marker_scaling(self):
        """
        Applies the current fixed marker scaling to the untransformed shape.
        """
        self.current_shape[:, self.fixed_marker_index, :] /= self.fixed_marker_scaling
    
    
    @property
    def markers(self):
        """
        Returns the non-fixed markers.
        """
        
        marker_index = self.marker_index
        
        return self.current_shape[:,marker_index,:]

    @property
    def right_markers(self):
        """
        Returns the right side markers.
        """
        
        right_marker_index = self.right_marker_index
        
        return self.current_shape[:,right_marker_index,:]
    
    @property
    def left_markers(self):
        """
        Returns the left side markers.
        """
        
        left_marker_index = self.left_marker_index
        
        return self.current_shape[:,left_marker_index,:]
    
    @property
    def default_markers(self):
        """
        Returns the default markers.
        """
        
        marker_index = self.marker_index
        
        return self.default_shape[:,marker_index,:]

    @property
    def default_right_markers(self):
        """
        Returns the right side markers.
        """
        
        right_marker_index = self.right_marker_index
        
        return self.default_shape[:,right_marker_index,:]

    @property
    def default_left_markers(self):
        """
        Returns the right side markers.
        """
        
        left_marker_index = self.left_marker_index
        
        return self.default_shape[:,left_marker_index,:]


    @property
    def fixed_markers(self):
        """
        Returns the fixed markers.
        """
        return self.current_shape[:,self.fixed_marker_index,:]




# ----- Plot Functions -----
 


