import numpy as np

class Hawk3D:
    body_sections = {
        "left_handwing": [
            "left_wingtip", 
            "left_primary", 
            "left_secondary",
        ],
        "right_handwing": [
            "right_wingtip", 
            "right_primary", 
            "right_secondary",
        ],
        "left_armwing": [
            "left_primary", 
            "left_secondary", 
            "left_tailbase", 
            "left_shoulder",
        ],
        "right_armwing": [
            "right_primary", 
            "right_secondary", 
            "right_tailbase",
            "right_shoulder",
        ],
        "body": [
            "right_shoulder", 
            "left_shoulder", 
            "left_tailbase", 
            "right_tailbase",
        ],
        "head": [
            "right_shoulder", 
            "hood", 
            "left_shoulder",
        ],
        "tail": [
            "right_tailtip", 
            "left_tailtip", 
            "left_tailbase", 
            "right_tailbase",
        ],
    }


    names_fixed_keypoints = [
        "left_shoulder", 
        "left_tailbase", 
        "right_tailbase", 
        "right_shoulder",
        "hood", 
        "tailpack"
    ]

    names_right_keypoints = [
        "right_wingtip", 
        "right_primary", 
        "right_secondary",
        "right_tailtip",
    ]

    names_left_keypoints = [
        "left_wingtip", 
        "left_primary", 
        "left_secondary",
        "left_tailtip",
    ]

    names_keypoints = [
        "left_wingtip",   "right_wingtip", 
        "left_primary",   "right_primary", 
        "left_secondary", "right_secondary", 
        "left_tailtip",   "right_tailtip"]

    def __init__(self, filename):
        self.names_all_keypoints, self.all_keypoints = self.load_data(filename)
        self.fixed_keypoints = self.get_keypoints_by_names(self.names_fixed_keypoints)
        self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)
        self.left_keypoints = self.get_keypoints_by_names(self.names_left_keypoints)
        self.keypoints = self.get_keypoints_by_names(self.names_keypoints)


        # Initialise polygons for plotting
        self._init_polygons()

    def load_data(self,filename):
        with open(filename, 'r') as file:
            data = np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')
        
        # Get the marker names from the first row of the csv file, get every 3rd name
        # and remove the '_x' from the names
        names_all_keypoints = data[0].reshape(-1, 3)
        names_all_keypoints = list(np.char.strip(names_all_keypoints[:, 0], '_x'))

        # Load marker coordinates and reshape to [n,3] matrix where n is the
        # number of markers
        all_keypoints = data[1].astype(float)
        all_keypoints = all_keypoints.reshape(-1, 3) # [n,3]

        return names_all_keypoints, all_keypoints

    
    def get_keypoints_by_names(self,names_to_find=None):
        """
        Returns the keypoints with the given names.
        """

        marker_names = self.names_all_keypoints
        keypoints = self.all_keypoints

        if names_to_find is None:
            names_to_find = marker_names

        indices = [marker_names.index(name) for name in names_to_find]

        return keypoints[indices]
    

    def _validate_keypoints(self, keypoints):

        if keypoints.shape[-1] != 3:
            raise ValueError("Keypoints not in 3D.")

        if len(np.shape(keypoints)) == 2:
            keypoints = keypoints.reshape(1, -1, 3)
                
        if len(keypoints.shape) != 3:
            keypoints = keypoints.reshape(-1, keypoints.shape[1], 3)

        if keypoints.shape[1] == len(self.names_right_keypoints):
            keypoints = self._mirror_keypoints(keypoints)

        if keypoints.shape[1] != len(self.names_keypoints):
            print(keypoints.shape)
            raise ValueError("Keypoints missing.")
        
        return keypoints

    def _mirror_keypoints(self, keypoints):
        """
        Mirrors keypoints across the y-axis.
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
        
    def _init_polygons(self):
        """
        Initialise polygons for plotting. 
        Each polygon is created with the indices of vertices forming it.
        These indices are used to subset the keypoint array when plotting.
        """
        self._polygons = {}
        for section_name, keypoint_names in self.body_sections.items():
            self._polygons[section_name] = self.get_keypoints_by_names(keypoint_names)

