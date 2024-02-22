import numpy as np
from scipy.spatial.transform import Rotation as R


class KeypointManager:

    right_marker_names = [
        "right_wingtip", 
        "right_primary", 
        "right_secondary",
        "right_tailtip"]
    
    left_marker_names = [
        "left_wingtip", 
        "left_primary", 
        "left_secondary",
        "left_tailtip"]
    
    marker_names = [
        "left_wingtip",   "right_wingtip", 
        "left_primary",   "right_primary", 
        "left_secondary", "right_secondary", 
        "left_tailtip",   "right_tailtip"]
    
    fixed_marker_names = [
        "left_shoulder", 
        "left_tailbase", 
        "right_tailbase", 
        "right_shoulder",
        "hood", 
        "tailpack"
    ]


    def __init__(self, filename):

        # Get the keypoint data from file.
        self.load_data(filename)
        
        
    def load_data(self,filename):

        # load the data
        with open(filename, 'r') as file:
            data = np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')
        
        # Get the marker names from the first row of the csv file, 
        # get every 3rd name and remove the '_x' from the names
        csv_keypoint_names = data[0].reshape(-1, 3)
        csv_keypoint_names = list(np.char.strip(csv_keypoint_names[:, 0], '_x'))

        # Save the keypoint names
        self.csv_keypoint_names = csv_keypoint_names


        # Load marker coordinates and reshape to [n,3] matrix where n is the
        # number of markers
        keypoints = data[1].astype(float)
        keypoints = keypoints.reshape(-1, 3) # [n,3]

        # Get the indices of the markers
        self.right_marker_index = self.get_keypoint_indices(self.right_marker_names)
        self.left_marker_index  = self.get_keypoint_indices(self.left_marker_names)
        self.marker_index       = self.get_keypoint_indices(self.marker_names)
        self.fixed_marker_index = self.get_keypoint_indices(self.fixed_marker_names)

        # Save the keypoints, and a safe copy. 
        self.keypoints_original = np.copy(keypoints)
        self.keypoints = keypoints

        

    def get_keypoint_indices(self,names_to_find=None):
            """
            Returns the indices of the keypoints with the given names.
            """
            csv_keypoint_names = self.csv_keypoint_names

            # If no names are given, use all the marker names (not the fixed ones)
            if names_to_find is None:
                names_to_find = self.marker_names

            indices = [csv_keypoint_names.index(name) for name in names_to_find]

            return indices
    
    def validate_keypoints(self,keypoints):

        # First check they are not empty
        if self.is_empty_keypoints(keypoints):
            raise ValueError("No keypoints given.")
        
        # Check they are in 3D
        if keypoints.shape[-1] != 3:
            raise ValueError("Keypoints not in 3D.")
        
        # If [4,3] or [8,3] is given, reshape to [1,4,3] or [1,8,3]
        if len(np.shape(keypoints)) == 2:
            keypoints = keypoints.reshape(1, -1, 3)

        # If [1,4,3] is given, mirror it to make [1,8,3]
        if keypoints.shape[1] == len(self.right_marker_names):
            keypoints = self._mirror_keypoints(keypoints)

        # Then check there are enough keypoints
        if keypoints.shape[1] != len(self.marker_names):
            raise ValueError("Keypoints missing.")
        
        return keypoints
        
    def update_keypoints(self,user_keypoints):
        """ 
        Assumes the keypoints from the user are in the same order as the
        marker names.
        State changes Hawk3D.keypoints. 
        """


        # First validate the keypoints. This will mirror them 
        # if only the right side is given. Also checks they are in 3D and 
        # will return [n,8,3]. 

        user_keypoints = self.validate_keypoints(user_keypoints)

        # Update the keypoints with the user marker info. 
        self.keypoints[self.marker_index] = user_keypoints


    def transform_keypoints(self,
                            horzDist=None,
                            bodypitch=None, 
                            vertDist=None):
        
        """
        If any of the parameters are None, they are not applied. 
        """
        
        # Make an unaltered copy of the keypoints
        if horzDist is not None or bodypitch is not None or vertDist is not None:
            self.keypoints_no_translation = np.copy(self.keypoints)
        else:
            self.keypoints_no_translation = None
            print("No transformation applied.")
            return

        if bodypitch is not None:
            self.add_pitchRotation(bodypitch)
        
        if horzDist is not None:
            self.add_horzDist(horzDist)
            print("Horizontal translation applied.")

        if vertDist is not None:
            self.add_vertDist(vertDist)

        return self.keypoints
    

    def restore_keypoints(self):

        """
        Makes the keypoints return to the previous 
        state with no translation or rotation.
        """

        if self.keypoints_no_translation is None:
            print("No previous keypoints to restore.")
            return
        
        self.keypoints = self.keypoints_no_translation
        self.keypoints_no_translation = None

    def add_pitchRotation(self, bodypitch):

        if bodypitch is None:
            return
    
        rotmat = R.from_euler('x', bodypitch, degrees=True)
        rot_keypoints = rotmat.apply(self.keypoints)

        
        self.keypoints = rot_keypoints
        


    def add_horzDist(self, horzDist):
        if horzDist is None:
            return

        self.keypoints[:,1] += horzDist
    
    def add_vertDist(self, vertDist):
        if vertDist is None:
            return

        self.keypoints[:,2] += vertDist
        


    @staticmethod
    def _mirror_keypoints(keypoints):
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

    @staticmethod
    def is_empty_keypoints(keypoints):
        if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
            return False
        else:
            return True
        
    def validate_frames(self, keypoints_frames):
        """
        Validates the keypoints_frames and returns a list of keypoints for each frame.
        """
        if not isinstance(keypoints_frames, list):
            keypoints_frames = [keypoints_frames]
        for ii, keypoints in enumerate(keypoints_frames):

            # The if statement is to ensure that all keypoint sequences have the same length
            if ii > 0 and keypoints.shape[0] != keypoints_frames[ii - 1].shape[0]:
                raise ValueError("All keypoint sequences must have the same length")
            
            keypoints_frames[ii] = self._validate_keypoints(keypoints)
        return keypoints_frames
    

# class KeypointManagerOLD:
    
#     names_fixed_keypoints = [
#         "left_shoulder", 
#         "left_tailbase", 
#         "right_tailbase", 
#         "right_shoulder",
#         "hood", 
#         "tailpack"
#     ]

#     names_right_keypoints = [
#         "right_wingtip", 
#         "right_primary", 
#         "right_secondary",
#         "right_tailtip",
#     ]

#     names_keypoints = [
#         "left_wingtip",   "right_wingtip", 
#         "left_primary",   "right_primary", 
#         "left_secondary", "right_secondary", 
#         "left_tailtip",   "right_tailtip"]
    
#     names_xyz_keypoints = [
#         "wingtip_x",   "wingtip_y",   "wingtip_z",
#         "primary_x",   "primary_y",   "primary_z",
#         "secondary_x", "secondary_y", "secondary_z",
#         "tailtip_x",   "tailtip_y",   "tailtip_z"]
    

#     def __init__(self, filename):
#         self.names_all_keypoints, self.all_keypoints = self.load_data(filename)
#         self.fixed_keypoints = self.get_keypoints_by_names(self.names_fixed_keypoints)
#         self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)
        
#         # Expecting the user to provide these. However, if they are not provided,
#         # the average keypoints will be used as loaded from the file. 
#         self.avg_keypoints = self.get_keypoints_by_names(self.names_keypoints)
#         self.keypoints = self.avg_keypoints
        

#     def load_data(self,filename):
#         with open(filename, 'r') as file:
#             data = np.loadtxt(file, delimiter=',', skiprows=0, dtype='str')
        
#         # Get the marker names from the first row of the csv file, get every 3rd name
#         # and remove the '_x' from the names
#         names_all_keypoints = data[0].reshape(-1, 3)
#         names_all_keypoints = list(np.char.strip(names_all_keypoints[:, 0], '_x'))

#         # Load marker coordinates and reshape to [n,3] matrix where n is the
#         # number of markers
#         all_keypoints = data[1].astype(float)
#         all_keypoints = all_keypoints.reshape(-1, 3) # [n,3]

#         self.all_keypoints_original = np.copy(all_keypoints)

#         return names_all_keypoints, all_keypoints

#     def get_keypoint_indices(self,names_to_find=None):
#         """
#         Returns the indices of the keypoints with the given names.
#         """

#         marker_names = self.names_all_keypoints

#         if names_to_find is None:
#             names_to_find = marker_names

#         indices = [marker_names.index(name) for name in names_to_find]

#         return indices
    

#     def get_keypoints_by_names(self,names_to_find=None):
#         """
#         Returns the keypoints with the given names.
#         """
        
#         indices = self.get_keypoint_indices(names_to_find)

#         return self.all_keypoints[indices]
    

#     def is_empty_keypoints(self, keypoints):
#         if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
#             return False
#         else:
#             return True

#     def update_keypoints(self, user_keypoints):

#         """
#         Updates the keypoints with the given user_keypoints.
#         ASSUMES THAT THE USER KEYPOINTS ARE GIVEN IN THE SAME ORDER AS THE
#         NAMES OF THE KEYPOINTS.
#         Either just the right or both keypoints can be given. 
#         [expected shape: [1, 4, 3] or [1, 8, 3]
#         """

#         if len(np.shape(user_keypoints)) == 2:
#             user_keypoints = user_keypoints.reshape(1, -1, 3)

#         if self.is_empty_keypoints(user_keypoints):
#             # Throw error
#             raise ValueError("No keypoints given.")
        

#         # If only the right keypoints are given, mirror them
#         if user_keypoints.shape[1] == len(self.names_right_keypoints):
#             user_keypoints = self._mirror_keypoints(user_keypoints)

#         # If the left and right keypoints are given. 
#         if user_keypoints.shape[1] == len(self.names_keypoints):
            
#             # Check the keypoints are valid
#             user_keypoints = self._validate_keypoints(user_keypoints)

#             # Update all_keypoints 
#             for ii, name in enumerate(self.names_keypoints):
#                 index = self.names_all_keypoints.index(name)
#                 self.all_keypoints[index] = user_keypoints[0,ii]

#             # Update the keypoints too. 
#             self.keypoints = self.get_keypoints_by_names(self.names_keypoints)

#             # Update the right keypoints too.
#             self.right_keypoints = self.get_keypoints_by_names(self.names_right_keypoints)
#         else:
#             raise ValueError("Wrong number of keypoints given.")

#     def _validate_keypoints(self, keypoints):

#         if keypoints.shape[-1] != 3:
#             raise ValueError("Keypoints not in 3D.")

#         if len(np.shape(keypoints)) == 2:
#             keypoints = keypoints.reshape(1, -1, 3)
                
#         if len(keypoints.shape) != 3:
#             keypoints = keypoints.reshape(-1, keypoints.shape[1], 3)

#         # If the keypoints are only for the right side, mirror them
#         if keypoints.shape[1] == len(self.names_right_keypoints):
#             keypoints = self._mirror_keypoints(keypoints)

#         if keypoints.shape[1] != len(self.names_keypoints):
#             print(keypoints.shape)
#             raise ValueError("Keypoints missing.")
        
#         return keypoints

#     def _mirror_keypoints(self, keypoints):
#         """
#         Mirrors keypoints across the y-axis.
#         """
#         mirrored = np.copy(keypoints)
#         mirrored[:, :, 0] *= -1

#         nFrames, nMarkers, nCoords = np.shape(keypoints)

#         # Create [n,8,3] array
#         new_keypoints = np.empty((nFrames, nMarkers * 2, nCoords),
#                                  dtype=keypoints.dtype)
        
#         new_keypoints[:, 0::2, :] = mirrored
#         new_keypoints[:, 1::2, :] = keypoints

#         return new_keypoints
    
#     @staticmethod
#     def add_horzDist(keypoints,horzDist):
#         """
#         Translates the keypoints with a given horzDist. 
#         Just changes the y.
#         """
#         if horzDist is None:
#             horzDist = 0

#         keypoints[:,1] += horzDist

#         return keypoints
        
#     @staticmethod
#     def add_vertDist(keypoints,vertDist):
#         """
#         Translates keypoints with a given vertDist. 
#         Just changes the z.
#         """
#         if vertDist is None:
#             vertDist = 0
        
#         keypoints[:,2] += vertDist

#         return keypoints

#     @staticmethod
#     def add_pitchRotation(keypoints,bodypitch):
#         """
#         Updates the keypoints (all) with a given bodypitch.
#         """

#         if bodypitch is None:
#             return
    
#         rotmat = R.from_euler('x', bodypitch, degrees=True)
#         rot_keypoints = rotmat.apply(keypoints)

#         return rot_keypoints

#     def validate_frames(self, keypoints_frames):
#         """
#         Validates the keypoints_frames and returns a list of keypoints for each frame.
#         """
#         if not isinstance(keypoints_frames, list):
#             keypoints_frames = [keypoints_frames]
#         for ii, keypoints in enumerate(keypoints_frames):

#             # The if statement is to ensure that all keypoint sequences have the same length
#             if ii > 0 and keypoints.shape[0] != keypoints_frames[ii - 1].shape[0]:
#                 raise ValueError("All keypoint sequences must have the same length")
            
#             keypoints_frames[ii] = self._validate_keypoints(keypoints)
#         return keypoints_frames

