import numpy as np
import pandas as pd



class HawkData:

    def __init__(self, csv_path):
        self.markers = self.load_marker_frames(csv_path)
        
        self.load_frame_data(csv_path)

        self.check_data()

    def load_marker_frames(self, csv_path):
        """Load the unilateral markers dataset.

        Returns
        -------
        data : pandas.DataFrame
            The data frame containing the unilateral markers dataset.
        """
        # Load the data
        markers_csv = pd.read_csv(csv_path)

        # Rename the columns
        markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_1", "_x")
        markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_2", "_y")
        markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_3", "_z")

        # Get the index of the columns that contain the markers
        marker_index = markers_csv.columns[markers_csv.columns.str.contains('_x|_y|_z')]
        markers_csv = markers_csv[marker_index]

        # Make a numpy array with the markers
        unilateral_markers = markers_csv.to_numpy()

        # Reshape to 3D
        unilateral_markers = unilateral_markers.reshape(-1,4,3)
        
        return unilateral_markers

    def load_frame_data(self, csv_path):
        """Load the frame info from the dataset.

        Returns
        -------
        data : pandas.DataFrame
            The data frame containing the unilateral markers dataset.
        """
        # Load the data
        markers_csv = pd.read_csv(csv_path)

        self.horzDist = markers_csv['HorzDistance'].to_numpy()
        self.frameID = markers_csv['frameID']
        self.leftBool = markers_csv['Left'].to_numpy()
        self.body_pitch = markers_csv['body_pitch'].to_numpy()
        self.obstacleBool = markers_csv['Obstacle'].to_numpy()
        self.IMUBool = markers_csv['IMU'].to_numpy()
        self.time = markers_csv['time'].to_numpy()

    def check_data(self):
        """
        Check that the data are the same length.
        """

        num_frames = self.markers.shape[0]

        if self.horzDist.shape[0] != num_frames:
            raise ValueError("horzDist must be the same length as keypoints_frames.")
        
        if len(self.frameID) != num_frames:
            raise ValueError("frameID must be the same length as keypoints_frames.")
        
        if self.leftBool.shape[0] != num_frames:
            raise ValueError("leftBool must be the same length as keypoints_frames.")
        
        if self.body_pitch.shape[0] != num_frames:
            raise ValueError("body_pitch must be the same length as keypoints_frames.")
        
        if self.obstacleBool.shape[0] != num_frames:
            raise ValueError("obstacleBool must be the same length as keypoints_frames.")
        
        if self.IMUBool.shape[0] != num_frames:
            raise ValueError("IMUBool must be the same length as keypoints_frames.")
        
    def filter_by(self,
                  hawk=None,
                  perchDist=None, 
                  obstacle=False, 
                  year=None, 
                  Left=None,
                  IMU=False):
        """
        Returns boolean array of indices to filter the data.
        """

        def filter_by_bool(variable, bool_value):

            if bool_value is None:
                # Simply return the full array bool mask if passed None
                is_selected = np.ones(variable.shape, dtype=bool)
                return is_selected
            
            is_selected = variable == bool_value
            return is_selected

        # Initialise the filter
        filter = np.ones(len(self.frameID), dtype=bool)

        # Filter by hawk_ID
        if hawk is not None:
            filter = np.logical_and(filter, self.filter_by_hawk_ID(hawk))

        # Filter by perchDist
        if perchDist is not None:
            filter = np.logical_and(filter, self.filter_by_perchDist(perchDist))

        # Filter by obstacleToggle
        # if obstacle is not None:
        filter = np.logical_and(filter, filter_by_bool(self.obstacleBool, obstacle))

        # Filter by IMUToggle
        # if IMU is not None:
        filter = np.logical_and(filter, filter_by_bool(self.IMUBool, IMU))

        # Filter by Left
        # if Left is not None:
        filter = np.logical_and(filter, filter_by_bool(self.leftBool, Left))

        # Filter by year
        # if year is not None:
        filter = np.logical_and(filter, self.filter_by_year(year))
        
        return filter


            
    def filter_by_hawk_ID(self, hawk: str):

        def get_hawkID(hawk_name):

            if hawk_name.isdigit():
                # Transform the hawk_ID into a string with the correct format
                hawk_ID = str(hawk_name).zfill(2) + "_"
                
            # The user may have provided the full name of the hawk, or just the first few letters
            # And so returns the matching ID.

            if "dr" in hawk_name.lower():
                hawk_ID = "01_"  
            if "rh" in hawk_name.lower():
                hawk_ID = "02_"
            if "ru" in hawk_name.lower():
                hawk_ID = "03_"  
            if "to" in hawk_name.lower():
                hawk_ID = "04_"  
            if "ch" in hawk_name.lower():
                hawk_ID = "05_"
            
            return hawk_ID

        frameID = self.frameID

        if hawk is None:
            is_selected = np.ones(len(frameID), dtype=bool)
            return is_selected
        else:
            hawk_ID = get_hawkID(hawk)

        is_selected = frameID.str.startswith(hawk_ID)

        return is_selected
    
    def filter_by_perchDist(self, perchDist):

        # If perchDist is None, return the full array bool mask
        if perchDist is None:
            is_selected = np.ones(self.horzDist.shape, dtype=bool)
            return is_selected

        # Get any number from the perchDist string. The user may have given 
        # "12m" or "12 m" or "12"
        if perchDist.isdigit():
            perchDist = int(perchDist)
        else:
            perchDist = int(''.join(filter(str.isdigit, perchDist)))

        # Build back up the string to search for
        # Make sure the integer is padded such that it is 2 digits in length
        perchDist_str = "_" + str(perchDist).zfill(2) + "_"
        
        # Now looks for _05_ or _12_ etc in the frameID. Note, 05_09_ would be 
        # charmander flying 9m so we need to make sure we don't select that by leading 
        # and trailing _ . HawkID should always be found with "startswith". 

        is_selected = self.frameID.str.contains(perchDist_str)
        
        return is_selected

    def filter_by_year(self, year):
        frameID = self.frameID

        if year is None:
            is_selected = np.ones(len(frameID), dtype=bool)
            return is_selected

        # Data from 2017 and 2020 have different frameID formats, 
        # there's an extra _ in the frameID for 2020
        if year == 2017:
            is_selected = frameID.str.count('_') == 3
        elif year == 2020:
            is_selected = frameID.str.count('_') == 4
        else:
            raise ValueError("Year must be 2017 or 2020.")
        
        return is_selected