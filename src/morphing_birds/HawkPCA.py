import numpy as np
import pandas as pd
from sklearn.decomposition import PCA




class HawkDataTest:
    def __init__(self, csv_path):
        self.load_marker_frames(csv_path)
        
        self.load_frame_info(csv_path)

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

        # Save the number of markers
        # Should be 4 if these are unilateral data and 8 if bilateral
        self.n_markers = len(markers_csv.columns) // 3

        # Save the cleaned up dataframe
        self.markers_df = markers_csv.copy()
        
        # Make a numpy array with the markers
        markers = markers_csv.to_numpy()

        # Reshape to 3D
        markers = markers.reshape(-1,self.n_markers,3)
        
        # Save numpy array of markers
        self.markers = markers

        return markers

    def load_frame_info(self,csv_path):
        """
        Load the frame info from the dataset.
        """
        
        info_csv = pd.read_csv(csv_path).copy()

        # Remove marker data (they are stored separately)
        # Drop everything with "rot_xyz" in the name
        info_csv = info_csv.drop(info_csv.columns[info_csv.columns.str.contains('rot_xyz')], axis=1)

        # Makes horizontal distance NEGATIVE
        info_csv['HorzDistance'] = -info_csv['HorzDistance']
        
        self.frameinfo = info_csv.copy()

        self.time       = info_csv['time'].to_numpy()
        self.horzDist   = info_csv['HorzDistance'].to_numpy()
        self.vertDist   = info_csv['VertDistance'].to_numpy()
        self.body_pitch = info_csv['body_pitch'].to_numpy()
        self.frameID    = info_csv['frameID']
        self.seqID      = info_csv['seqID']
        self.birdID     = info_csv['BirdID'].to_numpy()
        self.perchDist  = info_csv['PerchDistance'].to_numpy()
        self.year       = info_csv['Year'].to_numpy()
        self.obstacleBool = info_csv['Obstacle'].to_numpy()
        self.IMUBool    = info_csv['IMU'].to_numpy()
        self.naiveBool  = info_csv['Naive'].to_numpy()
        
        # If the data is unilateral, there will be a left column
        if 'Left' in info_csv.columns:
            self.leftBool   = info_csv['Left'].to_numpy()

    def check_data(self):
        """
        Check that the data are the same length.
        """

        num_frames = self.markers.shape[0]

        if self.horzDist.shape[0] != num_frames:
            raise ValueError("horzDist must be the same length as keypoints_frames.")
        
        if len(self.frameID) != num_frames:
            raise ValueError("frameID must be the same length as keypoints_frames.")
        
        if "Left" in self.frameinfo.columns:
            if self.leftBool.shape[0] != num_frames:
                raise ValueError("leftBool must be the same length as keypoints_frames.")
        
        if self.body_pitch.shape[0] != num_frames:
            raise ValueError("body_pitch must be the same length as keypoints_frames.")
        
        if self.obstacleBool.shape[0] != num_frames:
            raise ValueError("obstacleBool must be the same length as keypoints_frames.")
        
        if self.IMUBool.shape[0] != num_frames:
            raise ValueError("IMUBool must be the same length as keypoints_frames.")
        
        if self.naiveBool.shape[0] != num_frames:
            raise ValueError("naiveBool must be the same length as keypoints_frames.")
        
        if self.vertDist.shape[0] != num_frames:
            raise ValueError("vertDist must be the same length as keypoints_frames.")

    def filter_by(self,
                  hawkname=None,
                  birdID=None,
                  perchDist=None, 
                  obstacle=None, 
                  year=None, 
                  Left=None,
                  IMU=None,
                  naive=None):
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

        # Filter by hawk name
        if hawkname is not None:
            filter = np.logical_and(filter, self.filter_by_hawkname(hawkname))

        # Filter by birdID
        if birdID is not None:
            filter = np.logical_and(filter, filter_by_bool(self.birdID, birdID))

        # Filter by perchDist
        if perchDist is not None:
            # Check if int
            if isinstance(perchDist, int):
                # If the user has just given a number (5)
                filter = np.logical_and(filter, filter_by_bool(self.perchDist, perchDist))
            else:
                # If the user has given a string (5m)
                filter = np.logical_and(filter, self.filter_by_perchDist(perchDist))

        # Filter by obstacleToggle
        if obstacle is not None:
            filter = np.logical_and(filter, filter_by_bool(self.obstacleBool, obstacle))

        # Filter by IMUToggle
        if IMU is not None:
            filter = np.logical_and(filter, filter_by_bool(self.IMUBool, IMU))

        # Filter by Left
        # if Left is not None:
        if self.n_markers == 4:
            filter = np.logical_and(filter, filter_by_bool(self.leftBool, Left))

        # Filter by year
        # if year is not None:
        filter = np.logical_and(filter, filter_by_bool(self.year, year))
        
        # Filter by naive
        # if year is not None:
        filter = np.logical_and(filter, filter_by_bool(self.naiveBool, naive))
        

        return filter

 
    def filter_by_hawkname(self, hawk: str):

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
        
        if hawk is None:
            is_selected = np.ones(len(self.frameID), dtype=bool)
            return is_selected
        else:
            hawk_ID = get_hawkID(hawk)

        is_selected = self.frameID.str.startswith(hawk_ID)

        return is_selected
    

    def filter_by_perchDist(self, perchDist):

        
        # If perchDist is None, return the full array bool mask
        if perchDist is None:
            is_selected = np.ones(self.frameID, dtype=bool)
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


    @property
    def marker_names(self):
        """
        Returns the marker names.
        """
        return self.markers_df.columns.tolist()

class HawkPCATest:

    def __init__(self,
                 hawkdata_instance: HawkDataTest):
        self.data = hawkdata_instance.markers
        self.mu = None

    def get_mu(self):

        self.mu = np.mean(self.pca_input, axis=0)

    def get_input(self,data=None):
        
        if data is None:
            data = self.data.copy()

        # Reshape the data to be [n, nMarkers*3]
        self.n_markers = data.shape[1]
        pca_input = data.reshape(-1,self.n_markers*3)

        return pca_input
    
    def run_pca(self,data=None,filter=None):
        
        self.pca_input = self.get_input(data)

        if filter is not None:
            self.pca_input = self.pca_input[filter]

        self.n_frames = self.pca_input.shape[0]

        self.get_mu()
        
        pca = PCA()
        pca_output = pca.fit(self.pca_input)

        # Another word for eigenvectors is components.
        self.principal_components = pca_output.components_
        
        # Another word for scores is projections.
        self.scores = pca_output.transform(self.pca_input)
        
        # Test shape of outputs
        self.test_pca()

    def test_pca(self):

        assert self.mu.shape[0] == self.n_markers*3, "mu is not the right shape."
       
        assert self.principal_components.shape[0] == self.n_markers*3, "principal_components is not the right shape."
        assert self.principal_components.shape[1] == self.n_markers*3, "principal_components is not the right shape."

        assert self.scores.shape[0] == self.n_frames, "scores first dim is not the right shape."
        assert self.scores.shape[1] == self.n_markers*3, "scores second dim is not the right shape."


    def get_score_range(self, num_frames=30):

        num_components = self.scores.shape[1]

        min_score = np.mean(self.scores, axis=0) - (2 * np.std(self.scores, axis=0))
        max_score = np.mean(self.scores, axis=0) + (2 * np.std(self.scores, axis=0))

        half_length = num_frames // 2 + 1

        # Initialize score_frames with the shape [n, 12]
        self.score_frames = np.zeros([num_frames, num_components])


        for ii in range(num_components):
        
            # Create forward and backward ranges for each component using np.linspace
            forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
            
            # # If num_frames is odd, we add an extra element to 'forward'
            # if num_frames % 2 != 0:
            #     forward = np.append(forward, max_score[ii])
            
            backward = forward[::-1]  # Reverse the forward range

            # Combine forward and backward, and assign to the i-th column
            self.score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

        return self.score_frames

    def select_components(self, components_list):

        selected_components = self.principal_components[:,components_list]
        
        return selected_components

    def reconstruct(self,components_list=None, score_frames=None):

        if components_list is None:
            components_list = range(12)
        
        if score_frames is None:
            score_frames = self.score_frames
        else:

            # Check score_frames are a numpy array
            if not isinstance(score_frames, np.ndarray):
                raise TypeError("score_frames must be a numpy array.")

            # Check the score_frames are the right shape
            if score_frames.shape[1] != self.principal_components.shape[0]:
                raise ValueError("score_frames must have the same number of columns as components_list.")
            
            if len(score_frames.shape) != 2:
                raise ValueError("score_frames must be 2d.")
            

        selected_PCs = self.principal_components[components_list]
        selected_scores = score_frames[:,components_list]

        num_frames = score_frames.shape[0]
        reconstructed_frames = np.empty((0,4,3))


        selected_PCs = selected_PCs.reshape(1, -1, self.n_markers, 3)  # [1, nPCs, 4, 3]
        selected_scores = selected_scores.reshape(num_frames, -1, 1, 1)  # [n, nPCs, 1, 1]
        mu = self.mu.reshape(1, self.n_markers, 3)  # [1, 4, 3]

        reconstructed_frames = mu + np.sum(selected_scores * selected_PCs, axis=1)

        return reconstructed_frames



class HawkPCA:

    """
    Class to run PCA on the Hawk3D data.
    """
    
    def __init__(self, HawkData, KeypointManager):
        self.data = HawkData
        self.mu = KeypointManager.right_keypoints

        # Make the dimensions fit for PCA
        self.mu = self.mu.reshape(1,12)

    def get_input(self, data=None):

        if data is None:
            data = self.data.markers

        # The data is in the shape (n_frames, n_markers*n_dimensions)
        pca_input = data.reshape(-1,12)

        return pca_input
    
    def run_PCA(self, data=None):

        pca_input = self.get_input(data)
       
        pca = PCA()
        pca_output = pca.fit(pca_input)

        self.pca_input = pca_input

        # Another word for eigenvectors is components.
        self.principal_components = pca_output.components_

        # Another word for scores is projections.
        self.scores = pca_output.transform(pca_input)

    def get_score_range(self, num_frames=30):

        num_components = self.scores.shape[1]

        min_score = np.mean(self.scores, axis=0) - (2 * np.std(self.scores, axis=0))
        max_score = np.mean(self.scores, axis=0) + (2 * np.std(self.scores, axis=0))

        half_length = num_frames // 2 + 1

        # Initialize score_frames with the shape [n, 12]
        self.score_frames = np.zeros([num_frames, num_components])


        for ii in range(num_components):
        
            # Create forward and backward ranges for each component using np.linspace
            forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
            
            # # If num_frames is odd, we add an extra element to 'forward'
            # if num_frames % 2 != 0:
            #     forward = np.append(forward, max_score[ii])
            
            backward = forward[::-1]  # Reverse the forward range

            # Combine forward and backward, and assign to the i-th column
            self.score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

    def select_components(self, components_list):

        selected_components = self.principal_components[:,components_list]
        
        return selected_components

    def reconstruct(self,components_list=None, score_frames=None):

        if components_list is None:
            components_list = range(12)
        
        if score_frames is None:
            score_frames = self.score_frames
        else:

            # Check score_frames are a numpy array
            if not isinstance(score_frames, np.ndarray):
                raise TypeError("score_frames must be a numpy array.")

            # Check the score_frames are the right shape
            if score_frames.shape[1] != self.principal_components.shape[0]:
                raise ValueError("score_frames must have the same number of columns as components_list.")
            
            if len(score_frames.shape) != 2:
                raise ValueError("score_frames must be 2d.")
            

        selected_PCs = self.principal_components[components_list]
        selected_scores = score_frames[:,components_list]

        
        num_frames = score_frames.shape[0]
        reconstructed_frames = np.empty((0,4,3))

        # for ii in range(num_frames):
        #     score = selected_scores[ii]
            
            
        #     selected_PCs = selected_PCs.reshape(1,4,3)
        #     mu = self.mu.reshape(1,4,3)

        #     frame = mu + score * selected_PCs
        #     frame = frame.reshape(-1,4,3)

        #     reconstructed_frames = np.append(reconstructed_frames,frame,axis=0)
            
        selected_PCs = selected_PCs.reshape(1, -1, 4, 3)  # [1, nPCs, 4, 3]
        selected_scores = selected_scores.reshape(num_frames, -1, 1, 1)  # [n, nPCs, 1, 1]
        mu = self.mu.reshape(1, 4, 3)  # [1, 4, 3]

        reconstructed_frames = mu + np.sum(selected_scores * selected_PCs, axis=1)

        return reconstructed_frames

    def get_results_table(self,filter=None):


        # Using the original dataframe from the csv, concat the PCA scores
        # Filter makes sure the data is the same size as the PCA input. 
        self.results_table = self.concat_scores(filter)

        # Add a bins column based on the horizontal distance
        self.bin_by_horz_distance()

        return self.results_table

    def concat_scores(self, filter=None):

        """
        Returns a pandas dataframe with the scores added to the original data.
        """

        # If filter is None, check the input is the same
        # size as the original data and warn the user.
        if filter is None:
            if self.data.markers.shape[0] != self.scores.shape[0]:
                raise ValueError("Please input the filter you used to run the PCA.")

        # Get the data pandas dataframe, just the useful columns
        col_names = ['frameID','time','HorzDistance','body_pitch','Obstacle','IMU','Left']
        data = self.data.dataframe[col_names]

        # Apply the filter if given
        if filter is not None:
            data = data[filter]
        else:
            data = data

        num_components = self.scores.shape[1]

        # Add the scores to the dataframe. Give the column the name 'PC1' etc.
        PC_names = ["PC" + str(i) for i in range(1, num_components+1)]

        score_df = pd.DataFrame(self.scores, columns=PC_names)

        data = pd.concat([data, score_df], axis=1)

        return data
    
    def bin_by_horz_distance(self, size_bin=0.05):

        """
        Bin the horizontal distance into bins of size size_bin.
        Using the HawkPCA results dataframe.
        """

        bins = np.arange(-12.2,0.2, size_bin)
        bins = np.around(bins, 3)
        labels = bins.astype(str).tolist()
        # make label one smaller
        labels.pop(0)

        self.results_table['bins'] = pd.cut(self.results_table['HorzDistance'], 
                                       bins, 
                                       right=False, 
                                       labels = labels, 
                                       include_lowest=True)
    
    def filter_results_by(self,
                  hawk=None,
                  perchDist=None, 
                  obstacle=False, 
                  year=None, 
                  Left=None,
                  IMU=False):
        """
        Returns boolean array of indices to filter the data.
        Somewhat of a repeat of HawkData.filter_by but that is not compatible with
        pandas, maybe refactor it there.
        """

        # Get the data pandas dataframe
        data = self.results_table
        frameID = data.frameID

        # Initialise the filter
        filter = np.ones(len(frameID), dtype=bool)

        # Filter by hawk
        if hawk is not None:
            filter = np.logical_and(filter, HawkData.filter_by_hawk_ID(frameID, hawk))

        # Filter by perchDist
        if perchDist is not None:
            filter = np.logical_and(filter, HawkData.filter_by_perchDist(frameID, perchDist))

        # Filter by year
        if year is not None:
            filter = np.logical_and(filter, HawkData.filter_by_year(frameID,year))

        # Filter by obstacle
        if obstacle is not None:
            filter = np.logical_and(filter, data.Obstacle==obstacle)
            
        # Filter by Left
        if Left is not None:
            filter = np.logical_and(filter, data.Left==Left)
            
        # Filter by IMU
        if IMU is not None:
            filter = np.logical_and(filter, data.IMU==IMU)
        
        return data[filter]
         
    def binned_results(self, results_table=None):
 
        """
        Returns a dataframe with the mean scores and std scores for each bin, plus 
        the mean for horizontal distance, time and body pitch.
        """

        def mean_score_by_bin(results_table):

            # Get the PC names
            PC_columns = results_table.filter(like='PC')

            # observed=True is to prevent a warning about future behaviour. 
            # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
            # It will only group by the bins that are present in the data, 
            # so in this case it is irelevant.
            
            mean_scores = PC_columns.groupby(results_table['bins'], observed=True).mean()

            return mean_scores
        
        def std_score_by_bin(results_table):

            # Get the PC names
            PC_columns = results_table.filter(like='PC')

            # observed=True is to prevent a warning about future behaviour. 
            # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
            
            std_scores = PC_columns.groupby(results_table['bins'], observed=True).mean()

            return std_scores
        
        def mean_col_by_bin(results_table,column_name):
            
            mean_column = results_table.groupby('bins', observed=True)[column_name].mean()

            return mean_column


        if results_table is None:
            results_table = self.results_table
            print("Binning full data, no filters applied.")


        binned_means_scores = mean_score_by_bin(results_table)
        binned_std_scores = std_score_by_bin(results_table)

        # Concatenate HorzDist means etc to the binned_means_scores dataframe
        col_names = ["HorzDistance","time","body_pitch"]

        for col_name in col_names:
            binned_means_scores[col_name] = mean_col_by_bin(results_table,col_name)
            
        # Remove rows with NaN values
        binned_means_scores = binned_means_scores.dropna(axis=0, how='any')
        binned_std_scores = binned_std_scores.dropna(axis=0, how='any')

        return binned_means_scores, binned_std_scores

    def reconstruct_reduced_dims(self, binned_means_scores, selected_PCs=None):
        
        # Make a numpy array with binned scores. 
        # This will be the input to the reconstruction.

        # Get the PC names
        PC_columns = binned_means_scores.filter(like='PC')

        # Transform to numpy array
        score_frames = PC_columns.to_numpy()

        reconstructed_frames = self.reconstruct(selected_PCs, score_frames)


        return reconstructed_frames

        
