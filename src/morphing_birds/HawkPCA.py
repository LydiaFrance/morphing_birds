import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


from .Hawk3D import HawkData



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

        
