import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ------- Loading data -------

def process_data(csv_path):
    markers, markers_df = load_marker_frames(csv_path)
    frame_info, frame_info_df = load_frame_info(csv_path)

    if check_data(markers, frame_info):
        print("Data verified.")
    else:
        print("Data verification failed.")

    return markers, frame_info, markers_df, frame_info_df

# ....... Helper functions .......
def load_marker_frames(csv_path):
    markers_csv = pd.read_csv(csv_path)
    markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_1", "_x")
    markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_2", "_y")
    markers_csv.columns = markers_csv.columns.str.replace("_rot_xyz_3", "_z")

    print("Loaded marker data and renamed columns.")
    
    marker_cols = markers_csv.columns[markers_csv.columns.str.contains('_x|_y|_z')]
    markers_csv = markers_csv[marker_cols]
    n_markers = len(marker_cols) // 3

    markers = markers_csv.to_numpy().reshape(-1, n_markers, 3)
    
    return markers, markers_csv

def load_frame_info(csv_path):
    """Load and return frame info from a CSV file, cleaning as necessary."""
    frame_info_csv = pd.read_csv(csv_path)
    frame_info_csv.drop(columns=frame_info_csv.columns[frame_info_csv.columns.str.contains('rot_xyz')], inplace=True)
    
    frame_info_csv['HorzDistance'] *= -1  # Make horizontal distance negative

    frame_info = {
        'time':         frame_info_csv.get('time').to_numpy(),
        'horzDist':     frame_info_csv.get('HorzDistance').to_numpy(),
        'vertDist':     frame_info_csv.get('VertDistance').to_numpy(),
        'body_pitch':   frame_info_csv.get('body_pitch').to_numpy(),
        'frameID':      frame_info_csv.get('frameID'),
        'seqID':        frame_info_csv.get('seqID'),
        'birdID':       frame_info_csv.get('BirdID').to_numpy(),
        'perchDist':    frame_info_csv.get('PerchDistance').to_numpy(),
        'year':         frame_info_csv.get('Year').to_numpy(),
        'obstacleBool': frame_info_csv.get('Obstacle').to_numpy(),
        'IMUBool':      frame_info_csv.get('IMU').to_numpy(),
        'naiveBool':    frame_info_csv.get('Naive').to_numpy(),
        'leftBool':     frame_info_csv.get('Left') if 'Left' in frame_info_csv.columns else None
    }

    print("Loaded frame info and cleaned columns.")

    return frame_info, frame_info_csv

def check_data(markers, frame_info):
    """
    Check if all numpy array data elements in frame_info and markers have the same length.

    Parameters:
    - frame_info (dict): A dictionary containing various pieces of frame-related data, some of which are numpy arrays.
    - markers (numpy.ndarray): A numpy array containing markers data.

    Returns:
    - bool: True if all arrays have the same length, False otherwise.
    """
    # Gather all numpy arrays including markers
    all_data = list(frame_info.values()) + [markers]
    
    # Filter the list to include only numpy arrays and get their lengths
    array_lengths = [len(data) for data in all_data if isinstance(data, np.ndarray)]
    
    # Check if all numpy array lengths are the same
    if len(set(array_lengths)) == 1:
        return True
    else:
        # If lengths differ, print the mismatch information
        mismatch_info = {key: len(value) for key, value in frame_info.items() if isinstance(value, np.ndarray)}
        mismatch_info['markers'] = len(markers)
        print("Mismatch in data lengths found:", mismatch_info)
        return False


# ------- Filtering data -------

def filter_by(frame_info, 
              hawkname=None, 
              birdID=None, 
              perchDist=None, 
              obstacle=None, 
              year=None, 
              Left=None, 
              IMU=None, 
              naive=None):
    """ 
    Apply multiple filters based on given criteria and return indices as a boolean array. 
    """
    
    filter_mask = np.ones(len(frame_info['frameID']), dtype=bool)

    if hawkname is not None:
        filter_mask &= filter_by_hawkname(frame_info['frameID'], hawkname)

    if birdID is not None:
        filter_mask &= filter_by_bool(frame_info['birdID'], birdID)

    if perchDist is not None:
        filter_mask &= filter_by_perchDist(frame_info['frameID'], perchDist)

    if obstacle is not None:
        filter_mask &= filter_by_bool(frame_info['obstacleBool'], obstacle)

    if IMU is not None:
        filter_mask &= filter_by_bool(frame_info['IMUBool'], IMU)

    if Left is not None:
        # Check it is a unilateral dataset
        if frame_info["leftBool"] is None:
            raise ValueError("Left filter is only available for unilateral datasets.")
        filter_mask &= filter_by_bool(frame_info['leftBool'], Left)

    if year is not None:
        filter_mask &= filter_by_bool(frame_info['year'], year)

    if naive is not None:
        filter_mask &= filter_by_bool(frame_info['naiveBool'], naive)


    return filter_mask

# ....... Helper functions .......
def filter_by_bool(variable, bool_value):
    """ Helper function to filter data based on a boolean value. """
    if bool_value is None:
        return np.ones(variable.shape, dtype=bool)
    return variable == bool_value

def get_hawkID(hawk_name):
    """Determine the hawk ID from the provided name or numerical identifier."""
    if hawk_name.isdigit():
        # Transform the hawk_ID into a string with the correct format
        return hawk_name.zfill(2) + "_"

    # Mapping from hawk name initials to their IDs
    hawk_map = {
        "dr": "01_",
        "rh": "02_",
        "ru": "03_",
        "to": "04_",
        "ch": "05_"
    }

    # Return the matching ID based on the first two characters (lowercased)
    for key, value in hawk_map.items():
        if key in hawk_name.lower():
            return value

    # Return an empty string if no match is found (this should be handled carefully)
    return ""

def filter_by_hawkname(frameID, hawk):
    """Filter frameID entries based on a hawk name or ID."""
    if hawk is None:
        # Return a boolean array of True values if no specific hawk is specified
        return np.ones(len(frameID), dtype=bool)
    
    hawk_ID = get_hawkID(hawk)
    # Return a boolean array where the frameID starts with the determined hawk_ID
    return frameID.str.startswith(hawk_ID)

def filter_by_perchDist(frameID, perchDist):
    """
    Filter frameID entries based on a specified perch distance.
    
    Parameters:
    - frameID (pandas.Series): A Series containing frame identifiers.
    - perchDist (str or int): A string or integer indicating the perch distance,
      which may include non-digit characters (e.g., '12m', '12 m').
    
    Returns:
    - np.array: A boolean array where True indicates the frameID that matches
      the specified perch distance.
    """
    
    # If perchDist is None, return the full array bool mask
    if perchDist is None:
        return np.ones(len(frameID), dtype=bool)

    # Normalize perchDist to an integer
    if isinstance(perchDist, str):
        # Extract digits from the string and convert to integer
        perchDist = int(''.join(filter(str.isdigit, perchDist)))
    elif isinstance(perchDist, int):
        perchDist = perchDist
    else:
        raise ValueError("perchDist must be either an integer or a string containing digits.")

    # Format the perch distance as a string padded to 2 digits within underscores
    perchDist_str = "_" + str(perchDist).zfill(2) + "_"
    
    # Search for this pattern in frameID using str.contains
    is_selected = frameID.str.contains(perchDist_str)

    return is_selected



    @property
    def marker_names(self):
        """
        Returns the marker names.
        """
        return self.markers_df.columns.tolist()

# ------- PCA -------

def run_PCA(markers):

    # Reshape the data to be [n, nMarkers*3]
    pca_input = get_PCA_input(markers)

    # Run PCA
    pca = PCA()
    pca_output = pca.fit(pca_input)

    # Another word for eigenvectors is components.
    principal_components = pca_output.components_
    
    # Another word for scores is projections.
    scores = pca_output.transform(pca_input)

    # Check the shape of the output
    test_PCA_output(pca_input, principal_components, scores)

    return principal_components, scores, pca


# ....... Helper functions .......

def get_PCA_input_sizes(pca_input):
    """
    Get the sizes of the input data.
    """
    
    n_frames = pca_input.shape[0]
    n_markers = pca_input.shape[1]/3
    n_vars = pca_input.shape[1]

    return n_frames, n_markers, n_vars

def get_PCA_input(markers):
    """
    Reshape the data to be [n, nMarkers*3]
    """
    n_markers = markers.shape[1]
    pca_input = markers.reshape(-1, n_markers*3)

    return pca_input


def test_PCA_output(pca_input, principal_components, scores):
    """
    Test the shape of the PCA output.
    """
    n_frames, n_markers, n_vars = get_PCA_input_sizes(pca_input)

    assert n_vars == n_markers*3, "n_vars is not equal to n_markers*3."
    assert principal_components.shape[0] == n_vars, "principal_components is not the right shape."
    assert principal_components.shape[1] == n_vars, "principal_components is not the right shape."
    assert scores.shape[0] == n_frames, "scores first dim is not the right shape."
    assert scores.shape[1] == n_vars, "scores second dim is not the right shape."

def get_score_range(scores, num_frames=30):
    """
    Generate a series of scores for animations within a specified range.
    
    Parameters:
    - scores (numpy.ndarray): Array containing score values from which ranges are derived.
    - num_frames (int): Number of frames to generate scores for.
    
    Returns:
    - numpy.ndarray: An array of score values over the specified frame range.
    """
    num_components = scores.shape[1]

    min_score = np.mean(scores, axis=0) - (2 * np.std(scores, axis=0))
    max_score = np.mean(scores, axis=0) + (2 * np.std(scores, axis=0))

    half_length = num_frames // 2 + 1

    # Initialize score_frames with the shape [num_frames, num_components]
    score_frames = np.zeros([num_frames, num_components])

    for ii in range(num_components):
        # Create forward and backward ranges for each component using np.linspace
        forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
        backward = forward[::-1]  # Reverse the forward range

        # Combine forward and backward, and assign to the i-th column
        score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

    return score_frames

def reconstruct(score_frames, principal_components, mu, components_list=None):
    """
    Reconstruct frames based on principal components and score frames.

    Parameters:
    - principal_components (numpy.ndarray): The principal components matrix.
    - n_markers (int): Number of markers (points) per frame.
    - score_frames (numpy.ndarray): The score frames for reconstruction.
    - components_list (list): List of component indices to use for reconstruction. Defaults to all components.

    Returns:
    - numpy.ndarray: The reconstructed frames.
    """

    if components_list is None:
        components_list = range(principal_components.shape[1])

    if not isinstance(score_frames, np.ndarray):
        raise TypeError("score_frames must be a numpy array.")

    if len(score_frames.shape) != 2:
        raise ValueError("score_frames must be 2d.")

    assert score_frames.shape[1] == principal_components.shape[0], "score_frames must have the same number of columns as components_list."
    assert len(components_list) <= principal_components.shape[1], "components_list must not exceed the number of principal components."
    assert len(mu.shape)==3, "mu must be a 3d array: [1,nMarkers,3]."

    n_markers = mu.shape[1]
    n_dims = mu.shape[2]
    n_frames = score_frames.shape[0]


    # Select principal components and scores based on the provided list
    # principal_components is [n_components, n_markers*3]
    selected_PCs = principal_components[components_list,:]
    # score_frames is [n_frames, n_components]
    selected_scores = score_frames[:, components_list]

    reconstruction = np.dot(selected_scores,selected_PCs)  # [n, 12]
    reconstruction = reconstruction.reshape(-1, n_markers, n_dims)  # Reshape to [n, 4, 3]

    reconstructed_frames = mu + reconstruction  # Broadcasting [1, 4, 3] over [n, 4, 3]

    assert reconstructed_frames.shape[0] == n_frames, "Reconstructed frames do not match the number of frames."
    assert reconstructed_frames.shape[1] == n_markers, "Reconstructed frames do not match the number of markers."
    assert reconstructed_frames.shape[2] == n_dims, "Reconstructed frames do not match the number of dimensions."

    return reconstructed_frames


# class HawkPCATest:

#     def __init__(self,
#                  hawkdata_instance: HawkDataTest):
#         self.data = hawkdata_instance.markers
#         self.mu = None

#     def get_mu(self):

#         self.mu = np.mean(self.pca_input, axis=0)

#     def get_input(self,data=None):
        
#         if data is None:
#             data = self.data.copy()

#         # Reshape the data to be [n, nMarkers*3]
#         self.n_markers = data.shape[1]
#         pca_input = data.reshape(-1,self.n_markers*3)

#         return pca_input
    
#     def run_pca(self,data=None,filter=None):
        
#         self.pca_input = self.get_input(data)

#         if filter is not None:
#             self.pca_input = self.pca_input[filter]

#         self.n_frames = self.pca_input.shape[0]

#         self.get_mu()
        
#         pca = PCA()
#         pca_output = pca.fit(self.pca_input)

#         # Another word for eigenvectors is components.
#         self.principal_components = pca_output.components_
        
#         # Another word for scores is projections.
#         self.scores = pca_output.transform(self.pca_input)
        
#         # Test shape of outputs
#         self.test_pca()

#     def test_pca(self):

#         assert self.mu.shape[0] == self.n_markers*3, "mu is not the right shape."
       
#         assert self.principal_components.shape[0] == self.n_markers*3, "principal_components is not the right shape."
#         assert self.principal_components.shape[1] == self.n_markers*3, "principal_components is not the right shape."

#         assert self.scores.shape[0] == self.n_frames, "scores first dim is not the right shape."
#         assert self.scores.shape[1] == self.n_markers*3, "scores second dim is not the right shape."


#     def get_score_range(self, num_frames=30):

#         num_components = self.scores.shape[1]

#         min_score = np.mean(self.scores, axis=0) - (2 * np.std(self.scores, axis=0))
#         max_score = np.mean(self.scores, axis=0) + (2 * np.std(self.scores, axis=0))

#         half_length = num_frames // 2 + 1

#         # Initialize score_frames with the shape [n, 12]
#         self.score_frames = np.zeros([num_frames, num_components])


#         for ii in range(num_components):
        
#             # Create forward and backward ranges for each component using np.linspace
#             forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
            
#             # # If num_frames is odd, we add an extra element to 'forward'
#             # if num_frames % 2 != 0:
#             #     forward = np.append(forward, max_score[ii])
            
#             backward = forward[::-1]  # Reverse the forward range

#             # Combine forward and backward, and assign to the i-th column
#             self.score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

#         return self.score_frames

#     def select_components(self, components_list):

#         selected_components = self.principal_components[:,components_list]
        
#         return selected_components

#     def reconstruct(self,components_list=None, score_frames=None):

#         if components_list is None:
#             components_list = range(12)
        
#         if score_frames is None:
#             score_frames = self.score_frames
#         else:

#             # Check score_frames are a numpy array
#             if not isinstance(score_frames, np.ndarray):
#                 raise TypeError("score_frames must be a numpy array.")

#             # Check the score_frames are the right shape
#             if score_frames.shape[1] != self.principal_components.shape[0]:
#                 raise ValueError("score_frames must have the same number of columns as components_list.")
            
#             if len(score_frames.shape) != 2:
#                 raise ValueError("score_frames must be 2d.")
            

#         selected_PCs = self.principal_components[components_list]
#         selected_scores = score_frames[:,components_list]

#         num_frames = score_frames.shape[0]
#         reconstructed_frames = np.empty((0,4,3))


#         selected_PCs = selected_PCs.reshape(1, -1, self.n_markers, 3)  # [1, nPCs, 4, 3]
#         selected_scores = selected_scores.reshape(num_frames, -1, 1, 1)  # [n, nPCs, 1, 1]
#         mu = self.mu.reshape(1, self.n_markers, 3)  # [1, 4, 3]

#         reconstructed_frames = mu + np.sum(selected_scores * selected_PCs, axis=1)

#         return reconstructed_frames



# class HawkPCA:

#     """
#     Class to run PCA on the Hawk3D data.
#     """
    
#     def __init__(self, HawkData, KeypointManager):
#         self.data = HawkData
#         self.mu = KeypointManager.right_keypoints

#         # Make the dimensions fit for PCA
#         self.mu = self.mu.reshape(1,12)

#     def get_input(self, data=None):

#         if data is None:
#             data = self.data.markers

#         # The data is in the shape (n_frames, n_markers*n_dimensions)
#         pca_input = data.reshape(-1,12)

#         return pca_input
    
#     def run_PCA(self, data=None):

#         pca_input = self.get_input(data)
       
#         pca = PCA()
#         pca_output = pca.fit(pca_input)

#         self.pca_input = pca_input

#         # Another word for eigenvectors is components.
#         self.principal_components = pca_output.components_

#         # Another word for scores is projections.
#         self.scores = pca_output.transform(pca_input)

#     def get_score_range(self, num_frames=30):

#         num_components = self.scores.shape[1]

#         min_score = np.mean(self.scores, axis=0) - (2 * np.std(self.scores, axis=0))
#         max_score = np.mean(self.scores, axis=0) + (2 * np.std(self.scores, axis=0))

#         half_length = num_frames // 2 + 1

#         # Initialize score_frames with the shape [n, 12]
#         self.score_frames = np.zeros([num_frames, num_components])


#         for ii in range(num_components):
        
#             # Create forward and backward ranges for each component using np.linspace
#             forward = np.linspace(min_score[ii], max_score[ii], num=half_length)
            
#             # # If num_frames is odd, we add an extra element to 'forward'
#             # if num_frames % 2 != 0:
#             #     forward = np.append(forward, max_score[ii])
            
#             backward = forward[::-1]  # Reverse the forward range

#             # Combine forward and backward, and assign to the i-th column
#             self.score_frames[:, ii] = np.concatenate((forward, backward[:num_frames - half_length]))

#     def select_components(self, components_list):

#         selected_components = self.principal_components[:,components_list]
        
#         return selected_components

#     def reconstruct(self,components_list=None, score_frames=None):

#         if components_list is None:
#             components_list = range(12)
        
#         if score_frames is None:
#             score_frames = self.score_frames
#         else:

#             # Check score_frames are a numpy array
#             if not isinstance(score_frames, np.ndarray):
#                 raise TypeError("score_frames must be a numpy array.")

#             # Check the score_frames are the right shape
#             if score_frames.shape[1] != self.principal_components.shape[0]:
#                 raise ValueError("score_frames must have the same number of columns as components_list.")
            
#             if len(score_frames.shape) != 2:
#                 raise ValueError("score_frames must be 2d.")
            

#         selected_PCs = self.principal_components[components_list]
#         selected_scores = score_frames[:,components_list]

        
#         num_frames = score_frames.shape[0]
#         reconstructed_frames = np.empty((0,4,3))

#         # for ii in range(num_frames):
#         #     score = selected_scores[ii]
            
            
#         #     selected_PCs = selected_PCs.reshape(1,4,3)
#         #     mu = self.mu.reshape(1,4,3)

#         #     frame = mu + score * selected_PCs
#         #     frame = frame.reshape(-1,4,3)

#         #     reconstructed_frames = np.append(reconstructed_frames,frame,axis=0)
            
#         selected_PCs = selected_PCs.reshape(1, -1, 4, 3)  # [1, nPCs, 4, 3]
#         selected_scores = selected_scores.reshape(num_frames, -1, 1, 1)  # [n, nPCs, 1, 1]
#         mu = self.mu.reshape(1, 4, 3)  # [1, 4, 3]

#         reconstructed_frames = mu + np.sum(selected_scores * selected_PCs, axis=1)

#         return reconstructed_frames

#     def get_results_table(self,filter=None):


#         # Using the original dataframe from the csv, concat the PCA scores
#         # Filter makes sure the data is the same size as the PCA input. 
#         self.results_table = self.concat_scores(filter)

#         # Add a bins column based on the horizontal distance
#         self.bin_by_horz_distance()

#         return self.results_table

#     def concat_scores(self, filter=None):

#         """
#         Returns a pandas dataframe with the scores added to the original data.
#         """

#         # If filter is None, check the input is the same
#         # size as the original data and warn the user.
#         if filter is None:
#             if self.data.markers.shape[0] != self.scores.shape[0]:
#                 raise ValueError("Please input the filter you used to run the PCA.")

#         # Get the data pandas dataframe, just the useful columns
#         col_names = ['frameID','time','HorzDistance','body_pitch','Obstacle','IMU','Left']
#         data = self.data.dataframe[col_names]

#         # Apply the filter if given
#         if filter is not None:
#             data = data[filter]
#         else:
#             data = data

#         num_components = self.scores.shape[1]

#         # Add the scores to the dataframe. Give the column the name 'PC1' etc.
#         PC_names = ["PC" + str(i) for i in range(1, num_components+1)]

#         score_df = pd.DataFrame(self.scores, columns=PC_names)

#         data = pd.concat([data, score_df], axis=1)

#         return data
    
#     def bin_by_horz_distance(self, size_bin=0.05):

#         """
#         Bin the horizontal distance into bins of size size_bin.
#         Using the HawkPCA results dataframe.
#         """

#         bins = np.arange(-12.2,0.2, size_bin)
#         bins = np.around(bins, 3)
#         labels = bins.astype(str).tolist()
#         # make label one smaller
#         labels.pop(0)

#         self.results_table['bins'] = pd.cut(self.results_table['HorzDistance'], 
#                                        bins, 
#                                        right=False, 
#                                        labels = labels, 
#                                        include_lowest=True)
    
#     def filter_results_by(self,
#                   hawk=None,
#                   perchDist=None, 
#                   obstacle=False, 
#                   year=None, 
#                   Left=None,
#                   IMU=False):
#         """
#         Returns boolean array of indices to filter the data.
#         Somewhat of a repeat of HawkData.filter_by but that is not compatible with
#         pandas, maybe refactor it there.
#         """

#         # Get the data pandas dataframe
#         data = self.results_table
#         frameID = data.frameID

#         # Initialise the filter
#         filter = np.ones(len(frameID), dtype=bool)

#         # Filter by hawk
#         if hawk is not None:
#             filter = np.logical_and(filter, HawkData.filter_by_hawk_ID(frameID, hawk))

#         # Filter by perchDist
#         if perchDist is not None:
#             filter = np.logical_and(filter, HawkData.filter_by_perchDist(frameID, perchDist))

#         # Filter by year
#         if year is not None:
#             filter = np.logical_and(filter, HawkData.filter_by_year(frameID,year))

#         # Filter by obstacle
#         if obstacle is not None:
#             filter = np.logical_and(filter, data.Obstacle==obstacle)
            
#         # Filter by Left
#         if Left is not None:
#             filter = np.logical_and(filter, data.Left==Left)
            
#         # Filter by IMU
#         if IMU is not None:
#             filter = np.logical_and(filter, data.IMU==IMU)
        
#         return data[filter]
         
#     def binned_results(self, results_table=None):
 
#         """
#         Returns a dataframe with the mean scores and std scores for each bin, plus 
#         the mean for horizontal distance, time and body pitch.
#         """

#         def mean_score_by_bin(results_table):

#             # Get the PC names
#             PC_columns = results_table.filter(like='PC')

#             # observed=True is to prevent a warning about future behaviour. 
#             # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
#             # It will only group by the bins that are present in the data, 
#             # so in this case it is irelevant.
            
#             mean_scores = PC_columns.groupby(results_table['bins'], observed=True).mean()

#             return mean_scores
        
#         def std_score_by_bin(results_table):

#             # Get the PC names
#             PC_columns = results_table.filter(like='PC')

#             # observed=True is to prevent a warning about future behaviour. 
#             # See https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby-specify
            
#             std_scores = PC_columns.groupby(results_table['bins'], observed=True).mean()

#             return std_scores
        
#         def mean_col_by_bin(results_table,column_name):
            
#             mean_column = results_table.groupby('bins', observed=True)[column_name].mean()

#             return mean_column


#         if results_table is None:
#             results_table = self.results_table
#             print("Binning full data, no filters applied.")


#         binned_means_scores = mean_score_by_bin(results_table)
#         binned_std_scores = std_score_by_bin(results_table)

#         # Concatenate HorzDist means etc to the binned_means_scores dataframe
#         col_names = ["HorzDistance","time","body_pitch"]

#         for col_name in col_names:
#             binned_means_scores[col_name] = mean_col_by_bin(results_table,col_name)
            
#         # Remove rows with NaN values
#         binned_means_scores = binned_means_scores.dropna(axis=0, how='any')
#         binned_std_scores = binned_std_scores.dropna(axis=0, how='any')

#         return binned_means_scores, binned_std_scores

#     def reconstruct_reduced_dims(self, binned_means_scores, selected_PCs=None):
        
#         # Make a numpy array with binned scores. 
#         # This will be the input to the reconstruction.

#         # Get the PC names
#         PC_columns = binned_means_scores.filter(like='PC')

#         # Transform to numpy array
#         score_frames = PC_columns.to_numpy()

#         reconstructed_frames = self.reconstruct(selected_PCs, score_frames)


#         return reconstructed_frames

        
