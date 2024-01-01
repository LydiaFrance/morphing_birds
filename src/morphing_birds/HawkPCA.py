import numpy as np
from sklearn.decomposition import PCA

from .Hawk3D import *


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
        num_components = pca_input.shape[1]

        pca = PCA()
        pca_output = pca.fit(pca_input)


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

    def reconstruct(self,components_list=None):

        if components_list is None:
            components_list = range(12)
        
        selected_PCs = self.principal_components[components_list]
        selected_scores = self.score_frames[:,components_list]

        
        num_frames = self.score_frames.shape[0]
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
        

        # reconstructed_frames = reconstructed.reshape(-1,4,3)

        return reconstructed_frames

    