import numpy as np



def format_keypoint_frames(animal3d_instance, keypoints_frames):

        """
        Formats the keypoints_frames to be [n,8,3] where n is the number of frames 
        and 8 is the number of keypoints and 3 is the number of dimensions.
        If just 4 keypoints are given, the function will mirror the keypoints to make the left side.
        """

        if len(np.shape(keypoints_frames)) == 2:
            keypoints_frames = keypoints_frames.reshape(1, -1, 3)
            print("Warning: Only one frame given.")

    # Mirror the keypoints_frames if only the right is given. 
        if keypoints_frames.shape[1] == len(animal3d_instance.right_marker_names):
            keypoints_frames = animal3d_instance.mirror_keypoints(keypoints_frames)

        return keypoints_frames

def check_transformation_frames(num_frames, transformation_frames):

        """
        Checks that the transformation frames are the same length as the keypoints frames.
        If passed None, create an array of zeros for the transformation.
        """

        if transformation_frames is None:
            transformation_frames = np.zeros(num_frames)
        
        if len(transformation_frames) != num_frames:
            raise ValueError("Transformation frames must be the same length as keypoints_frames.")
        
        
        return transformation_frames

