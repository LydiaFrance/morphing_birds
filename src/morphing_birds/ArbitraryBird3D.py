from .Animal3D import Animal3D
from .hawk_skeleton_definition import HawkSkeletonDefinition

class ArbitraryBird3D(Animal3D):
    """
    Class for creating custom bird configurations with arbitrary marker positions.
    """
    def __init__(self, moving_markers: dict, fixed_markers: dict):
        """
        Create a new bird with custom marker positions.
        
        Parameters:
        - moving_markers: dict
            Maps marker names to [x,y,z] coordinates for moving markers
            e.g., {"left_wingtip": [1.0, 2.0, 3.0], ...}
        - fixed_markers: dict
            Maps fixed marker names to [x,y,z] coordinates
            e.g., {"left_shoulder": [1.0, 2.0, 3.0], ...}
        """
        # Use HawkSkeletonDefinition for marker structure
        skeleton_definition = HawkSkeletonDefinition()
        
        # Initialize with custom positions
        super().__init__(
            skeleton_definition,
            custom_marker_positions=moving_markers,
            custom_fixed_positions=fixed_markers
        )