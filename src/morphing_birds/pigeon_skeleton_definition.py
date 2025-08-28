from .SkeletonDefinition import SkeletonDefinition

"""
The `SkeletonDefinition` class is imported because it serves as the foundational 
blueprint for defining the structure of an animal's skeleton in our application. 

By inheriting from `SkeletonDefinition`, the `HawkSkeletonDefinition` class can 
use the methods and attributes defined in the parent class, allowing for a consistent 
interface when working with different animal morphologies. 

This design promotes code reusability and organisation, making it easier to manage 
and extend the functionality for various animal types. 

In summary, importing `SkeletonDefinition` is essential for creating a structured 
and maintainable codebase that can handle the complexities of animal shape definitions.
"""


class PigeonSkeletonDefinition(SkeletonDefinition):
    """
    The `PigeonSkeletonDefinition` class is a specific implementation of the 
    `SkeletonDefinition` class, tailored to represent the shape of the pigeon. 

    This class extends the `SkeletonDefinition` class, adding specific marker 
    names and body section definitions that are unique to pigeons. 

    It also includes methods to retrieve marker names and indices for both 
    left and right sides of the pigeon, ensuring a consistent interface for
    different animal types within the application.

    In summary, this class serves as a bridge between the general 
    functionality provided by `SkeletonDefinition` and the specific 
    requirements of kestrel morphology, enhancing code reusability and 
    maintainability.
    """

    def __init__(self):
        """
        Initialise the PigeonSkeletonDefinition with three categories of markers:
        1. Ignored Markers: Not used at all (not stored, not plotted)
        2. Fixed Markers: Loaded from mean shape, used for plotting but not for analysis
        3. Active Markers: Used for both plotting and analysis
        """
        self.marker_name_change = {
            # Head markers
            "head": "Head",

            # Body markers
            "centre_backpack": "Body_Start",
            "centre_body_base": "Body_End",

            # Tail markers
            "left_tailtip": "Left_Tail",
            "right_tailtip": "Right_Tail",
            "centre_tailtip": "Middle_Tail",
            
            # Right Arm Wing markers
            "right_lastsecondary_tip" : "Right_Shoulder_Trailing_Edge",
            "right_elbow" : "Right_Shoulder_End",
            "right_shoulder" : "Right_Shoulder_Start",
            
            # Left Arm Wing markers
            "left_lastsecondary_tip" : "Left_Shoulder_Trailing_Edge",
            "left_elbow" : "Left_Shoulder_End",
            "left_shoulder" : "Left_Shoulder_Start",
            

            # Right Hand Wing markers
            "right_wingtip": "Right_Tip",
            "right_wrist": "Right_Wrist_Leading_Edge",
            "right_secondary": "Right_Wrist_Trailing_Edge",

            # Left Hand Wing markers
            "left_wingtip": "Left_Tip",
            "left_wrist": "Left_Wrist_Leading_Edge",
            "left_secondary": "Left_Wrist_Trailing_Edge",

            # Tailbase markers (calculated from shoulder positions)
            "left_tailbase": "Left_Tailbase",
            "right_tailbase": "Right_Tailbase",

            }
        
        # Create reverse mapping for lookup
        self.marker_name_change_reverse = {v: k for k, v in self.marker_name_change.items()}
        
        # === Define markers that are completely ignored (not stored, not plotted) ===
        self.ignored_marker_names = ["centre_tailtip"]
        
        # === Define fixed markers (stored and plotted, but not used in analysis) ===
        # These markers are loaded from mean shape and kept fixed
        self.fixed_marker_names = ["head", "centre_backpack", "centre_body_base", "left_tailbase", "right_tailbase"]
        
        # For simple mode, we fix additional markers
        self.fixed_marker_names_simple = self.fixed_marker_names + ["right_shoulder", "left_shoulder"]
        
        # === Define body sections for visualization ===
        body_sections = {
            # Full mode sections
            "head": ["right_shoulder", "head", "left_shoulder"], 
            "body": ["right_shoulder", "right_tailbase", "centre_body_base", "left_tailbase", "left_shoulder"], 
            "tail": ["right_tailbase","centre_body_base","left_tailbase","left_tailtip",  "right_tailtip"],
            "right_armwing": ["right_shoulder", "right_elbow", "right_wrist", "right_secondary", "right_lastsecondary_tip", "right_tailbase"], 
            "left_armwing": ["left_shoulder", "left_elbow", "left_wrist", "left_secondary", "left_lastsecondary_tip", "left_tailbase"], 
            "left_handwing": ["left_wrist", "left_secondary", "left_wingtip"], 
            "right_handwing": ["right_wrist", "right_secondary", "right_wingtip"], 
            
            # Simple mode sections WITH WRIST(comparable to hawks)
            "head_simple": ["right_shoulder", "head", "left_shoulder"],
            "body_simple": ["right_shoulder", "left_shoulder", "left_tailbase", "right_tailbase"], 
            "tail_simple": ["left_tailbase","left_tailtip", "right_tailtip", "right_tailbase"],
            "left_armwing_simple": ["left_wrist", "left_secondary", "left_tailbase", "left_shoulder"], 
            "right_armwing_simple": ["right_wrist", "right_secondary", "right_tailbase", "right_shoulder"], 
            "left_handwing_simple": ["left_wrist", "left_secondary", "left_wingtip"], 
            "right_handwing_simple": ["right_wrist", "right_secondary", "right_wingtip"],

            # Simple mode sections (comparable to hawks)
            # "head_simple": ["right_shoulder", "head", "left_shoulder"],
            # "body_simple": ["right_shoulder", "left_shoulder", "left_lastsecondary_tip", "right_lastsecondary_tip"], 
            # "tail_simple": ["right_lastsecondary_tip", "left_lastsecondary_tip", "left_tail_tip", "right_tail_tip"],
            # "left_armwing_simple": ["left_firstprimary_base", "left_secondary_tip", "left_lastsecondary_tip", "left_shoulder"], 
            # "right_armwing_simple": ["right_firstprimary_base", "right_secondary_tip", "right_lastsecondary_tip", "right_shoulder"], 
            # "left_handwing_simple": ["left_firstprimary_base", "left_secondprimary_tip", "left_secondary_tip"], 
            # "right_handwing_simple": ["right_firstprimary_base", "right_secondprimary_tip", "right_secondary_tip"]
        
        
        }

        # Initialise the parent class with only the active markers
        all_marker_names = list(self.marker_name_change.keys())
        active_markers = [name for name in all_marker_names 
                        if name not in self.ignored_marker_names 
                        and name not in self.fixed_marker_names]
        

        
        super().__init__(active_markers, self.fixed_marker_names, body_sections)

    def get_original_marker_name(self, readable_marker_name: str) -> str:
        """
        Returns the original marker name for a given readable marker name.
        """
        return self.marker_name_change_reverse[readable_marker_name]
    
    def get_readable_marker_name(self, original_marker_name: str) -> str:
        """
        Returns the readable marker name for a given original marker name.
        """
        return self.marker_name_change[original_marker_name]
    
    def get_fixed_marker_names(self) -> list:
        """
        Returns the fixed marker names.
        """
        return self.fixed_marker_names
    
    def get_marker_names(self) -> list:
        """
        Returns a list of all marker names.
        """
        return self.marker_names

    def get_right_marker_names(self) -> list:
        """
        Returns a list of all right side marker names.
        """
        return [name for name in self.marker_names if name.startswith("right_")]
    
    def get_left_marker_names(self) -> list:
        """
        Returns a list of all left side marker names.
        """
        return [name for name in self.marker_names if name.startswith("left_")]
    
    def get_ignored_marker_names(self) -> list:
        """
        Returns a list of all ignored marker names.
        """
        return self.ignored_marker_names


    def get_marker_names_simple(self) -> list:
        """
        Returns a compact, left/right-paired set of markers for simple mode.

        Maintenance notes (reference only):
        - Keep left/right pairs adjacent and in the same relative order for all pairs.
        - Match hawk semantics for downstream algorithms; for pigeons, "primary" is
          represented by the wrist landmark.
        - Keep center-line markers out of simple mode.

        Order (matching hawk semantics: primary=wrist):
        [left_wingtip, right_wingtip,
         left_wrist, right_wrist,
         left_secondary, right_secondary,
         left_tailtip, right_tailtip]
        """
        desired_order = [
            "left_wingtip", "right_wingtip",
            "left_wrist", "right_wrist",
            "left_secondary", "right_secondary",
            "left_tailtip", "right_tailtip",
        ]
        
        # Get all markers from simple sections
        simple_sections = [section for section in self.body_sections if section.endswith("_simple")]
        available_markers = set()
        for section in simple_sections:
            available_markers.update(self.body_sections[section])
            
        # Remove fixed markers (simple uses same fixed set unless overridden)
        available_markers = available_markers - set(self.fixed_marker_names_simple)

        # Verify all canonical markers are available
        missing_markers = set(desired_order) - available_markers
        if missing_markers:
            raise ValueError(f"Missing expected markers in simple sections: {missing_markers}")
        
        # Return markers in canonical order
        return desired_order
    
    def get_marker_names_full(self, verbose: bool = False) -> list:
        """
        Returns ordered active markers for full mode (no centre/fixed).
        Includes shoulders/elbows/wrists/secondaries/lastsecondary tips/wingtips/tail tips.

        Maintenance notes (reference only):
        - Favor distal→proximal along the wing, and lateral→center→lateral across tail.
        - Maintain left/right alternation for paired markers.
        - Pigeons use wrist as the primary analogue to align with hawk pipelines.
        - Center tail is included if available and not fixed.

        Order (filtered to what exists in body sections, matching hawk semantics where primary=wrist):
        [left_wingtip, right_wingtip,
         left_wrist, right_wrist,
         left_secondary, right_secondary,
         left_lastsecondary_tip, right_lastsecondary_tip,
         left_elbow, right_elbow,
         left_shoulder, right_shoulder,
         centre_body_base,
         left_tailtip, right_tailtip]
        """
        desired_order = [
            "left_wingtip", "right_wingtip",
            "left_wrist", "right_wrist",
            "left_secondary", "right_secondary",
            "left_lastsecondary_tip", "right_lastsecondary_tip",
            "left_elbow", "right_elbow",
            "left_shoulder", "right_shoulder",
            "centre_body_base",
            "left_tailtip", "right_tailtip",
        ]
        
        # Get all markers from non-simple sections
        non_simple_sections = [section for section in self.body_sections if not section.endswith("_simple")]
        available_markers = set()
        for section in non_simple_sections:
            available_markers.update(self.body_sections[section])
            
        # Remove fixed and ignored markers
        available_markers = available_markers - set(self.fixed_marker_names) - set(self.ignored_marker_names)
        
        # Add debugging information
        if verbose:
            print("\nIn get_marker_names_full:")
            print(f"Desired order length: {len(desired_order)}")
            print(f"Available markers before filtering: {len(available_markers)}")
            print(f"Fixed markers being removed: {self.fixed_marker_names}")
        
        # Filter the canonical order to only include available active markers
        canonical_order = [marker for marker in desired_order if marker in available_markers]
        if verbose:
            print(f"Final canonical order length: {len(canonical_order)}")
            print(f"Markers in canonical order but not available: {[m for m in desired_order if m not in available_markers]}")
            print(f"Markers available but not in canonical order: {[m for m in available_markers if m not in desired_order]}")
        
        return canonical_order
    
    def get_hawk_version_of_marker_name(self, marker_name: str) -> str:
        """
        Returns the hawk marker name for a given kestrel marker name.
        """
        return self.marker_name_change_to_hawk[marker_name]

    def get_marker_pairs_and_centers(self, use_simple: bool = None) -> tuple:
        """
        Returns lists of marker pairs and center markers based on the mode.
        
        Parameters:
        - use_simple (bool, optional): Whether to use simple mode. If None, uses current setting.
        
        Returns:
        - tuple: (left_markers, right_markers, center_markers)
            - left_markers: List of left-side marker names in order
            - right_markers: List of right-side marker names in order (matching left_markers)
            - center_markers: List of center marker names
            
        Raises:
        - ValueError: If a left marker doesn't have a matching right marker
        - ValueError: If marker names don't follow the expected pattern (left_*, right_*, centre_*)
        """
        # Get the appropriate marker list
        if use_simple:
            marker_names = self.get_marker_names_simple()
            # Simple mode has no center markers, just left-right pairs
            left_markers = marker_names[::2]  # Even indices
            right_markers = marker_names[1::2]  # Odd indices
            center_markers = []
        else:
            marker_names = self.get_marker_names_full(verbose=False)
            # In full mode, we need to identify center markers
            left_markers = []
            right_markers = []
            center_markers = []
            
            # First pass: categorize markers
            for marker in marker_names:
                if not (marker.startswith('left_') or marker.startswith('right_') or marker.startswith('centre_')):
                    raise ValueError(f"Invalid marker name pattern: {marker}. Must start with 'left_', 'right_', or 'centre_'")
                
                if marker.startswith('left_'):
                    # For each left marker, find its right pair
                    right_marker = 'right_' + marker[5:]
                    if right_marker in marker_names:
                        left_markers.append(marker)
                        right_markers.append(right_marker)
                    else:
                        raise ValueError(f"Left marker {marker} has no matching right marker {right_marker}")
                elif marker.startswith('centre_'):
                    center_markers.append(marker)
                # Skip right_ markers as they're handled with left_ markers
        
        # Validate that we have matching numbers of left and right markers
        if len(left_markers) != len(right_markers):
            raise ValueError(f"Mismatch in number of left ({len(left_markers)}) and right ({len(right_markers)}) markers")
        
        # Validate that each left-right pair follows the same pattern
        for left, right in zip(left_markers, right_markers):
            if not (left.startswith('left_') and right.startswith('right_')):
                raise ValueError(f"Invalid marker pair: {left} - {right}")
            if left[5:] != right[6:]:
                raise ValueError(f"Mismatched marker pair: {left} - {right}. Suffixes should be identical")
        
        return left_markers, right_markers, center_markers
