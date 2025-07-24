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


class KestrelSkeletonDefinition(SkeletonDefinition):
    """
    The `KestrelSkeletonDefinition` class is a specific implementation of the 
    `SkeletonDefinition` class, tailored to represent the shape of the kestrel. 

    This class extends the `SkeletonDefinition` class, adding specific marker 
    names and body section definitions that are unique to kestrels. 

    It also includes methods to retrieve marker names and indices for both 
    left and right sides of the kestrel, ensuring a consistent interface for
    different animal types within the application.

    In summary, this class serves as a bridge between the general 
    functionality provided by `SkeletonDefinition` and the specific 
    requirements of kestrel morphology, enhancing code reusability and 
    maintainability.
    """

    def __init__(self):
        """
        Initialize the KestrelSkeletonDefinition with three categories of markers:
        1. Ignored Markers: Not used at all (not stored, not plotted)
        2. Fixed Markers: Loaded from mean shape, used for plotting but not for analysis
        3. Active Markers: Used for both plotting and analysis
        """
        self.marker_name_change = {
            # Head markers
            "head": "h1", 
            "head_mid": "he",
            "left_head": "h2",
            "right_head": "h3",

            # Body markers
            "left_backpack": "b3",
            "right_backpack": "b2",
            "centre_backpack": "be",
            "centre_back_backpack": "b1",

            # Tail markers
            "centre_tail_base": "t_c_1",
            "centre_tail_mid": "t_c_2",
            "centre_tail_tip": "t_c_3",
            "left_tail_base": "t_l_1",
            "left_tail_mid": "t_l_2",
            "left_tail_tip": "t_l_3",
            "right_tail_base": "t_r_1",
            "right_tail_mid": "t_r_2",
            "right_tail_tip": "t_r_3",
            "left_tailpack": "tl",
            "right_tailpack": "tr",
            "centre_tailpack": "tc",
            
            # Right Arm Wing markers
            "right_shoulder": "r_sh",
            "right_wrist": "r_w",
            "right_elbow": "r_e",
            "right_alula": "r_al_1",
            "right_alula_lower": "r_al_3",
            "right_secondary_tip": "r_s1_1",
            "right_lastsecondary_tip": "r_sb_1",

            # Left Arm Wing markers
            "left_shoulder": "l_sh",
            "left_wrist": "l_w",
            "left_elbow": "l_e",
            "left_alula": "l_al_1",
            "left_alula_lower": "l_al_3",
            "left_secondary_tip": "l_s1_1",
            "left_lastsecondary_tip": "l_sb_1",

            # Right Hand Wing markers
            "right_firstprimary_tip": "r_p1_3",
            "right_firstprimary_mid": "r_p1_2",
            "right_firstprimary_base": "r_p1_1",
            "right_secondprimary_tip": "r_p2_3",
            "right_secondprimary_mid": "r_p2_2",
            "right_secondprimary_base": "r_p2_1",
            "right_fourthprimary_tip": "r_p4_1",
            "right_lastprimary_tip": "r_p9_1",

            # Left Hand Wing markers
            "left_firstprimary_tip": "l_p1_3",
            "left_firstprimary_mid": "l_p1_2",
            "left_firstprimary_base": "l_p1_1",
            "left_secondprimary_tip": "l_p2_3",
            "left_secondprimary_mid": "l_p2_2",
            "left_secondprimary_base": "l_p2_1",
            "left_fourthprimary_tip": "l_p4_1",
            "left_lastprimary_tip": "l_p9_1"
            }
        
        # Create reverse mapping for lookup
        self.marker_name_change_reverse = {v: k for k, v in self.marker_name_change.items()}
        
        # === Define markers that are completely ignored (not stored, not plotted) ===
        self.ignored_marker_names = [
            # Extra head markers
            "head_mid", "left_head", "right_head",
            # Pack markers
            "left_backpack", "right_backpack", "centre_backpack", "centre_back_backpack",
            "left_tailpack", "right_tailpack", "centre_tailpack"
        ]
        
        # === Define fixed markers (stored and plotted, but not used in analysis) ===
        # These markers are loaded from mean shape and kept fixed
        self.fixed_marker_names = ["head", "left_shoulder", "right_shoulder"]
        
        # For simple mode, we fix additional markers
        self.fixed_marker_names_simple = self.fixed_marker_names + [
            "left_lastsecondary_tip", "right_lastsecondary_tip"
        ]
        
        # === Define body sections for visualization ===
        body_sections = {
            # Full mode sections
            "head": ["right_shoulder", "head", "left_shoulder"], 
            "body": ["right_shoulder", "left_shoulder", "left_lastsecondary_tip","right_lastsecondary_tip"], 
            "tail": ["right_lastsecondary_tip", "left_lastsecondary_tip", "left_tail_tip", "right_tail_tip"],
            "right_armwing": ["right_shoulder", "right_elbow", "right_wrist", "right_lastprimary_tip", "right_secondary_tip", "right_lastsecondary_tip", "right_shoulder", "right_wrist"], 
            "left_armwing": ["left_shoulder", "left_elbow", "left_wrist", "left_lastprimary_tip", "left_secondary_tip", "left_lastsecondary_tip", "left_shoulder", "left_wrist"], 
            "left_handwing": ["left_wrist", "left_firstprimary_base", "left_firstprimary_mid", "left_firstprimary_tip", "left_secondprimary_tip", "left_secondprimary_mid", "left_secondprimary_base", "left_fourthprimary_tip", "left_lastprimary_tip"], 
            "right_handwing": ["right_wrist", "right_firstprimary_base", "right_firstprimary_mid", "right_firstprimary_tip", "right_secondprimary_tip", "right_secondprimary_mid", "right_secondprimary_base", "right_fourthprimary_tip", "right_lastprimary_tip"], 
            "left_alula": ["left_wrist", "left_alula", "left_alula_lower"],
            "right_alula": ["right_wrist", "right_alula", "right_alula_lower"],
            
            # Simple mode sections WITH WRIST(comparable to hawks)
            # "head_simple": ["right_shoulder", "head", "left_shoulder"],
            # "body_simple": ["right_shoulder", "left_shoulder", "left_lastsecondary_tip", "right_lastsecondary_tip"], 
            # "tail_simple": ["right_lastsecondary_tip", "left_lastsecondary_tip", "left_tail_tip", "right_tail_tip"],
            # "left_armwing_simple": ["left_wrist", "left_secondary_tip", "left_lastsecondary_tip", "left_shoulder"], 
            # "right_armwing_simple": ["right_wrist", "right_secondary_tip", "right_lastsecondary_tip", "right_shoulder"], 
            # "left_handwing_simple": ["left_wrist", "left_secondprimary_tip", "left_secondary_tip"], 
            # "right_handwing_simple": ["right_wrist", "right_secondprimary_tip", "right_secondary_tip"]

            # Simple mode sections (comparable to hawks)
            "head_simple": ["right_shoulder", "head", "left_shoulder"],
            "body_simple": ["right_shoulder", "left_shoulder", "left_lastsecondary_tip", "right_lastsecondary_tip"], 
            "tail_simple": ["right_lastsecondary_tip", "left_lastsecondary_tip", "left_tail_tip", "right_tail_tip"],
            "left_armwing_simple": ["left_firstprimary_base", "left_secondary_tip", "left_lastsecondary_tip", "left_shoulder"], 
            "right_armwing_simple": ["right_firstprimary_base", "right_secondary_tip", "right_lastsecondary_tip", "right_shoulder"], 
            "left_handwing_simple": ["left_firstprimary_base", "left_secondprimary_tip", "left_secondary_tip"], 
            "right_handwing_simple": ["right_firstprimary_base", "right_secondprimary_tip", "right_secondary_tip"]
        
        
        }

        # Initialize the parent class with only the active markers
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
        Returns a list of all marker names in the simple body sections.
        The markers are returned in a specific order that must be maintained:
        [left_secondprimary_tip, right_secondprimary_tip, 
         left_firstprimary_base, right_firstprimary_base,
         left_secondary_tip, right_secondary_tip,
         left_tail_tip, right_tail_tip]
        """
        # Define the canonical order of markers
        canonical_order = [
            "left_secondprimary_tip", "right_secondprimary_tip",
            # "left_wrist", "right_wrist",
            "left_firstprimary_base", "right_firstprimary_base",
            "left_secondary_tip", "right_secondary_tip",
            "left_tail_tip", "right_tail_tip"
        ]
        
        # Get all markers from simple sections
        simple_sections = [section for section in self.body_sections if section.endswith("_simple")]
        available_markers = set()
        for section in simple_sections:
            available_markers.update(self.body_sections[section])
            
        # Remove fixed markers
        available_markers = available_markers - set(self.fixed_marker_names_simple)
        
        # Verify all canonical markers are available
        missing_markers = set(canonical_order) - available_markers
        if missing_markers:
            raise ValueError(f"Missing expected markers in simple sections: {missing_markers}")
        
        # Return markers in canonical order
        return canonical_order
    
    def get_marker_names_full(self) -> list:
        """
        Returns a list of all marker names in the full body sections.
        The markers are returned in a specific order that must be maintained,
        alternating between left and right sides where applicable.
        
        The order follows anatomical structure:
        1. Hand wing (primaries) from outermost to innermost
        2. Alula
        3. Mid-wing and wrist
        4. Secondaries
        5. Tail feathers (base to tip, including center)
        
        Only active markers (not fixed or ignored) that are used in body sections are included.
        """
        # Define the desired canonical order
        desired_order = [
            # Hand wing (Primaries), outermost to innermost
            "left_firstprimary_tip", "right_firstprimary_tip",
            "left_firstprimary_mid", "right_firstprimary_mid",
            "left_firstprimary_base", "right_firstprimary_base",
            
            "left_secondprimary_tip", "right_secondprimary_tip",
            "left_secondprimary_mid", "right_secondprimary_mid",
            "left_secondprimary_base", "right_secondprimary_base",
            
            "left_fourthprimary_tip", "right_fourthprimary_tip",
            "left_lastprimary_tip", "right_lastprimary_tip",
            
            # Alula
            "left_alula", "right_alula",
            "left_alula_lower", "right_alula_lower",
            
            # Mid-wing + wrist
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            
            # Secondaries
            "left_secondary_tip", "right_secondary_tip",
            "left_lastsecondary_tip", "right_lastsecondary_tip",
            
            # Tail feathers: base → tip
            "left_tail_base", "right_tail_base",
            "left_tail_tip", "right_tail_tip",
        ]
        
        # Get all markers from non-simple sections
        non_simple_sections = [section for section in self.body_sections if not section.endswith("_simple")]
        available_markers = set()
        for section in non_simple_sections:
            available_markers.update(self.body_sections[section])
            
        # Remove fixed and ignored markers
        available_markers = available_markers - set(self.fixed_marker_names) - set(self.ignored_marker_names)
        
        # Add debugging information
        print("\nIn get_marker_names_full:")
        print(f"Desired order length: {len(desired_order)}")
        print(f"Available markers before filtering: {len(available_markers)}")
        print(f"Fixed markers being removed: {self.fixed_marker_names}")
        
        # Filter the canonical order to only include available active markers
        canonical_order = [marker for marker in desired_order if marker in available_markers]
        
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
            marker_names = self.get_marker_names_full()
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
