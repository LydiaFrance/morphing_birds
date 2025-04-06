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
        In this class, we define specific markers that correspond to 
        key anatomical points on a kestrel, such as the wingtips and tail 
        tips as measured with motion capture. Fixed markers are for 
        visualisation and not included in analysis, e.g. shoulders, tailbase. 
        
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
            "right_armwing_mid": "r_e",
            "right_alula": "r_al_1",
            "right_alula_lower": "r_al_3",
            "right_secondary_tip": "r_s1_1",
            "right_lastsecondary_tip": "r_sb_1",

            # Left Arm Wing markers
            "left_shoulder": "l_sh",
            "left_wrist": "l_w",
            "left_armwing_mid": "l_e",
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

        marker_names = list(self.marker_name_change.keys())

        # fixed markers
        # we will treat these as fixed markers as they are not useful for animation

        fixed_marker_names = ["left_backpack", "right_backpack", 
                              "centre_backpack", "centre_back_backpack", 
                              "left_tailpack", "right_tailpack", "centre_tailpack" 
                              ]
        
        
        # Define the kestrel sections for animation
        body_sections = {
            "head": ["right_shoulder", "head", "left_shoulder"], 
            "body": ["right_shoulder", "left_shoulder", "left_tail_base","centre_tail_base", "right_tail_base"], 
            "tail": ["right_tail_base", "centre_tail_base", "left_tail_base", "left_tail_tip", "centre_tail_tip", "right_tail_tip"],
            "right_armwing": ["right_shoulder", "right_wrist", "right_armwing_mid", "right_lastprimary_tip", "right_secondary_tip", "right_lastsecondary_tip"], 
            "left_armwing": ["left_shoulder", "left_wrist", "left_armwing_mid", "left_lastprimary_tip", "left_secondary_tip", "left_lastsecondary_tip"], 
            "left_handwing": ["left_wrist", "left_firstprimary_base", "left_firstprimary_mid", "left_firstprimary_tip", "left_secondprimary_tip", "left_secondprimary_mid", "left_secondprimary_base", "left_fourthprimary_tip", "left_lastprimary_tip"], 
            "right_handwing": ["right_wrist", "right_firstprimary_base", "right_firstprimary_mid", "right_firstprimary_tip", "right_secondprimary_tip", "right_secondprimary_mid", "right_secondprimary_base", "right_fourthprimary_tip", "right_lastprimary_tip"], 
            "left_alula": ["left_wrist", "left_alula", "left_alula_lower"],
            "right_alula": ["right_wrist", "right_alula", "right_alula_lower"],
            
            # An alternative way to define the body sections
            # This uses comparable markers to the hawks
            "head_simple": ["right_shoulder", "head", "left_shoulder"],
            "body_simple": ["right_shoulder", "left_shoulder", "left_lastsecondary_tip", "right_lastsecondary_tip"], 
            "tail_simple": ["right_lastsecondary_tip", "left_lastsecondary_tip", "left_tail_tip", "right_tail_tip"],
            "left_armwing_simple": ["left_firstprimary_base", "left_secondary_tip", "left_lastsecondary_tip", "left_shoulder"], 
            "right_armwing_simple": ["right_firstprimary_base", "right_secondary_tip", "right_lastsecondary_tip", "right_shoulder"], 
            "left_handwing_simple": ["left_firstprimary_base", "left_secondprimary_tip", "left_secondary_tip"], 
            "right_handwing_simple": ["right_firstprimary_base", "right_secondprimary_tip", "right_secondary_tip"], 
            
        }
        
        # First, start with fixed markers as ignored
        ignored_marker_names = fixed_marker_names.copy()

        # Create a set of all markers used in any body section
        used_in_body_sections = set()
        for section_markers in body_sections.values():
            used_in_body_sections.update(section_markers)

        # Add any marker that isn't used in any body section to ignored_marker_names
        for marker_name in marker_names:
            if marker_name not in used_in_body_sections:
                ignored_marker_names.append(marker_name)

        self.ignored_marker_names = ignored_marker_names
        marker_names = [marker_name for marker_name in marker_names if marker_name not in ignored_marker_names]

        # We will also have an additional set of marker names that 
        # are treated as fixed markers when we run the kestrel data like a hawk. 
        self.fixed_marker_names_simple = ["left_shoulder", "right_shoulder", "right_lastsecondary_tip", "left_lastsecondary_tip", "head"]
        
        # Define additional fixed markers to be excluded from motion data
        # This is separate from fixed_marker_names and fixed_marker_names_simple,
        # which are used for defining the polygon structure
        # self.additional_fixed_markers = []
        
        # Default setting: to fix shoulders only, uncomment the following line:
        self.additional_fixed_markers = ["left_shoulder", "right_shoulder", "head"]

        # To fix both shoulders and tail base, uncomment the following line:
        # self.additional_fixed_markers = ["left_shoulder", "right_shoulder", "left_tail_base", "right_tail_base"]

        # We also want to translate the kestrel marker names to the hawk marker names
        self.marker_name_change_to_hawk = {
            "left_secondary_tip" : "left_secondary",
            "right_secondary_tip" : "right_secondary",
            "left_tail_tip" : "left_tailtip",
            "right_tail_tip" : "right_tailtip",
            "right_firstprimary_base" : "right_primary",
            "left_firstprimary_base" : "left_primary",
            "right_secondprimary_tip" : "right_wingtip",
            "left_secondprimary_tip" : "left_wingtip"
        }
        self.marker_name_change_to_kestrel = {v: k for k, v in self.marker_name_change_to_hawk.items()}

        # super() is used to call the __init__ method of the 
        # class SkeletonDefinition (the parent class). 
        super().__init__(marker_names, fixed_marker_names, body_sections)

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
    
    def get_hawk_version_of_marker_name(self, marker_name: str) -> str:
        """
        Returns the hawk marker name for a given kestrel marker name.
        """
        return self.marker_name_change_to_hawk[marker_name]
