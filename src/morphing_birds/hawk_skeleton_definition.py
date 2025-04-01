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


class HawkSkeletonDefinition(SkeletonDefinition):
    """
    The `HawkSkeletonDefinition` class is a specific implementation of the 
    `SkeletonDefinition` class, tailored to represent the shape of the hawk. 
    By extending `SkeletonDefinition`, it inherits 
    all the methods and attributes necessary for defining markers and body 
    sections, while also allowing for customisation specific to hawks.

    This class defines various markers that were placed on key anatomical points 
    on a hawk's body, such as wingtips, primary feathers, and tail tips. 
    Additionally, it specifies additional fixed markers which are for reference 
    in the visualisation and do not move and are not included in analysis.

    The `HawkSkeletonDefinition` class also includes methods to retrieve 
    marker names based on their side (left or right), facilitating the 
    manipulation and visualisation of the hawk's shape in a 3D space. 
    This design ensures that the hawk's skeletal representation has a
    consistent interface for different animal types within the application.

    In summary, this class serves as a bridge between the general 
    functionality provided by `SkeletonDefinition` and the specific 
    requirements of hawk morphology, enhancing code reusability and 
    maintainability.
    """

    def __init__(self):

        """
        In this class, we define specific markers that correspond to 
        key anatomical points on a hawk, such as the wingtips and tail 
        tips as measured with motion capture. Fixed markers are for 
        visualisation and not included in analysis, e.g. shoulders, tailbase. 

        The `__init__` method is the constructor for the class, 
        which is called when an instance of `HawkSkeletonDefinition` is created. 
        It sets up the marker names, fixed marker names, and body sections that 
        are unique to the hawk shape. 

        - `marker_names`: This list contains the names of all the movable markers on the hawk.
        - `fixed_marker_names`: This list includes markers that remain stationary and are used for 
          reference during visualisation.
        - `body_sections`: This dictionary maps the names of body sections to their corresponding 
          markers, allowing for organised representation of the hawk's shape.

        By using this class, users can easily access and manipulate the hawk's body structure, 
        for tasks such as animation and visualisation in a 3D environment, and biomechanics analysis.
        """

        marker_names = [
            "left_wingtip", "right_wingtip", 
            "left_primary", "right_primary", 
            "left_secondary", "right_secondary", 
            "left_tailtip", "right_tailtip"
        ]
        fixed_marker_names = [
            "left_shoulder", "left_tailbase", "right_tailbase", 
            "right_shoulder", "hood", "tailpack"
        ]
        body_sections = {
            "left_handwing": ["left_wingtip", "left_primary", "left_secondary"],
            "right_handwing": ["right_wingtip", "right_primary", "right_secondary"],
            "left_armwing": ["left_primary", "left_secondary", "left_tailbase", "left_shoulder"],
            "right_armwing": ["right_primary", "right_secondary", "right_tailbase", "right_shoulder"],
            "body": ["right_shoulder", "left_shoulder", "left_tailbase", "right_tailbase"],
            "head": ["right_shoulder", "hood", "left_shoulder"],
            "tail": ["right_tailtip", "left_tailtip", "left_tailbase", "right_tailbase"],
        }

        # super() is used to call the __init__ method of the 
        # class SkeletonDefinition (the parent class). 
        super().__init__(marker_names, fixed_marker_names, body_sections)


    def get_all_marker_names(self):
        """
        Returns a list of all marker names from the hawk shape definition.
        """
        return self.marker_names + self.fixed_marker_names

    def get_right_marker_names(self):
        """
        Returns a list of all right side marker names from the hawk shape definition.
        """
        return [name for name in self.marker_names if "right" in name]
    
    def get_left_marker_names(self):
        """
        Returns a list of all left side marker names from the hawk shape definition.
        """
        return [name for name in self.marker_names if "left" in name]