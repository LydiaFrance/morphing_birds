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

        body_sections = {"left_handwing": [], 
                         "right_handwing": [], 
                         "left_armwing": [], 
                         "right_armwing": [], 
                         "body": [], 
                         "head": [], 
                         "tail": []}
        
        