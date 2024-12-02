"""
morphing_birds: Run PCA on morphing bird wings and tail in flight.
"""
from __future__ import annotations

from .SkeletonDefinition import SkeletonDefinition
from .Animal3D import Animal3D
from .hawk_skeleton_definition import HawkSkeletonDefinition
from .spider_skeleton_definition import SpiderSkeletonDefinition
from .Hawk3D import Hawk3D
from .Spider3D import Spider3D
from .plotting import (
    plot, 
    interactive_plot, 
    plot_multiple, 
    plot_plotly,
    animate, 
    animate_compare, 
    animate_plotly, 
    animate_plotly_compare, 
    save_plotly_animation,
    create_dash_app
)

__all__ = ("__version__", 
           "SkeletonDefinition", "Animal3D", 
           "Hawk3D", "HawkSkeletonDefinition", 
           "SpiderSkeletonDefinition", "Spider3D",
           "plot", "interactive_plot", "plot_multiple", "plot_plotly",
           "animate", "animate_compare", "create_dash_app", "animate_plotly",
           "animate_plotly_compare", "save_plotly_animation")
__version__ = "0.1.0"
