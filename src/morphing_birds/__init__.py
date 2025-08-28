"""
morphing_birds: Run PCA on morphing bird wings and tail in flight.
"""
from __future__ import annotations

from .SkeletonDefinition import SkeletonDefinition
from .Animal3D import Animal3D
from .ArbitraryBird3D import ArbitraryBird3D
from .hawk_skeleton_definition import HawkSkeletonDefinition
from .spider_skeleton_definition import SpiderSkeletonDefinition
from .kestrel_skeleton_definition import KestrelSkeletonDefinition
from .pigeon_skeleton_definition import PigeonSkeletonDefinition
from .Hawk3D import Hawk3D
from .Spider3D import Spider3D
from .Kestrel3D import Kestrel3D
from .Pigeon3D import Pigeon3D
from .plotting import (
    plot, 
    interactive_plot, 
    plot_multiple, 
    plot_plotly,
    plot_plotly_compare,
    plot_compare_plotly,
    plot_partial_plotly,
    plot_plotly_with_trace,
    animate, 
    animate_compare, 
    animate_plotly, 
    animate_plotly_compare,
    animate_plotly_partial,
    save_plotly_animation,
    create_dash_app
)

__all__ = ("__version__", 
           "SkeletonDefinition", "Animal3D", "ArbitraryBird3D",
           "Hawk3D", "HawkSkeletonDefinition", "Kestrel3D", "KestrelSkeletonDefinition",
           "Pigeon3D", "PigeonSkeletonDefinition", "SpiderSkeletonDefinition", "Spider3D",
           "plot", "interactive_plot", "plot_multiple", "plot_plotly", "plot_plotly_compare", "plot_partial_plotly",
           "plot_compare_plotly", "animate", "animate_compare", "create_dash_app", "plot_plotly_with_trace",
           "animate_plotly", "animate_plotly_compare", "save_plotly_animation",
           "animate_plotly_partial")
__version__ = "0.1.0"
