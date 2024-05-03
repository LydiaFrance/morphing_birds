"""
morphing_birds: Run PCA on morphing bird wings and tail in flight.
"""
from __future__ import annotations

from .Hawk3D import Hawk3D, plot, interactive_plot, plot_multiple, animate, animate_compare

__all__ = ("__version__", "Hawk3D", "plot", "interactive_plot", "plot_multiple", "animate", "animate_compare")
__version__ = "0.1.0"
