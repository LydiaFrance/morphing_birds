"""
morphing_birds: Run PCA on morphing bird wings and tail in flight.
"""
from __future__ import annotations

from .Hawk3D import Hawk3D, plot, interactive_plot, animate
# from .HawkData import HawkData
# from .HawkPCA import HawkPCA
# from .Keypoints import KeypointManager
# from .PCAFigures import PCAFigures
from .trytofix import HawkDataTest, HawkPCATest

# from .Figures import plot_components


__all__ = ("trytofix",
           "__version__")
__version__ = "0.1.0"
