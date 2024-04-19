"""
morphing_birds: Run PCA on morphing bird wings and tail in flight.
"""
from __future__ import annotations

from .Hawk3D import Hawk3D, plot, interactive_plot, animate
from .HawkPCA import process_data, filter_by, run_PCA, get_score_range, reconstruct
from .PCAFigures import plot_components_grid
# from .trytofix import 

# from .Figures import plot_components


__all__ = ("trytofix",
           "__version__")
__version__ = "0.1.0"
