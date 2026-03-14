"""morphing_birds: Config-driven 3D animal morphology toolkit."""

from __future__ import annotations

from .skeleton import SkeletonDefinition
from .animal import Animal3D
from .bilateral import (
    mirror_to_bilateral,
    make_unilateral,
    make_bilateral,
    validate_left_right,
)
from .transforms import TransformState
from .scaling import (
    unit_conversion_factor,
    compute_wingspan,
    compute_body_length,
    UNIT_FACTORS,
)
from .data_loading import (
    load_from_csv,
    load_from_dataframe,
    load_from_dict,
    load_mean_shape_csv,
)
from .configs import load_config, list_configs
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
    create_dash_app,
)

__all__ = (
    "__version__",
    # Core
    "SkeletonDefinition",
    "Animal3D",
    "TransformState",
    # Bilateral
    "mirror_to_bilateral",
    "make_unilateral",
    "make_bilateral",
    "validate_left_right",
    # Scaling
    "unit_conversion_factor",
    "compute_wingspan",
    "compute_body_length",
    "UNIT_FACTORS",
    # Data loading
    "load_from_csv",
    "load_from_dataframe",
    "load_from_dict",
    "load_mean_shape_csv",
    # Config
    "load_config",
    "list_configs",
    # Plotting
    "plot",
    "interactive_plot",
    "plot_multiple",
    "plot_plotly",
    "plot_plotly_compare",
    "plot_compare_plotly",
    "plot_partial_plotly",
    "plot_plotly_with_trace",
    "animate",
    "animate_compare",
    "animate_plotly",
    "animate_plotly_compare",
    "animate_plotly_partial",
    "save_plotly_animation",
    "create_dash_app",
)
__version__ = "0.2.0"
