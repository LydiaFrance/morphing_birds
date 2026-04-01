"""morphing_birds: Config-driven 3D animal morphology toolkit."""

from __future__ import annotations

from .animal import Animal3D
from .bilateral import (
    make_bilateral,
    make_unilateral,
    mirror_to_bilateral,
    validate_left_right,
)
from .configs import list_configs, load_config
from .data_loading import (
    load_from_csv,
    load_from_dataframe,
    load_from_dict,
    load_mean_shape_csv,
)
from .plotting import (
    animate,
    animate_compare,
    animate_plotly,
    animate_plotly_compare,
    animate_plotly_partial,
    camera_from_plotly,
    create_dash_app,
    export_svg,
    interactive_plot,
    plot,
    plot_compare_plotly,
    plot_multiple,
    plot_partial_plotly,
    plot_plotly,
    plot_plotly_compare,
    plot_plotly_with_trace,
    save_plotly_animation,
)
from .scaling import (
    UNIT_FACTORS,
    unit_conversion_factor,
)
from .skeleton import SkeletonDefinition
from .transforms import TransformState

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
    "export_svg",
    "camera_from_plotly",
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
