"""Generic 3D animal class.

Replaces ``Animal3D``, ``Hawk3D``, ``Pigeon3D``, ``Kestrel3D``,
``Spider3D``, and ``ArbitraryBird3D`` with a single config-driven class.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from .data_loading import (
    load_from_csv,
    load_from_dataframe,
    load_from_dict,
    load_mean_shape_csv,
)
from .scaling import compute_body_length, compute_wingspan, unit_conversion_factor
from .skeleton import SkeletonDefinition
from .transforms import TransformState


class Animal3D:
    """A 3D model of an animal, driven by a :class:`SkeletonDefinition`.

    Parameters
    ----------
    skeleton : SkeletonDefinition or str
        A ``SkeletonDefinition`` instance, or a builtin name
        (``'hawk'``, ``'pigeon'``, ``'kestrel'``, ``'spider'``).
    data : str, np.ndarray, pd.DataFrame, dict, or None
        Initial shape data.  Can be a CSV path, a ``[n_markers, 3]`` or
        ``[1, n_markers, 3]`` array, a DataFrame, or a dict of
        ``{marker_name: [x, y, z]}``.
    variant : str or None
        Named variant from the config (e.g. ``'simple'``).
    column_mapping : dict or None
        Override column mapping for CSV/DataFrame loading.
    """

    def __init__(
        self,
        skeleton: SkeletonDefinition | str,
        data: str | np.ndarray | pd.DataFrame | dict | None = None,
        variant: str | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> None:
        # Resolve skeleton
        if isinstance(skeleton, str):
            skeleton = SkeletonDefinition.from_builtin(skeleton)
        if variant is not None:
            skeleton = skeleton.get_variant(variant)

        self.skeleton: SkeletonDefinition = skeleton
        self._column_mapping_override = column_mapping

        # Runtime-modifiable exclusion set (copy from skeleton)
        self._analysis_exclude: set[str] = set(skeleton.analysis_exclude)

        # Shape arrays: [1, n_markers, 3]
        n = skeleton.n_markers
        self.default_shape: np.ndarray = np.zeros((1, n, 3))
        self.current_shape: np.ndarray = np.zeros((1, n, 3))
        self.untransformed_shape: np.ndarray = np.zeros((1, n, 3))

        # Transformation state
        self._transform = TransformState()

        # Colour sections for backward-compatible plotting
        self.colour_sections: list[str] = self._default_colour_sections()

        # Polygon indices (body section name -> list of marker indices)
        self.polygons: dict[str, list[int]] = skeleton.get_body_section_indices()

        # Load initial data if provided
        if data is not None:
            self._load_initial_data(data)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_initial_data(self, data) -> None:
        """Load initial shape data from various sources."""
        if isinstance(data, str):
            # CSV path
            shape = load_mean_shape_csv(
                data, self.skeleton, self._column_mapping_override
            )
        elif isinstance(data, dict):
            shape = load_from_dict(data, self.skeleton)
        elif isinstance(data, pd.DataFrame):
            shape = load_from_dataframe(
                data, self.skeleton, self._column_mapping_override
            )
        elif isinstance(data, np.ndarray):
            shape = self._validate_array(data)
        else:
            msg = f"Unsupported data type: {type(data)}"
            raise TypeError(msg)

        self.default_shape = shape.copy()
        self.current_shape = shape.copy()
        self.untransformed_shape = shape.copy()

    def _validate_array(self, arr: np.ndarray) -> np.ndarray:
        """Validate and reshape a raw numpy array to ``[1, n_markers, 3]``."""
        if arr.size == 0:
            msg = "No keypoints provided."
            raise ValueError(msg)
        if arr.shape[-1] != 3:
            msg = "Keypoints must be in 3D space (last dim must be 3)."
            raise ValueError(msg)
        if arr.ndim == 2:
            arr = arr.reshape(1, -1, 3)
        if arr.shape[1] != self.skeleton.n_markers:
            # Might be analysis-only data — try to handle
            msg = f"Expected {self.skeleton.n_markers} markers, got {arr.shape[1]}."
            raise ValueError(msg)
        return arr

    def load_data(
        self,
        path: str,
        column_mapping: dict[str, str] | None = None,
    ) -> None:
        """Load a single-frame shape from a CSV file.

        Parameters
        ----------
        path : str
            Path to CSV file.
        column_mapping : dict, optional
            Override column mapping.
        """
        mapping = column_mapping or self._column_mapping_override
        shape = load_mean_shape_csv(path, self.skeleton, mapping)
        self.default_shape = shape.copy()
        self.current_shape = shape.copy()
        self.untransformed_shape = shape.copy()

    def load_motion_data(
        self,
        path: str,
        column_mapping: dict[str, str] | None = None,
    ) -> np.ndarray:
        """Load multi-frame motion data from a CSV file.

        Parameters
        ----------
        path : str
            Path to CSV file.
        column_mapping : dict, optional
            Override column mapping.

        Returns
        -------
        np.ndarray
            Motion data, shape ``(n_frames, n_markers, 3)``.
        """
        mapping = column_mapping or self._column_mapping_override
        return load_from_csv(path, self.skeleton, mapping)

    def update_to_frame(self, motion_data: np.ndarray, frame_idx: int) -> None:
        """Update the current shape to a specific frame from motion data.

        Parameters
        ----------
        motion_data : np.ndarray
            Motion data, shape ``(n_frames, n_markers, 3)``.
        frame_idx : int
            Index of the frame to use.
        """
        if frame_idx >= motion_data.shape[0]:
            msg = (
                f"Frame index {frame_idx} out of range "
                f"(max {motion_data.shape[0] - 1})."
            )
            raise ValueError(msg)
        self.current_shape = motion_data[frame_idx : frame_idx + 1].copy()
        self.untransformed_shape = self.current_shape.copy()

    # ------------------------------------------------------------------
    # Marker access
    # ------------------------------------------------------------------

    @property
    def analysis_marker_names(self) -> list[str]:
        """Ordered list of non-excluded marker names (convenience alias)."""
        return [
            m
            for m in self.skeleton.all_marker_names
            if m not in self._analysis_exclude
        ]

    @property
    def analysis_indices(self) -> list[int]:
        """Indices of analysis markers in ``current_shape``."""
        exclude = self._analysis_exclude
        return [
            i
            for i, n in enumerate(self.skeleton.all_marker_names)
            if n not in exclude
        ]

    @property
    def display_only_indices(self) -> list[int]:
        """Indices of display-only (excluded) markers."""
        exclude = self._analysis_exclude
        return [
            i
            for i, n in enumerate(self.skeleton.all_marker_names)
            if n in exclude
        ]

    @property
    def markers(self) -> np.ndarray:
        """Analysis marker coordinates from ``current_shape``."""
        return self.current_shape[:, self.analysis_indices, :]

    @property
    def fixed_markers(self) -> np.ndarray:
        """Display-only marker coordinates from ``current_shape``."""
        return self.current_shape[:, self.display_only_indices, :]

    @property
    def default_markers(self) -> np.ndarray:
        """Analysis marker coordinates from ``default_shape``."""
        return self.default_shape[:, self.analysis_indices, :]

    @property
    def right_markers(self) -> np.ndarray:
        """Right-side marker coordinates from ``current_shape``."""
        right_names = self.skeleton.get_right_markers()
        # Filter to analysis markers only
        right_analysis = [n for n in right_names if n not in self._analysis_exclude]
        indices = self.skeleton.marker_indices(right_analysis)
        return self.current_shape[:, indices, :]

    @property
    def left_markers(self) -> np.ndarray:
        """Left-side marker coordinates from ``current_shape``."""
        left_names = self.skeleton.get_left_markers()
        left_analysis = [n for n in left_names if n not in self._analysis_exclude]
        indices = self.skeleton.marker_indices(left_analysis)
        return self.current_shape[:, indices, :]

    @property
    def default_right_markers(self) -> np.ndarray:
        """Right-side marker coordinates from ``default_shape``."""
        right_names = self.skeleton.get_right_markers()
        right_analysis = [n for n in right_names if n not in self._analysis_exclude]
        indices = self.skeleton.marker_indices(right_analysis)
        return self.default_shape[:, indices, :]

    @property
    def default_left_markers(self) -> np.ndarray:
        """Left-side marker coordinates from ``default_shape``."""
        left_names = self.skeleton.get_left_markers()
        left_analysis = [n for n in left_names if n not in self._analysis_exclude]
        indices = self.skeleton.marker_indices(left_analysis)
        return self.default_shape[:, indices, :]

    def marker_name_at(self, index: int) -> str:
        """Return the marker name at the given index."""
        return self.skeleton.all_marker_names[index]

    def marker_index_of(self, name: str) -> int:
        """Return the index of the given marker name."""
        return self.skeleton.marker_index(name)

    def analysis_marker_labels(self, axes: list[str] | None = None) -> list[str]:
        """Generate ``marker x axis`` labels for PCA heatmaps.

        Uses only currently non-excluded markers.
        """
        if axes is None:
            axes = ["x", "y", "z"]
        labels: list[str] = []
        for name in self.analysis_marker_names:
            display = self.skeleton.get_display_name(name)
            for ax in axes:
                labels.append(f"{display}_{ax}")
        return labels

    # ------------------------------------------------------------------
    # Runtime marker exclusion
    # ------------------------------------------------------------------

    def exclude_markers(self, names: list[str]) -> None:
        """Add markers to the analysis exclusion set.

        Parameters
        ----------
        names : list[str]
            Marker names to exclude from analysis.
        """
        for n in names:
            if n not in self.skeleton.all_marker_names:
                msg = f"Unknown marker: '{n}'"
                raise ValueError(msg)
        self._analysis_exclude.update(names)

    def include_markers(self, names: list[str]) -> None:
        """Remove markers from the analysis exclusion set.

        Parameters
        ----------
        names : list[str]
            Marker names to re-include in analysis.
        """
        self._analysis_exclude -= set(names)

    def set_variant(self, name: str) -> None:
        """Apply a named variant, updating exclusions and body sections.

        Parameters
        ----------
        name : str
            Name of the variant from the skeleton config.
        """
        variant_skeleton = self.skeleton.get_variant(name)
        self._analysis_exclude = set(variant_skeleton.analysis_exclude)
        self.polygons = variant_skeleton.get_body_section_indices()
        # Keep the same skeleton reference for marker order
        # but update body_sections
        self.skeleton.body_sections = variant_skeleton.body_sections

    # ------------------------------------------------------------------
    # Transforms (two modes)
    # ------------------------------------------------------------------

    def transform_display_only(
        self,
        bodypitch: float = 0,
        bodyyaw: float = 0,
        bodyroll: float = 0,
        horzDist: float = 0,
        vertDist: float = 0,
    ) -> None:
        """Transform only display-only (excluded) markers.

        Use when moving markers already contain rotations from mocap data,
        and display-only markers need to be rotated to match.

        Parameters
        ----------
        bodypitch : float
            Rotation around x-axis (degrees).
        bodyyaw : float
            Rotation around z-axis (degrees).
        bodyroll : float
            Rotation around y-axis (degrees).
        horzDist : float
            Horizontal translation (y-axis).
        vertDist : float
            Vertical translation (z-axis).
        """
        self._apply_transform(
            indices=self.display_only_indices,
            bodypitch=bodypitch,
            bodyyaw=bodyyaw,
            bodyroll=bodyroll,
            horzDist=horzDist,
            vertDist=vertDist,
        )

    def transform_all(
        self,
        bodypitch: float = 0,
        bodyyaw: float = 0,
        bodyroll: float = 0,
        horzDist: float = 0,
        vertDist: float = 0,
    ) -> None:
        """Transform all markers (analysis + display-only).

        Use when no rotation is baked into the data and you want to rotate
        the entire animal for visualisation.

        Parameters
        ----------
        bodypitch : float
            Rotation around x-axis (degrees).
        bodyyaw : float
            Rotation around z-axis (degrees).
        bodyroll : float
            Rotation around y-axis (degrees).
        horzDist : float
            Horizontal translation (y-axis).
        vertDist : float
            Vertical translation (z-axis).
        """
        all_indices = list(range(self.skeleton.n_markers))
        self._apply_transform(
            indices=all_indices,
            bodypitch=bodypitch,
            bodyyaw=bodyyaw,
            bodyroll=bodyroll,
            horzDist=horzDist,
            vertDist=vertDist,
        )


    def _apply_transform(
        self,
        indices: list[int],
        bodypitch: float = 0,
        bodyyaw: float = 0,
        bodyroll: float = 0,
        horzDist: float = 0,
        vertDist: float = 0,
    ) -> None:
        """Apply rotation + translation to specific marker indices."""
        # Reset to untransformed state first
        self.current_shape = self.untransformed_shape.copy()
        self._transform.reset()

        if horzDist or vertDist:
            self._transform.add_translation(y=horzDist, z=vertDist)

        if bodypitch:
            self._transform.add_rotation(bodypitch, axis="x")
        if bodyyaw:
            self._transform.add_rotation(bodyyaw, axis="z")
        if bodyroll:
            self._transform.add_rotation(bodyroll, axis="y")

        if not indices:
            return

        coords = self.current_shape[0, indices, :]
        transformed = self._transform.apply_to(coords)
        self.current_shape[0, indices, :] = transformed

    @property
    def origin(self) -> np.ndarray:
        """Current transform origin."""
        return self._transform.origin.copy()

    @property
    def transformation_matrix(self) -> np.ndarray:
        """Current 4x4 transformation matrix."""
        return self._transform.matrix.copy()

    def reset_transformation(self) -> None:
        """Reset transforms, restoring current_shape to untransformed_shape."""
        self._transform.reset()
        self.current_shape = self.untransformed_shape.copy()

    def restore_default(self) -> None:
        """Restore all shapes to the original loaded default."""
        self.current_shape = self.default_shape.copy()
        self.untransformed_shape = self.current_shape.copy()
        self._transform.reset()


    # ------------------------------------------------------------------
    # Keypoint update (backward compat)
    # ------------------------------------------------------------------

    def update_keypoints(self, user_keypoints: np.ndarray | None) -> None:
        """Update analysis markers with user-provided data.

        Parameters
        ----------
        user_keypoints : np.ndarray or None
            New marker positions.  If ``None``, resets to default.
        """
        if user_keypoints is None:
            self.restore_default()
            return

        validated = self._validate_user_keypoints(user_keypoints)
        self.current_shape[:, self.analysis_indices, :] = validated
        self.untransformed_shape = self.current_shape.copy()

    def _validate_user_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Validate and possibly mirror user-provided keypoints."""
        if keypoints.size == 0:
            msg = "No keypoints provided."
            raise ValueError(msg)
        if keypoints.shape[-1] != 3:
            msg = "Keypoints must be in 3D space."
            raise ValueError(msg)
        if keypoints.ndim == 2:
            keypoints = keypoints.reshape(1, -1, 3)

        n_analysis = len(self.analysis_indices)
        n_right = len([
            n for n in self.skeleton.get_right_markers()
            if n not in self._analysis_exclude
        ])

        # If half the expected markers, mirror to create both sides
        if keypoints.shape[1] == n_right:
            keypoints = self.mirror_keypoints(keypoints)

        if keypoints.shape[1] != n_analysis:
            msg = f"Expected {n_analysis} analysis markers, got {keypoints.shape[1]}."
            raise ValueError(msg)

        return keypoints

    def mirror_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Mirror right-side keypoints to create left-side markers.

        Assumes markers are ordered ``left_1, right_1, left_2, right_2, ...``

        Parameters
        ----------
        keypoints : np.ndarray
            Right-side only keypoints, shape ``(n_frames, n_right, 3)``.

        Returns
        -------
        np.ndarray
            Full bilateral keypoints, shape ``(n_frames, n_right * 2, 3)``.
        """
        mirrored = np.copy(keypoints)
        mirrored[:, :, 0] *= -1

        n_frames, n_markers, n_coords = keypoints.shape
        full = np.empty((n_frames, n_markers * 2, n_coords), dtype=keypoints.dtype)
        full[:, 0::2, :] = mirrored  # left at even indices
        full[:, 1::2, :] = keypoints  # right at odd indices
        return full

    # ------------------------------------------------------------------
    # Overwrite / copy
    # ------------------------------------------------------------------

    def overwrite_keypoints(self, keypoints: np.ndarray) -> None:
        """Overwrite default, current, and untransformed shapes.

        Analysis markers come from *keypoints*; display-only markers are
        preserved from the existing default.

        Parameters
        ----------
        keypoints : np.ndarray
            Shape ``(1, n_analysis_markers, 3)``.
        """
        if keypoints.shape[0] != 1:
            msg = "Keypoints must have shape (1, n_markers, 3)."
            raise ValueError(msg)
        if keypoints.shape[2] != 3:
            msg = "Keypoints must be 3D coordinates."
            raise ValueError(msg)

        new_shape = np.zeros_like(self.default_shape)
        new_shape[:, self.analysis_indices, :] = keypoints
        new_shape[:, self.display_only_indices, :] = self.default_shape[
            :, self.display_only_indices, :
        ]

        self.default_shape = new_shape.copy()
        self.current_shape = new_shape.copy()
        self.untransformed_shape = new_shape.copy()

    def copy(self) -> Animal3D:
        """Return a deep copy."""
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_polygon_coords(self, section_name: str) -> np.ndarray:
        """Retrieve coordinates for a body section polygon.

        Parameters
        ----------
        section_name : str
            Name of the body section.

        Returns
        -------
        np.ndarray
            Coordinates of shape ``(n_vertices, 3)``.
        """
        if section_name not in self.polygons:
            msg = f"Section name '{section_name}' not recognised."
            raise ValueError(msg)
        indices = self.polygons[section_name]
        return self.current_shape[0, indices, :]

    def get_bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the bounding box of the current shape.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(min_coords, max_coords)``, each of shape ``(3,)``.
        """
        min_coords = self.current_shape.min(axis=(0, 1))
        max_coords = self.current_shape.max(axis=(0, 1))
        return min_coords, max_coords

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def set_scale(
        self,
        unit_from: str | None = None,
        unit_to: str | None = None,
        normalise_by: str | None = None,
        factor: float | None = None,
    ) -> None:
        """Scale marker positions.

        Parameters
        ----------
        unit_from, unit_to : str, optional
            Unit conversion (e.g. ``'mm'`` -> ``'m'``).
        normalise_by : str, optional
            Biological normalisation — ``'wingspan'`` or ``'body_length'``.
        factor : float, optional
            Direct scaling factor.
        """
        if unit_from and unit_to:
            f = unit_conversion_factor(unit_from, unit_to)
        elif normalise_by == "wingspan":
            ws = compute_wingspan(self.current_shape, self.skeleton)
            f = 1.0 / ws if ws > 0 else 1.0
        elif normalise_by == "body_length":
            bl = compute_body_length(self.current_shape, self.skeleton)
            f = 1.0 / bl if bl > 0 else 1.0
        elif factor is not None:
            f = factor
        else:
            return

        self.default_shape *= f
        self.current_shape *= f
        self.untransformed_shape *= f

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Run validation rules from the skeleton config.

        Raises
        ------
        ValueError
            If any validation rule is violated.
        """
        if not self.skeleton.validation_rules:
            return

        names = self.skeleton.all_marker_names
        shape = self.current_shape[0]  # [n_markers, 3]

        for rule in self.skeleton.validation_rules:
            self._check_rule(rule, names, shape)

    @staticmethod
    def _check_rule(
        rule: str, names: list[str], shape: np.ndarray
    ) -> None:
        """Parse and check a single validation rule string."""
        # Rules are like "left_wingtip.x < right_wingtip.x"
        axis_map = {"x": 0, "y": 1, "z": 2}

        if "<" in rule:
            parts = rule.split("<")
            op = "<"
        elif ">" in rule:
            parts = rule.split(">")
            op = ">"
        else:
            return  # Skip unknown rule formats

        left_ref = parts[0].strip()
        right_ref = parts[1].strip()

        def resolve(ref: str) -> float:
            marker_name, axis = ref.rsplit(".", 1)
            idx = names.index(marker_name)
            return float(shape[idx, axis_map[axis]])

        left_val = resolve(left_ref)
        right_val = resolve(right_ref)

        if op == "<" and not (left_val < right_val):
            msg = f"Validation failed: {rule} ({left_val} >= {right_val})"
            raise ValueError(msg)
        if op == ">" and not (left_val > right_val):
            msg = f"Validation failed: {rule} ({left_val} <= {right_val})"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------

    @staticmethod
    def remove_nan_frames(motion_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove frames containing NaN values.

        Parameters
        ----------
        motion_data : np.ndarray
            Shape ``(n_frames, n_markers, 3)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(clean_data, valid_frames_mask)``
        """
        valid = np.asarray(~np.isnan(motion_data).any(axis=(1, 2)))
        return motion_data[valid], valid

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_colour_sections(self) -> list[str]:
        """Derive default colour_sections from section_styles config."""
        styles = self.skeleton.section_styles
        if not styles:
            return ["handwing", "tail"]

        # Sections that inherit the caller's colour (no explicit colour override)
        coloured = []
        for key, style in styles.items():
            if key == "default":
                continue
            if "colour" not in style:
                coloured.append(key)
        return coloured if coloured else ["handwing", "tail"]

