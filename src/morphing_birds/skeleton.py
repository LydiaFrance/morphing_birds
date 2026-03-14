"""Config-driven skeleton definition.

Replaces the old SkeletonDefinition + per-species skeleton definition
subclasses with a single class that reads its configuration from a
dictionary (typically loaded from YAML).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .configs import load_config


class SkeletonDefinition:
    """Pure data container describing an animal's marker layout.

    Constructed from a configuration dictionary (or loaded from YAML).
    No subclassing required — different animals are different configs.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys: ``name``, ``markers``,
        ``analysis_exclude``, ``body_sections``, etc.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: dict) -> None:
        cfg = config  # shorthand

        self.name: str = cfg.get("name", "unnamed")
        self._all_marker_names: list[str] = list(cfg["markers"])
        self._analysis_exclude: set[str] = set(cfg.get("analysis_exclude", []))
        self.body_sections: dict[str, list[str]] = dict(cfg.get("body_sections", {}))
        self.column_mapping: dict[str, str] = dict(cfg.get("column_mapping", {}))
        self.ignored_columns: list[str] = list(cfg.get("ignored_columns", []))
        self.display_names: dict[str, str] = dict(cfg.get("display_names", {}))
        self.section_styles: dict[str, dict] = dict(cfg.get("section_styles", {}))
        self.variants: dict[str, dict] = dict(cfg.get("variants", {}))
        self.validation_rules: list[str] = list(cfg.get("validation_rules", []))
        self.laterality: str | dict = cfg.get("laterality", "prefix")

        # Store the full config for variant application
        self._config = cfg

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> SkeletonDefinition:
        """Load a skeleton definition from a YAML file.

        Parameters
        ----------
        path : str
            Path to the YAML file.
        """
        with Path(path).open() as f:
            cfg = yaml.safe_load(f)
        return cls(cfg)

    @classmethod
    def from_builtin(cls, name: str) -> SkeletonDefinition:
        """Load a shipped builtin config by name.

        Parameters
        ----------
        name : str
            One of ``'hawk'``, ``'pigeon'``, ``'kestrel'``, ``'spider'``.
        """
        cfg = load_config(name)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Marker lists (order matters — determines array column order)
    # ------------------------------------------------------------------

    @property
    def all_marker_names(self) -> list[str]:
        """Every marker, in config order."""
        return list(self._all_marker_names)

    @property
    def analysis_exclude(self) -> set[str]:
        """Markers excluded from analysis (display-only)."""
        return set(self._analysis_exclude)

    @property
    def analysis_markers(self) -> list[str]:
        """Analysis markers: all minus *analysis_exclude*, preserving order."""
        return [m for m in self._all_marker_names if m not in self._analysis_exclude]

    @property
    def display_only_markers(self) -> list[str]:
        """The *analysis_exclude* set as an ordered list."""
        return [m for m in self._all_marker_names if m in self._analysis_exclude]

    @property
    def n_markers(self) -> int:
        """Total number of markers."""
        return len(self._all_marker_names)

    @property
    def n_analysis_markers(self) -> int:
        """Number of analysis (non-excluded) markers."""
        return len(self.analysis_markers)

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def marker_index(self, name: str) -> int:
        """Return the index of *name* in ``all_marker_names``."""
        return self._all_marker_names.index(name)

    def marker_indices(self, names: list[str]) -> list[int]:
        """Return indices for a list of marker names."""
        return [self._all_marker_names.index(n) for n in names]

    @property
    def analysis_indices(self) -> list[int]:
        """Indices of analysis markers within ``all_marker_names``."""
        exclude = self._analysis_exclude
        return [i for i, n in enumerate(self._all_marker_names) if n not in exclude]

    @property
    def display_only_indices(self) -> list[int]:
        """Indices of display-only markers within ``all_marker_names``."""
        exclude = self._analysis_exclude
        return [i for i, n in enumerate(self._all_marker_names) if n in exclude]

    # ------------------------------------------------------------------
    # Display name helpers
    # ------------------------------------------------------------------

    def get_display_name(self, canonical_name: str) -> str:
        """Return the display name for *canonical_name*, falling back to the
        canonical name itself if no display name is configured."""
        return self.display_names.get(canonical_name, canonical_name)

    def analysis_marker_labels(self, axes: list[str] | None = None) -> list[str]:
        """Generate ``marker x axis`` labels for PCA heatmaps.

        Parameters
        ----------
        axes : list[str], optional
            Axis suffixes.  Defaults to ``["x", "y", "z"]``.

        Returns
        -------
        list[str]
            E.g. ``["Wingtip_x", "Wingtip_y", "Wingtip_z", ...]``
        """
        if axes is None:
            axes = ["x", "y", "z"]
        labels: list[str] = []
        for name in self.analysis_markers:
            display = self.get_display_name(name)
            for ax in axes:
                labels.append(f"{display}_{ax}")
        return labels

    # ------------------------------------------------------------------
    # Left / right / centre detection
    # ------------------------------------------------------------------

    def get_left_markers(self) -> list[str]:
        """Return left-side marker names based on *laterality* config."""
        if self.laterality == "prefix":
            return [n for n in self._all_marker_names if n.startswith("left_")]
        if isinstance(self.laterality, dict) and "left" in self.laterality:
            suffixes = tuple(self.laterality["left"])
            return [n for n in self._all_marker_names if n.endswith(suffixes)]
        return []

    def get_right_markers(self) -> list[str]:
        """Return right-side marker names based on *laterality* config."""
        if self.laterality == "prefix":
            return [n for n in self._all_marker_names if n.startswith("right_")]
        if isinstance(self.laterality, dict) and "right" in self.laterality:
            suffixes = tuple(self.laterality["right"])
            return [n for n in self._all_marker_names if n.endswith(suffixes)]
        return []

    def get_centre_markers(self) -> list[str]:
        """Return markers with no left/right counterpart."""
        left = set(self.get_left_markers())
        right = set(self.get_right_markers())
        sided = left | right
        return [n for n in self._all_marker_names if n not in sided]

    def get_marker_pairs(self) -> list[tuple[str, str]]:
        """Return paired ``(left, right)`` marker tuples.

        Pairing is determined by the *laterality* config.  For ``prefix``
        mode, ``left_X`` is paired with ``right_X``.  For dict-based
        laterality (e.g. spiders), pairing is by base name with matching
        suffix numbers.
        """
        if self.laterality == "prefix":
            pairs: list[tuple[str, str]] = []
            rights = set(self.get_right_markers())
            for left in self.get_left_markers():
                right_candidate = "right_" + left[5:]
                if right_candidate in rights:
                    pairs.append((left, right_candidate))
            return pairs
        # Dict-based laterality (spider-style)
        if isinstance(self.laterality, dict):
            right_suffixes = self.laterality.get("right", [])
            left_suffixes = self.laterality.get("left", [])
            pairs = []
            for rs, ls in zip(right_suffixes, left_suffixes, strict=True):
                for name in self._all_marker_names:
                    if name.endswith(rs):
                        base = name[: -len(rs)]
                        partner = base + ls
                        if partner in self._all_marker_names:
                            pairs.append((partner, name))
            return pairs
        return []

    # ------------------------------------------------------------------
    # Body section helpers
    # ------------------------------------------------------------------

    def get_body_section_indices(self) -> dict[str, list[int]]:
        """Map each body section to marker indices in ``all_marker_names``."""
        result: dict[str, list[int]] = {}
        for section, markers in self.body_sections.items():
            result[section] = [self._all_marker_names.index(m) for m in markers]
        return result

    # ------------------------------------------------------------------
    # Variant support
    # ------------------------------------------------------------------

    def get_variant(self, name: str) -> SkeletonDefinition:
        """Return a new ``SkeletonDefinition`` with the named variant applied.

        Variants can override ``analysis_exclude`` and ``body_sections``.

        Parameters
        ----------
        name : str
            Name of the variant (must exist in the config's ``variants``).
        """
        if name not in self.variants:
            msg = f"Unknown variant '{name}' for skeleton '{self.name}'. Available: {list(self.variants.keys())}"
            raise ValueError(msg)
        variant_cfg = self.variants[name]
        new_config = copy.deepcopy(self._config)

        # Apply variant overrides
        if "analysis_exclude" in variant_cfg:
            new_config["analysis_exclude"] = variant_cfg["analysis_exclude"]
        if "body_sections" in variant_cfg:
            new_config["body_sections"] = variant_cfg["body_sections"]

        return SkeletonDefinition(new_config)

    # ------------------------------------------------------------------
    # Column mapping helpers
    # ------------------------------------------------------------------

    @property
    def reverse_column_mapping(self) -> dict[str, str]:
        """CSV column name -> canonical name mapping."""
        return {v: k for k, v in self.column_mapping.items()}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SkeletonDefinition(name={self.name!r}, "
            f"markers={self.n_markers}, "
            f"analysis={self.n_analysis_markers}, "
            f"sections={list(self.body_sections.keys())})"
        )
