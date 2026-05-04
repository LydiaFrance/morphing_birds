"""Generic data loading for CSV, DataFrame, and dict sources.

Extracts and unifies the CSV loading logic that was previously
duplicated across Hawk3D, Pigeon3D, Kestrel3D, and Spider3D.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .skeleton import SkeletonDefinition


def load_from_csv(
    path: str,
    skeleton: SkeletonDefinition,
    column_mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Load marker data from a CSV file.

    The CSV is expected to have columns named ``<marker>_x``, ``<marker>_y``,
    ``<marker>_z`` (or without the underscore: ``<marker>x``, etc.).

    Parameters
    ----------
    path : str
        Path to the CSV file.
    skeleton : SkeletonDefinition
        Skeleton definition (provides marker order and default column mapping).
    column_mapping : dict, optional
        Override mapping from canonical name -> CSV column name.  Merged on
        top of the skeleton's ``column_mapping``.

    Returns
    -------
    np.ndarray
        Marker data, shape ``(n_frames, n_markers, 3)``.
    """
    df = pd.read_csv(path)
    return load_from_dataframe(df, skeleton, column_mapping)


def load_from_dataframe(
    df: pd.DataFrame,
    skeleton: SkeletonDefinition,
    column_mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Load marker data from a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with marker coordinate columns.
    skeleton : SkeletonDefinition
        Skeleton definition.
    column_mapping : dict, optional
        Override mapping from canonical name -> CSV column name.

    Returns
    -------
    np.ndarray
        Marker data, shape ``(n_frames, n_markers, 3)``.
    """
    # Build effective column mapping
    mapping = dict(skeleton.column_mapping)
    if column_mapping:
        mapping.update(column_mapping)

    all_names = skeleton.all_marker_names
    n_frames = len(df)
    n_markers = len(all_names)
    data = np.full((n_frames, n_markers, 3), np.nan)

    for idx, canonical_name in enumerate(all_names):
        # Skip ignored columns
        if canonical_name in skeleton.ignored_columns:
            continue

        # Determine the CSV column base name
        csv_name = mapping.get(canonical_name, canonical_name)

        # Try both underscore and no-underscore column formats
        coords = _extract_xyz(df, csv_name)
        if coords is not None:
            data[:, idx, :] = coords

    return data


def get_matched_csv_columns(
    df: pd.DataFrame,
    skeleton: SkeletonDefinition,
    column_mapping: dict[str, str] | None = None,
) -> set[str]:
    """Return the set of CSV column names that were matched to markers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with marker coordinate columns.
    skeleton : SkeletonDefinition
        Skeleton definition.
    column_mapping : dict, optional
        Override mapping from canonical name -> CSV column name.

    Returns
    -------
    set[str]
        CSV column names that were successfully matched.
    """
    mapping = dict(skeleton.column_mapping)
    if column_mapping:
        mapping.update(column_mapping)

    matched: set[str] = set()
    for canonical_name in skeleton.all_marker_names:
        if canonical_name in skeleton.ignored_columns:
            continue
        csv_name = mapping.get(canonical_name, canonical_name)
        cols = _matched_xyz_columns(df, csv_name)
        matched.update(cols)
    return matched


def load_from_dict(
    positions: dict[str, list | np.ndarray],
    skeleton: SkeletonDefinition,
) -> np.ndarray:
    """Load marker data from a dictionary of positions.

    This replaces the old ``ArbitraryBird3D`` approach.

    Parameters
    ----------
    positions : dict
        Maps canonical marker names to ``[x, y, z]`` coordinate arrays.
    skeleton : SkeletonDefinition
        Skeleton definition.

    Returns
    -------
    np.ndarray
        Marker data, shape ``(1, n_markers, 3)``.
    """
    all_names = skeleton.all_marker_names
    n_markers = len(all_names)
    data = np.zeros((1, n_markers, 3))

    for name, pos in positions.items():
        if name not in all_names:
            msg = f"Unknown marker name: '{name}'. Valid markers: {all_names}"
            raise ValueError(msg)
        idx = all_names.index(name)
        data[0, idx] = np.asarray(pos)

    return data


def load_mean_shape_csv(
    path: str,
    skeleton: SkeletonDefinition,
    column_mapping: dict[str, str] | None = None,
) -> np.ndarray:
    """Load a single-row mean shape CSV.

    Handles the case where the CSV may store data as a single row of
    ``x, y, z, x, y, z, ...`` values (old hawk format) or as a standard
    multi-column DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    skeleton : SkeletonDefinition
        Skeleton definition.
    column_mapping : dict, optional
        Override column mapping.

    Returns
    -------
    np.ndarray
        Shape ``(1, n_markers, 3)``.
    """
    df = pd.read_csv(path)
    result = load_from_dataframe(df, skeleton, column_mapping)

    # If we got NaN for some markers, they might be in the ignored list
    # That's fine — leave them as NaN or zero
    # Replace remaining NaNs with zeros for display-only markers
    return np.nan_to_num(result, nan=0.0)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _extract_xyz(df: pd.DataFrame, base_name: str) -> np.ndarray | None:
    """Try to extract x, y, z columns for *base_name* from *df*.

    Tries ``base_name_x`` / ``base_namex`` formats.

    Returns
    -------
    np.ndarray or None
        Array of shape ``(n_rows, 3)`` if found, else ``None``.
    """
    # Try underscore format first: name_x, name_y, name_z
    x_col = f"{base_name}_x"
    y_col = f"{base_name}_y"
    z_col = f"{base_name}_z"

    if x_col in df.columns and y_col in df.columns and z_col in df.columns:
        return np.column_stack(
            [
                df[x_col].values,
                df[y_col].values,
                df[z_col].values,
            ]
        )

    # Try no-underscore format: namex, namey, namez
    x_col = f"{base_name}x"
    y_col = f"{base_name}y"
    z_col = f"{base_name}z"

    if x_col in df.columns and y_col in df.columns and z_col in df.columns:
        return np.column_stack(
            [
                df[x_col].values,
                df[y_col].values,
                df[z_col].values,
            ]
        )

    return None


def _matched_xyz_columns(df: pd.DataFrame, base_name: str) -> list[str]:
    """Return the CSV column names matched for *base_name*, or empty list.

    Mirrors the logic of :func:`_extract_xyz` but returns column names
    instead of data.
    """
    # Try underscore format first
    cols = [f"{base_name}_x", f"{base_name}_y", f"{base_name}_z"]
    if all(c in df.columns for c in cols):
        return cols

    # Try no-underscore format
    cols = [f"{base_name}x", f"{base_name}y", f"{base_name}z"]
    if all(c in df.columns for c in cols):
        return cols

    return []
