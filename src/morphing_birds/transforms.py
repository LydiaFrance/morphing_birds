"""Two-mode transformation system for Animal3D.

Provides a ``TransformState`` class that encapsulates a 4x4 homogeneous
transformation matrix plus an origin, and helper functions for applying
transforms to subsets of markers.
"""

from __future__ import annotations

import numpy as np


class TransformState:
    """Encapsulates a 4x4 homogeneous transformation matrix + origin.

    All rotations are specified in degrees and applied around the origin.
    """

    def __init__(self) -> None:
        self.matrix: np.ndarray = np.eye(4)
        self.origin: np.ndarray = np.zeros(3)

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def add_rotation(self, degrees: float, axis: str = "x") -> None:
        """Accumulate a rotation around the given axis.

        Parameters
        ----------
        degrees : float
            Rotation angle in degrees.
        axis : str
            ``'x'``, ``'y'``, or ``'z'``.
        """
        degrees = float(np.asarray(degrees).flat[0])
        radians = np.deg2rad(degrees)
        c, s = np.cos(radians), np.sin(radians)

        if axis == "x":
            rot = np.array([
                [1, 0,  0, 0],
                [0, c, -s, 0],
                [0, s,  c, 0],
                [0, 0,  0, 1],
            ])
        elif axis == "y":
            rot = np.array([
                [ c, 0, s, 0],
                [ 0, 1, 0, 0],
                [-s, 0, c, 0],
                [ 0, 0, 0, 1],
            ])
        elif axis == "z":
            rot = np.array([
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0,  0, 0, 1],
            ])
        else:
            msg = f"Unknown axis '{axis}'. Must be 'x', 'y', or 'z'."
            raise ValueError(msg)

        self.matrix = self.matrix @ rot

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def add_translation(self, x: float = 0, y: float = 0, z: float = 0) -> None:
        """Accumulate a translation.

        Parameters
        ----------
        x, y, z : float
            Translation along each axis.
        """
        trans = np.eye(4)
        trans[0, 3] = float(x)
        trans[1, 3] = float(y)
        trans[2, 3] = float(z)
        self.matrix = self.matrix @ trans
        self.origin = self.origin + np.array([float(x), float(y), float(z)])

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply_to(self, coords: np.ndarray) -> np.ndarray:
        """Apply the accumulated transform to an ``[n, 3]`` coordinate array.

        Parameters
        ----------
        coords : np.ndarray
            Coordinates of shape ``(n, 3)``.

        Returns
        -------
        np.ndarray
            Transformed coordinates, same shape as input.
        """
        n = coords.shape[0]
        homogeneous = np.hstack((coords, np.ones((n, 1))))
        transformed = homogeneous @ self.matrix.T
        return transformed[:, :3]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset to the identity transform."""
        self.matrix = np.eye(4)
        self.origin = np.zeros(3)
