"""Tests for the TransformState class."""

import numpy as np
import pytest
from morphing_birds.transforms import TransformState


class TestTransformState:
    """Test transformation operations."""

    def test_identity(self):
        t = TransformState()
        np.testing.assert_array_almost_equal(t.matrix, np.eye(4))

    def test_rotation_x(self):
        t = TransformState()
        coords = np.array([[1.0, 0.0, 0.0]])
        t.add_rotation(90, axis="x")
        result = t.apply_to(coords)
        # 90 degree rotation around x: (1,0,0) stays at (1,0,0)
        np.testing.assert_array_almost_equal(result, [[1.0, 0.0, 0.0]])

    def test_rotation_z(self):
        t = TransformState()
        coords = np.array([[1.0, 0.0, 0.0]])
        t.add_rotation(90, axis="z")
        result = t.apply_to(coords)
        # 90 degree rotation around z: (1,0,0) -> (0,1,0)
        np.testing.assert_array_almost_equal(result, [[0.0, 1.0, 0.0]], decimal=5)

    def test_translation(self):
        t = TransformState()
        coords = np.array([[0.0, 0.0, 0.0]])
        t.add_translation(x=1.0, y=2.0, z=3.0)
        result = t.apply_to(coords)
        np.testing.assert_array_almost_equal(result, [[1.0, 2.0, 3.0]])

    def test_combined_transform(self):
        t = TransformState()
        coords = np.array([[1.0, 0.0, 0.0]])
        t.add_translation(x=1.0)
        t.add_rotation(90, axis="z")
        result = t.apply_to(coords)
        # Matrix is accumulated right-to-left: T @ R applied as (coord @ (T @ R).T)
        # Translation shifts x by 1: (1,0,0) stays, then rotation:
        # Result: x=1, y=1, z=0 (rotate original, then translate)
        np.testing.assert_array_almost_equal(result, [[1.0, 1.0, 0.0]], decimal=5)

    def test_reset(self):
        t = TransformState()
        t.add_rotation(45, axis="x")
        t.add_translation(x=5)
        t.reset()
        np.testing.assert_array_almost_equal(t.matrix, np.eye(4))
        np.testing.assert_array_almost_equal(t.origin, [0, 0, 0])

    def test_apply_to_multiple_points(self):
        t = TransformState()
        t.add_translation(y=1.0)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        result = t.apply_to(coords)
        np.testing.assert_array_almost_equal(
            result,
            [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0]],
        )

    def test_invalid_axis_raises(self):
        t = TransformState()
        with pytest.raises(ValueError, match="Unknown axis"):
            t.add_rotation(90, axis="w")
