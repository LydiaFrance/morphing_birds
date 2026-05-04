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
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        result = t.apply_to(coords)
        np.testing.assert_array_almost_equal(
            result,
            [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0]],
        )

    def test_invalid_axis_raises(self):
        t = TransformState()
        with pytest.raises(ValueError, match="Unknown axis"):
            t.add_rotation(90, axis="w")

    # ------------------------------------------------------------------
    # Rotation composition and commutativity
    # ------------------------------------------------------------------

    def test_rotation_composition_is_non_commutative(self):
        """Rotations around different axes should not commute."""
        coords = np.array([[1.0, 2.0, 3.0]])

        t1 = TransformState()
        t1.add_rotation(90, axis="x")
        t1.add_rotation(90, axis="z")
        result_xz = t1.apply_to(coords)

        t2 = TransformState()
        t2.add_rotation(90, axis="z")
        t2.add_rotation(90, axis="x")
        result_zx = t2.apply_to(coords)

        # The two orderings must produce different results
        assert not np.allclose(
            result_xz, result_zx
        ), "Rotation composition should be non-commutative"

    # ------------------------------------------------------------------
    # Full-circle and zero-degree rotations
    # ------------------------------------------------------------------

    def test_rotation_360_degrees_returns_to_original(self):
        """A 360-degree rotation should act as the identity."""
        coords = np.array([[1.0, 2.0, 3.0]])
        for axis in ("x", "y", "z"):
            t = TransformState()
            t.add_rotation(360, axis=axis)
            result = t.apply_to(coords)
            np.testing.assert_array_almost_equal(
                result,
                coords,
                decimal=10,
                err_msg=f"360-degree rotation around {axis} should be identity",
            )

    def test_rotation_0_degrees_is_identity(self):
        """A 0-degree rotation should leave coordinates unchanged."""
        coords = np.array([[4.0, 5.0, 6.0]])
        for axis in ("x", "y", "z"):
            t = TransformState()
            t.add_rotation(0, axis=axis)
            result = t.apply_to(coords)
            np.testing.assert_array_almost_equal(
                result,
                coords,
                decimal=10,
                err_msg=f"0-degree rotation around {axis} should be identity",
            )

    # ------------------------------------------------------------------
    # Negative rotations reverse positive rotations
    # ------------------------------------------------------------------

    def test_negative_rotation_reverses_positive(self):
        """Applying +θ then -θ should return to the original coordinates."""
        coords = np.array([[1.0, 2.0, 3.0]])
        for axis in ("x", "y", "z"):
            t = TransformState()
            t.add_rotation(45, axis=axis)
            t.add_rotation(-45, axis=axis)
            result = t.apply_to(coords)
            np.testing.assert_array_almost_equal(
                result,
                coords,
                decimal=10,
                err_msg=f"Positive then negative rotation around {axis} "
                f"should cancel out",
            )

    # ------------------------------------------------------------------
    # Large translations — numerical stability
    # ------------------------------------------------------------------

    def test_large_translation_no_numerical_issues(self):
        """Very large translations should not introduce numerical artefacts."""
        coords = np.array([[0.0, 0.0, 0.0]])
        big = 1e12
        t = TransformState()
        t.add_translation(x=big, y=-big, z=big)
        result = t.apply_to(coords)
        np.testing.assert_array_almost_equal(
            result,
            [[big, -big, big]],
            decimal=0,
        )

    # ------------------------------------------------------------------
    # Single point vs multi-point arrays
    # ------------------------------------------------------------------

    def test_apply_to_single_point(self):
        """Transform should work on a single-row (1, 3) array."""
        t = TransformState()
        t.add_translation(x=1.0, y=2.0, z=3.0)
        single = np.array([[7.0, 8.0, 9.0]])
        result = t.apply_to(single)
        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result, [[8.0, 10.0, 12.0]])

    def test_apply_to_many_points(self):
        """Transform should work on a large (n, 3) array."""
        t = TransformState()
        t.add_rotation(90, axis="z")
        n = 100
        coords = np.random.default_rng(42).standard_normal((n, 3))
        result = t.apply_to(coords)
        assert result.shape == (n, 3)

    # ------------------------------------------------------------------
    # Reset after complex transforms
    # ------------------------------------------------------------------

    def test_reset_after_complex_transforms(self):
        """Reset should restore a clean identity state after many operations."""
        t = TransformState()
        t.add_rotation(30, axis="x")
        t.add_rotation(60, axis="y")
        t.add_translation(x=10, y=-5, z=3)
        t.add_rotation(90, axis="z")
        t.add_translation(x=-2, y=7)
        t.reset()

        np.testing.assert_array_almost_equal(t.matrix, np.eye(4))
        np.testing.assert_array_almost_equal(t.origin, [0, 0, 0])

        # Applying the reset transform should leave points unchanged
        coords = np.array([[1.0, 2.0, 3.0]])
        result = t.apply_to(coords)
        np.testing.assert_array_almost_equal(result, coords)

    # ------------------------------------------------------------------
    # Transform preserves array shape
    # ------------------------------------------------------------------

    def test_transform_preserves_shape(self):
        """Output array must have the same shape as the input."""
        t = TransformState()
        t.add_rotation(45, axis="y")
        t.add_translation(x=1, y=2, z=3)
        for n in (1, 5, 50):
            coords = np.zeros((n, 3))
            result = t.apply_to(coords)
            assert (
                result.shape == coords.shape
            ), f"Shape mismatch for n={n}: {result.shape} != {coords.shape}"

    # ------------------------------------------------------------------
    # Per-axis rotation — expected coordinate changes
    # ------------------------------------------------------------------

    def test_rotation_around_x_preserves_x(self):
        """Rotation around X should leave the X coordinate unchanged."""
        coords = np.array([[1.0, 1.0, 1.0]])
        t = TransformState()
        t.add_rotation(45, axis="x")
        result = t.apply_to(coords)
        np.testing.assert_almost_equal(result[0, 0], 1.0)
        # Y and Z should change
        assert not np.isclose(result[0, 1], 1.0) or not np.isclose(result[0, 2], 1.0)

    def test_rotation_around_y_preserves_y(self):
        """Rotation around Y should leave the Y coordinate unchanged."""
        coords = np.array([[1.0, 1.0, 1.0]])
        t = TransformState()
        t.add_rotation(45, axis="y")
        result = t.apply_to(coords)
        np.testing.assert_almost_equal(result[0, 1], 1.0)
        # X and Z should change
        assert not np.isclose(result[0, 0], 1.0) or not np.isclose(result[0, 2], 1.0)

    def test_rotation_around_z_preserves_z(self):
        """Rotation around Z should leave the Z coordinate unchanged."""
        coords = np.array([[1.0, 1.0, 1.0]])
        t = TransformState()
        t.add_rotation(45, axis="z")
        result = t.apply_to(coords)
        np.testing.assert_almost_equal(result[0, 2], 1.0)
        # X and Y should change
        assert not np.isclose(result[0, 0], 1.0) or not np.isclose(result[0, 1], 1.0)

    def test_rotation_y_90_expected_values(self):
        """90-degree rotation around Y: (1,0,0) -> (0,0,-1)."""
        coords = np.array([[1.0, 0.0, 0.0]])
        t = TransformState()
        t.add_rotation(90, axis="y")
        result = t.apply_to(coords)
        np.testing.assert_array_almost_equal(result, [[0.0, 0.0, -1.0]], decimal=5)
