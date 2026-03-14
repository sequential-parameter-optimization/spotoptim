# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Robot Arm function."""

import numpy as np
import pytest
from spotoptim.function.so import robot_arm_obstacle


class TestRobotArm:
    """Test suite for the Robot Arm function."""

    def test_output_shape_single(self):
        """Test output shape for single 1D input."""
        n_dims = 10
        X = np.random.rand(n_dims)
        result = robot_arm_obstacle(X)
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_output_shape_batch(self):
        """Test output shape for batch input."""
        n_dims = 10
        n_samples = 5
        X = np.random.rand(n_samples, n_dims)
        result = robot_arm_obstacle(X)
        assert result.shape == (n_samples,)
        assert np.all(np.isfinite(result))

    def test_input_dimension_validation(self):
        """Test that invalid dimensions raise ValueError."""
        X = np.random.rand(1, 11)  # Incorrect size
        with pytest.raises(ValueError, match="requires exactly 10 dimensions"):
            robot_arm_obstacle(X)

    def test_scaling_range(self):
        """Test that internal scaling maps [0, 1] to [-pi, pi].

        We can infer this by checking a path that should simply go straight.
        If all angles are 0 (in scaled space), the arm is a straight line along X.

        Input 0 -> -pi
        Input 0.5 -> 0
        Input 1 -> pi
        """
        # Case: Straight line along X axis.
        # This requires relative angles to be 0.
        # So inputs should be 0.5 (scaled to 0).
        X = np.ones((1, 10)) * 0.5

        # End effector position should be (10, 0)
        # Target is (5, 5).
        # Distance^2 = (10-5)^2 + (0-5)^2 = 25 + 25 = 50.
        # Obstacles:
        # (2,2), r=1 -> dist to (x,0) is sqrt((x-2)^2 + 2^2) >= 2 > 1. No collision.
        # (4,3), r=1.5 -> dist to (x,0) is sqrt((x-4)^2 + 3^2) >= 3 > 1.5. No collision.
        # (3,6), r=1 -> dist to (x,0) is sqrt((x-3)^2 + 6^2) >= 6 > 1. No collision.

        # So penalty should be 0. Cost should be exactly 50?
        # Wait, there is a buffer 0.1 in code.
        # Check closest approach to obstacle 1 at x=2: dist = 2. r+0.1 = 1.1. No violation.

        cost = robot_arm_obstacle(X)[0]
        assert np.isclose(
            cost, 50.0, atol=1e-4
        ), f"Expected cost 50.0 for straight arm, got {cost}"

    def test_obstacle_collision(self):
        """Test that hitting an obstacle incurs high penalty."""
        # Obstacle at (2, 2) with radius 1.
        # Let's put a joint exactly there.
        # Joint 1 is at L1 * (cos(theta1), sin(theta1)).
        # We need length 1 vector to point to (2,2)? magnitude is sqrt(8) > 1. Can't reach with 1 link.
        # Joint 2 is at L1(...) + L2(...).

        # Easier strategy: force arm to go through the obstacle.
        # Angle to (2,2) is 45 deg = pi/4.
        # Set all angles to pi/4 (relative).
        # Wait, relative means they sum up.
        # If theta1 = pi/4, theta2 = 0 => abs_theta2 = pi/4.
        # So set all relative angles to 0 except first one?
        # If theta1 = pi/4 (0.625 input?), then arm is straight line at 45 deg.
        # Points: (0.7, 0.7), (1.4, 1.4), (2.1, 2.1), ...
        # (2.1, 2.1) is inside (2,2) with radius 1.

        # 0.5 input = 0 rad.
        # Want pi/4. Mapping: val_scaled = val * 2pi - pi.
        # pi/4 = x * 2pi - pi => 1.25 pi = x * 2pi => x = 1.25/2 = 0.625.

        X = np.ones((1, 10)) * 0.5
        X[0, 0] = 0.625  # First link 45 deg, rest 0 relative (straight line)

        cost = robot_arm_obstacle(X)[0]

        # Without penalty, distance to (5,5) from end effector (~7.07, ~7.07)
        # (7.07-5)^2 + (7.07-5)^2 approx 4+4=8.
        # With penalty: we have joints at ~ (2.1, 2.1) inside obstacle (2,2).
        # Should be huge.

        assert cost > 500, f"Collision should yield high cost, got {cost}"

    def test_deterministic(self):
        """Test that function is deterministic."""
        X = np.random.rand(5, 10)
        val1 = robot_arm_obstacle(X)
        val2 = robot_arm_obstacle(X)
        np.testing.assert_array_equal(val1, val2)
