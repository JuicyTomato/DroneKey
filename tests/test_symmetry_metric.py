import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronepose.eval.metrics import SYMMETRIC_DRONES, rotation_error_deg, symmetric_rotation_error_deg

R0 = Rotation.from_euler("xyz", [10, -20, 35], degrees=True).as_matrix()


def body(axis, deg):
    return Rotation.from_euler(axis, deg, degrees=True).as_matrix()


def test_identity_and_quarter_turns_are_equivalent():
    assert symmetric_rotation_error_deg(R0, R0) == pytest.approx(0, abs=1e-4)
    for k in (1, 2, 3):
        assert symmetric_rotation_error_deg(R0 @ body("z", 90 * k), R0) == pytest.approx(0, abs=1e-4)


def test_half_turn_flip_about_a_body_axis_is_equivalent():
    for axis in ("x", "y"):
        assert symmetric_rotation_error_deg(R0 @ body(axis, 180), R0) == pytest.approx(0, abs=1e-4)


def test_other_errors_are_not_hidden():
    assert symmetric_rotation_error_deg(R0 @ body("z", 45), R0) == pytest.approx(45, abs=1e-4)
    assert symmetric_rotation_error_deg(R0 @ body("x", 30), R0) == pytest.approx(30, abs=1e-4)
    assert symmetric_rotation_error_deg(R0 @ body("z", 90 + 10), R0) == pytest.approx(10, abs=1e-4)


def test_never_larger_than_the_plain_error():
    for deg in (0, 20, 70, 130, 200):
        R = R0 @ body("y", deg)
        assert symmetric_rotation_error_deg(R, R0) <= rotation_error_deg(R, R0) + 1e-9


def test_only_the_tello_is_marked_symmetric():
    assert SYMMETRIC_DRONES == frozenset({"08"})
