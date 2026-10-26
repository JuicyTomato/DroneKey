import numpy as np

from dronepose.eval.units import GRAVITY_FROM_CAMERA


def test_gravity_matches_forecast():
    from dronepose.forecast.frames import GRAVITY_FROM_CAMERA as G_forecast   # read-only
    assert np.array_equal(GRAVITY_FROM_CAMERA, G_forecast)
