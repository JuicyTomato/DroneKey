"""Physics baseline: a quadrotor accelerates horizontally along its tilt, a_h = g * tan(tilt), with hover thrust.

Position and velocity come from the CV-k fit; the acceleration from the last observed up axis is held constant over
the horizon, vertical acceleration 0. On 6DroneSyn attitude and motion are unrelated (keyframed, PROJECT_MEMORY
§3.11), so this baseline is expected not to help there; it matters on physically generated or real trajectories.
"""
import numpy as np

from dronepose.forecast.baselines import fit_state


def tilt_acceleration(u: np.ndarray, g: float, max_tilt_deg: float) -> np.ndarray:
    """Horizontal acceleration (W,3) implied by the body up axis u (W,3) in the gravity frame; tilt saturated."""
    u = np.asarray(u, float)
    uh = u[:, :2]
    nh = np.linalg.norm(uh, axis=1)
    tilt = np.minimum(np.arctan2(nh, u[:, 2]), np.radians(max_tilt_deg))   # u_z <= 0 (upside down) -> saturated
    direction = np.divide(uh, nh[:, None], out=np.zeros_like(uh), where=nh[:, None] > 0)
    a = np.zeros_like(u)
    a[:, :2] = g * np.tan(tilt)[:, None] * direction
    return a


def physics_predict(obs_t, obs_p, obs_u, fut_t, k: int, g: float, max_tilt_deg: float) -> np.ndarray:
    p0, v0 = fit_state(obs_t, obs_p, k, 1)
    a = tilt_acceleration(np.asarray(obs_u, float)[:, -1], g, max_tilt_deg)
    tau = (np.asarray(fut_t, float) - obs_t[-1])[None, :, None]
    return p0[:, None] + v0[:, None] * tau + 0.5 * a[:, None] * tau ** 2
