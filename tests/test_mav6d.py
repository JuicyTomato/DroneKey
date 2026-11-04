import numpy as np

from dronepose.data import mav6d


def _label(tmp_path, t_vicon, q):
    p = tmp_path / "l.txt"
    p.write_text(" ".join(str(x) for x in [1, 0, 0, 0, 0, 0, 0, 1, 1, *t_vicon, *q]))
    return p


def test_read_label_applies_vicon_to_camera(tmp_path):
    R, t = mav6d.read_label(_label(tmp_path, [0.5, -0.2, 1.0], [0, 0, 0, 1]))
    Rc, tc = mav6d.CAM_FROM_VICON[:, :3], mav6d.CAM_FROM_VICON[:, 3]
    assert np.allclose(R, Rc)
    assert np.allclose(t, Rc @ [0.5, -0.2, 1.0] + tc)


def test_project_matches_util_py_formula(tmp_path):
    R, t = mav6d.read_label(_label(tmp_path, [-0.8, 0.37, 1.04], [0.105, 0.053, 0.371, 0.921]))
    pts = np.array([[0.0, 0.0, 0.0], [0.1, -0.05, 0.02]])
    cam = (R @ pts.T).T + t
    xp, yp = cam[:, 0] / cam[:, 2], cam[:, 1] / cam[:, 2]
    k1, k2, p1, p2, k3 = mav6d.DIST
    r2 = xp ** 2 + yp ** 2
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xp * radial + 2 * p1 * xp * yp + p2 * (r2 + 2 * xp ** 2)
    yd = yp * radial + p1 * (r2 + 2 * yp ** 2) + 2 * p2 * xp * yp
    K = mav6d.K
    ref = np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)   # util.py: no skew, like OpenCV
    assert np.allclose(mav6d.project(pts, R, t), ref, atol=1e-6)


def test_undistort_inverts_project(tmp_path):
    R, t = mav6d.read_label(_label(tmp_path, [-0.8, 0.37, 1.04], [0.105, 0.053, 0.371, 0.921]))
    pts = np.array([[0.1, 0.1, 0.0], [-0.1, 0.12, -0.03]])
    cam = (R @ pts.T).T + t
    assert np.allclose(mav6d.undistort(mav6d.project(pts, R, t)), cam[:, :2] / cam[:, 2:], atol=1e-6)
