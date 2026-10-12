"""Draw keypoints (GT green, predicted red), body axes and the forecast trajectory (yellow)."""
import cv2
import numpy as np

from dronepose.common.geometry import project


def draw_overlay(img, K, kpts_gt, kpts_pred, R, t, future, axis_len: float = 0.15) -> np.ndarray:
    out = img.copy()
    for (x, y) in np.asarray(kpts_gt, float):
        cv2.circle(out, (int(x), int(y)), 7, (0, 255, 0), 2)
    for i, (x, y) in enumerate(np.asarray(kpts_pred, float)[:, :2]):
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.putText(out, str(i + 1), (int(x) + 6, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if R is not None and t is not None:
        R, t = np.asarray(R), np.asarray(t)
        ends = project(K, np.vstack([t, t + axis_len * R.T]))     # rows of R.T = body axes in camera frame
        o = tuple(ends[0].astype(int))
        for end, col in zip(ends[1:], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):   # x red, y green, z blue
            cv2.arrowedLine(out, o, tuple(end.astype(int)), col, 3, tipLength=0.2)
    if future is not None and len(future) > 1:
        pts = project(K, np.asarray(future, float)).astype(np.int32)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 3)
    return out
