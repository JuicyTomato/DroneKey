"""Turn a YOLO-Pose result on a crop into frame-coordinate keypoints (pure NumPy, testable without ultralytics)."""
import numpy as np

from dronepose.data.crops import CropBox, crop_to_frame
from dronepose.keypoints.predictors import no_detection


def crop_result_to_frame(xy: np.ndarray | None, kp_conf: np.ndarray | None, box_conf: np.ndarray | None,
                         box: CropBox, size: int) -> np.ndarray:
    if xy is None or box_conf is None or len(box_conf) == 0:
        return no_detection()
    i = int(np.argmax(box_conf))
    conf = np.ones(4) if kp_conf is None else np.asarray(kp_conf[i], float)
    return np.hstack([crop_to_frame(np.asarray(xy[i], float), box, size), conf[:, None]])
