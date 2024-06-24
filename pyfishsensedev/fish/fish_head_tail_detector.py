from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class FishHeadTailDetector:
    def _pca_find_left_right(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # Find all the nonzero points.  These are the mask.
        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        # Crop the mask using the nonzero
        mask_crop = mask[y_min:y_max, x_min:x_max]
        y, x = mask_crop.nonzero()

        # Necessary for using PCA
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x = x - x_mean
        y = y - y_mean
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        # PCA
        coords = np.vstack([x, y])
        cov = np.cov(coords)
        evals, evecs = np.linalg.eig(cov)

        # Choose the largest eigenvalue
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue

        height, width = mask_crop.shape

        # Scale the line until we reach the end.
        scale = height if height > width else width

        coord1 = np.array([-x_v1 * scale * 2, -y_v1 * scale * 2])
        coord2 = np.array([x_v1 * scale * 2, y_v1 * scale * 2])

        # Calculate the line
        coord1[0] -= x_min
        coord2[0] += x_min

        coord1[1] -= y_min
        coord2[1] += y_min

        m = y_v1 / x_v1
        b = coord1[1] - m * coord1[0]

        y, x = mask_crop.nonzero()
        y_target = m * x + b

        # Find the full line
        points_along_line = np.where(np.abs(y - y_target) < 1)
        x = x[points_along_line]
        y = y[points_along_line]

        coords = np.stack([x, y])
        left_coord = coords[:, np.argmin(x)]
        right_coord = coords[:, np.argmax(x)]

        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        left_coord[0] += x_min
        right_coord[0] += x_min
        left_coord[1] += y_min
        right_coord[1] += y_min

        return left_coord, right_coord

    def _pca_find_left_right(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Find the initial left/right coord
        left_coord, right_coord, _ = self._pca_find_left_right(mask)

        # Draw a line
        line_direction = (right_coord - left_coord) / np.linalg.norm(
            right_coord - left_coord
        )
        perp_direction = np.array([line_direction[1], line_direction[0]])
        mid_point = (left_coord) + (right_coord - left_coord) / 2

        height, width = mask.shape
        m = perp_direction[1] / perp_direction[0]
        b = mid_point[1] - m * mid_point[0]

        x_coord = np.tile(np.arange(width), (height, 1))
        y_coord = np.tile(np.arange(height), (width, 1)).T

        left_mask = np.logical_and((m * x_coord + b) > y_coord, mask)
        right_mask = np.logical_and((m * x_coord + b) < y_coord, mask)

        left_left_coord, _, left_eigen_value = self._pca_find_left_right(left_mask)
        _, right_right_coord, right_eigen_value = self._pca_find_left_right(right_mask)

        # The greater eigen value means more symmetry.
        # Maybe fish tails are more symmetrical than head?
        if left_eigen_value < right_eigen_value:
            tail_coord = right_right_coord
            y, x = left_mask.nonzero()
            pos = x.argmin()
            x = x[pos]
            y = y[pos]
            head_coord = np.array([x, y])

        else:
            tail_coord = left_left_coord
            y, x = right_mask.nonzero()
            pos = x.argmax()
            x = x[pos]
            y = y[pos]
            head_coord = np.array([x, y])

        return tail_coord, head_coord


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    import torch

    from pyfishsensedev.fish import FishSegmentationFishialPyTorch
    from pyfishsensedev.image import ImageRectifier
    from pyfishsensedev.image.image_processors.raw_processor import RawProcessor
    from pyfishsensedev.laser.laser_detector import LaserDetector

    raw_processor = RawProcessor()
    raw_processor_dark = RawProcessor(enable_histogram_equalization=False)
    image_rectifier = ImageRectifier(Path("./data/lens-calibration.pkg"))
    laser_detector = LaserDetector(
        Path("./data/models/laser_detection.pth"),
        Path("./data/lens-calibration.pkg"),
        Path("./data/laser-calibration.pkg"),
    )

    img = raw_processor.load_and_process(Path("./data/P8030201.ORF"))
    img_dark = raw_processor_dark.load_and_process(Path("./data/P8030201.ORF"))
    img = image_rectifier.rectify(img)
    img_dark = image_rectifier.rectify(img_dark)

    img8 = ((img.astype("float64") / 65535) * 255).astype("uint8")
    img_dark8 = ((img_dark.astype("float64") / 65535) * 255).astype("uint8")
    coords = laser_detector.find_laser(img_dark8)

    fish_segmentation_inference = FishSegmentationFishialPyTorch(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    segmentations = fish_segmentation_inference.inference(img8)

    mask = np.zeros_like(segmentations, dtype=bool)
    mask[segmentations == segmentations[coords[1], coords[0]]] = True

    fish_head_tail_detector = FishHeadTailDetector()
    left_coord, right_coord = fish_head_tail_detector.find_head_tail(mask, img8)

    plt.imshow(img8)
    plt.plot(left_coord[0], left_coord[1], "r.")
    plt.plot(right_coord[0], right_coord[1], "b.")
    plt.show()
