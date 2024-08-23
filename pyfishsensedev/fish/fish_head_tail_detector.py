from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line, plot_points
import numpy as np
import shapely
from shapely import ops

from pyfishsensedev.fish.fish_geometry import FishGeometry

class FishHeadTailDetector:
    def __init__(self, mask: np.ndarray):
        self.geo = FishGeometry(mask)

    def correct_tail_coord(self): # TODO
        self.geo.set_tail_corrected(np.asarray(self.geo.get_tail_coord()))

        return {
            'tail': self.geo.get_tail_corrected()
            }

    def correct_head_coord(self):
        # any point on head_poly_sliced is better or just as good as the initial estimated point
        head_poly_sliced = ops.split(self.geo.get_head_poly(), self.geo.get_headpoint_line()).geoms

        # polygon closest to nosepoint is the one we want
        distances = [shapely.distance(p, self.geo.get_nose_point()) for p in head_poly_sliced]
        head_poly_sliced = head_poly_sliced[distances.index(min(distances))]

        # we guess the tip of head_poly_sliced by finding its nearest point to another point far ahead
        try:
            _, head_corrected = ops.nearest_points(self.geo.get_nose_point(), head_poly_sliced.convex_hull.boundary)
        except:
            # if a point can't be extracted, we default to the original estimation
            head_corrected = shapely.geometry.Point(self.geo.get_head_coord())
        self.geo.set_head_corrected([head_corrected.x, head_corrected.y])

        return {
            'head': self.geo.get_head_corrected()
            }
    
    def correct_endpoints(self):
        self.correct_head_coord()

        self.correct_tail_coord()

        return {
            'head': self.geo.get_head_corrected(),
            'tail': self.geo.get_tail_corrected()
            }

    def classify_endpoints(self, endpoints=None) -> Dict[str, np.ndarray]:
        endpoints = self.geo.get_estimated_endpoints() if endpoints == None else endpoints
        self.geo.set_estimated_endpoints(endpoints)

        # get halves
        halves = self.geo.get_halves()

        # get the convex versions
        halves_convex = self.geo.get_halves_convex()

        # get the differences
        halves_difference = [shapely.difference(halves_convex[0], halves[0]),
                             shapely.difference(halves_convex[1], halves[1])]

        # compare the areas and set head/tail polys
        half_areas = [halves_difference[0].area], [halves_difference[1].area]
        self.geo.set_tail_poly(halves[half_areas.index(max(half_areas))])
        self.geo.set_head_poly(halves[half_areas.index(min(half_areas))])

        # assign the classification to the endpoints
        head_coord = self.geo.get_head_coord(endpoints)
        tail_coord = self.geo.get_tail_coord(endpoints)

        # compute the confidence score
        confidence = float((max(half_areas)[0] - min(half_areas)[0]) / max(half_areas)[0] / 2.0 + 0.5)
        confidence = int(confidence * 100) / 100

        return {
                'head': head_coord,
                'tail': tail_coord,
                'confidence': confidence
                }

    def estimate_endpoints(self) -> Tuple[np.ndarray, np.ndarray, float]:
        # Find all the nonzero points.  These are the mask.
        y, x = self.geo.mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        # Crop the mask using the nonzero
        mask_crop = self.geo.mask[y_min:y_max, x_min:x_max]
        y, x = mask_crop.nonzero()

        # Necessary for using PCA
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x = x - x_mean
        y = y - y_mean
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        # PCA
        coords = np.vstack([x, y])
        cov = np.cov(coords) # covariance matrix
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

        y, x = self.geo.mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        left_coord[0] += x_min
        right_coord[0] += x_min
        left_coord[1] += y_min
        right_coord[1] += y_min

        self.geo.set_estimated_endpoints([left_coord, right_coord])
        return self.geo.get_estimated_endpoints()
    
    def get_head_tail(self):
        self.estimate_endpoints()
        self.classify_endpoints()
        corrected = self.correct_endpoints()
        return corrected

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
    image_rectifier = ImageRectifier(Path(".../data/calibration/fsl-01d-lens-raw.pkg"))
    laser_detector = LaserDetector(
        Path("../laser/models/laser_detection.pth"),
        Path(".../data/calibration/fsl-01d-lens-raw.pkg"),
        Path(".../data/calibration/fsl-01d-laser.pkg"),
    )

    img = raw_processor.load_and_process(Path(".../data/P8030201.ORF"))
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

    fish_head_tail_detector = FishHeadTailDetector(mask)

    # run through the process
    fish_head_tail_detector.estimate_endpoints()
    fish_head_tail_detector.classify_endpoints()
    corrections = fish_head_tail_detector.correct_endpoints()
    head_coord = corrections['head']
    tail_coord = corrections['tail']

    # or just use one function
    # corrections = fish_head_tail_detector.get_head_tail()
    # head_coord = corrections['head']
    # tail_coord = corrections['tail']

    plt.imshow(img8)
    plt.plot(head_coord[0], head_coord[1], "r.")
    plt.plot(tail_coord[0], tail_coord[1], "b.")
    plt.show()
