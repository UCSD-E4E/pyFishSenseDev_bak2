from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line, plot_points
import numpy as np
import cv2
import shapely
from shapely import ops
from sklearn.decomposition import PCA

class FishHeadTailDetector:
    def classify_coords(self, mask: np.ndarray, coords: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        left_coord, right_coord = coords

        # find the perimeter of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = contours[0].reshape(-1, 2)

        # find the line perpendicular to the line connecting the points
        ab = shapely.geometry.LineString(coords)
        vert_min = abs(perimeter[:,1].min() - ab.centroid.y)
        vert_max = abs(perimeter[:,1].max() - ab.centroid.y)
        vert_len = vert_min if vert_min > vert_max else vert_max
        ab_left = ab.parallel_offset(vert_len*1.5, 'left')
        ab_right = ab.parallel_offset(vert_len*1.5, 'right')
        ab_perp = shapely.geometry.LineString([ab_left.centroid, ab_right.centroid])

        # create a polygon
        polygon = shapely.geometry.Polygon(perimeter)

        # split the polygon by the perpendicular line
        halves = ops.split(polygon, ab_perp).geoms
        if len(halves) > 2:
            halves = sorted(halves, key=lambda p: p.area, reverse=True)[:2]

        # get the convex versions
        halves_convex = [halves[0].convex_hull, halves[1].convex_hull]

        # get the differences
        halves_difference = [shapely.difference(halves_convex[0], halves[0]), shapely.difference(halves_convex[1], halves[1])]

        # compare the areas
        half_areas = [halves_difference[0].area], [halves_difference[1].area]
        tail_poly = halves[half_areas.index(max(half_areas))]
        head_poly = halves[half_areas.index(min(half_areas))]

        # determine which coord is head/tail
        if abs(shapely.distance(shapely.Point(left_coord), head_poly)) < abs(shapely.distance(shapely.Point(left_coord), tail_poly)):
            head_coord = left_coord
            tail_coord = right_coord
        else:
            head_coord = right_coord
            tail_coord = left_coord

        # compute the confidence score
        confidence = float((max(half_areas)[0] - min(half_areas)[0]) / max(half_areas)[0] / 2.0 + 0.5)
        confidence = int(confidence * 100) / 100

        # plot_polygon(halves_difference[0], add_points=False, color="#ff0000", facecolor='#ff0000')
        # plot_polygon(halves_difference[1], add_points=False, color='#0000ff', facecolor='#0000ff')

        # START ADJUST COORDINATES

        # make head_polygon convex
        head_poly_convex = head_poly.convex_hull

        # find the furthest points from ab_perp.centroid
        point1 = head_coord
        point2 = head_coord
        largest_dist1 = largest_dist2 = -1
        for p in head_poly_convex.boundary.coords:
            dist = abs(shapely.distance(shapely.Point(p), ab_perp.centroid))
            if dist > largest_dist1:
                largest_dist1 = dist
                point1 = p
            elif dist > largest_dist2:
                largest_dist2 = dist
                point2 = p
        #head_coord_corrected = shapely.LineString([shapely.Point(point1), shapely.Point(point2)]).centroid
        head_coord_corrected = shapely.Point(point1)

        return {
                'head': np.asarray(head_coord),
                'tail': np.asarray(tail_coord),
                'confidence': confidence,
                'head_poly': head_poly,
                'tail_poly': tail_poly,
                'head_corrected': np.asarray([head_coord_corrected.x, head_coord_corrected.y])
                }


    def find_head_tail(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
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

        y, x = mask.nonzero()
        x_min, x_max, y_min, y_max = [x.min(), x.max(), y.min(), y.max()]

        left_coord[0] += x_min
        right_coord[0] += x_min
        left_coord[1] += y_min
        right_coord[1] += y_min

        return left_coord, right_coord


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

    fish_head_tail_detector = FishHeadTailDetector()
    left_coord, right_coord = fish_head_tail_detector.find_head_tail(mask, img8)

    plt.imshow(img8)
    plt.plot(left_coord[0], left_coord[1], "r.")
    plt.plot(right_coord[0], right_coord[1], "b.")
    plt.show()
