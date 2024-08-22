from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line, plot_points
import numpy as np
import cv2
import shapely
from shapely import ops

class FishGeometry:
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.perimeter = self.get_perimeter()
        self.pca_endpoints = None # the estimated endpoints from PCA 
        self.ab = None # the line connecting the endpoints 
        self.ab_perp = None # the line perpendicular to ab 
        self.polygon = None # a polygonal representation of the fish mask 
        self.halves = None # two polygon halves sliced by ab_perp 
        self.halves_convex = None # the respective convex hulls of the halves
        self.halves_difference = None # the polygons contained in halves_convex but not halves 
        self.tail_coord = None # the estimated tail coord from PCA after binary classification
        self.head_coord = None # the estimated head coord from PCA after binary classification
        self.tail_poly = None # the polygon half containing tail_coord
        self.head_poly = None # the polygon half containing head_coord
        self.headpoint_line = None # a line parallel to ab_perp with its centroid being head_coord
        self.tailpoint_line = None # a line parallel to ab_perp with its centroid being tail_coord
        self.nose_point = None # a point further out from the head_coord
        self.tail_point = None # a point further out form the tail_coord
        self.head_corrected = None
        self.tail_corrected = None
    
    def get_perimeter(self):
        # find the perimeter of the mask
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.perimeter = contours[0].reshape(-1, 2)

class FishHeadTailDetector:
    def __init__(self, mask: np.ndarray):
        self.geo = FishGeometry(mask)

    def correct_head_coord(self):
        # draw two lines parallel to ab_perp a little bit aways from the original endpoints
        hor_min = abs(self.geo.perimeter[:,0].min() - self.geo.ab_perp.centroid.x)
        hor_max = abs(self.geo.perimeter[:,0].max() - self.geo.ab_perp.centroid.x)
        hor_len = hor_min if hor_min > hor_max else hor_max
        nose_line1 = self.geo.ab_perp.parallel_offset(hor_len*2.0, 'right') # hor_len*2.0
        nose_line2 = self.geo.ab_perp.parallel_offset(hor_len*2.0, 'left') # hor_len*2.0

        # choose the line near the head and get the centroid 
        if (shapely.distance(nose_line1.centroid, shapely.Point(self.geo.head_coord)) < 
            shapely.distance(nose_line1.centroid, shapely.Point(self.geo.tail_coord))):
            self.geo.nose_point = nose_line1.centroid
            self.geo.tail_point = nose_line2.centroid
        else:
            self.geo.nose_point = nose_line2.centroid
            self.geo.tail_point = nose_line1.centroid

        # get the perpendicular line of ab from the headpoint
        half_len = shapely.distance(self.geo.ab.centroid, shapely.Point(self.geo.head_coord).centroid)
        endpoint_line1 = self.geo.ab_perp.parallel_offset(half_len, 'right')
        endpoint_line2 = self.geo.ab_perp.parallel_offset(half_len, 'left')

        # again choose the right line
        if (shapely.distance(endpoint_line1.centroid, shapely.Point(self.geo.head_coord)) < 
            shapely.distance(endpoint_line1.centroid, shapely.Point(self.geo.tail_coord))):
            self.geo.headpoint_line = endpoint_line1
            self.geo.tailpoint_line = endpoint_line2
        else:
            self.geo.headpoint_line = endpoint_line2
            self.geo.tailpoint_line = endpoint_line1

        # slice the headpoly by that line
        # any point on head_poly_sliced is better or just as good as the initial estimated point
        head_poly_sliced = ops.split(self.geo.head_poly, self.geo.headpoint_line).geoms
        # polygon closest to nosepoint is the one we want
        distances = [shapely.distance(p, self.geo.nose_point) for p in head_poly_sliced]
        head_poly_sliced = head_poly_sliced[distances.index(min(distances))]

        # we guess the tip of head_poly_sliced by finding its nearest point to another point far ahead
        try:
            _, head_corrected = ops.nearest_points(self.geo.nose_point, head_poly_sliced.convex_hull.boundary)
        except:
            # if a point can't be extracted, we default to the original estimation
            head_corrected = shapely.geometry.Point(self.geo.head_coord)
        self.geo.head_corrected = [head_corrected.x, head_corrected.y]

        return {
            'head_corrected': np.asarray(self.geo.head_corrected)
            }
    
    def correct_endpoints(self):
        self.correct_head_coord()
        return {
            'head_corrected': np.asarray(self.geo.head_corrected)
            }

    def classify_endpoints(self) -> Dict[str, np.ndarray]:
        left_coord, right_coord = self.geo.pca_endpoints

        # find the perimeter of the mask
        contours, _ = cv2.findContours(self.geo.mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.geo.perimeter = contours[0].reshape(-1, 2)

        # find the line perpendicular to the line connecting the points
        self.geo.ab = shapely.geometry.LineString([left_coord, right_coord])
        vert_min = abs(self.geo.perimeter[:,1].min() - self.geo.ab.centroid.y)
        vert_max = abs(self.geo.perimeter[:,1].max() - self.geo.ab.centroid.y)
        vert_len = vert_min if vert_min > vert_max else vert_max
        ab_left = self.geo.ab.parallel_offset(vert_len*1.5, 'left')
        ab_right = self.geo.ab.parallel_offset(vert_len*1.5, 'right')
        self.geo.ab_perp = shapely.geometry.LineString([ab_left.centroid, ab_right.centroid])

        # create a polygon
        self.geo.polygon = shapely.geometry.Polygon(self.geo.perimeter)

        # split the polygon by the perpendicular line
        self.geo.halves = ops.split(self.geo.polygon, self.geo.ab_perp).geoms
        if len(self.geo.halves) > 2:
            self.geo.halves = sorted(self.geo.halves, key=lambda p: p.area, reverse=True)[:2]

        # get the convex versions
        self.geo.halves_convex = [self.geo.halves[0].convex_hull, self.geo.halves[1].convex_hull]

        # get the differences
        self.geo.halves_difference = [shapely.difference(self.geo.halves_convex[0], self.geo.halves[0]),
                             shapely.difference(self.geo.halves_convex[1], self.geo.halves[1])]

        # compare the areas
        half_areas = [self.geo.halves_difference[0].area], [self.geo.halves_difference[1].area]
        self.geo.tail_poly = self.geo.halves[half_areas.index(max(half_areas))]
        self.geo.head_poly = self.geo.halves[half_areas.index(min(half_areas))]

        # determine which coord is head/tail
        if (abs(shapely.distance(shapely.Point(left_coord), self.geo.head_poly)) <
            abs(shapely.distance(shapely.Point(left_coord), self.geo.tail_poly))):
            self.geo.head_coord = left_coord
            self.geo.tail_coord = right_coord
        else:
            self.geo.head_coord = right_coord
            self.geo.tail_coord = left_coord

        # compute the confidence score
        confidence = float((max(half_areas)[0] - min(half_areas)[0]) / max(half_areas)[0] / 2.0 + 0.5)
        confidence = int(confidence * 100) / 100

        return {
                'head': np.asarray(self.geo.head_coord),
                'tail': np.asarray(self.geo.tail_coord),
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

        self.geo.pca_endpoints = (left_coord, right_coord)
        return self.geo.pca_endpoints

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
    left_coord, right_coord = fish_head_tail_detector.estimate_endpoints(mask, img8)

    plt.imshow(img8)
    plt.plot(left_coord[0], left_coord[1], "r.")
    plt.plot(right_coord[0], right_coord[1], "b.")
    plt.show()
