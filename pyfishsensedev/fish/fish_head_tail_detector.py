from pathlib import Path
from typing import Tuple, Dict

import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon, plot_line, plot_points
import numpy as np
import cv2
import shapely
from shapely import ops

class FishHeadTailDetector:
    def classify_coords(self, mask: np.ndarray, coords: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        """Given a fish mask and left/right coords, this function will classify each coord as head/tail.
            Input:
                mask: np.ndarray
                coords: Tuple[left_coord, right_coord]
            Output (dict):
                'head': np.ndarray
                    [x, y]
                'tail': np.ndarray
                    [x, y]
                'confidence': float
        """
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

        # let's recalculate ab so that it's long enough to slice the polygon horizontally
        hor_min = abs(perimeter[:,0].min() - ab_perp.centroid.x)
        hor_max = abs(perimeter[:,0].max() - ab_perp.centroid.x)
        hor_len = hor_min if hor_min > hor_max else hor_max
        ab_perp_left = ab_perp.parallel_offset(hor_len*1.5, 'left')
        ab_perp_right = ab_perp.parallel_offset(hor_len*1.5, 'right')
        ab_corrected = shapely.geometry.LineString([ab_perp_left.centroid, ab_perp_right.centroid])

        # create a polygon
        polygon = shapely.geometry.Polygon(perimeter)

        # simplify the polygon
        #polygon = polygon.simplify(tolerance=3.5) # 2.5

        # split the polygon by the perpendicular line
        halves = ops.split(polygon, ab_perp).geoms

        # split those halfs by ab to get 4 sections
        quads = []
        for half in halves:
            for quad in ops.split(half, ab_corrected).geoms: quads.append(quad)

        # find the nearest points from each centroid to the boundary
        neareset_points = []
        nearest_distances = []
        for quad in quads:
            _, point = ops.nearest_points(quad.centroid, polygon.boundary)
            neareset_points.append(point)
            nearest_distances.append(abs(shapely.distance(quad.centroid, point)))

        # determine which left/right coordinate is closest to the chosen centroid
        left_coord = shapely.Point(left_coord)
        right_coord = shapely.Point(right_coord)
        idx = nearest_distances.index(min(nearest_distances))
        if abs(shapely.distance(quads[idx].centroid, left_coord)) < abs(shapely.distance(quads[idx].centroid, right_coord)):
            tail_coord = left_coord
            head_coord = right_coord
        else:
            tail_coord = right_coord
            head_coord = left_coord

        # calculate the confidence score
        h1_quad = min([nearest_distances[0], nearest_distances[1]])
        h2_quad = min([nearest_distances[2], nearest_distances[3]])
        max_quad = max([h1_quad, h2_quad])
        min_quad = min([h1_quad, h2_quad])
        confidence = float((max_quad - min_quad) / max_quad / 2.0 + 0.5)
        confidence = int(confidence * 100) / 100

        #Visualize the quadrants
        # plt.imshow(mask)
        # plot_polygon(quads[0], color='#ff0000', add_points=False)
        # plot_points(shapely.geometry.Point(quads[0].centroid), color="#ff0000") # red
        # plot_line(shapely.geometry.LineString([quads[0].centroid, neareset_points[0]]))

        # plot_polygon(quads[1], color='#0000ff', add_points=False)
        # plot_points(shapely.geometry.Point(quads[1].centroid), color="#0000ff") # blue
        # plot_line(shapely.geometry.LineString([quads[1].centroid, neareset_points[1]]))

        # plot_polygon(quads[2], color='#008000', add_points=False)
        # plot_points(shapely.geometry.Point(quads[2].centroid), color="#008000") # green
        # plot_line(shapely.geometry.LineString([quads[2].centroid, neareset_points[2]]))

        # plot_polygon(quads[3], color='#FFA500', add_points=False)
        # plot_points(shapely.geometry.Point(quads[3].centroid), color="#FFA500") # orange
        # plot_line(shapely.geometry.LineString([quads[3].centroid, neareset_points[3]]))

        # plt.show()
        # plt.close()

        # Extract x and y coordinates
        head_coord = [head_coord.x, head_coord.y]
        tail_coord = [tail_coord.x, tail_coord.y]

        return {'head': np.asarray(head_coord), 'tail': np.asarray(tail_coord), 'confidence': confidence}


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
