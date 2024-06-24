import numpy as np

from pyfishsensedev.fish.fish_head_tail_detector import FishHeadTailDetector


def test_find_head_tail():
    data = np.load("./tests/data/fish_segmentation.npz")
    segmentation = data["segmentations"]

    left_truth = np.array([1075, 1112])
    right_truth = np.array([2319, 1052])

    find_head_tail_detector = FishHeadTailDetector()
    left, right = find_head_tail_detector.find_head_tail(segmentation)

    np.testing.assert_array_equal(left, left_truth)
    np.testing.assert_array_equal(right, right_truth)
