import numpy as np
import torch

from pyfishsensedev.fish.fish_segmentation_inference import FishSegmentationInference


def test_fishial_inference():
    data = np.load("./tests/data/fish_test.npz")
    img8 = data["arr_0"]
    truth = data["arr_1"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fish_segmentation_interface = FishSegmentationInference(device)
    result = fish_segmentation_interface.inference(img8)

    np.testing.assert_array_equal(result, truth)