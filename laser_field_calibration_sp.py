import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Laser calibration implemented using SuperPoint feature extraction and descripting
# paired with LightGlue matching.

extraction_model_path = ""
matching_model_path = ""

# Given an image, extract features using SuperPoint and return points and descriptors
def extract_features(img: np.ndarray) -> tuple:
    
    pass

# Given features of two images, return correlations
def match_features(img0_features, img1_features):
    pass

def main():
    # Get image paths
    slate_path = "./test_data/slate.jpg"
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <image path>")
        return
    cal_path = sys.argv[1]

    # Read images
    cal = cv2.imread(cal_path) # likely needs to be cropped
    slate = cv2.imread(slate_path) # likely needs to be scaled

    # Gather and match features
    cal_keypoints, cal_descriptors = extract_features(cal)
    slate_keypoints, slate_descriptors = extract_features(slate)

    # Visualize the keypoints

if __name__ == "__main__":
    main()
