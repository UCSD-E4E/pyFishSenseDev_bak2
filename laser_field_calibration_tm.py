import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import os
import sys

# Implementation of the laser calibration system using the PnP method.
# Uses template matching to identify 2D points on calibration images.

# Given an image, apply a Gaussian blur and detect edges
def preprocess_img(img: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    img_blurred = cv2.GaussianBlur(img, (5,5), 0)

    # Get edges
    img_edges = cv2.Canny(img_blurred, 150, 200)

    # Draw contours
    img_contours = img_edges.copy()
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 3)

    return img_contours

# Given an image and a template, locate the template in the image
# Returns the top-left and bottom-right image coordinates
def detect_corners_of_template(img: np.ndarray, template: np.ndarray):
    # Resize the template
    template = cv2.resize(template, (0,0), fx=0.1, fy=0.1)
    t_h, t_w = template.shape[:2]

    # Preprocess the images
    img_processed = preprocess_img(img)
    template_processed = preprocess_img(template)

    # Use TM_CCOEFF template matching
    result = cv2.matchTemplate(img_processed, template_processed, cv2.TM_CCOEFF)
    _, _, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    bottom_right = (top_left[0] + t_w, top_left[1] + t_h)

    return (top_left, bottom_right)

def main():
    slate_path = "./test_data/slate.jpg"
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <image path>")
        return
    img_path = sys.argv[1]

    img = cv2.imread(img_path)
    slate = cv2.imread(slate_path)

    top_left, bottom_right = detect_corners_of_template(img, slate)

    cv2.rectangle(img, top_left, bottom_right, (0,0,255), 10)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    main()