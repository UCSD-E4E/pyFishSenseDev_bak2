import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Implementation of the laser calibration system using SIFT.

# Given an image and template containing feature points/descriptions we want to look for,
# return a matching list of correlating points.
def find_correlating_points(img: np.ndarray, template: np.ndarray):
    # Copy images as RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    # Grayscale the images
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.GaussianBlur(template_gray, (5,5), 0)

    # Find feature points and descriptors
    orb = cv2.ORB_create()
    img_feature_points, img_descriptor = orb.detectAndCompute(img_gray, None)
    template_feature_points, template_descriptor = orb.detectAndCompute(template_gray, None)
    
    # Draw keypoints and display images
    cv2.drawKeypoints(img_gray, img_feature_points, img_rgb, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.drawKeypoints(template_gray, template_feature_points, template_rgb, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img_rgb)
    #plt.show()
    #plt.imshow(template_rgb)
    #plt.show()

    # Match points
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(template_descriptor, img_descriptor)
    matches = sorted(matches, key = lambda x : x.distance)
    result = cv2.drawMatches(template_gray, template_feature_points, img_gray, img_feature_points, matches[:15], None, flags=2)

    # Draw matches and display images
    plt.imshow(result)
    plt.show()

def main():
    slate_path = "./test_data/blg_template.jpg"
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <image path>")
        return
    cal_path = sys.argv[1]

    # Read images
    cal = cv2.imread(cal_path)
    cal = cv2.resize(cal, (0,0), fx=0.1, fy=0.1)
    slate = cv2.imread(slate_path)
    #slate = cv2.resize(slate, (0,0), fx=0.1, fy=0.1) # just for visualization

    find_correlating_points(cal, slate)

if __name__ == "__main__":
    main()