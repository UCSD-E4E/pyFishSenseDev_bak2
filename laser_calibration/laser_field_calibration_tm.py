import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

# Implementation of the laser calibration system using the PnP method.

# Given an image, apply a Gaussian blur
def preprocess_img(img: np.ndarray) -> np.ndarray:
    img_copy = img.copy()

    # Convert to grayscale
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur
    img_blurred = cv2.GaussianBlur(img_copy, (5,5), 0)

    return img_blurred

# Given a pre-processed image, return its contours and its respective grayscale image
def get_lines(preprocessed_img: np.ndarray) -> tuple[tuple, np.ndarray]:
    preprocessed_img_copy = preprocessed_img.copy()

    # Get edges
    img_edges = cv2.Canny(preprocessed_img_copy, 150, 200)

    # Get contours
    img_contours = img_edges.copy()
    contours, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contours, contours, -1, (255,255,255), 3)

    return (contours, img_contours)

# Given a pre-processed image and two corners, return a list of contours in the cropped image
# and its respective grayscale image
def get_cropped_lines(preprocessed_img: np.ndarray, corners: tuple) -> tuple[tuple, np.ndarray]:
    preprocess_img_copy = preprocessed_img.copy()

    # Crop the image
    top_left, bottom_right = corners
    img_cropped = preprocess_img_copy[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Get contours
    return get_lines(img_cropped)

# Given contour-converted images of a (resized) template and an image, locate the
# template in the image
# Returns the top-left and bottom-right image coordinates
def detect_corners_of_template(img: np.ndarray, template: np.ndarray):
    # The range and increment of template sizes we should compare
    # TODO: Make the scalars respective to the comparing image, not the template image
    scalar_min = 0.09
    scalar_max = 0.3
    scalar_step = 0.01
    
    scalars_to_test = [x / 100.0 for x in range(int(scalar_min*100), int(scalar_max*100+(scalar_step*100)), int(scalar_step*100))]
    print(f"Comparing {len(scalars_to_test)} different template sizes...")
    print(f"Scalars to test: {scalars_to_test}")

    matches = list()

    for s in scalars_to_test:
        # Resize the template
        template_copy = cv2.resize(template, (0,0), fx=s, fy=s)
        t_h, t_w = template_copy.shape[:2]
        print(f"Testing scalar {s} with template size {t_w}x{t_h}...")

        # Use TM_CCOEFF template matching
        result = cv2.matchTemplate(img, template_copy, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(f"Max value is {max_val}.")

        # Add results to the list
        matches.append((result, min_val, max_val, min_loc, max_loc, t_h, t_w))

    # Choose the best match
    best_match = max(matches, key=lambda x: x[2])
    best_result, best_min_val, best_max_val, best_min_loc, best_max_loc, best_t_h, best_t_w = best_match
    print(f"Best size is {best_t_w}x{best_t_h}.")
    
    # Calculate the top left and bottom right pixel
    top_left = best_max_loc
    bottom_right = (top_left[0] + best_t_w, top_left[1] + best_t_h)

    return (top_left, bottom_right)

def main():
    slate_path = "./test_data/slate.jpg"
    if len(sys.argv) != 2:
        print(f"Usage: python3 {os.path.basename(__file__)} <image path>")
        return
    cal_path = sys.argv[1]

    # Read images
    cal = cv2.imread(cal_path)
    slate = cv2.imread(slate_path)

    # Pre-process images
    cal_preprocessed = preprocess_img(cal)
    slate_preprocessed = preprocess_img(slate)

    # Extract and draw contours
    cal_contours, cal_contours_img = get_lines(cal_preprocessed)
    slate_contours, slate_contours_img = get_lines(slate_preprocessed)

    # Find where the slate is in cal
    top_left, bottom_right = detect_corners_of_template(cal_contours_img, slate_contours_img)

    # Display the results
    display = cal.copy()
    cv2.rectangle(display, top_left, bottom_right, (0,0,255), 10)
    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    plt.imshow(display)
    plt.show()

    # Get and display just the contours of the identified slate
    cropped_cal_contours, cropped_cal_contours_img = get_cropped_lines(cal_preprocessed, (top_left, bottom_right))
    plt.imshow(cropped_cal_contours_img, cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()