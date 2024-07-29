# Script to test commercial/non-commercial implementations of SuperPoint+LightGlue.
# Usage: python3 test_data.py [-c] <data_dir>
# [-c]: Flag to enable the non-commercial implementation of SuperPoint
# A slate image must be in the specified data directory. Results will be saved in the appropriate subdirectory.
# WARNING: Ensure the correct use of licensing.

import sys
import os
import argparse

import cv2
import matplotlib.pyplot as plt
import glob

import superpoint_inference

from utils import load_image
import viz2d

class Image():
    def __init__(self, img_path):
        self.image = load_image(img_path) # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
        self.name = os.path.basename(img_path)
        self.img_path = img_path

def visualize_matches(image0: Image, m_kpts0, image1: Image, m_kpts1, output_path, text=None, save_fig=False, show_fig=True):
    axes = viz2d.plot_images([image0.image, image1.image])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    if save_fig:
        plt.savefig(f"{output_path}/{image1.name}", bbox_inches='tight')
        print(f"    Output saved to {output_path}/{image1.name}.")
    if show_fig: plt.show()
    plt.close()

# Given a path to the data and an optional matching string, return a list of image objects
def load_images(data_path, matching_string=''):
    extensions = ['.PNG', '.png', '.JPG', '.jpg']
    path = f'{data_path}/{matching_string}*'
    glob_list = [glob.glob(path + ext) for ext in extensions]
    glob_list = sum(glob_list, [])
    glob_list.sort()
    imgs = [Image(img) for img in glob_list]
    return imgs

# Given an image path and a scale, resize the image and save it
def resize_and_save(img_path, scale):
    pass

def main():
    #sys.stderr = open(os.devnull, 'w')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dir_path", 
        type=str, 
        help="Path to a directory containing your data."
    )
    parser.add_argument(
        "-c",
        "--noncom-license",
        action="store_true",
        help="Use this if you want to use the non-commercial version of SuperPoint.",
    )
    args = parser.parse_args()
    data_dir = args.dir_path

    # get our Image objects
    print("Loading images...")
    cals, slates = load_images(data_dir), load_images(data_dir, matching_string='slate')
    if len(slates) == 0: sys.exit(f"Could not find a slate image in {data_dir} with the format slate{'{*}.{ext}'}")
    slate = slates[0] # we only want one slate
    print(f"Using {slate.name} as the template.")

    # run our images
    com_license = False if args.noncom_license else True
    results_dir = f"{data_dir}/results/{'commercial' if com_license else 'non_commercial'}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for c in cals:
        if 'slate' in c.name: continue # ensuring we're not processing a slate

        print(f"==== Processing {c.name} ====")

        slate_feats, cal_feats, matches01 = superpoint_inference.run_inference(slate.image, c.image, com_license=com_license)

        slate_keypoints, cal_keypoints, matches = slate_feats['keypoints'], cal_feats['keypoints'], matches01['matches']
        slate_matches, cal_matches = slate_keypoints[matches[..., 0]], cal_keypoints[matches[..., 1]]

        print(f"    Found {len(cal_keypoints)} keypoints and {len(cal_matches)} matches.")
        visualize_matches(slate, slate_matches, c, cal_matches, results_dir, save_fig=True, show_fig=False)

if __name__ == "__main__":
    main()