# Script to test commercial/non-commercial implementations of SuperPoint+LightGlue
# WARNING: Ensure the correct use of licensing

import sys
import os

import cv2
import matplotlib.pyplot as plt
import glob

import superpoint_inference_com

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
    if show_fig:
        plt.show()
    plt.close()

# Given a dir path of the calibration images, return a list of image objects
def load_images(cal_path):
    glob_list = glob.glob(cal_path + '/*.PNG') + glob.glob(cal_path + '/*.png') + glob.glob(cal_path + '/*.JPG') + glob.glob(cal_path + '/*.jpg')
    glob_list.sort()
    imgs = list()
    for img in glob_list:
        imgs.append(Image(img))
    return imgs

# Given an image path and a scale, resize the image and save it
def resize_and_save(img_path, scale):
    pass

def main():
    #sys.stderr = open(os.devnull, 'w')

    # Get image paths
    if len(sys.argv) != 4:
        print(f"Usage: python3 {os.path.basename(__file__)} <slate path> <image dir path> <output dir dest>")
        return
    slate_path = sys.argv[1]
    cal_dir_path = sys.argv[2]
    output_dir_path = sys.argv[3]

    # get our Image objects
    slate = Image(slate_path)
    cals = load_images(cal_dir_path)

    # run our images
    for c in cals:
        print(f"Processing {c.name}...")

        slate_feats, cal_feats, matches01 = superpoint_inference_com.run_inference(slate.image, c.image)

        slate_keypoints, cal_keypoints, matches = slate_feats['keypoints'], cal_feats['keypoints'], matches01['matches']
        slate_matches, cal_matches = slate_keypoints[matches[..., 0]], cal_keypoints[matches[..., 1]]

        print(f"    # Keypoints: {len(slate_keypoints)}")
        print(f"    # Matches: {len(cal_matches)}")
        print(f"    # Descriptors: {len(cal_feats['descriptors'])}")
        print(f"    Descriptors: {cal_feats['descriptors']}")
        print(f"    Matches: {cal_matches}")
        visualize_matches(slate, slate_matches, c, cal_matches, output_dir_path, save_fig=True, show_fig=False)

    print(f"All outputs saved to {output_dir_path}.")

if __name__ == "__main__":
    main()