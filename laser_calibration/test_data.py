# Script to test commercial/non-commercial implementations of SuperPoint+LightGlue.
# Usage: python3 test_data.py [-n] [-p] <data_dir>
# [-n]: Use this if you want to use the non-commercial version of SuperPoint.
# [-p]: Use this to disable saving results in a path name based on preprocessing configurations.
# A slate image must be in the specified data directory. Results will be saved in the appropriate subdirectory.
# WARNING: Ensure the correct use of licensing.

# TODO: Load configurations from a file.
# TODO: Save feature and match results to a .csv file.

import sys
import os
import argparse

import cv2
import matplotlib.pyplot as plt
import glob

import superpoint_inference

from utils import load_image
import viz2d

parser = argparse.ArgumentParser()
parser.add_argument(
    "dir_path", 
    type=str, 
    help="Path to a directory containing your data."
)
parser.add_argument(
    "-n",
    "--noncom-license",
    action="store_true",
    help="Use this if you want to use the non-commercial version of SuperPoint.",
)
parser.add_argument(
    "-p",
    "--disable-preprocessing-path-naming",
    action="store_true",
    help="Use this to disable saving results in a path name based on preprocessing configurations.",
)
args = parser.parse_args()

class Image():
    def __init__(self, img_path):
        self.image = load_image(img_path) # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
        self.name = os.path.basename(img_path)
        self.img_path = img_path

def visualize_matches(image0: Image, m_kpts0, image1: Image, m_kpts1, output_path, save_fig=False, show_fig=True):
    axes = viz2d.plot_images([image0.image, image1.image])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'{len(m_kpts1)} matches', fs=20)
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

def generate_results_path(data_dir, com_license, **conf):
    base_path = f"{data_dir}/results/{'commercial' if com_license else 'non_commercial'}"
    if args.disable_preprocessing_path_naming: return base_path
    ext = '/'
    if len(conf) == 0: ext = f'{ext}no_preprocessing'
    else:
        for k, i in conf.items():
            ext = f'{ext}{i}_{k}+'
        ext = ext[:-1]
    return base_path + ext

def main():
    #sys.stderr = open(os.devnull, 'w')
    data_dir = args.dir_path

    # get our Image objects
    print("Loading images...")
    cals, slates = load_images(data_dir), load_images(data_dir, matching_string='slate')
    if len(slates) == 0: sys.exit(f"Could not find a slate image in {data_dir} with the format slate{'{*}.{ext}'}")
    slate = slates[0] # we only want one slate
    print(f"Using {slate.name} as the template.")

    # run our images
    preprocess_conf = {
        'crop': 1.5,
        'gamma': 2.0,
    }
    com_license = False if args.noncom_license else True
    results_dir = generate_results_path(data_dir, com_license, **preprocess_conf)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    matches_count = 0
    less_six_matches_count = 0
    processed_count = 0
    lines = ["==== Images Processed ====\n"]
    for c in cals:
        if 'slate' in c.name: continue # ensure we're not processing a slate

        print(f"==== Processing {c.name} ====")
        lines.append(f"{c.name}:\n")
        processed_count += 1

        slate_feats, cal_feats, matches01 = superpoint_inference.run_inference(slate.image, c.image, com_license=com_license, preprocess_conf=preprocess_conf)

        slate_keypoints, cal_keypoints, matches = slate_feats['keypoints'], cal_feats['keypoints'], matches01['matches']
        slate_matches, cal_matches = slate_keypoints[matches[..., 0]], cal_keypoints[matches[..., 1]]

        matches_count += len(cal_matches)
        if len(cal_matches) < 6:
            less_six_matches_count += 1
        print(f"    Found {len(cal_keypoints)} features and {len(cal_matches)} matches.")
        lines.append(f"    Features: {len(cal_keypoints)}\n")
        lines.append(f"    Matches: {len(cal_matches)}\n")
        visualize_matches(slate, slate_matches, c, cal_matches, results_dir, save_fig=True, show_fig=False)
    
    # Write to results.txt
    print(f"Found a total of {matches_count} matches.")
    recap_lines = [f"==== SLATE MATCHING RESULTS ====\n\n",
                   f"Results for: {data_dir}\n",
                   f"Processed a total of {processed_count} images.\n\n",
                   f"Used {slate.name} as the template.\n\n",
                   f"Found a total of {matches_count} matches.\n\n",
                   f"There are {less_six_matches_count} images with less than 6 matches.\n\n"]
    if not com_license: recap_lines.insert(0, "WARNING: NON-COMMERCIAL USE OF SUPERPOINT!\n\n")
    if len(preprocess_conf) > 0:
        recap_lines.append("Preprocessing Config:\n")
        for k, i in preprocess_conf.items():
            recap_lines.append(f"    {k} = {i}\n")
        recap_lines.append("\n")
    f = open(results_dir + '/results.txt', 'w')
    f.writelines(recap_lines + lines)
    f.close()

if __name__ == "__main__":
    main()