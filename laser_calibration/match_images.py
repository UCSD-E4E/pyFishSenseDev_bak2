# Inference use of SuperPoint+LightGlue

import torch
import kornia
from types import SimpleNamespace
from utils import rbd

from models.lightglue import LightGlue
from models.superpoint_pytorch import SuperPoint
from models.superpoint import SuperPoint as NonCommercialSuperPoint

class Preprocess():
    def __init__(self, img, **conf):
        self.default_conf = {
            'contrast': None,
            'gamma': 2.0,
            'brightness': None,
            'sharpness': 0.5,
            'crop': 1.6,
        }
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.img = img
        self.process()
    def process(self):
        if self.conf.contrast != None:
            self.img = kornia.enhance.adjust_contrast(self.img, self.conf.contrast)
        if self.conf.gamma != None:
            self.img = kornia.enhance.adjust_gamma(self.img, self.conf.gamma)
        if self.conf.brightness != None:
            self.img = kornia.enhance.adjust_brightness(self.img, self.conf.brightness)
        if self.conf.sharpness != None:
            self.img = kornia.enhance.sharpness(self.img, self.conf.sharpness)
        if self.conf.crop != None:
            self.img = self.img.unsqueeze(0)
            original_h, original_w = self.img.shape[:2]
            center = [original_w // 2, original_h // 2]
            self.img = kornia.geometry.transform.scale(self.img, torch.Tensor([self.conf.crop]), center=torch.Tensor([center]))
            self.img = self.img.squeeze(0)

def run_inference(image0: torch.Tensor, image1: torch.Tensor, com_license=True, preprocess_conf={}):
    """Given a slate and calibration image, return their respective features and matches
    Input:
        image0: torch.Tensor
        image1: torch.Tensor
    Output:
        TODO
    """ 
    # First round of preprocessing
    image1_preprocessed = Preprocess(image1, **preprocess_conf)

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_grad_enabled(False)

        # SuperPoint+LightGlue
        extractor_parameters = {'max_num_keypoints': None,}
        matcher_parameters = {
            'features': 'superpoint',
            'depth_confidence': -1,
            'width_confidence': -1,
            'filter_threshold': 0.4 # 0.4
        }
        extractor = (SuperPoint(**extractor_parameters).eval().to(device) if com_license
                    else NonCommercialSuperPoint(**extractor_parameters).eval().to(device))
        matcher = LightGlue(**matcher_parameters).eval().to(device)

        # Extract features
        if not com_license: print("    WARNING: USING THE NON-COMMERCIAL VERSION OF SUPERPOINT!")
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1_preprocessed.img.to(device))

        # Get matches
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

        # Return to original scale
        if image1_preprocessed.conf.crop != None:
            feats1['keypoints'] = feats1['keypoints'] / image1_preprocessed.conf.crop

    feats0_keypoints, feats1_keypoints, matches = feats0['keypoints'], feats1['keypoints'], matches01['matches']
    feats0_matches, feats1_matches = feats0_keypoints[matches[..., 0]], feats1_keypoints[matches[..., 1]]
    
    return feats0_matches, feats1_matches