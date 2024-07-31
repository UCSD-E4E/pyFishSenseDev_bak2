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

class ImageMatcher():
    def __init__(self, template: torch.Tensor, com_license=True, processing_conf={}):
        self.image0 = template
        self.feats0 = None
        self.com_license=com_license
        self.preprocess_conf=processing_conf['preprocess']
        self.matcher_conf=processing_conf['matcher']
        self.extractor_conf=processing_conf['extractor']
    def process(self, image1: torch.Tensor):
        """Given a slate and calibration image, return their respective features and matches
        Input:
            image0: torch.Tensor
            image1: torch.Tensor
        Output:
            TODO
        """ 
        # First round of preprocessing
        image1_preprocessed = Preprocess(image1, **self.preprocess_conf)

        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch.set_grad_enabled(False)

            # SuperPoint+LightGlue
            def_extractor_conf = {'max_num_keypoints': None,}
            def_matcher_conf = {
                'features': 'superpoint',
                'depth_confidence': -1,
                'width_confidence': -1,
                'filter_threshold': 0.5 # 0.4
            }
            self.extractor_conf = {**def_extractor_conf, **self.extractor_conf}
            self.matcher_conf = {**def_matcher_conf, **self.matcher_conf}
            print(self.extractor_conf)
            print(self.matcher_conf)
            extractor = (SuperPoint(**self.extractor_conf).eval().to(device) if self.com_license
                        else NonCommercialSuperPoint(**self.extractor_conf).eval().to(device))
            matcher = LightGlue(**self.matcher_conf).eval().to(device)

            # Extract features
            if not self.com_license: print("    WARNING: USING THE NON-COMMERCIAL VERSION OF SUPERPOINT!")
            if self.feats0 == None: # so we don't have to extract template keypoints every time
                self.feats0 = extractor.extract(self.image0.to(device))
            else:
                for k,_ in self.feats0.items():
                    self.feats0[k] = torch.unsqueeze(self.feats0[k], 0)
            feats1 = extractor.extract(image1_preprocessed.img.to(device))

            # Get matches
            matches01 = matcher({"image0": self.feats0, "image1": feats1})
            self.feats0, feats1, matches01 = [
                rbd(x) for x in [self.feats0, feats1, matches01]
            ]  # remove batch dimension

            # Return to original scale
            if image1_preprocessed.conf.crop != None:
                feats1['keypoints'] = feats1['keypoints'] / image1_preprocessed.conf.crop

        feats0_keypoints, feats1_keypoints, matches = self.feats0['keypoints'], feats1['keypoints'], matches01['matches']
        feats0_matches, feats1_matches = feats0_keypoints[matches[..., 0]], feats1_keypoints[matches[..., 1]]
        
        return feats0_matches, feats1_matches