"""
Inference use of SuperPoint+LightGlue
Usage: Look at examples from example_inference.ipynb and process_data.py
Due to licensing issues, make sure you are using the correct version of SuperPoint.
"""
import torch

from models.lightglue import LightGlue
from models.superpoint_pytorch import SuperPoint
from models.superpoint import SuperPoint as NonCommercialSuperPoint

from utils import Preprocessor
from utils import rbd

class ImageMatcher():
    def __init__(self, template: torch.Tensor, com_license=True, processing_conf={}):
        self.image0 = template
        self.feats0 = None
        self.com_license=com_license
        self.preprocess_conf=processing_conf['preprocess'] if 'preprocess' in processing_conf else {}
        self.matcher_conf=processing_conf['matcher'] if 'matcher' in processing_conf else {}
        self.extractor_conf=processing_conf['extractor'] if 'extractor' in processing_conf else {}
    def __call__(self, image1: torch.Tensor):
        """Given a calibration image, return its matches with the template.
        Input:
            image1: torch.Tensor
        Output:
            feats0_matches
            feats1_matches
        """ 
        # Preprocess our input image
        preprocessor = Preprocessor(**self.preprocess_conf)
        image1_preprocessed = preprocessor(image1)

        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            torch.set_grad_enabled(False)

            # SuperPoint+LightGlue
            def_extractor_conf = {'max_num_keypoints': None,}
            def_matcher_conf = {
                'features': 'superpoint',
                'depth_confidence': -1,
                'width_confidence': -1,
                'filter_threshold': 0.5 # 0.5
            }
            self.extractor_conf = {**def_extractor_conf, **self.extractor_conf}
            self.matcher_conf = {**def_matcher_conf, **self.matcher_conf}
            extractor = (SuperPoint(**self.extractor_conf).eval().to(device) if self.com_license
                        else NonCommercialSuperPoint(**self.extractor_conf).eval().to(device))
            matcher = LightGlue(**self.matcher_conf).eval().to(device)

            # Extract features
            if not self.com_license: print("    WARNING: USING THE NON-COMMERCIAL VERSION OF SUPERPOINT!")
            if self.feats0 == None: # so we don't have to extract template keypoints every time
                self.feats0 = extractor.extract(self.image0.to(device))
            else:
                for k,_ in self.feats0.items():
                    self.feats0[k] = torch.unsqueeze(self.feats0[k], 0) # undo the squeezing we did before
            feats1 = extractor.extract(image1_preprocessed.to(device))

            # Get matches
            matches01 = matcher({"image0": self.feats0, "image1": feats1})
            self.feats0, feats1, matches01 = [
                rbd(x) for x in [self.feats0, feats1, matches01]
            ]  # remove batch dimension

            # Return to original scale
            if preprocessor.conf.scale != None:
                feats1['keypoints'] = feats1['keypoints'] / preprocessor.conf.scale

        feats0_keypoints, feats1_keypoints, matches = self.feats0['keypoints'], feats1['keypoints'], matches01['matches']
        feats0_matches, feats1_matches = feats0_keypoints[matches[..., 0]], feats1_keypoints[matches[..., 1]]
        
        return feats0_matches, feats1_matches