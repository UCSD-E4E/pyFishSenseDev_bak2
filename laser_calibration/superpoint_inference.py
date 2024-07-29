# Commercial inference use of SuperPoint+LightGlue

import torch

from utils import rbd
from models.lightglue import LightGlue
from models.superpoint_pytorch import SuperPoint
from models.superpoint import SuperPoint as NonCommercialSuperPoint

def run_inference(image0: torch.Tensor, image1: torch.Tensor, com_license=True):
    """Given a slate and calibration image, return their respective features and matches

    B: batch size
    M: number of keypoints (feats0)
    N: number of keypoints (feats1)
    D: dimensionality of descriptors
    C: number of channels
    H: height
    W: width

    Input:
        image0: torch.Tensor
        image1: torch.Tensor

    Output:
        feats0: dict
            keypoints: [B x M x 2]
            descriptors: [B x M x D]
            image: [B x C x H x W]
            image_size: [B x 2]
        feats1: dict
            keypoints: [B x N x 2]
            descriptors: [B x N x D]
            image: [B x C x H x W]
            image_size: [B x 2]
        matches01: dict
            matches0: [B x M]
            matching_scores0: [B x M]
            matches1: [B x N]
            matching_scores1: [B x N]
            matches: List[[Si x 2]]
            scores: List[[Si]]
            stop: int
            prune0: [B x M]
            prune1: [B x N]
    """

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_grad_enabled(False)

        # SuperPoint+LightGlue
        extractor_parameters = {'max_num_keypoints': None,}
        matcher_parameters = {
            'features': 'superpoint',
            'depth_confidence': -1,
            'width_confidence': -1,
            'filter_threshold': 0.1 # 0.4
        }
        extractor = (SuperPoint(**extractor_parameters).eval().to(device) if com_license
                    else NonCommercialSuperPoint(**extractor_parameters).eval().to(device))
        matcher = LightGlue(**matcher_parameters).eval().to(device)

        # Extract features
        if not com_license: print("    WARNING: USING THE NON-COMMERCIAL VERSION OF SUPERPOINT!")
        feats0 = extractor.extract(image0.to(device))
        feats1 = extractor.extract(image1.to(device))

        # Get matches
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension

    return feats0, feats1, matches01