from pathlib import Path
from types import SimpleNamespace

import cv2
import kornia
import numpy as np
import torch

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }

def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def load_image(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    return numpy_image_to_torch(read_image(path))

class Preprocessor():
    def __init__(self, **conf):
        self.default_conf = {
            'contrast': None,
            'gamma': 2.0,
            'brightness': None,
            'sharpness': 0.5,
            'scale': 1.6,
        }
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
    def __call__(self, img):
        if self.conf.contrast != None:
            img = kornia.enhance.adjust_contrast(img, self.conf.contrast)
        if self.conf.gamma != None:
            img = kornia.enhance.adjust_gamma(img, self.conf.gamma)
        if self.conf.brightness != None:
            img = kornia.enhance.adjust_brightness(img, self.conf.brightness)
        if self.conf.sharpness != None:
            img = kornia.enhance.sharpness(img, self.conf.sharpness)
        if self.conf.scale != None:
            img = img.unsqueeze(0)
            original_h, original_w = img.shape[:2]
            center = [original_w // 2, original_h // 2]
            img = kornia.geometry.transform.scale(img, torch.Tensor([self.conf.scale]), center=torch.Tensor([center]))
            img = img.squeeze(0)
        return img

class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
    @torch.no_grad()
    def extract(self, img: torch.Tensor, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1] # take last two and reverse order
        # resize to use with superpoint
        h, w = img.shape[-2:]
        img = kornia.geometry.transform.resize(
            img,
            1024,
            side='long',
            antialias=True,
            align_corners=None,
        )
        scales = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        return feats