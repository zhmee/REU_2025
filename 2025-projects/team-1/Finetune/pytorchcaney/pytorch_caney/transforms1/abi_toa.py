import torchvision.transforms as T
import torch.nn.functional as F
import torch

from .abi_toa_scale import MinMaxEmissiveScaleReflectance
from .abi_radiance_conversion import ConvertABIToReflectanceBT




class ResizeTensor:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, img):
        # img: tensor (C, H, W)
        img = img.unsqueeze(0)  # add batch dim
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        return img.squeeze(0)


# -----------------------------------------------------------------------------
# AbiToaTransform
# -----------------------------------------------------------------------------
class AbiToaTransform:
    """
    torchvision transform which transforms the input imagery into
    addition to generating a MiM mask
    """
    def __init__(self, img_size):
        self.img_size = (img_size, img_size)
        self.convert_bt = ConvertABIToReflectanceBT()
        self.scale_reflect = MinMaxEmissiveScaleReflectance()
        self.resize = ResizeTensor(self.img_size)

    def __call__(self, img):
        # img: numpy array (H, W, C), float32
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)

        img = self.convert_bt(img)
        img = self.scale_reflect(img)
        img = self.resize(img)

        return img  # tensor (C, img_size, img_size)
