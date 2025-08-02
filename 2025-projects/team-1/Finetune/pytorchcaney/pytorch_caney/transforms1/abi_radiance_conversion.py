import numpy as np
import torch


# -----------------------------------------------------------------------------
# vis_calibrate
# -----------------------------------------------------------------------------
def vis_calibrate(data):
    """Calibrate visible channels to reflectance."""
    solar_irradiance = np.array(2017)
    esd = np.array(0.99)
    factor = np.pi * esd * esd / solar_irradiance

    return data * np.float32(factor) * 100


# -----------------------------------------------------------------------------
# ir_calibrate
# -----------------------------------------------------------------------------
def ir_calibrate(data):
    """Calibrate IR channels to BT."""
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32)

    data = data.to(dtype=torch.float32)

    #fk1 = np.array(13432.1)
    #fk2 = np.array(1497.61)
    #bc1 = np.array(0.09102)
    #bc2 = np.array(0.99971)
    

    fk1 = torch.tensor(13432.1, dtype=data.dtype, device=data.device)
    fk2 = torch.tensor(1497.61, dtype=data.dtype, device=data.device)
    bc1 = torch.tensor(0.09102, dtype=data.dtype, device=data.device)
    bc2 = torch.tensor(0.99971, dtype=data.dtype, device=data.device)
    # if self.clip_negative_radiances:
    #     min_rad = self._get_minimum_radiance(data)
    #     data = data.clip(min=data.dtype.type(min_rad))

    res = (fk2 / np.log(fk1 / data + 1) - bc1) / bc2
    return res.to(torch.float32)


# -----------------------------------------------------------------------------
# ConvertABIToReflectanceBT
# -----------------------------------------------------------------------------
class ConvertABIToReflectanceBT(object):
    """
    Performs scaling of MODIS TOA data
    - Scales reflectance percentages to reflectance units (% -> (0,1))
    - Performs per-channel minmax scaling for emissive bands (k -> (0,1))
    """

    def __init__(self):

        self.reflectance_indices = [0, 1, 2, 3, 4, 6]
        self.emissive_indices = [5, 7, 8, 9, 10, 11, 12, 13]

    def __call__(self, img):
        # img: tensor (C,H,W), float32
        # convert to numpy

        img = img.permute(1,2,0).cpu().numpy() # H, W, C

        # Reflectance % to reflectance units
        img[:, :, self.reflectance_indices] = \
            vis_calibrate(img[:, :, self.reflectance_indices])

        # Brightness temp scaled to (0,1) range
        img[:, :, self.emissive_indices] = ir_calibrate(
            img[:, :, self.emissive_indices])

        # back to tensor
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).to(img.device)

        return img
