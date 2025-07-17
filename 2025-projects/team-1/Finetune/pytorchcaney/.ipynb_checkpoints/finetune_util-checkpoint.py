import torch
import numpy as np
from torch.utils.data import Dataset

class CloudPhaseDataset(Dataset):
    def __init__(self, npz_paths,in_chans=14, transform=None):
        # Note: use glob.glob( path to directory with all the npz files ) later in the python file to make a list of the paths
        self.npz_paths = npz_paths
        self.transform = transform

        # These are the 16 input radiance bands EDIT right now just to make sure the code and everything works its just the first 14 bands
        self.in_chans = in_chans
        self.label_key = "l2_cloud_phase"  # <- your target variable

        if self.in_chans == 14:
            # Drop ABI bands 8 and 13
            self.band_indices = [i for i in range(16) if i not in (7,12)]
        elif self.in_chans == 16:
            self.band_indices = list(range(16))
        else:
            raise ValueError("Unsupported input band count. Must be 14 or 16")

    def __len__(self):
        return len(self.npz_paths)

    def __getitem__(self, idx):
        path = self.npz_paths[idx]
        with np.load(path) as data:
             
            # Stack bands into a [C, H, W] input tensor
            # 'rad' is shape (H, W, 16), select the right bands.
            rad = data['rad'][..., self.band_indices]  # shape is (128, 128, 14 or 16)

            # Rearrange to (C, H, W) for PyTorch
            inputs = np.transpose(rad, (2, 0, 1)).astype(np.float32)  # shape (14, 128, 128)


            inputs = torch.tensor(inputs)  # shape: [14, 128, 128]

            # Read the cloud phase mask
            label = torch.tensor(data[self.label_key].astype(np.int64))  # shape: [H, W]

            if self.transform:
                inputs = self.transform(inputs)

            data.close()

        return inputs, label
