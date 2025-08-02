import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

from datetime import datetime
from pytorch_lightning.callbacks import Callback

class CloudDataset(Dataset):
    def __init__(self, npz_paths, task, in_chans=16, transform=None):
        # Note: use glob.glob( path to directory with all the npz files ) later in the python file to make a list of the paths
        self.npz_paths = npz_paths
        self.transform = transform

        # These are the 16 input radiance bands EDIT right now just to make sure the code and everything works its just the first 14 bands
        self.in_chans = in_chans
        
        if task.lower() == "phasepred" or task.lower() == "phase pred":
            self.label_key = "l2_cloud_phase"
        elif task.lower() == "cloudmask" or task.lower() == "cloud mask":
            self.label_key = "l2_cloud_mask"
        elif task.lower() == "cod":
            self.label_key = "l2_cod"
        elif task.lower() == "cps":
            self.label_key = "l2_cps"
        elif task.lower() == "multitask" or task.lower() == "multitask2":
            self.label_key = "multitask"
        else:
            raise ValueError(f"Task {task} not supported for CloudDataset. Try phasepred, cloudmask, cod, cps, or multitask.")

        
        if self.in_chans == 14:
            # Drop ABI bands 8 and 13
            self.band_indices = [1,2,0,4,5,6,3,8,9,10,11,13,14,15]
            #self.band_indices = [i for i in range(16) if i not in (7,12)]
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
            #inputs = np.transpose(rad, (2, 0, 1)).astype(np.float32)  # shape (14, 128, 128)
            # ^ commented this out bc the transform expects H, W, C
            inputs = rad.astype(np.float32)

            inputs = torch.tensor(inputs)  # shape: [14, 128, 128]
            #print("Inputs shape:", inputs.shape)

            label = {}
            if self.label_key in ["l2_cloud_phase", "l2_cloud_mask"]:
                label = torch.tensor(data[self.label_key].astype(np.int64))  # classification
            elif self.label_key in ["l2_cod", "l2_cps"]:
                label = torch.tensor(np.log1p(data[self.label_key]).astype(np.float32))  # regression
                #inputs = inputs.permute(3, 0, 1, 2)
            elif self.label_key.lower() == "multitask":
                label["phase"] = torch.tensor(data["l2_cloud_phase"].astype(np.int64))
                label["mask"] = torch.tensor(data["l2_cloud_mask"].astype(np.int64))
                label["cod"] = torch.tensor(np.log1p(data["l2_cod"]).astype(np.float32))
                label["cps"] = torch.tensor(np.log1p(data["l2_cps"]).astype(np.float32))
            else:
                raise ValueError(f"Unexpected label key: {self.label_key}")

            
            if torch.isnan(inputs).any():
                print("NaNs in inputs after transform")
            if torch.isinf(inputs).any():
                print("Infs in inputs after transform")
            data.close()
        inputs = inputs.permute(2,0,1)
        return inputs, label#, str(path)


class InputConv(nn.Module):
    def __init__(self, in_channels=16, out_channels=14, hidden_dims=[64, 128],kernel_size=1,dropout=0):
        super().__init__()

        layers = []
        prev_dim = in_channels
        for h in hidden_dims:
            layers.append(nn.Conv2d(prev_dim, h, kernel_size=kernel_size,padding=kernel_size // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Conv2d(prev_dim, out_channels, kernel_size=kernel_size,padding= kernel_size // 2))  # final projection
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 16, H, W)
        #print("========== DEBUG: x shape in forward method of CNN:",x.shape)
        result = self.conv(x)
        #print("========== DEBUG: self.conv(x) shape", result.shape)
        return result  # (B, 14, H, W)


class InputMLP(nn.Module):
    def __init__(self, in_channels=16, out_channels=14, hidden_dims=[32]):
        super().__init__()
        
        layers = []
        prev_dim = in_channels
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim,h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim,out_channels))
        self.mlp = nn.Sequential(*layers)

    def forward(self,x):
        # x is (B, 16, 128, 128)
        #print(f"[ChannelMLP] x initial shape: {x.shape}")
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)         # (B, 128, 128, 16)

        
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)         # (B, 14, 128, 128) – convert back to format the encoder expects
        return x







class TimerLogger(Callback):
    def __init__(self, timer):
        self.timer = timer

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = datetime.now()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_duration = datetime.now() - self.epoch_start_time
        seconds = epoch_duration.total_seconds()
        pl_module.log("train_epoch_time_sec", seconds, prog_bar=False, on_epoch=True)

    def on_train_end(self, trainer, pl_module):
        # Log to the CSVLogger — must log inside training hooks
        train_time = self.timer.time_elapsed("train")
        logger = trainer.logger
        if logger is not None and hasattr(logger, "log_metrics"):
            logger.log_metrics({"total_train_time_sec": train_time}, step=trainer.global_step)
