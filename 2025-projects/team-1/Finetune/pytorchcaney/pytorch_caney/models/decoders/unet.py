import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_factory import ModelFactory


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        #print(f"UP x shape: {x.shape}, skip shape: {skip.shape}")
        
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        #print(f"AFTER cat: {x.shape}")
        return self.conv(x)


@ModelFactory.decoder("unet")
class SwinUNet(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder

        self.decoder4 = DecoderBlock(4096, 2048, 2048)  # 4 → 8
        self.decoder3 = DecoderBlock(2048, 1024, 1024)  # 8 → 16
        self.decoder2 = DecoderBlock(1024, 512, 512)   # 16 → 32
        self.decoder1 = DecoderBlock(512,0, 256)    # 32 → 64

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 64 → 128
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        feats = self.encoder.extra_features(x)  # [32x32, 16x16, 8x8, 4x4]

        #for i, f in enumerate(feats):
        #    print(f"Stage {i}: {f.shape}")

        x4 = feats[3]  # 4x4
        x3 = self.decoder4(x4, feats[2])  # 8x8
        x2 = self.decoder3(x3, feats[1])  # 16x16
        x1 = self.decoder2(x2, feats[0])  # 32x32
        x0 = self.decoder1(x1, None)  # 64x64 
        out = self.final_up(x0)           # 
        #print("Shape after final up", out.shape)
        #out = self.head(out)

        return out
