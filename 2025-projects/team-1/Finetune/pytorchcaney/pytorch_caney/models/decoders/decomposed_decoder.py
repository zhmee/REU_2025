import torch.nn as nn

from ..model_factory import ModelFactory


@ModelFactory.decoder("DecomposedFcnDecoder")
class DecomposedFcnDecoder(nn.Module):
    def __init__(self, out_chans=64, num_features: int = 1024):
        super(DecomposedFcnDecoder, self).__init__()
        self.output_channels = out_chans
        self.decoder = nn.Sequential(
            DecomposedTransposedConv(num_features, 1024, 2048),   # 8x8 -> 16x16
            DecomposedTransposedConv(2048, 1024, 512),            # 16x16 -> 32x32
            DecomposedTransposedConv(512, 512, 256),              # 32x32 -> 64x64
            DecomposedTransposedConv(256, 256, 128),              # 64x64 -> 128x128
            DecomposedTransposedConv(128, 128, self.output_channels)  # 128x128 -> final
        )

    def forward(self, x):
        return self.decoder(x)


class DecomposedTransposedConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels,
                 kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.deconv_h = nn.ConvTranspose2d(
            in_channels, mid_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            output_padding=(output_padding, 0)
        )
        self.deconv_w = nn.ConvTranspose2d(
            mid_channels, out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            output_padding=(0, output_padding)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.deconv_h(x))
        x = self.relu(self.deconv_w(x))
        return x

