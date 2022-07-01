import torch
from torch import nn

from dpipe import layers


class UNet(nn.Module):
    def __init__(self, init_bias: float = -3):
        super().__init__()
        self.init_bias = init_bias

        self.unet = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            layers.FPN(
                layer=layers.ResBlock3d,
                downsample=nn.MaxPool3d(2, ceil_mode=True),
                upsample=nn.Identity,
                merge=lambda left, down: torch.add(
                    *layers.interpolate_to_left(left, down, 'trilinear')),
                structure=[
                    [[8, 8, 8], [8, 8, 8]],
                    [[8, 16, 16], [16, 16, 8]],
                    [[16, 32, 32], [32, 32, 16]],
                    [[32, 64, 64], [64, 64, 32]],
                    [[64, 128, 128], [128, 128, 64]],
                    [[128, 256, 256], [256, 256, 128]],
                    [[256, 512, 512], [512, 512, 256]],
                    [[512, 1024, 1024], [1024, 1024, 512]],
                    [1024, 1024]
                ],
                kernel_size=3,
                padding=1
            ),
        )

        self.head = layers.PreActivation3d(8, 1, kernel_size=1)
        if init_bias is not None:
            self.head.layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    def forward_features(self, x):
        return self.unet(x)

    def forward(self, x):
        return self.head(self.forward_features(x))
