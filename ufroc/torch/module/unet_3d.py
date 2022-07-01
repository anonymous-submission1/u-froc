import warnings
from functools import partial

import torch
from torch import nn
import numpy as np
from dpipe.layers.resblock import ResBlock2d, ResBlock3d, ResBlock


class UNet(nn.Module):
    def __init__(self, ndim: int = 3, n_chans_in: int = 1, n_chans_out: int = 1, n_filters_init: int = 16,
                 norm_module: str = 'bn', patch_size: tuple = None,
                 return_features_from: tuple = (3,), cat_features: bool = True, init_bias: float = None):
        super().__init__()

        # TODO: move to check warps
        if ndim not in (2, 3,):
            raise ValueError(f'`ndim` should be in (2, 3). However, {ndim} is given.')
        self.ndim = ndim

        self.n_chans_in = n_chans_in
        self.n_chans_out = n_chans_out
        self.n_features = n_filters_init
        self.norm_module = norm_module
        self.patch_size = patch_size

        self.return_features_from = return_features_from
        self.cat_features = cat_features
        self.init_bias = init_bias

        print(f'Features will be returned from {return_features_from}', flush=True)

        n = n_filters_init

        filters = (n, n, 2*n, 2*n, 4*n, 4*n, 8*n, 8*n, 16*n, 16*n, 8*n, 8*n, 4*n, 4*n, 2*n, 2*n, n, n, n, n_chans_out)
        scales = (1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 1)
        self.layer2filters = {i: f for i, f in enumerate(filters, start=1)}
        self.layer2scale = {i: d for i, d in enumerate(scales, start=1)}

        self.s1, self.s2, self.s4, self.s8, self.s16 = get_resizing_features_modules(ndim, 'x1')
        self.scale2module = {d: m for d, m in zip((1, 2, 4, 8, 16), (self.s1, self.s2, self.s4, self.s8, self.s16))}

        resblock = get_resblock_module(ndim=ndim, norm_module=norm_module)
        downsample = get_downsample(ndim=ndim)
        upsample = get_upsample(ndim=ndim)

        patch_sizes = {i: (div_int_probably_none(patch_size, s, norm_module=norm_module))
                       for i, s in enumerate(scales, start=1)}

        self.init_path = nn.Sequential(
            resblock(n_chans_in, n, kernel_size=3, padding=1, patch_size=patch_sizes[1]),                  # 1
            resblock(n, n, kernel_size=3, padding=1, patch_size=patch_sizes[2]),                           # 2
        )

        self.down1 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n, n * 2, kernel_size=3, padding=1, patch_size=patch_sizes[3]),                       # 3
            resblock(n * 2, n * 2, kernel_size=3, padding=1, patch_size=patch_sizes[4])                    # 4
        )

        self.down2 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 2, n * 4, kernel_size=3, padding=1, patch_size=patch_sizes[5]),                   # 5
            resblock(n * 4, n * 4, kernel_size=3, padding=1, patch_size=patch_sizes[6])                    # 6
        )

        self.down3 = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 4, n * 8, kernel_size=3, padding=1, patch_size=patch_sizes[7]),                   # 7
            resblock(n * 8, n * 8, kernel_size=3, padding=1, patch_size=patch_sizes[8])                    # 8
        )

        self.bottleneck = nn.Sequential(
            downsample(2, 2, ceil_mode=True),
            resblock(n * 8, n * 16, kernel_size=3, padding=1, patch_size=patch_sizes[9]),                  # 9
            resblock(n * 16, n * 16, kernel_size=3, padding=1, patch_size=patch_sizes[10]),                # 10
            resblock(n * 16, n * 8, kernel_size=3, padding=1, patch_size=patch_sizes[11]),                 # 11
            upsample(scale_factor=2, align_corners=True),
        )

        self.up3 = nn.Sequential(
            resblock(n * 8, n * 8, kernel_size=3, padding=1, patch_size=patch_sizes[12]),                  # 12
            resblock(n * 8, n * 4, kernel_size=3, padding=1, patch_size=patch_sizes[13]),                  # 13
            upsample(scale_factor=2, align_corners=True),
        )

        self.up2 = nn.Sequential(
            resblock(n * 4, n * 4, kernel_size=3, padding=1, patch_size=patch_sizes[14]),                  # 14
            resblock(n * 4, n * 2, kernel_size=3, padding=1, patch_size=patch_sizes[15]),                  # 15
            upsample(scale_factor=2, align_corners=True),
        )

        self.up1 = nn.Sequential(
            resblock(n * 2, n * 2, kernel_size=3, padding=1, patch_size=patch_sizes[16]),                  # 16
            resblock(n * 2, n, kernel_size=3, padding=1, patch_size=patch_sizes[17]),                      # 17
            upsample(scale_factor=2, align_corners=True),
        )

        self.out_path = nn.Sequential(
            resblock(n, n, kernel_size=3, padding=1, patch_size=patch_sizes[18]),                          # 18
            resblock(n, n, kernel_size=3, padding=1, patch_size=patch_sizes[19]),                          # 19
            resblock(n, n_chans_out, kernel_size=1, padding=0, bias=True, patch_size=patch_sizes[20]),     # 20
        )

        if self.init_bias is not None:
            self.out_path[2].conv_path[1].layer.bias = nn.Parameter(torch.tensor([init_bias], dtype=torch.float32))

    @staticmethod
    def forward_block(x, block):
        outputs = []
        for layer in block:
            x = layer(x)
            if isinstance(layer, ResBlock):
                outputs.append(x)
        return outputs, x

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if return_features:
            warnings.filterwarnings('ignore')
            xs_init, x_init = self.forward_block(x, self.init_path)
            xs_down1, x_down1 = self.forward_block(x_init, self.down1)
            xs_down2, x_down2 = self.forward_block(x_down1, self.down2)
            xs_down3, x_down3 = self.forward_block(x_down2, self.down3)
            xs_bottleneck, x_bottleneck = self.forward_block(x_down3, self.bottleneck)
            xs_up3, x_up3 = self.forward_block(x_bottleneck + x_down3, self.up3)
            xs_up2, x_up2 = self.forward_block(x_up3 + x_down2, self.up2)
            xs_up1, x_up1 = self.forward_block(x_up2 + x_down1, self.up1)
            xs_out, x_out = self.forward_block(x_up1 + x_init, self.out_path)
            warnings.filterwarnings('default')

            xs = [_x for _xs in (xs_init, xs_down1, xs_down2, xs_down3, xs_bottleneck, xs_up3, xs_up2, xs_up1, xs_out)
                  for _x in _xs]
            if self.cat_features:
                return_features = torch.cat([self.scale2module[self.layer2scale[l]](xs[l - 1])
                                             for l in self.return_features_from], dim=1)
            else:
                return_features = [xs[l] for l in self.return_features_from]

            return x_out, return_features

        else:
            warnings.filterwarnings('ignore')
            x0 = self.init_path(x)
            x1 = self.down1(x0)
            x2 = self.down2(x1)
            x3 = self.down3(x2)
            x3_up = self.up3(self.bottleneck(x3) + x3)
            x2_up = self.up2(x3_up + x2)
            x1_up = self.up1(x2_up + x1)
            x_out = self.out_path(x1_up + x0)
            warnings.filterwarnings('default')

            return x_out


# TODO: remove legacy


def get_resblock_module(ndim: int = 3, norm_module: str = 'bn'):
    norm_module = get_norm_module(ndim=ndim, norm_module=norm_module)
    _resblock = ResBlock2d if (ndim == 2) else ResBlock3d

    def resblock(*args, **kwargs):
        patch_size = kwargs.pop('patch_size', None)
        _resblock_normed = partial(_resblock, batch_norm_module=norm_module) if (patch_size is None) else\
            partial(_resblock, batch_norm_module=partial(norm_module, patch_size=patch_size))
        return _resblock_normed(*args, **kwargs)

    return resblock


def get_norm_module(ndim: int = 3, norm_module: str = 'bn'):
    if norm_module == 'bn':
        return nn.BatchNorm2d if (ndim == 2) else nn.BatchNorm3d
    elif norm_module == 'in':
        return nn.InstanceNorm2d if (ndim == 2) else nn.InstanceNorm3d
    elif norm_module == 'ln':
        return layer_norm
    elif norm_module == 'gn':
        return group_norm
    else:
        raise ValueError(f'Given `norm_module` ({norm_module}) is not supported.')


def get_downsample(ndim: int = 3):
    return nn.MaxPool2d if (ndim == 2) else nn.MaxPool3d


def get_upsample(ndim: int = 3):
    return partial(nn.Upsample, mode='bilinear' if (ndim == 2) else 'trilinear')


def layer_norm(num_features, patch_size):
    return nn.LayerNorm(normalized_shape=[num_features, *patch_size])


def group_norm(num_features, num_groups: int = 16):
    if num_features % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_features)


def div_int_probably_none(a, b, norm_module: str = 'bn'):
    return None if ((a is None) or (norm_module != 'ln')) else tuple([int(e) for e in (np.asarray(a) / b)])


def get_resizing_features_modules(ndim: int, resize_features_to: str):
    if ndim not in (2, 3, ):
        raise ValueError(f'`ndim` should be in (2, 3). However, {ndim} is given.')

    ds16 = nn.AvgPool2d(16, 16, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(16, 16, ceil_mode=True)
    ds8 = nn.AvgPool2d(8, 8, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(8, 8, ceil_mode=True)
    ds4 = nn.AvgPool2d(4, 4, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(4, 4, ceil_mode=True)
    ds2 = nn.AvgPool2d(2, 2, ceil_mode=True) if (ndim == 2) else nn.AvgPool3d(2, 2, ceil_mode=True)
    identity = nn.Identity()
    us2 = nn.Upsample(scale_factor=2, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us4 = nn.Upsample(scale_factor=4, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us8 = nn.Upsample(scale_factor=8, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)
    us16 = nn.Upsample(scale_factor=16, mode='bilinear' if (ndim == 2) else 'trilinear', align_corners=True)

    if resize_features_to == 'x16':
        return ds16, ds8, ds4, ds2, identity
    elif resize_features_to == 'x8':
        return ds8, ds4, ds2, identity, us2
    elif resize_features_to == 'x4':
        return ds4, ds2, identity, us2, us4
    elif resize_features_to == 'x2':
        return ds2, identity, us2, us4, us8
    elif resize_features_to == 'x1':
        return identity, us2, us4, us8, us16
    else:
        resize_features_to__options = ('x1', 'x2', 'x4', 'x8', 'x16')
        raise ValueError(f'`resize_features_to` should be in {resize_features_to__options}. '
                         f'However, {resize_features_to} is given.')
