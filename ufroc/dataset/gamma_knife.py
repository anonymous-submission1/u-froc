from pathlib import Path

import numpy as np
import pandas as pd
from skimage import measure
from dpipe.im.shape_ops import zoom
from dpipe.io import load
from dpipe.dataset.wrappers import Proxy

from ufroc.paths import GAMMA_KNIFE_MET_DATA_PATH


class GammaKnife:
    def __init__(self, data_path=GAMMA_KNIFE_MET_DATA_PATH, metadata_rpath='metadata_07Jun2022.csv',
                 index_col='SeriesInstanceUID', image_col='t1c', mask_col='target',
                 spacing_cols=('x', 'y', 'z'), n_tumors_col='n_tumors'):
        self.data_path = Path(data_path)
        self.metadata_rpath = metadata_rpath
        self.index_col = index_col
        self.image_col = image_col
        self.mask_col = mask_col
        self.spacing_cols = spacing_cols
        self.n_tumors_col = n_tumors_col

        self.df = pd.read_csv(self.data_path / metadata_rpath, index_col=index_col)
        self.ids = tuple(self.df.index.tolist())

    def load_image(self, i):
        return np.float32(load(self.data_path / self.df.loc[i][self.image_col]))

    def load_segm(self, i):
        return np.float32(load(self.data_path / self.df.loc[i][self.mask_col]) > 0.5)

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        return np.array([self.df[c].loc[i] for c in self.spacing_cols])

    def load_n_tumors(self, i):
        return self.df.loc[i][self.n_tumors_col]


class Change(Proxy):
    def _change(self, x, i):
        raise NotImplementedError

    def load_image(self, i):
        return self._change(self._shadowed.load_image(i), i)

    def load_segm(self, i):
        return np.float32(self._change(self._shadowed.load_segm(i), i) >= .5)


class Rescale3D(Change):
    def __init__(self, shadowed, new_voxel_spacing=1., order=1):
        super().__init__(shadowed)
        self.new_voxel_spacing = np.broadcast_to(new_voxel_spacing, 3).astype(float)
        self.order = order

    def _scale_factor(self, i):
        old_voxel_spacing = self._shadowed.load_spacing(i)
        scale_factor = old_voxel_spacing / self.new_voxel_spacing
        return np.nan_to_num(scale_factor, nan=1)

    def _change(self, x, i):
        return zoom(x, self._scale_factor(i), order=self.order)

    def load_spacing(self, i):
        old_spacing = self.load_orig_spacing(i)
        spacing = self.new_voxel_spacing.copy()
        spacing[np.isnan(spacing)] = old_spacing[np.isnan(spacing)]
        return spacing

    def load_orig_spacing(self, i):
        return self._shadowed.load_spacing(i)


class TumorCenters(Proxy):
    def __init__(self, shadowed):
        super().__init__(shadowed)

    def load_tumor_centers(self, i):
        labels, n_labels = measure.label(self.load_segm(i) > 0.5, connectivity=3, return_num=True)
        return np.int16([np.round(np.mean(np.argwhere(labels == i), axis=0)) for i in range(1, n_labels + 1)])
