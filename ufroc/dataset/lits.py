from pathlib import Path

import numpy as np
import pandas as pd
from dpipe.im.box import mask2bounding_box, add_margin, limit_box
from dpipe.io import load
from dpipe.im.shape_ops import crop_to_box

from ufroc.dataset.gamma_knife import Change
from ufroc.paths import LITS_DATA_PATH


class LiTS:
    def __init__(self, data_path=LITS_DATA_PATH, metadata_rpath='meta.csv',
                 index_col='ID', image_col='CT', mask_col='mask',
                 spacing_cols=('x', 'y', 'z'), n_tumors_col='n_tumors'):
        self.liver_mask_id = 1
        self.tumor_mask_id = 2

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
        return np.float32(load(self.data_path / self.df.loc[i][self.mask_col]) == self.tumor_mask_id)

    def load_shape(self, i):
        return np.int32(np.shape(self.load_segm(i)))

    def load_spacing(self, i):
        return np.array([self.df[c].loc[i] for c in self.spacing_cols])

    def load_n_tumors(self, i):
        return self.df.loc[i][self.n_tumors_col]

    def load_liver_mask(self, i):
        return np.float32(load(self.data_path / self.df.loc[i][self.mask_col]) == self.liver_mask_id)


class CropToLiver(Change):
    def __init__(self, shadowed, margin=1):
        super().__init__(shadowed)
        self.margin = margin

    def _change(self, x, i):
        liver_mask = self._shadowed.load_liver_mask(i)
        box = limit_box(add_margin(mask2bounding_box(liver_mask), margin=self.margin), limit=liver_mask.shape)
        return crop_to_box(x, box)

    def load_orig_image(self, i):
        return self._shadowed.load_image(i)

    def load_orig_segm(self, i):
        return self._shadowed.load_segm(i)
