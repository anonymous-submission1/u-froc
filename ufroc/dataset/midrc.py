from pathlib import Path

import numpy as np
import pandas as pd
from dpipe.io import load

from ufroc.paths import MIDRC_DATA_PATH


class MIDRC:
    def __init__(self, data_path=MIDRC_DATA_PATH, metadata_rpath='meta.csv',
                 index_col='ID', image_col='CT', mask_col='mask',
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

    @staticmethod
    def load_n_tumors(i):
        return 1
