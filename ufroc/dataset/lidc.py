from pathlib import Path

import numpy as np
import pandas as pd
# from skimage.segmentation import flood
from dpipe.io import load
from dpipe.im.shape_ops import crop_to_box  # , pad
# from dpipe.im.box import mask2bounding_box

from ufroc.dataset.gamma_knife import Change
from ufroc.paths import LIDC_DATA_PATH, LUNG_BBOXES_PATH


class LIDC:
    def __init__(self, data_path=LIDC_DATA_PATH, metadata_rpath='meta.csv',
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

    def load_n_tumors(self, i):
        return self.df.loc[i][self.n_tumors_col]


# class CropToLungs(Change):
#     def __init__(self, shadowed, lungs_threshold=-600, lungs_fraction_threshold=0.005):
#         super().__init__(shadowed)
#         self.lungs_threshold = lungs_threshold
#         self.lungs_fraction_threshold = lungs_fraction_threshold
#
#     def _bbox(self, i):
#         x = self._shadowed.load_image(i)
#         # find lungs
#         lungs_mask = x < self.lungs_threshold
#         # find air
#         air_mask = flood(pad(lungs_mask, padding=1, axis=(0, 1), padding_values=True),
#                          seed_point=(0, 0, 0))[1:-1, 1:-1]
#         lungs_mask = lungs_mask & ~air_mask
#
#         if not lungs_mask.any():
#             #             print(f'Warning: no lungs were found! Case: {i}', flush=True)
#             lungs_mask[0, 0, 0] = lungs_mask[-1, -1, -1] = True
#
#         box = mask2bounding_box(lungs_mask)
#         return box
#
#     def _change(self, x, i):
#         return crop_to_box(x, self._bbox(i))
#
#     def load_orig_image(self, i):
#         return self._shadowed.load_image(i)


class CropToLungs(Change):
    def __init__(self, shadowed, dataset_name='lidc', bboxes_path=LUNG_BBOXES_PATH):
        super().__init__(shadowed)
        self.lungs_bboxes = load(bboxes_path / f'{dataset_name}.json')

    def _change(self, x, i):
        return crop_to_box(x, self.lungs_bboxes[i])

    def load_orig_image(self, i):
        return self._shadowed.load_image(i)
