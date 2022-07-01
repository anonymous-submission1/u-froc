import os
import argparse
from pathlib import Path
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd

import pylidc as pl
from pylidc.utils import consensus
from dpipe.io import save
from dicom_csv import (expand_volumetric, drop_duplicated_instances,
                       drop_duplicated_slices, order_series, stack_images,
                       get_slice_locations, get_pixel_spacing)


def get_image_from_dicom(series, series_instance_uid, drop_dupl_slices):
    series = expand_volumetric(series)
    series = drop_duplicated_instances(series)

    if drop_dupl_slices:
        _original_num_slices = len(series)
        series = drop_duplicated_slices(series)
        if len(series) < _original_num_slices:
            warnings.warn(f'Dropped duplicated slices for series {series_instance_uid}.')

    series = order_series(series)

    image = stack_images(series, -1).astype(np.int16)
    pixel_spacing = get_pixel_spacing(series).tolist()
    slice_locations = get_slice_locations(series)

    return image, pixel_spacing, slice_locations


def main(src, dst):
    src = Path(src)
    dst = Path(dst)
    os.makedirs(dst / 'images')
    os.makedirs(dst / 'masks')

    # write path to ~/.pylidcrc file (nesessary for pylidc work)
    data = f'[dicom]\npath = {src}'
    with open(os.path.expanduser("~/.pylidcrc"), "w") as text_file:
        text_file.write(data)

    meta = []

    for scan in tqdm(pl.query(pl.Scan).all()):

        _id = scan.series_instance_uid

        series = scan.load_all_dicom_images(verbose=False)
        image, pixel_spacing, slice_locations = get_image_from_dicom(series, _id, drop_dupl_slices=True)

        # calculate spacing
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])

        mask = np.zeros(image.shape, dtype=bool)
        n_tumors = 0
        for anns in scan.cluster_annotations():
            mask |= consensus(anns, pad=np.inf)[0]
            n_tumors = len(anns)

        mask = np.flip(mask, -1)

        assert mask.shape == image.shape

        # update metadata
        meta.append({'ID': _id, 'CT': f'images/{_id}.npy.gz', 'mask': f'masks/{_id}.npy.gz',
                     'x': spacing[0], 'y': spacing[1], 'z': spacing[2], 'n_tumors': n_tumors})

        save(image, dst / 'images' / f'{_id}.npy.gz', compression=1)
        save(mask, dst / 'masks' / f'{_id}.npy.gz', compression=1)

    meta = pd.DataFrame.from_records(meta, index='ID')
    meta.to_csv(dst / 'meta.csv', index_label='ID')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.input, args.output)
