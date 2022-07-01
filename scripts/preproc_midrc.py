import os
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

import numpy as np
import pandas as pd
from skimage.draw import polygon
import pydicom

import mdai
from dpipe.io import save
from dicom_csv import (expand_volumetric, drop_duplicated_instances, 
                       drop_duplicated_slices, order_series, stack_images, 
                       get_slice_locations, get_pixel_spacing, get_tag, join_tree)


def get_image_data_from_dicom(series, series_instance_uid, drop_dupl_slices):
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
    
    sop_uids = [str(get_tag(i, 'SOPInstanceUID')) for i in series]

    return image, pixel_spacing, slice_locations, sop_uids


def main(src, dst):
    src = Path(src)
    dst = Path(dst)
    os.makedirs(dst / 'images')
    os.makedirs(dst / 'masks')
    
    meta = []
    
    joined = join_tree(src / 'MIDRC-RICORD-1A', verbose=1)
    annotations = mdai.common_utils.json_to_dataframe(
        src / 'MIDRC-RICORD-1a_annotations_labelgroup_all_2020-Dec-8.json')['annotations']

    for series_uid, rows in tqdm(list(joined.groupby('SeriesInstanceUID'))):
        files = {str(src / 'MIDRC-RICORD-1A' / row.PathToFolder / row.FileName) for _, row in rows.iterrows()}
        series = list(map(pydicom.dcmread, files))
        
        try:
            image, pixel_spacing, slice_locations, sop_uids = get_image_data_from_dicom(series, series_uid, True)
        except Exception as e:
            print(f'Preparing ct {series_uid} failed with {e.__class__.__name__}: {str(e)}.')
            continue

        # calculate spacing
        diffs, counts = np.unique(np.round(np.diff(slice_locations), decimals=5), return_counts=True)
        spacing = np.float32([pixel_spacing[0], pixel_spacing[1], -diffs[np.argsort(counts)[-1]]])
        
        # masks
        mask = np.zeros(image.shape, dtype=bool)
        for label, _rows in annotations[(annotations.SeriesInstanceUID == series_uid) &
                                        (annotations.scope == 'INSTANCE')].groupby('labelName'):
            
            if label in ['Infectious opacity', 'Infectious TIB/micronodules']:
                new_mask = np.zeros(image.shape, dtype=bool)
                for _, row in _rows.iterrows():
                    slice_index = sop_uids.index(row['SOPInstanceUID'])
                    if row['data'] is None:
                        warnings.warn(
                            f'{label} annotations for series {series_uid} contains None for slice {slice_index}.')
                        continue
                    ys, xs = np.array(row['data']['vertices']).T[::-1]
                    new_mask[(*polygon(ys, xs, image.shape[:2]), slice_index)] = True

                if new_mask is not None:
                    mask |= new_mask
                    
        if mask.sum() == 0:
            print(f'CT {series_uid} has empty mask, skipping it')
            continue
        
        # update metadata
        meta.append({'ID': series_uid, 'CT': f'images/{series_uid}.npy.gz', 'mask': f'masks/{series_uid}.npy.gz',
                     'x': spacing[0], 'y': spacing[1], 'z': spacing[2]})

        save(image, dst / 'images' / f'{series_uid}.npy.gz', compression=1)
        save(mask, dst / 'masks' / f'{series_uid}.npy.gz', compression=1)

    meta = pd.DataFrame.from_records(meta, index='ID')
    meta.to_csv(dst / 'meta.csv', index_label='ID')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    main(args.input, args.output)
