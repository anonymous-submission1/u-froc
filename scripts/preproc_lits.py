import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nb
from tqdm import tqdm
from skimage.measure import label

from dpipe.io import save


# BACKGROUND_MASK_ID = 0
# LIVER_MASK_ID = 1
TUMOR_MASK_ID = 2

NOT_FLIP_IDS = list(map(str, range(28, 48)))  # manually checked ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path_raw', type=str, required=True)
    parser.add_argument('-o', '--path_prep', type=str, required=True)
    args = parser.parse_known_args()[0]

    src = Path(args.path_raw)
    dst = Path(args.path_prep)

    os.makedirs(dst / 'images')
    os.makedirs(dst / 'masks')

    records = []
    for batch_id, rpath in enumerate(['media/nas/01_Datasets/CT/LITS/Training Batch 1',
                                      'media/nas/01_Datasets/CT/LITS/Training Batch 2'], start=1):
        all_fnames = [fname for fname in os.listdir(src / rpath)]
        ids = np.unique(list(map(lambda s: s.strip('.nii').split('-')[-1], all_fnames)))
        id2paths = {_id: {'image': src / rpath / f'volume-{_id}.nii',
                          'mask': src / rpath / f'segmentation-{_id}.nii'}
                    for _id in ids}

        for _id, paths in tqdm(id2paths.items()):
            image_nii = nb.as_closest_canonical(nb.nifti1.Nifti1Image.from_filename(paths['image']))
            mask_nii = nb.as_closest_canonical(nb.nifti1.Nifti1Image.from_filename(paths['mask']))

            spacing = image_nii.header.get_zooms()

            if _id in NOT_FLIP_IDS:
                image = np.moveaxis(image_nii.get_fdata(), 1, 0)[::-1, :, ::-1].astype(np.int16)
                mask = np.moveaxis(mask_nii.get_fdata(), 1, 0)[::-1, :, ::-1].astype(np.int16)
            else:
                image = np.moveaxis(image_nii.get_fdata(), 1, 0)[::-1, ::-1, ::-1].astype(np.int16)
                mask = np.moveaxis(mask_nii.get_fdata(), 1, 0)[::-1, ::-1, ::-1].astype(np.int16)

            _, n_tumors = label(mask == TUMOR_MASK_ID, connectivity=3, return_num=True)

            image_rpath = f'images/{_id}.npy.gz'
            mask_rpath = f'masks/{_id}.npy.gz'
            records.append({'ID': f'lits_{_id}', 'CT': image_rpath, 'mask': mask_rpath, 'x': spacing[0],
                            'y': spacing[1], 'z': spacing[2], 'n_tumors': n_tumors, 'batch_id': batch_id})

            save(image, dst / image_rpath, compression=1)
            save(mask, dst / mask_rpath, compression=1)

    meta = pd.DataFrame.from_records(records, index='ID')
    meta.to_csv(dst / 'meta.csv', index_label='ID')


if __name__ == '__main__':
    main()
