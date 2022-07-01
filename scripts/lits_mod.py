import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.morphology import ball, binary_erosion
from dpipe.io import load, save

from ufroc.paths import LITS_DATA_PATH, LITS_MOD_DATA_PATH
from ct_augmentation import to_sinogram, from_sinogram
from ct_augmentation import apply_conv_filter, simulate_dose, simulate_ct_dose


# TODO: fill background (-2048) with old value


LIVER_MASK_ID = 1


def generate_balls(x, y, rs, background_value, mask_value):
    n = rs.randint(1, 3)
    radiuses = rs.randint(2, 5, size=n)

    mask = np.zeros_like(x) + background_value

    liver_mask = y == LIVER_MASK_ID
    liver_mask_boundary_voxels = np.array((np.int8(liver_mask) - np.int8(binary_erosion(liver_mask))).nonzero()).T
    liver_mask_boundary_voxels_idx = np.arange(len(liver_mask_boundary_voxels))

    for r in radiuses:
        b = ball(r) * mask_value
        # c = rs.randint([r + 1, ] * 3, np.asarray(mask.shape) - r - 1)
        c = liver_mask_boundary_voxels[rs.choice(liver_mask_boundary_voxels_idx)]
        mask[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1, c[2] - r:c[2] + r + 1] = b

    return mask


def main():
    data_path = LITS_DATA_PATH
    meta = pd.read_csv(data_path / 'meta.csv', index_col='ID')

    meta_zero = meta[meta['n_tumors'] == 0]

    output_path = LITS_MOD_DATA_PATH
    (output_path / 'images').mkdir(exist_ok=True)
    (output_path / 'masks').mkdir(exist_ok=True)
    # TODO: create `images` and `masks` directories

    rs = np.random.RandomState(0xBadFace)

    meta_zero.to_csv(output_path / 'meta.csv', index_label='ID')

    for _id in tqdm(meta_zero.index):
        r = meta_zero.loc[_id]

        # if (output_path / r['CT']).exists():
        #     continue

        x = load(data_path / r['CT'])
        y = load(data_path / r['mask'])

        save(y, output_path / r['mask'], compression=1)

        # low-dose + kernel:
        a, b = [(30, 3), (-0.7, 0.5)][rs.choice([0, 1])]
        intensity, sigma = rs.uniform(1e-3, 1), rs.uniform()
        x = simulate_ct_dose(x, intensity=intensity, sigma=sigma, a=a, b=b, axes=(0, 1), random_state=rs)

        # add artifacts:
        fill_value = -2048
        artifact_intensity = 3000
        artifacts = generate_balls(x, y, rs, background_value=fill_value, mask_value=artifact_intensity)

        x_mod = np.copy(x)
        x_mod[artifacts == artifact_intensity] = artifact_intensity

        x_sinogram = to_sinogram(x, axes=(0, 1))
        a_sinogram = to_sinogram(artifacts, axes=(0, 1))
        z_sinogram = to_sinogram(np.zeros_like(x) + fill_value, axes=(0, 1))

        a_sinogram = a_sinogram - z_sinogram

        fill_max_scale = rs.uniform(0.7, 0.8)
        x_mod_sinogram = to_sinogram(x_mod, axes=(0, 1))
        x_sinogram[a_sinogram > a_sinogram.min()] = fill_max_scale * x_mod_sinogram.max()

        x = from_sinogram(x_sinogram, fill_value=fill_value, axes=(0, 1))

        save(x, output_path / r['CT'], compression=1)


if __name__ == '__main__':
    main()
