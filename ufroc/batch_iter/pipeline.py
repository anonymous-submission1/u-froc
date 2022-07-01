import numpy as np
from dpipe.im.patch import sample_box_center_uniformly
from dpipe.im.box import get_centered_box
from dpipe.im.shape_ops import crop_to_box


SPATIAL_DIMS = (-3, -2, -1)


def sample_center_uniformly(shape, patch_size, spatial_dims, random_state):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size, random_state=random_state)
    else:
        return spatial_shape // 2


def center_choice_ts(inputs, y_patch_size, random_state: np.random.RandomState, nonzero_fraction: float = 0.5,
                     center_random_shift=(19, 19, 19)):
    x, y, centers = inputs

    y_patch_size = np.array(y_patch_size)
    center_random_shift = np.array(center_random_shift)

    if len(centers) > 0 and random_state.uniform() < nonzero_fraction:
        center = centers[random_state.choice(np.arange(len(centers)))]
        # shift augm:
        max_shift = y_patch_size // 2
        low = np.maximum(max_shift, center - center_random_shift)
        high = np.minimum(np.array(y.shape) - max_shift, center + center_random_shift + 1)
        center = center if np.any(low >= high) else random_state.randint(low=low, high=high, size=len(SPATIAL_DIMS))
    else:
        center = sample_center_uniformly(y.shape, patch_size=y_patch_size,
                                         spatial_dims=SPATIAL_DIMS, random_state=random_state)

    return x, y, center


def extract_patch(inputs, patch_sizes, padding_values, spatial_dims=SPATIAL_DIMS):
    *inputs, center = inputs
    return [crop_to_box(inp, box=get_centered_box(center, patch_size),
                        padding_values=padding_value, axis=spatial_dims)
            for inp, patch_size, padding_value in zip(inputs, patch_sizes, padding_values)]


def get_random_patch_of_slices(inputs, z_patch_size, random_state: np.random.RandomState):
    x, y = inputs
    z_min = random_state.randint(low=0, high=max(1, x.shape[-1] - z_patch_size))
    return x[..., z_min:z_min + z_patch_size], y[..., z_min:z_min + z_patch_size]


def center_crop(inputs, crop_shape=(256, 320)):
    x, y = inputs
    crop_shape = np.array(crop_shape)
    diff = np.maximum(x.shape[:-1] - crop_shape, 0)
    start, end = diff // 2, x.shape[:-1] - diff // 2 - diff % 2
    return x[start[0]:end[0], start[1]:end[1], ...], y[start[0]:end[0], start[1]:end[1], ...]
