import numpy as np


def scale_mri(image: np.ndarray, q_min: int = 1, q_max: int = 99) -> np.ndarray:
    image = np.clip(np.float32(image), *np.percentile(np.float32(image), [q_min, q_max]))
    image -= np.min(image)
    image /= np.max(image)
    return np.float32(image)


def scale_ct(x: np.ndarray, min_value: float = -1350, max_value: float = 300) -> np.ndarray:
    x = np.clip(x, a_min=min_value, a_max=max_value)
    x -= np.min(x)
    x /= np.max(x)
    return np.float32(x)


def get_n_tumors(dataset, ids):
    return [dataset.load_n_tumors(i) for i in ids]
