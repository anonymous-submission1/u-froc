import warnings
import random
from pathlib import Path

import numpy as np
import torch
from dpipe.io import PathLike


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_root(*paths: PathLike) -> Path:
    for path in paths:
        path = Path(path)
        if path.exists():
            return path
    raise FileNotFoundError('No appropriate root found.')


def np_sigmoid(x):
    """Applies sigmoid function to the incoming value(-s)."""
    warnings.filterwarnings('ignore')
    y = 1 / (1 + np.exp(-x))
    warnings.filterwarnings('default')
    return y
