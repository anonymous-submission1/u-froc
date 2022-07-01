from .utils import choose_root


# ########################################### SUPPLEMENTARY PATHS ############################################


LUNG_BBOXES_PATH = choose_root(
    # '/path/to/data', # put here path to bounding boxes
    '/',  # avoiding `FileNotFoundError`
)

EXP_BASE_PATH = choose_root(
    # '/path/to/repo', # put here local path to folder with experiments
    '/',  # avoiding `FileNotFoundError`
)

# ################################################ DATA PATHS ################################################


GAMMA_KNIFE_MET_DATA_PATH = choose_root(
    # '/path/to/data', # put here path to dataset
    '/',  # avoiding `FileNotFoundError`
)

EGD_DATA_PATH = choose_root(
    # '/path/to/data', # put here path to dataset
    '/',  # avoiding `FileNotFoundError`
)

LIDC_DATA_PATH = choose_root(
    # '/path/to/data', # put here path to dataset
    '/',  # avoiding `FileNotFoundError`
)

MIDRC_DATA_PATH = choose_root(
    # '/path/to/data', # put here path to dataset
    '/',  # avoiding `FileNotFoundError`
)

LITS_DATA_PATH = choose_root(
    # '/path/to/data', # put here path to dataset
    '/',  # avoiding `FileNotFoundError`
)

LITS_MOD_DATA_PATH = choose_root(
    # '/path/to/data', # put here path to dataset
    '/',  # avoiding `FileNotFoundError`
)
