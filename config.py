"""
This file contains experiment parameters. Standard parameters are written as comment for each.
"""

WHICH_ACQ_FUNCS = [
    'predictive_entropy',
    'mutual_information',
    'variation_ratios',
    'mean_standard_deviation',
    'random',
]

# Experiment parameters
SEED_SEQUENCE = [1, 2, 3]  # [1, 2, 3]
RUN_ON_FULL = True  # True
N_RUNS = 3  # 3
TRAIN_SIZE = 20  # 20
VAL_SIZE = 1_000  # 1_000
DATA_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/data/mnist/'
MODEL_SAVE_PATH_BASE = '/Users/pascalpilz/Documents/Bsc Thesis/models/'
EXP_SAVE_PATH_BASE = './Experiment Results/'

# Active learning parameters
N_ACQUISITION_STEPS = 100  # 100
N_SAMPLES_TO_ACQUIRE = 10  # 10
POOL_SUBSET_SIZE = 5_000  # 5_000
TEST_SUBSET_SIZE = 3_000  # 3_000
NUM_MC_SAMPLES = 64  # 64

# Training parameters
N_EPOCHS = 400  # 200
EARLY_STOPPING = 50  # 50
WHICH_MODEL = 'LeNet'  # 'LeNet'
