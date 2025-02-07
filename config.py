"""
This file contains global variables.
"""

SEED = 1

DATA_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/data/mnist/'
MODEL_SAVE_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/models/'
EXPERIMENT_SAVE_PATH = './Experiment Results/'

TRAIN_SIZE = 20
VAL_SIZE = 100
POOL_SIZE = 60_000 - (TRAIN_SIZE + VAL_SIZE)

N_EPOCHS = 50

T = 8
N_ACQUISITION_STEPS = 2
N_SAMPLES_TO_ACQUIRE = 10  # 100

ACQ_FUNCS = [# 'predictive_entropy',
             'mutual_information',
             'variation_ratios',
             # 'mean_standard_deviation',
             # 'random'
             ]
