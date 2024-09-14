from utils import run_experiments
from acquisition_functions import get_experiments

SEED = 1
MODEL_SAVE_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/models/'
EXPERIMENT_SAVE_PATH = './Experiment Results/'


experiments = get_experiments(which_acq_funcs=['predictive_entropy',
                                               'mutual_information',
                                               'variation_ratios',
                                               'mean_standard_deviation',
                                               'random'
                                               ],
                              n_acquisition_steps=100,
                              n_samples_to_acquire=10,
                              pool_subset_size=5_000,
                              test_subset_size=3_000,
                              T=64,
                              n_epochs=200,
                              early_stopping=-1,
                              val_size=1_000,
                              n_runs=3,
                              seed=SEED,)

experiment_results, save_path = run_experiments(experiments)
