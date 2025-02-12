import yaml
from tqdm.auto import tqdm
from datetime import datetime

import torch
import numpy as np

from src.reproducibility import set_seed
from src.training_and_testing import train_and_test_full_dataset
from src.data_loading_and_processing import get_datasets
from src.active_learning import run_active_learning


def get_experiment(
        # experiment parameters
        which_acq_funcs: list[str],
        seed: int,
        n_runs: int,
        train_size: int,
        val_size: int,
        data_path: str,
        exp_save_path_base: str,
        model_save_path_base: str,

        # active learning parameters
        n_acquisition_steps: int,
        n_samples_to_acquire: int,
        pool_subset_size: int,
        test_subset_size: int,
        num_mc_samples: int,

        # training parameters
        n_epochs: int,
        early_stopping: int,

        # (optional) precomputed test bounds
        test_inf: list[float] | None = None,
        test_acc: list[float] | None = None
) -> dict:
    """
    Initializes an experiment configuration dictionary.

    Parameters
    ----------
    which_acq_funcs : List[str]
        List of acquisition function names.
    seed : int
        Random seed for reproducibility.
    n_runs : int
        Number of experiment repetitions.
    train_size : int
        Number of training samples.
    val_size : int
        Number of validation samples.
    data_path : str
        Path to the dataset.
    exp_save_path_base : str
        Directory to save experiment results.
    model_save_path_base : str
        Path to save trained models.

    n_acquisition_steps : int
        Number of acquisition steps in active learning.
    n_samples_to_acquire : int
        Number of samples to acquire per acquisition step.
    pool_subset_size : int
        Subset size of the pool for acquisition.
    test_subset_size : int
        Number of test samples.
    num_mc_samples : int
        Number of Monte Carlo dropout samples.

    n_epochs : int
        Number of training epochs.
    early_stopping : int
        Early stopping patience.

    test_inf : Optional[list[float]], default=None
        Test mutual information lower bound per run in n_runs (if precomputed).
        If None, the mutual information is computed during the experiment.
    test_acc : Optional[list[float]], default=None
        Test accuracy upper bound per run in n_runs (if precomputed).
        If None, the accuracy is computed during the experiment.

    Returns
    -------
    dict
        A structured dictionary containing experiment parameters and placeholders for results.
    """

    if test_inf is None or test_acc is None:
        test_inf = []
        test_acc = []
    else:
        assert len(test_inf) == n_runs
        assert len(test_acc) == n_runs

    experiment = {

        'params': {
            # stores experiment parameters
            'exp': {
                'which_acq_funcs': which_acq_funcs,
                'seed': seed,
                'n_runs': n_runs,
                'train_size': train_size,
                'val_size': val_size,
                'data_path': data_path,
                'exp_save_path_base': exp_save_path_base,
                'model_save_path_base': model_save_path_base,
            },
            # stored active learning parameters
            'al': {
                'n_acquisition_steps': n_acquisition_steps,
                'n_samples_to_acquire': n_samples_to_acquire,
                'pool_subset_size': pool_subset_size,
                'test_subset_size': test_subset_size,
                'num_mc_samples': num_mc_samples,
            },
            # stored training parameters
            'train': {
                'n_epochs': n_epochs,
                'early_stopping': early_stopping,
            },
        },

        'results': {
            # stores the results for each acquisition function
            'acq': {
                func_name: {
                    'test_inf': [],
                    'test_acc': []
                } for func_name in which_acq_funcs
            },
            # bounds are obtained by training on the full training set
            'bounds': {
                'test_inf': test_inf,
                'test_acc': test_acc
            }
        }

    }

    return experiment


def save_experiment(experiment, filename):

    # convert numpy arrays to lists for YAML compatibility
    def _convert_numpy(obj):
        # convert arrays to lists
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        # recursively convert dicts
        elif isinstance(obj, dict):
            return {k: _convert_numpy(v) for k, v in obj.items()}
        # recursively convert lists
        elif isinstance(obj, list):
            return [_convert_numpy(v) for v in obj]
        # return as-is if not numpy
        return obj

    experiment = _convert_numpy(experiment)

    with open(filename, 'w') as f:
        yaml.dump(experiment, f, default_flow_style=False, sort_keys=False)


def load_experiment(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(experiment):

    exp_params = experiment['params']['exp']

    set_seed(exp_params['seed'])

    exp_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_save_path = exp_params['exp_save_path_base'] + exp_id + '.yaml'

    # for each experiment we perform three runs
    for i in tqdm(range(exp_params['n_runs']), desc='Runs per Experiment', leave=False):

        model_save_path = exp_params['model_save_path_base'] + exp_id + f'/run-{i}/'

        # get dataset new for each run, such that train and pool is shuffled newly
        X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test \
            = get_datasets(data_path=exp_params['data_path'],
                           init_train_size=exp_params['train_size'],
                           val_size=exp_params['val_size'])

        # get val and test set and loader, these stay constant
        val_set = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set))
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

        # if the test bounds are not precomputed, we train on the full training set
        if len(experiment['results']['bounds']['test_inf']) != exp_params['n_runs']\
                or len(experiment['results']['bounds']['test_acc']) != exp_params['n_runs']:

            inf, acc = train_and_test_full_dataset(X_train, y_train,
                                                   X_pool, y_pool,
                                                   val_loader=val_loader,
                                                   test_loader=test_loader,
                                                   model_save_path=model_save_path,
                                                   num_mc_samples=experiment['params']['al']['num_mc_samples'],
                                                   **experiment['params']['train'])
            experiment['results']['bounds']['test_inf'].append(inf)
            experiment['results']['bounds']['test_acc'].append(acc)
            save_experiment(experiment, exp_save_path)

        # perform active learning for each acquisition function
        for acq_func_name in tqdm(exp_params['which_acq_funcs'], desc='Experiments'):

            acq_func = acq_funcs[acq_func_name]
            inf, acc = run_active_learning(X_train, y_train,
                                           X_pool, y_pool,
                                           val_loader, test_loader,
                                           acquisition_function=acq_func,
                                           model_save_path_base=model_save_path,
                                           **experiment['params']['al'],
                                           **experiment['params']['train'],
                                           )

            experiment['results']['acq'][acq_func_name]['test_inf'] = inf
            experiment['results']['acq'][acq_func_name]['test_acc'] = acc
            save_experiment(experiment, exp_save_path)

    return experiment
