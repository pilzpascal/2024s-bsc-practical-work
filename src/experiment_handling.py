import pickle
from tqdm.auto import tqdm
from datetime import datetime

import torch
import numpy as np

from src.data_loading_and_processing import get_datasets
from src.active_learning import run_active_learning
from src.acquisition_functions import (random,
                                       variation_ratios,
                                       mutual_information,
                                       predictive_entropy,
                                       mean_standard_deviation)


def get_experiments(which_acq_funcs: list[str],

                    seed: int,
                    n_runs: int,

                    train_size: int,
                    val_size: int,
                    n_acquisition_steps: int,
                    n_samples_to_acquire: int,
                    pool_subset_size: int,
                    test_subset_size: int,
                    T: int,

                    n_epochs: int,
                    early_stopping: int,
                    model_save_path: str,
                    experiment_save_path: str):

    base_experiment = {'n_acquisition_steps': n_acquisition_steps,
                       'n_samples_to_acquire': n_samples_to_acquire,
                       'pool_subset_size': pool_subset_size,
                       'test_subset_size': test_subset_size,
                       'T': T,
                       'n_epochs': n_epochs,
                       'early_stopping': early_stopping,
                       'val_size': val_size,
                       'n_runs': n_runs,
                       'seed': seed}

    experiments = []
    for func in [random, variation_ratios, mutual_information, predictive_entropy, mean_standard_deviation]:
        if func.__name__ in which_acq_funcs:
            experiment_params = base_experiment.copy()
            experiment_params['acquisition_function'] = func
            experiments.append(experiment_params)

    return experiments


def run_experiments(experiments,
                    model_save_path,
                    experiment_save_path,
                    data_path):

    exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_save_path = experiment_save_path + f'expID-{exp_id}'

    for experiment in tqdm(experiments, desc='Experiments'):

        n_runs = experiment.pop('n_runs')
        seed = experiment.pop('seed')
        val_size = experiment.pop('val_size')

        # we need +1 since for the first iteration we don't have an acquisition step yet
        n_acq_steps = experiment['n_acquisition_steps'] + 1
        # for each experiment we perform three runs and average the results
        accuracies = torch.zeros((n_runs, n_acq_steps))
        infos = torch.zeros((n_runs, n_acq_steps))

        torch.manual_seed(seed)
        np.random.seed(seed)

        for i in tqdm(range(n_runs), desc='Runs per Experiment', leave=False):

            # get dataset new for each run, such that train and pool is shuffled newly
            X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test = get_datasets(val_size=val_size,
                                                                                          data_path=data_path)

            # get val and test set and loader, these stay constant
            val_set = torch.utils.data.TensorDataset(X_val, y_val)
            test_set = torch.utils.data.TensorDataset(X_test, y_test)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set))
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

            mod_save_path = model_save_path + f'expID-{exp_id}/run-{i}/'
            info, acc = run_active_learning(X_train, y_train,
                                            X_pool, y_pool,
                                            val_loader, test_loader,
                                            model_save_path=mod_save_path,
                                            **experiment)

            accuracies[i] = acc
            infos[i] = info

        # we add this information later otherwise we'd get an issue when we pass **experiment to run_active_learning
        experiment.update({'results': {'test_acc': accuracies, 'test_info': infos},
                           'val_size': val_size,
                           'seed': seed,
                           'n_runs': n_runs,
                           'model_save_path': model_save_path,
                           'experiment_save_path': experiment_save_path})

        # everything after each experiment is done
        with open(exp_save_path, 'wb') as handle:
            pickle.dump(experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # TODO: implement upper bound for acc and lower bound for mutual info
    experiments.append({'test_acc_ubound': None,
                        'test_inf_lbound': None})

    with open(exp_save_path, 'wb') as handle:
        pickle.dump(experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return experiments, exp_save_path
