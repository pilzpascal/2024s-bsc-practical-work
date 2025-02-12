from tqdm.auto import tqdm

import torch

from src.training_and_testing import test_model, get_trained_model
from src.acquisition_functions import get_info_and_predictions
from src.acquisition_functions import (random,
                                       variation_ratios,
                                       mutual_information,
                                       predictive_entropy,
                                       mean_standard_deviation)
acq_funcs = {
    func.__name__: func for func in
    [random, variation_ratios, mutual_information, predictive_entropy, mean_standard_deviation]
}


def perform_acquisition(infos,
                        X_train, y_train,
                        X_pool, y_pool,
                        n_samples_to_acquire: int,
                        subset_idx: int | None
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # get indices of top n_samples_to_acquire elements of infos
    idx = torch.topk(torch.Tensor(infos), n_samples_to_acquire).indices

    # if infos was not calculated over a subset (i.e. subset_idx=None) use the full pool, otherwise use the subset
    if subset_idx is None:
        subset_idx = torch.arange(X_pool.shape[0])
    subset_idx = torch.Tensor(subset_idx).int()

    # divide the pool set into the same subset used to calculate infos
    subset_X_pool = X_pool[subset_idx]
    subset_y_pool = y_pool[subset_idx]
    remaining_X_pool = X_pool[~torch.isin(torch.arange(len(X_pool)), subset_idx)]
    remaining_y_pool = y_pool[~torch.isin(torch.arange(len(y_pool)), subset_idx)]

    # from subset choose the ones corresponding to the index gotten from infos
    chosen_subset_X_pool = subset_X_pool[idx]
    chosen_subset_y_pool = subset_y_pool[idx]
    not_chosen_subset_X_pool = subset_X_pool[~torch.isin(torch.arange(len(subset_X_pool)), idx)]
    not_chosen_subset_y_pool = subset_y_pool[~torch.isin(torch.arange(len(subset_X_pool)), idx)]

    # recombine new pool and train sets
    new_X_train = torch.concatenate([X_train, chosen_subset_X_pool], 0)
    new_y_train = torch.concatenate([y_train, chosen_subset_y_pool], 0)
    new_X_pool = torch.concatenate([remaining_X_pool, not_chosen_subset_X_pool], 0)
    new_y_pool = torch.concatenate([remaining_y_pool, not_chosen_subset_y_pool], 0)

    return new_X_train, new_y_train, new_X_pool, new_y_pool


def run_active_learning(X_train, y_train,
                        X_pool, y_pool,
                        val_loader, test_loader,
                        acquisition_function_name: str,
                        model_save_path_base: str,

                        n_acquisition_steps: int,
                        n_samples_to_acquire: int,
                        pool_subset_size: int,
                        test_subset_size: int,
                        num_mc_samples: int,

                        n_epochs: int,
                        early_stopping: int,
                        ) -> tuple[list, list]:

    test_inf = []
    test_acc = []

    acquisition_function = acq_funcs[acquisition_function_name]
    model_save_path_acq = model_save_path_base + f'{acquisition_function_name}/'

    # we need n_acquisition_steps+1 since the first iteration does not do an acquisition step
    for _ in tqdm(range(n_acquisition_steps+1), leave=False,
                  desc=f'Acquisition Steps for {' '.join(acquisition_function_name.split('_')).title()}'):

        model = get_trained_model(X_train, y_train,
                                  val_loader=val_loader,
                                  n_epochs=n_epochs,
                                  early_stopping=early_stopping,
                                  model_save_path_base=model_save_path_acq)

        inf, acc = test_model(model, test_loader, num_mc_samples, subset=test_subset_size, show_pbar=True)
        test_inf.append(inf)
        test_acc.append(acc)

        pool_infos, _, subset_idx = get_info_and_predictions(
            model, X_pool,
            acquisition_function, num_mc_samples=num_mc_samples,
            subset=pool_subset_size, show_pbar=True
        )

        X_train, y_train, X_pool, y_pool = perform_acquisition(
            pool_infos, X_train, y_train, X_pool, y_pool,
            n_samples_to_acquire=n_samples_to_acquire, subset_idx=subset_idx
        )

    return test_inf, test_acc
