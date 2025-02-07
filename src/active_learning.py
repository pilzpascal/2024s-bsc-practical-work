from tqdm.auto import tqdm

import torch

from src.training_and_testing import test_model, get_trained_model
from src.acquisition_functions import get_info_and_predictions


def perform_acquisition(infos, X_train, y_train, X_pool, y_pool,
                        n_samples_to_acquire=10, subset_idx=None):

    # get indices of top n_samples_to_acquire elements of infos
    idx = torch.topk(torch.Tensor(infos), n_samples_to_acquire).indices
    # if infos was not calculated over a subset, then we use the full pool, otherwise use the subset
    if subset_idx is None:
        subset_idx = torch.arange(X_pool.shape(0))
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
                        acquisition_function, model_save_path,
                        n_acquisition_steps=100, n_samples_to_acquire=10,
                        T=64, early_stopping=10, n_epochs=100,
                        pool_subset_size=None, test_subset_size=None):

    # we only return the test accuracies
    # we need n_acquisition_steps+1 since the first iteration does not do an acquisition step
    test_info = torch.zeros(n_acquisition_steps+1)
    test_acc = torch.zeros(n_acquisition_steps+1)

    acq_func_name = acquisition_function.__name__
    training_model_save_path = model_save_path + f'{acq_func_name}/'

    for i in tqdm(range(n_acquisition_steps+1), leave=False,
                  desc=f'Acquisition Steps for {' '.join(acq_func_name.split('_')).title()}'):

        # get the running training set, with full batch learning
        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set))

        model = get_trained_model(train_loader=train_loader,
                                  val_loader=val_loader,
                                  n_epochs=n_epochs,
                                  early_stopping=early_stopping,
                                  model_save_path=training_model_save_path)

        info, acc, _ = test_model(model, test_loader, T=T, subset=test_subset_size, show_pbar=True)
        test_info[i] = info
        test_acc[i] = acc

        infos, _, subset_idx = get_info_and_predictions(model, X_pool, acquisition_function,
                                                        T=T, subset=pool_subset_size, show_pbar=True)
        X_train, y_train, X_pool, y_pool \
            = perform_acquisition(infos, X_train, y_train, X_pool, y_pool,
                                  n_samples_to_acquire=n_samples_to_acquire, subset_idx=subset_idx)

    return test_info, test_acc