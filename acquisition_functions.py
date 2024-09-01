import torch

import numpy as np


def perform_acquisition(model, acquisition_function,
                        X_train, y_train, X_pool, y_pool,
                        n_samples_to_acquire=10):
    idx = acquisition_function(model, X_pool, n_samples_to_acquire=n_samples_to_acquire)

    chosen_X_pool = X_pool[idx]
    chosen_y_pool = y_pool[idx]

    new_X_train = torch.concatenate([X_train, chosen_X_pool], 0)
    new_y_train = torch.concatenate([y_train, chosen_y_pool], 0)

    new_X_pool = X_pool[~torch.isin(torch.arange(len(X_pool)), idx)]
    new_y_pool = y_pool[~torch.isin(torch.arange(len(y_pool)), idx)]

    return new_X_train, new_y_train, new_X_pool, new_y_pool


def random(model, X_pool, n_samples_to_acquire=10):
    idx = np.random.choice(range(len(X_pool)), size=n_samples_to_acquire, replace=False)
    idx = torch.from_numpy(idx)
    return idx
