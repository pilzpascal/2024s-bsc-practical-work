import torch

import numpy as np


def perform_acquisition(model, acquisition_function,
                        X_train, y_train, X_pool, y_pool,
                        n_samples_to_acquire=10):

    vals = acquisition_function(model, X_pool)
    idx = torch.topk(vals, n_samples_to_acquire).indices

    chosen_X_pool = X_pool[idx]
    chosen_y_pool = y_pool[idx]

    new_X_train = torch.concatenate([X_train, chosen_X_pool], 0)
    new_y_train = torch.concatenate([y_train, chosen_y_pool], 0)

    new_X_pool = X_pool[~torch.isin(torch.arange(len(X_pool)), idx)]
    new_y_pool = y_pool[~torch.isin(torch.arange(len(y_pool)), idx)]

    return new_X_train, new_y_train, new_X_pool, new_y_pool


def random(model, X_pool):
    vals = torch.rand(X_pool.shape[0])
    return vals

def bald(mode, X_pool):
    pass
