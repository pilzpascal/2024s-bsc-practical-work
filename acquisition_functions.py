import torch
import numpy as np
from tqdm.auto import tqdm

import utils


def predictive_entropy(tensor_outputs, mean_outputs):

    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)

    return entropy_mean


def mutual_information(tensor_outputs, mean_outputs):

    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)
    mean_entropy = torch.mean(torch.sum(tensor_outputs * torch.log(tensor_outputs + 1e-8), dim=2), dim=0)
    mutual_info = entropy_mean + mean_entropy

    return mutual_info


def variation_ratios(tensor_outputs, mean_outputs):

    max_y = mean_outputs.max(dim=1).values
    var_ratio = torch.ones_like(max_y) - max_y
    var_ratio = var_ratio

    return var_ratio


def mean_standard_deviation(tensor_outputs, mean_outputs):

    stds = torch.std(tensor_outputs, dim=0)
    mean_stds = torch.mean(stds, dim=1)

    return mean_stds


def random(tensor_outputs, mean_outputs):
    return torch.rand(tensor_outputs.shape[1])


def get_info_and_predictions(model, X, acquisition_function, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    infos = []
    preds = []
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Mean Standard Deviation'):
        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

        infos += acquisition_function(tensor_outputs, mean_outputs).tolist()
        preds += predictions.tolist()

    return infos, preds, subset_idx


def perform_acquisition(infos, X_train, y_train, X_pool, y_pool, n_samples_to_acquire=10, subset_idx=None):

    if subset_idx is None:
        # TODO: take care of what happens if we do the acquisition only for a subset of the pool
        subset_idx = np.arange(X_pool.shape(0))

    idx = torch.topk(torch.Tensor(infos), n_samples_to_acquire).indices

    chosen_X_pool = X_pool[idx]
    chosen_y_pool = y_pool[idx]

    new_X_train = torch.concatenate([X_train, chosen_X_pool], 0)
    new_y_train = torch.concatenate([y_train, chosen_y_pool], 0)

    new_X_pool = X_pool[~torch.isin(torch.arange(len(X_pool)), idx)]
    new_y_pool = y_pool[~torch.isin(torch.arange(len(y_pool)), idx)]

    return new_X_train, new_y_train, new_X_pool, new_y_pool
