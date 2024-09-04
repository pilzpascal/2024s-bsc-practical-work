import torch
import numpy as np
from tqdm.auto import tqdm

import utils


def predictive_entropy(model, X, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    entropies = []
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Predictive Entropy'):

        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

        entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)

        entropies += entropy_mean.tolist()

    return torch.Tensor(entropies), predictions, subset_idx


def mutual_information(model, X, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    mutual_infos = []
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Mutual Information'):

        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

        entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)
        mean_entropy = torch.mean(torch.sum(tensor_outputs * torch.log(tensor_outputs + 1e-8), dim=2), dim=0)

        mutual_infos += (entropy_mean + mean_entropy).tolist()

    return torch.Tensor(mutual_infos), predictions, subset_idx


def variation_ratios(model, X, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    var_ratios = []
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Variation Ratios'):

        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

        max_y = mean_outputs.max(dim=1).values
        var_ratio = torch.ones_like(max_y) - max_y

        var_ratios += var_ratio.tolist()

    return torch.Tensor(var_ratios), predictions, subset_idx


def mean_standard_deviation(model, X, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    mean_stds = []
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Mean Standard Deviation'):

        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

        stds = torch.std(tensor_outputs, dim=0)

        mean_stds += torch.mean(stds, dim=1)

    return torch.Tensor(mean_stds), predictions, subset_idx


def random(model, X, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Mean Standard Deviation'):
        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

    return torch.rand(X[subset_idx].shape[0]), predictions, subset_idx


def get_info_and_predictions(model, X, acquisition_function, T=100, subset=None, show_pbar=False):

    dataloader, subset_idx = utils.get_subset_dataloader(subset, X)
    infos = []
    model.eval()

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Mean Standard Deviation'):
        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)
        predictions = mean_outputs.argmax(dim=1)

        infos = torch.concat(infos, acquisition_function(tensor_outputs, mean_outputs))

    return torch.Tensor(infos), predictions, subset_idx


def perform_acquisition(infos, X_train, y_train, X_pool, y_pool, n_samples_to_acquire=10):

    idx = torch.topk(infos, n_samples_to_acquire).indices

    chosen_X_pool = X_pool[idx]
    chosen_y_pool = y_pool[idx]

    new_X_train = torch.concatenate([X_train, chosen_X_pool], 0)
    new_y_train = torch.concatenate([y_train, chosen_y_pool], 0)

    new_X_pool = X_pool[~torch.isin(torch.arange(len(X_pool)), idx)]
    new_y_pool = y_pool[~torch.isin(torch.arange(len(y_pool)), idx)]

    return new_X_train, new_y_train, new_X_pool, new_y_pool
