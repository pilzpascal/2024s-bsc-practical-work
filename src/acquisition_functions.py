from tqdm.auto import tqdm

import torch
import numpy as np

from src.data_loading_and_processing import get_subset


def predictive_entropy(tensor_outputs, mean_outputs):
    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-10), dim=1)
    return entropy_mean


def mutual_information(tensor_outputs, mean_outputs):
    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-10), dim=1)
    mean_entropy = torch.mean(
        torch.sum(tensor_outputs * torch.log(tensor_outputs + 1e-10), dim=2),
        dim=0)
    mutual_info = entropy_mean + mean_entropy
    return mutual_info


def variation_ratios(tensor_outputs, mean_outputs):
    max_y = mean_outputs.max(dim=1).values
    var_ratio = 1. - max_y
    return var_ratio


def mean_standard_deviation(tensor_outputs, mean_outputs):
    stds = torch.std(tensor_outputs, dim=0)
    mean_stds = torch.mean(stds, dim=1)
    return mean_stds


def random(tensor_outputs, mean_outputs):
    return torch.rand(tensor_outputs.shape[1])


def get_info_and_predictions(model, data, acquisition_function,
                             subset: int | list | np.ndarray | None = None,
                             T: int = 64,
                             show_pbar: bool = False)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the information per sample for a given acquisition and predictions from the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use
    data : torch.utils.data.DataLoader
        The data to use
    acquisition_function : callable
        The acquisition function to use
    T : int
        The number of samples to use for MC Dropout
    subset : int, list, np.ndarray, or None
        The number of samples to use for the subset, or the indices of
        the subset to use, or None. If None, then the full dataset is used.
    show_pbar : bool
        Whether to show the tqdm progress bar or not

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The information, predictions, and subset indices
    """

    inputs, subset_idx = get_subset(data, subset)
    list_outputs = []
    model.eval()

    with torch.no_grad():

        for _ in tqdm(range(T), disable=not show_pbar, desc='MC Dropout', leave=False):
            list_outputs.append(torch.softmax(model(inputs, use_dropout=True), dim=1))

        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)

        preds = mean_outputs.argmax(dim=1)
        infos = acquisition_function(tensor_outputs, mean_outputs).tolist()

    return infos, preds, subset_idx
