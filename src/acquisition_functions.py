from tqdm.auto import tqdm

import torch
import scipy
import numpy as np

from src.data_loading_and_processing import get_subset


def predictive_entropy(tensor_outputs, eps=1e-10) -> torch.Tensor:  # (T, B, C)
    mean_outputs = torch.mean(tensor_outputs, dim=0)  # mean over MC samples
    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + eps), dim=-1)  # -1 is class dim
    return entropy_mean


def mutual_information(tensor_outputs, eps=1e-10) -> torch.Tensor:  # (T, B, C)
    mean_outputs = torch.mean(tensor_outputs, dim=0)  # mean over MC samples
    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + eps), dim=-1)  # -1 is class dim
    mean_entropy = torch.mean(
        torch.sum(tensor_outputs * torch.log(tensor_outputs + eps), dim=-1),  # -1 is class dim
        dim=0
    )
    mutual_info = entropy_mean + mean_entropy
    return mutual_info


def variation_ratios(tensor_outputs) -> torch.Tensor:  # (T, B, C)
    preds = tensor_outputs.argmax(dim=2).numpy()
    _, counts = scipy.stats.mode(preds, axis=0)  # count of the most common class
    N = preds.shape[0]  # total number of cases in all classes, T
    var_ratio = (1. - counts / N).squeeze()
    var_ratio = torch.Tensor(var_ratio)
    return var_ratio


def mean_standard_deviation(tensor_outputs) -> torch.Tensor:  # (T, B, C)
    stds = torch.std(tensor_outputs, dim=0)  # std over MC samples
    mean_stds = torch.mean(stds, dim=-1)  # -1 is class dim
    return mean_stds


def random(tensor_outputs) -> torch.Tensor:  # (T, B, C)
    return torch.rand(tensor_outputs.shape[1])  # 1 is batch dim


def get_info_and_predictions(
        model,
        data,
        acquisition_function: callable,
        subset: int | list | np.ndarray | None,
        num_mc_samples: int,
        show_pbar: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    num_mc_samples : int
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

    model.eval()
    with torch.no_grad():

        # Note: parallelizing this does not speed it up. I tested it.
        list_outputs = []
        for _ in tqdm(range(num_mc_samples), disable=not show_pbar, desc='MC Dropout', leave=False):
            list_outputs.append(
                torch.softmax(
                    model(inputs, use_dropout=True),
                    dim=1
                )
            )

    tensor_outputs = torch.stack(list_outputs, dim=0)
    preds = torch.mean(tensor_outputs, dim=0).argmax(dim=1)
    infos = acquisition_function(tensor_outputs)

    return infos, preds, subset_idx
