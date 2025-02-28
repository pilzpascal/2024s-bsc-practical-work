from pandas.io.formats.style import subset_args
from tqdm.auto import tqdm

import torch
import scipy
import numpy as np

from src.data_loading_and_processing import get_subset


def predictive_entropy(outputs_mc, eps=1e-10) -> torch.Tensor:  # (T, B, C)
    mean_mc = outputs_mc.mean(dim=0)  # 0 is MC sample dim

    pred_entropy = -torch.sum(
        mean_mc * torch.log(mean_mc + eps),
        dim=-1  # -1 is class dim
    )

    return pred_entropy


def mutual_information(outputs_mc, eps=1e-10) -> torch.Tensor:  # (T, B, C)
    pred_entropy = predictive_entropy(outputs_mc, eps=eps)

    mean_entropy = torch.mean(
        torch.sum(
            outputs_mc * torch.log(outputs_mc + eps),
            dim=-1  # -1 is class dim
        ),
        dim=0  # 0 is MC sample dim
    )

    mutual_info = pred_entropy + mean_entropy
    return mutual_info


def variation_ratios(outputs_mc) -> torch.Tensor:  # (T, B, C)
    preds = outputs_mc.argmax(dim=-1).numpy()  # -1 is class dim
    _, counts = scipy.stats.mode(preds, axis=0)  # count of the most common class (mode)
    N = preds.shape[0]  # total number of cases in all classes, T
    var_ratio = (1. - counts / N).squeeze()
    var_ratio = torch.Tensor(var_ratio)
    return var_ratio


def mean_standard_deviation(outputs_mc) -> torch.Tensor:  # (T, B, C)
    stds = outputs_mc.std(dim=0)  # 0 is MC sample dim
    mean_stds = stds.mean(dim=-1)  # -1 is class dim
    return mean_stds


def random(outputs_mc) -> torch.Tensor:  # (T, B, C)
    return torch.rand(outputs_mc.shape[1])  # 1 is batch dim


def get_info_and_predictions(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
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
    dataloader : torch.Tensor
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

    subset_inputs, subset_idx = get_subset(dataloader, subset)

    model.eval()
    with torch.no_grad():

        # Note: parallelizing this by tiling the batch does not speed it up. I tested it.
        subset_outputs_mc = []
        for _ in tqdm(range(num_mc_samples), disable=not show_pbar, desc='MC Dropout', leave=False):
            subset_outputs = model(subset_inputs, use_dropout=True)  # (B, C)
            subset_outputs_mc.append(torch.softmax(subset_outputs, dim=1))

    subset_outputs_mc = torch.stack(subset_outputs_mc)  # (T, B, C)
    preds = torch.mean(subset_outputs_mc, dim=0).argmax(dim=1)  # (T, B, C) -> (B, C) -> (B)
    infos = acquisition_function(subset_outputs_mc)  # (T, B, C) -> (B)

    return infos, preds, subset_idx
