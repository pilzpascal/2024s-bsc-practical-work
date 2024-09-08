import torch


def predictive_entropy(tensor_outputs, mean_outputs):
    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)
    return entropy_mean


def mutual_information(tensor_outputs, mean_outputs):
    entropy_mean = -torch.sum(mean_outputs * torch.log(mean_outputs + 1e-8), dim=1)
    mean_entropy = torch.mean(
        torch.sum(tensor_outputs * torch.log(tensor_outputs + 1e-8), dim=2),
        dim=0)
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
