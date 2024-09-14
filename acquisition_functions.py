import torch
torch.use_deterministic_algorithms(mode=True)


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
    return var_ratio


def mean_standard_deviation(tensor_outputs, mean_outputs):
    stds = torch.std(tensor_outputs, dim=0)
    mean_stds = torch.mean(stds, dim=1)
    return mean_stds


def random(tensor_outputs, mean_outputs):
    return torch.rand(tensor_outputs.shape[1])


def get_experiments(which_acq_funcs: list[str], n_acquisition_steps: int, n_samples_to_acquire: int,
                    pool_subset_size: int, test_subset_size: int, T: int, n_epochs: int, early_stopping: int,
                    val_size: int, n_runs: int, seed: int):

    base_experiment = {'n_acquisition_steps': n_acquisition_steps,
                       'n_samples_to_acquire': n_samples_to_acquire,
                       'pool_subset_size': pool_subset_size,
                       'test_subset_size': test_subset_size,
                       'T': T,
                       'n_epochs': n_epochs,
                       'early_stopping': early_stopping,
                       'val_size': val_size,
                       'n_runs': n_runs,
                       'seed': seed}

    experiments = []
    for func in [random, variation_ratios, mutual_information, predictive_entropy, mean_standard_deviation]:
        if func.__name__ in which_acq_funcs:
            experiment_params = base_experiment.copy()
            experiment_params['acquisition_function'] = func
            experiments.append(experiment_params)

    return experiments
