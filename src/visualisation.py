import os

import torch
import numpy as np

from matplotlib import pyplot as plt
from tabulate import tabulate


def compute_samples_for_error(
        experiment: dict,
        acc_thresholds: list[float]
) -> dict:
    """
    Compute the number of samples required to reach specific error thresholds for each
    acquisition function, including variability measures (mean and std across repetitions).

    Parameters
    ----------
    experiment : dict
        The experiment dictionary containing results.
    acc_thresholds : list[float]
        List of error thresholds (e.g., [0.10, 0.05] for 10% and 5%).

    Returns
    -------
    results_summary : dict
        Dictionary of the form:
        {
            'threshold': {
                'acq_func': {'mean': value, 'std': value}
            }
        }
    """

    train_size = experiment['params']['exp']['train_size']
    n_samples_to_acquire = experiment['params']['al']['n_samples_to_acquire']
    results_summary = {}

    for threshold in acc_thresholds:
        results_summary[threshold] = {}

        for acq_func, results in experiment['results']['acq'].items():
            acc_runs = np.array(results['test_acc'])

            if len(acc_runs) == 0:
                continue

            n_acq_steps = np.argmax(acc_runs > threshold, axis=1)
            n_acq_steps_mean = np.nanmean(n_acq_steps)
            n_acq_steps_std = np.nanstd(n_acq_steps)

            results_summary[threshold][acq_func] = {
                'mean': n_acq_steps_mean * n_samples_to_acquire + train_size,
                'std': n_acq_steps_std * n_samples_to_acquire + train_size
            }

    return results_summary


def _ax_label_helper(
        ax: plt.Axes,
        train_size: int,
        n_acq_steps: int,
        n_samples_to_acquire: int
) -> None:

    """
    Helper function to set common parameters for the axes of the plots.

    Parameters
    ----------
    ax : plt.Axes
        The axes to set the parameters for.
    train_size : int
        The initial size of the training set.
    n_acq_steps : int
        The number of acquisition steps.

    Returns
    -------

    """

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(
        ticks=np.linspace(0, n_acq_steps-1, min(11, n_acq_steps)),
        labels=np.linspace(
            0,
            n_samples_to_acquire*(n_acq_steps-1), min(11, n_acq_steps)
        ).astype(int) + train_size
    )
    ax.set_xlabel(f'Training set size (initial trian size={train_size})')
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.legend()


def visualise_experiment_results(
        experiment: dict,
        which_measures: list[str],
        save_path: str = None
) -> None:
    """
    Visualises the results of an experiment by plotting the test set accuracy and mutual information
    for each acquisition function over the number of samples acquired, including shaded error regions.

    Parameters
    ----------
    experiment : dict
        The experiment dictionary containing parameters and results.
    which_measures : list[str]
        List of measures to plot. Can contain 'acc' for accuracy and 'inf' for mutual information.
    save_path : str, optional
        If provided, saves the plot to the specified path instead of displaying it, by default None.
        If None, the plot is not saved but is displayed.
    """

    assert all([w in ['acc', 'inf'] for w in which_measures]), 'Invalid value for which'

    # Number of steps and sizes
    # we need +1 since for the first iteration we don't have an acquisition step yet
    train_size = experiment['params']['exp']['train_size']
    n_acq_steps = experiment['params']['al']['n_acquisition_steps'] + 1
    n_samples_to_acquire = experiment['params']['al']['n_samples_to_acquire']

    fig, axs = plt.subplots(len(which_measures), 1, figsize=(10, 5 * len(which_measures)))
    axs = [axs] if not isinstance(axs, np.ndarray) else axs

    # Collect min and max values across acquisition functions
    min_y = np.inf
    max_y = -np.inf

    # Differentiate if we plot accuracy or mutual information
    for i, w in enumerate(which_measures):
        for acq_func, results in experiment['results']['acq'].items():

            if len(results[f'test_{w}']) == 0:  # Skip incomplete data
                continue

            # Compute mean and std across repetitions
            y_mean = np.mean(results[f'test_{w}'], axis=0)
            y_std = np.std(results[f'test_{w}'], axis=0)

            if w == 'acc':
                y_mean = y_mean * 100
                y_std = y_std * 100
                min_y = min(min_y, np.nanmin(y_mean))
            elif w == 'inf':
                max_y = max(max_y, np.nanmax(y_mean))

            steps = np.arange(len(y_mean))

            # Plot mean curve
            # axs[i].plot(steps, y_mean, label=acq_func.replace('_', ' ').title())
            axs[i].plot(steps, y_mean, label=acq_func)

            # Add shaded error region (mean ± std)
            axs[i].fill_between(steps, y_mean - y_std, y_mean + y_std, alpha=0.2)

        # Full dataset reference line
        results_bounds = experiment['results']['bounds'][f'test_{w}']
        bound = np.mean(results_bounds)
        bound = bound * 100 if w == 'acc' else bound
        axs[i].axhline(y=bound, color='grey', linestyle='--',
                       label=f'Full dataset ({bound:.2f})')

        # Y-axis labels and limits
        if w == 'acc':
            axs[i].set_title('Test set accuracy')
            axs[i].set_ylabel('Accuracy [%]')
            axs[i].set_ylim(min_y * 0.9, 100)
        elif w == 'inf':
            axs[i].set_title('Test set mutual information')
            axs[i].set_ylabel('Mutual information')
            axs[i].set_ylim(0, max_y * 1.2)

    for ax in axs:
        _ax_label_helper(ax, train_size, n_acq_steps, n_samples_to_acquire)

    plt.tight_layout(h_pad=5)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_save_path = os.path.join(save_path, '_'.join(which_measures) + '.pdf')
        plt.savefig(image_save_path, format='pdf', dpi=300)
    else:
        plt.show()


def visualise_datasets(
        X_train, y_train,
        X_pool, y_pool,
        X_val, y_val,
        X_test, y_test
) -> None:

    """
    Visualises the datasets by showing their sizes, means, standard deviations,
    and label distributions. It also displays images from the initial training set.
    This function is useful for understanding the distribution and characteristics of the datasets
    before starting the active learning process.

    Parameters
    ----------
    X_train : torch.Tensor
    y_train : torch.Tensor
    X_pool : torch.Tensor
    y_pool : torch.Tensor
    X_val : torch.Tensor
    y_val : torch.Tensor
    X_test : torch.Tensor
    y_test : torch.Tensor

    Returns
    -------

    """

    # show each dataset size, mean, std in a nice table
    print(
        tabulate(
            [
                [name, len(ds), ds.mean(), ds.std()]
                for name, ds in
                [('train', X_train), ('pool', X_pool), ('val', X_val), ('test', X_test)]
            ],
            floatfmt='.3f', intfmt='_', headers=['size', 'mean', 'std']
        )
    )

    # show each dataset distribution of labels in one histogram
    fig, axs = plt.subplots(1, 1, figsize=(10, 3))

    axs.hist([y_train, y_val, y_pool, y_test], bins=range(0, 11),
             density=True, label=['train', 'val', 'pool', 'test'])
    axs.axhline(y=0.1, color='r', linestyle='--', alpha=0.6)

    plt.xticks(np.linspace(0.5, 9.5, 10), np.arange(0, 10))
    plt.legend(loc='lower right')
    fig.suptitle('Label Distribution per Dataset', fontsize=16)
    plt.show()

    # Show images of entire initial training set. Only consists of 20 images, two of each class
    fig, axs = plt.subplots(2, 10, figsize=(10, 3))
    for i in range(2):
        for j in range(10):
            ax = axs[i, j]
            ax.imshow(X_train[i + j * 2].squeeze(), cmap='gray')
            if i == 0:
                ax.set_title(f'{y_train[i + j * 2].item()}')
            ax.axis('off')
    fig.suptitle('Initial Training Set', fontsize=16)
    plt.tight_layout()
    plt.show()


def visualise_epochs_before_early_stopping(
        experiment: dict,
        window_width: int = 10
) -> None:

    """
    Visualises the number of epochs trained before early stopping for each acquisition function.

    Parameters
    ----------
    experiment : dict
        The experiment dictionary containing parameters and results.
    window_width : int, optional
        The width of the moving average window to smooth the number of epochs, by default 10.

    Returns
    -------

    """

    # we need +1 since for the first iteration we don't have an acquisition step yet
    exp_id = experiment['params']['exp']['exp_id']
    max_epochs = experiment['params']['train']['n_epochs']
    train_size = experiment['params']['exp']['train_size']
    n_acq_steps = experiment['params']['al']['n_acquisition_steps'] + 1
    n_samples_to_acquire = experiment['params']['al']['n_samples_to_acquire']

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for acq_func, results in experiment['results']['acq'].items():

        if len(results[f'test_acc']) == 0 or len(results[f'test_inf']) == 0:
            continue

        epochs = []
        for i in range(experiment['params']['exp']['n_runs']):
            files = os.listdir(
                experiment['params']['exp']['model_save_path_base']
                + f'{exp_id}/run-{i}/{acq_func}'
            )
            # +1 because epochs start counting from 0
            epochs.append(
                [
                    int(file.split('_')[1].split('-')[1]) + 1
                    for file in sorted(files)
                ]
            )

        # if we don't have results because the experiment stopped early we skip it
        try:
            epochs = np.array(epochs)
            data = epochs.mean(axis=0)
            data_std = epochs.std(axis=0)
        except ValueError:
            continue
        cumulative_sum = np.cumsum(np.insert(data, 0, 0))
        moving_average = (cumulative_sum[window_width:] - cumulative_sum[:-window_width]) / window_width

        cumulative_sum_std = np.cumsum(np.insert(data_std, 0, 0))
        moving_std = (cumulative_sum_std[window_width:] - cumulative_sum_std[:-window_width]) / window_width

        ax.plot(moving_average, label=acq_func.replace('_', ' ').title())
        ax.fill_between(
            np.arange(len(moving_average)),
            moving_average - moving_std,
            moving_average + moving_std,
            alpha=0.2
        )

    # plot the number of epochs trained for if using full dataset
    files = []
    for i in range(experiment['params']['exp']['n_runs']):
        files += os.listdir(
            experiment['params']['exp']['model_save_path_base']
            + f'{exp_id}/run-{i}/full_dataset'
        )

    epochs = [
        int(file.split('_')[1].split('-')[1]) + 1
        for file in sorted(files)
    ]
    bound = np.mean(epochs)
    ax.axhline(y=bound, color='grey', linestyle='--', label=f'Full Dataset ({int(bound)})')

    ax.set_ylim(0, max_epochs * 1.05)
    ax.set_ylabel('Epochs')
    ax.set_title(f'Moving Average of Epochs before Early Stopping (max={max_epochs})')
    _ax_label_helper(ax, train_size, n_acq_steps, n_samples_to_acquire)

    fig.tight_layout()
    plt.show()


def visualise_time_per_acquisition(
        experiment: dict,
) -> None:

    """
    Visualises the time taken per acquisition step for each acquisition function.

    Parameters
    ----------
    experiment : dict
        The experiment dictionary containing parameters and results.

    Returns
    -------

    """

    # !!! Note that the time for testing cannot be factored out !!!

    # we need +1 since for the first iteration we don't have an acquisition step yet
    n_acq_steps = experiment['params']['al']['n_acquisition_steps'] + 1

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    names, heights = [], []
    for key, value in experiment['results']['acq'].items():
        names.append(key.replace('_', ' ').title())
        heights.append((value['time'] / n_acq_steps) / 60)

    ax.bar(names, heights)

    ax.set_ylabel('time per step [minutes]')
    ax.set_title(f'Time taken per Acquisition Step')
    ax.yaxis.grid(True, linewidth=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.tight_layout()
    plt.show()


def visualise_most_and_least_informative_samples(
        X_data, y_data,
        vals,
        preds,
        n_most=10, n_least=10
) -> None:

    """
    Visualises the most and least informative samples based on their mutual information values.

    Parameters
    ----------
    X_data : torch.Tensor
    y_data : torch.Tensor
    vals : list[float]
    preds : torch.Tensor
    n_most : int, optional
        The number of most informative samples to visualise, by default 10
    n_least : int, optional
        The number of least informative samples to visualise, by default 10

    Returns
    -------

    """

    most_informative_idx = torch.topk(torch.Tensor(vals), n_most).indices
    least_informative_idx = torch.topk(-torch.Tensor(vals), n_least).indices

    most = (n_most, most_informative_idx, 'most')
    least = (n_least, least_informative_idx, 'least')

    for num, idx, most_least in [most, least]:
        if num > 0:
            X, y = X_data[idx], y_data[idx]
            fig, axs = plt.subplots(1, num, figsize=(num, 1))
            axs = axs if isinstance(axs, np.ndarray) else [axs]
            for i in range(num):
                ax = axs[i]
                ax.imshow(X[i].squeeze(), cmap='gray')
                ax.set_title(f'true {y[i].item()}\n'
                             f'pred {preds[idx[i]]}\n'
                             f'info {vals[idx[i]]:.1f}')
                ax.axis('off')
            fig.suptitle(f'{num} {most_least} informative samples', y=1.7, fontsize=16)

    plt.show()
