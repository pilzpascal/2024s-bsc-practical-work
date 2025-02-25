import os

import torch
import numpy as np

from matplotlib import pyplot as plt
from tabulate import tabulate


def get_number_of_samples_required_for_test_error(
        experiment: dict,
        target_accuracies: list[float]
) -> dict:

    train_size = experiment['params']['exp']['train_size']
    n_samples_to_acquire = experiment['params']['al']['n_samples_to_acquire']
    results = {}

    for acq_func, data in experiment['results']['acq'].items():
        results[acq_func] = {}

        for target_accuracy in target_accuracies:

            if len(data['test_acc']) == 0:
                continue

            acc = np.mean(data['test_acc'], axis=0)
            n_acquisition_steps = np.argmax(acc > target_accuracy)
            n_samples = n_acquisition_steps * n_samples_to_acquire + train_size

            results[acq_func][target_accuracy] = n_samples

    return results


def ax_label_helper(
        ax: plt.Axes,
        train_size: int,
        n_acq_steps: int,
        n_samples_to_acquire: int
) -> None:

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
    ax.set_xlabel(f'Training Set Size (initial trian size={train_size})')
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax.legend()


def visualise_experiment_results(
        experiment: dict,
        which_measures: list[str]
) -> None:

    assert all([w in ['acc', 'inf'] for w in which_measures]), 'Invalid value for which'

    # we need +1 since for the first iteration we don't have an acquisition step yet
    train_size = experiment['params']['exp']['train_size']
    n_acq_steps = experiment['params']['al']['n_acquisition_steps'] + 1
    n_samples_to_acquire = experiment['params']['al']['n_samples_to_acquire']

    fig, axs = plt.subplots(
        len(which_measures), 1,
        figsize=(10, 5 * len(which_measures))
    )
    axs = [axs] if not isinstance(axs, np.ndarray) else axs

    # collect min and max values across acquisition functions
    min_y = np.inf
    max_y = -np.inf

    # differentiate if we plot accuracy or mutual information
    for i, w in enumerate(which_measures):

        # plot accuracy of mutual information for each acquisition functions
        for acq_func, results in experiment['results']['acq'].items():

            # if we don't have results because the program stopped early we skip it
            if len(results[f'test_{w}']) == 0:
                continue

            if w == 'acc':
                y = np.array(np.mean(results[f'test_{w}'], axis=0)) * 100
                min_y = min(min_y, np.nanmin(y))
            elif w == 'inf':
                y = np.array(np.mean(results[f'test_{w}'], axis=0))
                max_y = max(max_y, np.nanmax(y))
            axs[i].plot(y, label=acq_func.replace('_', ' ').title())

        # if existent, get the calculated bounds for accuracy and mutual information
        # these are obtained by training on all 59_000 (-1_000 val) training samples
        results_bounds = experiment['results']['bounds'][f'test_{w}']
        bound = np.mean(results_bounds)
        bound = bound * 100 if w == 'acc' else bound
        axs[i].axhline(y=bound, color='grey', linestyle='--',
                       label=f'Full Dataset ({bound:.2f})')

        # set y-axis labels and limits depending on if we plot accuracy or mutual information
        if w == 'acc':
            axs[i].set_title('Test Set Accuracy')
            axs[i].set_ylabel('Accuracy [%]')
            axs[i].set_ylim(min_y * 0.9, 100)
        elif w == 'inf':
            axs[i].set_title('Test Set Mutual Information')
            axs[i].set_ylabel('Mutual Information')
            axs[i].set_ylim(0, max_y * 1.2)

    # set plot parameters independent of what we plot
    for ax in axs:
        ax_label_helper(ax, train_size, n_acq_steps, n_samples_to_acquire)

    plt.tight_layout(h_pad=5)
    plt.show()


def visualise_datasets(
        X_train, y_train,
        X_pool, y_pool,
        X_val, y_val,
        X_test, y_test
) -> None:

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
        except ValueError:
            continue
        cumulative_sum = np.cumsum(np.insert(data, 0, 0))
        moving_average = (cumulative_sum[window_width:] - cumulative_sum[:-window_width]) / window_width

        ax.plot(moving_average, label=acq_func.replace('_', ' ').title())

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
    ax_label_helper(ax, train_size, n_acq_steps, n_samples_to_acquire)

    fig.tight_layout()
    plt.show()


def visualise_most_and_least_informative_samples(
        X_data, y_data,
        vals,
        preds,
        n_most=10, n_least=10
) -> None:

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
