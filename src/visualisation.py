import pickle

import torch
import numpy as np

from matplotlib import pyplot as plt
from tabulate import tabulate


def visualise_datasets(
        X_train, y_train,
        X_pool, y_pool,
        X_val, y_val,
        X_test, y_test)\
        -> None:

    # show each dataset size, mean, std in a nice table
    print(tabulate([[name, len(ds), ds.mean(), ds.std()] for name, ds in
                    [('train', X_train), ('pool', X_pool), ('val', X_val), ('test', X_test)]],
                   floatfmt='.3f', intfmt='_', headers=['size', 'mean', 'std']))

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


def visualise_experiments(experiment: dict) -> None:

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # if existent, get the calculated upper bounds for accuracy and information
    # these are obtained by training on all 59_000 (1_000 val) training samples
    for i, value in enumerate(['test_acc_ubound', 'test_inf_lbound']):
        if experiment[-1][value] is not None:
            bound = torch.mean(experiment[-1][value], dim=0)
            # acc is percentage, mutual info is not
            bound = bound * 100 if value == 'test_acc_ubound' else bound
            axs[i].axhline(y=bound, color='grey', linestyle='--',
                           label=f'full dataset ({torch.mean(bound):.2f})')

    for experiment in experiment[:-1]:

        if 'results' not in experiment:
            continue

        results = experiment['results']
        # we need +1 since for the first iteration we don't have an acquisition step yet
        n_acq_steps = experiment['n_acquisition_steps'] + 1
        n_samples = experiment['n_samples_to_acquire']
        name = experiment['acquisition_function'].__name__

        # we show the mean of all the runs that were done, usually it's 3 runs
        x = torch.arange(n_acq_steps) * n_samples
        y_acc = torch.mean(results['test_acc'], dim=0) * 100
        y_inf = torch.mean(results['test_info'], dim=0)

        axs[0].plot(x, y_acc, label=' '.join(name.split('_')).title())
        axs[1].plot(x, y_inf, label=' '.join(name.split('_')).title())

        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            ax.set_xticks(ticks=np.linspace(0, x.max(), min(11, n_acq_steps)))
            ax.set_xlabel('Acquired Samples')
            ax.grid(True, which='both', linestyle='--', linewidth=0.7)
            # ax.legend(loc='lower right')
            ax.legend()

        axs[0].set_yticks(ticks=np.linspace(np.floor(y_acc.min() / 10) * 10, 100, 11))
        axs[0].set_ylabel('Test Set Accuracy (%)', fontsize=14)
        axs[0].set_title('Mean Test Accuracy Across Runs', fontsize=16)

        # axs[1].set_yticks([])
        axs[1].set_ylabel('Average Mutual Info per Test Sample', fontsize=14)
        axs[1].set_title('Mean Mutual Information Across Runs', fontsize=16)

    plt.tight_layout(h_pad=3)
    plt.show()


def visualise_most_and_least_informative_samples(
        vals, preds,
        X_data, y_data,
        n_most=10, n_least=10)\
        -> None:

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
