import os
import pickle
from tabulate import tabulate
from tqdm.auto import tqdm
from datetime import datetime

import numpy as np

import torchvision
import torch

from matplotlib import pyplot as plt

# custom import
from networks import LeNet
from acquisition_functions import mutual_information

torch.use_deterministic_algorithms(mode=True)


DATA_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/data/mnist/'
MODEL_SAVE_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/models/'
EXPERIMENT_SAVE_PATH = './Experiment Results/'
SEED = 1


def get_datasets(train_size=20, val_size=1_000, data_path=DATA_PATH):

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         # we pad because the images are 28x28 but the network expects 32x32
         torchvision.transforms.Pad(2),
         # 0.10003718 and 0.2752173 is mean and std respectively of train set after padding
         # (without padding it is 0.1307 and 0.3081 mean and std respectively)
         torchvision.transforms.Normalize((0.10003718,), (0.2752173,))])

    # getting the test and train set
    train_val_set = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform)

    pool_size = len(train_val_set) - (train_size + val_size)

    # splitting train further into train and val
    train_pool_set, val_set = torch.utils.data.random_split(train_val_set, [train_size + pool_size, val_size])

    # creating data loaders
    # having batch size equal to the size of the dataset to get the whole dataset in one batch
    # enables us to get the whole dataset at once when iterating over the loader
    train_pool_loader = torch.utils.data.DataLoader(train_pool_set, batch_size=len(train_pool_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    X_train_pool, y_train_pool = next(iter(train_pool_loader))
    X_val, y_val = next(iter(val_loader))
    X_test, y_test = next(iter(test_loader))

    # creating a random but balanced initial training set and pool from remaining train data
    idx = []
    for num in range(10):
        indices = torch.where(y_train_pool == num)[0]
        idx += list(np.random.choice(indices, 2, replace=False))
    idx = torch.tensor(idx)

    X_train = X_train_pool[idx]
    y_train = y_train_pool[idx]

    # pool is all samples from full training set that weren't chosen for initial training set
    X_pool = X_train_pool[~torch.isin(torch.arange(len(X_train_pool)), idx)]
    y_pool = y_train_pool[~torch.isin(torch.arange(len(X_train_pool)), idx)]

    return X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test


def get_subset(subset, X):

    # if X is already a dataloader we convert it into a torch tensor
    if isinstance(X, torch.utils.data.DataLoader):
        X = torch.utils.data.DataLoader(X.dataset, batch_size=len(X.dataset), shuffle=False)
        X = next(iter(X))[0]

    if isinstance(subset, int):
        subset_idx = np.random.choice(range(X.shape[0]), size=subset, replace=False)
    elif subset is None:
        subset_idx = np.arange(X.shape[0])
    else:
        raise ValueError('subset needs to be int or None.')

    subset = X[subset_idx]

    return subset, subset_idx


def get_info_and_predictions(model, data, acquisition_function, T=64, subset=None, show_pbar=False):

    inputs, subset_idx = get_subset(subset, data)
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


def test_model(model, dataloader, T=64, subset=None, show_pbar=False):

    infos, predictions, subset_idx = get_info_and_predictions(model, dataloader, subset=subset, show_pbar=show_pbar,
                                                              acquisition_function=mutual_information, T=T)
    # get average information per data point
    infos = torch.Tensor(infos).mean()

    labels = torch.utils.data.DataLoader(dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False)
    labels = next(iter(labels))[1][subset_idx]
    acc = (torch.Tensor(predictions) == labels).float().mean()

    return acc, infos, subset_idx


def perform_acquisition(infos, X_train, y_train, X_pool, y_pool,
                        n_samples_to_acquire=10, subset_idx=None):

    # get indices of top n_samples_to_acquire elements of infos
    idx = torch.topk(torch.Tensor(infos), n_samples_to_acquire).indices
    # if infos was not calculated over a subset, then we use the full pool, otherwise use the subset
    if subset_idx is None:
        subset_idx = torch.arange(X_pool.shape(0))
    subset_idx = torch.Tensor(subset_idx).int()

    # divide the pool set into the same subset used to calculate infos
    subset_X_pool = X_pool[subset_idx]
    subset_y_pool = y_pool[subset_idx]
    remaining_X_pool = X_pool[~torch.isin(torch.arange(len(X_pool)), subset_idx)]
    remaining_y_pool = y_pool[~torch.isin(torch.arange(len(y_pool)), subset_idx)]

    # from subset choose the ones corresponding to the index gotten from infos
    chosen_subset_X_pool = subset_X_pool[idx]
    chosen_subset_y_pool = subset_y_pool[idx]
    not_chosen_subset_X_pool = subset_X_pool[~torch.isin(torch.arange(len(subset_X_pool)), idx)]
    not_chosen_subset_y_pool = subset_y_pool[~torch.isin(torch.arange(len(subset_X_pool)), idx)]

    # recombine new pool and train sets
    new_X_train = torch.concatenate([X_train, chosen_subset_X_pool], 0)
    new_y_train = torch.concatenate([y_train, chosen_subset_y_pool], 0)
    new_X_pool = torch.concatenate([remaining_X_pool, not_chosen_subset_X_pool], 0)
    new_y_pool = torch.concatenate([remaining_y_pool, not_chosen_subset_y_pool], 0)

    return new_X_train, new_y_train, new_X_pool, new_y_pool


def get_trained_model(train_loader, val_loader, n_epochs=100,
                      early_stopping=10, model_save_path=MODEL_SAVE_PATH):

    model = LeNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    epochs_since_best_vloss = 0
    best_model_state_dict = model.state_dict()
    best_vloss = np.inf

    os.makedirs(model_save_path, exist_ok=True)
    model_path = model_save_path + f'trainsize-{len(train_loader.dataset):05d}_'
    # save_path gets redefined everytime we get a model with a new best_vloss
    save_path = model_path + f'epoch-####_valloss-{best_vloss:.3f}'

    if early_stopping == -1:
        epochs = tqdm(total=n_epochs, desc=f'Training Model with Training size {len(train_loader.dataset)}', leave=False)
    else:
        epochs = tqdm(desc=f'Training Model with Training size {len(train_loader.dataset)}', leave=False)

    for epoch in range(n_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # get validation loss
        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            for vinputs, vlabels in val_loader:
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / len(val_loader)

        # track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            epochs_since_best_vloss = 0
            best_vloss = avg_vloss
            save_path = model_path + f'epoch-{epoch:04d}_valloss-{best_vloss:.3f}'
            best_model_state_dict = model.state_dict()
        # Early stopping if no improvement for certain number of epochs
        else:
            epochs_since_best_vloss += 1
            if epochs_since_best_vloss == early_stopping:
                break

        epochs.update()
    epochs.close()

    torch.save(best_model_state_dict, save_path)

    best_model = LeNet()
    best_model.load_state_dict(best_model_state_dict)

    return best_model


def run_active_learning(X_train, y_train, X_pool, y_pool, val_loader, test_loader, acquisition_function,
                        n_acquisition_steps=100, n_samples_to_acquire=10, T=64, early_stopping=10, n_epochs=100,
                        pool_subset_size=None, test_subset_size=None, model_save_path=MODEL_SAVE_PATH):

    running_X_train = X_train.clone()
    running_y_train = y_train.clone()
    running_X_pool = X_pool.clone()
    running_y_pool = y_pool.clone()

    # we only return the test accuracies
    # we need n_acquisition_steps+1 since the first iteration does not do an acquisition step
    test_info = torch.zeros(n_acquisition_steps+1)
    test_acc = torch.zeros(n_acquisition_steps+1)

    for i in tqdm(range(n_acquisition_steps+1),
                  desc=f'Acquisition Steps for {' '.join(acquisition_function.__name__.split('_')).title()}',
                  leave=False):

        running_train_set = torch.utils.data.TensorDataset(running_X_train, running_y_train)
        running_train_loader = torch.utils.data.DataLoader(running_train_set, batch_size=10, shuffle=True)

        training_model_save_path = model_save_path + f'{acquisition_function.__name__}/'
        model = get_trained_model(train_loader=running_train_loader,
                                  val_loader=val_loader,
                                  n_epochs=n_epochs,
                                  early_stopping=early_stopping,
                                  model_save_path=training_model_save_path)

        info, acc, _ = test_model(model, test_loader, T=T, subset=test_subset_size, show_pbar=True)
        test_info[i] = info
        test_acc[i] = acc

        infos, _, subset_idx = get_info_and_predictions(model, running_X_pool, acquisition_function,
                                                        T=T, subset=pool_subset_size, show_pbar=True)
        running_X_train, running_y_train, running_X_pool, running_y_pool \
            = perform_acquisition(infos, running_X_train, running_y_train, running_X_pool, running_y_pool,
                                  n_samples_to_acquire=n_samples_to_acquire, subset_idx=subset_idx)

    return test_info, test_acc


def run_experiments(experiments, model_save_path=MODEL_SAVE_PATH, experiment_save_path=EXPERIMENT_SAVE_PATH):

    exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_save_path = experiment_save_path + f'expID-{exp_id}'

    for experiment in tqdm(experiments, desc='Experiments'):

        n_runs = experiment.pop('n_runs')
        seed = experiment.pop('seed')
        val_size = experiment.pop('val_size')

        # we need +1 since for the first iteration we don't have an acquisition step yet
        n_acq_steps = experiment['n_acquisition_steps'] + 1
        # for each experiment we perform three runs and average the results
        accuracies = torch.zeros((n_runs, n_acq_steps))
        infos = torch.zeros((n_runs, n_acq_steps))

        torch.manual_seed(seed)
        np.random.seed(seed)

        for i in tqdm(range(n_runs), desc='Runs per Experiment', leave=False):

            # get dataset new for each run, such that train and pool is shuffled newly
            X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test = get_datasets(val_size=val_size)

            # get val and test set and loader, these stay constant
            val_set = torch.utils.data.TensorDataset(X_val, y_val)
            test_set = torch.utils.data.TensorDataset(X_test, y_test)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set))
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))

            mod_save_path = model_save_path + f'expID-{exp_id}/run-{i}/'
            info, acc = run_active_learning(X_train, y_train,
                                            X_pool, y_pool,
                                            val_loader, test_loader,
                                            model_save_path=mod_save_path,
                                            **experiment)

            accuracies[i] = info
            infos[i] = acc

        # we add this information later other we'd get an issue when we pass **experiment to run_active_learning
        experiment.update({'results': {'test_acc': accuracies, 'test_info': infos},
                           'val_size': val_size,
                           'seed': seed,
                           'n_runs': n_runs,
                           'model_save_path': model_save_path,
                           'experiment_save_path': experiment_save_path})

        # save the whole experiments dict via pickle
        with open(exp_save_path, 'wb') as handle:
            pickle.dump(experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return experiments, exp_save_path


def visualise_datasets(X_train, y_train, X_pool, y_pool,
                       X_val, y_val, X_test, y_test):
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


def visualise_experiments(experiments):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    for experiment in experiments:

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
        # Normalise information values to [0,1]
        # y_inf = (y_inf - y_inf.min()) / (y_inf.max() - y_inf.min())

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
        axs[1].set_ylabel('Mean Information per Test Sample', fontsize=14)
        axs[1].set_title('Mean Information to be gained from Test Set Across Runs', fontsize=16)

    plt.tight_layout(h_pad=3)
    plt.show()


def visualise_most_and_least_informative_samples(vals, preds, X_data, y_data,
                                                 n_most=10, n_least=10):
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
