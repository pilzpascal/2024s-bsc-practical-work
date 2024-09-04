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
from acquisition_functions import perform_acquisition

TRAIN_SIZE = 20
VAL_SIZE = 100
POOL_SIZE = 60_000 - (TRAIN_SIZE + VAL_SIZE)

DATA_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/data/mnist/'
MODEL_SAVE_PATH = '/Users/pascalpilz/Documents/Bsc Thesis/models/'
EXPERIMENT_SAVE_PATH = './Experiment Results/'
SEED = 1


def get_datasets(train_size=TRAIN_SIZE,
                 pool_size=POOL_SIZE,
                 val_size=VAL_SIZE,
                 data_path=DATA_PATH):
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

    # splitting train further into train and val
    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size + pool_size, val_size])

    # creating data loaders
    # having batch size equal to the size of the dataset to get the whole dataset in one batch
    # enables us to get the whole dataset at once when iterating over the loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

    X_train_all, y_train_all = next(iter(train_loader))
    X_val, y_val = next(iter(val_loader))
    X_test, y_test = next(iter(test_loader))

    # creating a random but balanced initial training set and pool from remaining train data
    idx = []
    for num in range(10):
        indices = torch.where(y_train_all == num)[0]
        idx += list(np.random.choice(indices, 2, replace=False))
    idx = torch.tensor(idx)

    X_train = X_train_all[idx]
    y_train = y_train_all[idx]

    # pool is all samples from full training set that weren't chosen for initial training set
    X_pool = X_train_all[~torch.isin(torch.arange(len(X_train_all)), idx)]
    y_pool = y_train_all[~torch.isin(torch.arange(len(X_train_all)), idx)]

    return X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test


def get_subset_dataloader(subset, X):

    if isinstance(subset, list):
        subset_idx = subset
    elif isinstance(subset, int):
        subset_idx = np.random.choice(range(X.shape[0]), size=subset)
    else:
        subset_idx = np.arange(X.shape[0])

    if isinstance(X, torch.util.data.DataLoader):
        X = torch.concat([torch.concat(elem) for elem in list(iter(X))])

    dataset = torch.utils.data.TensorDataset(X[subset_idx])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False)
    return dataloader, subset_idx


def get_accuracy_and_info(model, dataloader, acquisition_function, T=100, subset=None):

    dataloader, subset_idx = get_subset_dataloader(subset, dataloader)

    for inputs in tqdm(dataloader, total=len(dataloader), disable=not show_pbar, desc='Predictive Entropy'):
        list_outputs = [torch.softmax(model(inputs[0], use_dropout=True), dim=1) for _ in range(T)]
        tensor_outputs = torch.stack(list_outputs, dim=0)
        mean_outputs = torch.mean(tensor_outputs, dim=0)

    model.eval()
    running_corrects = 0
    predications = []

    for i, (inputs, labels) in enumerate(dataloader):
        outputs = model(inputs)
        outputs = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(torch.Tensor(outputs == labels.data))
        predications += outputs.tolist()

    acc = running_corrects.float() / (len(dataloader) * 4)

    return acc


def train_one_epoch(model,
                    train_loader,
                    optimizer,
                    loss_fn,
                    print_at=1000,
                    verbose=False):
    running_loss = 0.
    last_loss = 0.

    for i, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % print_at == print_at - 1:
            last_loss = running_loss / print_at  # loss per batch
            if verbose:
                print(f'  batch {i + 1:7_d} train loss: {last_loss:6.4f}')
            running_loss = 0.

    return last_loss


def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_fn,
                 print_at=-1,
                 n_epochs=100,
                 verbose=False,
                 early_stopping=10,
                 model_save_path=MODEL_SAVE_PATH):

    # if print_at is -1 then we just print 5 times during training
    if print_at == -1:
        print_at = max(len(train_loader) // 5, 1)

    epochs_since_best_vloss = 0
    best_model_state_dict = model.state_dict()
    best_vloss = np.inf

    os.makedirs(model_save_path, exist_ok=True)
    model_path = model_save_path + f'trainsize-{len(train_loader.dataset):05d}_'
    save_path = model_path + f'epoch-####_valloss-{best_vloss:.3f}'

    epochs = tqdm(desc=f'Training Epochs with Training size {len(train_loader.dataset)}', leave=False)
    # epochs = tqdm(range(n_epochs), desc=f'Training size {len(train_loader.dataset)}')
    for epoch in range(n_epochs):

        model.train(True)
        avg_loss = train_one_epoch(model=model,
                                   train_loader=train_loader,
                                   optimizer=optimizer,
                                   loss_fn=loss_fn,
                                   print_at=print_at,
                                   verbose=verbose)

        # get validation loss
        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, (vinputs, vlabels) in enumerate(val_loader):
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        if verbose:
            print(f'LOSS train {avg_loss:6.5f} valid {avg_vloss:6.4f}\n')

        # track the best performance, and save the model's state. Early stopping if no improvement for certain number of epochs
        if avg_vloss < best_vloss:
            epochs_since_best_vloss = 0
            best_vloss = avg_vloss
            save_path = model_path + f'epoch-{epoch:04d}_valloss-{best_vloss:.3f}'
            best_model_state_dict = model.state_dict()
        else:
            epochs_since_best_vloss += 1
            if epochs_since_best_vloss == early_stopping:
                if verbose:
                    print(f'Early stopping at epoch {epoch}.')
                break

        epochs.update()
    epochs.close()

    torch.save(best_model_state_dict, save_path)

    if verbose:
        print('Done.')
    return save_path


def run_active_learning(X_train, y_train, X_pool, y_pool, val_loader, test_loader,
                        acquisition_function, n_acquisition_steps=100, n_samples_to_acquire=10,
                        n_epochs=100, verbose=False, training_verbose=False, model_save_path=MODEL_SAVE_PATH):
    running_X_train = X_train
    running_y_train = y_train

    running_X_pool = X_pool
    running_y_pool = y_pool

    # we only return the test accuracies
    # we need n_acquisition_steps+1 since the first iteration does not do an acquisition step
    results = torch.zeros(n_acquisition_steps+1)

    for i in tqdm(range(n_acquisition_steps+1), desc='Acquisition Steps', leave=False):

        net = LeNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        running_train_set = torch.utils.data.TensorDataset(running_X_train, running_y_train)
        running_train_loader = torch.utils.data.DataLoader(running_train_set, batch_size=4, shuffle=True)

        training_model_save_path = (model_save_path
                                    + f'{acquisition_function.__name__}/')
        best_model_path = run_training(net,
                                       train_loader=running_train_loader,
                                       val_loader=val_loader,
                                       optimizer=optimizer,
                                       loss_fn=loss_fn,
                                       n_epochs=n_epochs,
                                       verbose=training_verbose,
                                       early_stopping=10,
                                       model_save_path=training_model_save_path)

        # loading the model with the lowest val loss to get accuracy on test set
        best_model = LeNet()
        best_model.load_state_dict(torch.load(best_model_path, weights_only=True))
        acc, info = get_accuracy_and_info(best_model, test_loader, acquisition_function)
        results[i] = acc

        if verbose:
            print('\n' + '=' * 75)
            print(f'Acquisition step {i:3d} - train size: {running_X_train.shape[0]:6_d}, test accuracy: {acc:6.4f}')
            print('\n')

        infos, _, _ = acquisition_function(model=net, X=running_X_pool, T=100, subset=None)
        running_X_train, running_y_train, running_X_pool, running_y_pool \
            = perform_acquisition(infos=infos, n_samples_to_acquire=n_samples_to_acquire, X_train=running_X_train,
                                  y_train=running_y_train, X_pool=running_X_pool, y_pool=running_y_pool)

    return results


def run_experiments(experiments, n_runs=3, seed=SEED, model_save_path=MODEL_SAVE_PATH, show_dataset=False):

    exp_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    for experiment in tqdm(experiments, desc='Experiments'):

        # we need +1 since for the first iteration we don't have an acquisition step yet
        n_acq_steps = experiment['n_acquisition_steps'] + 1
        # for each experiment we perform three runs and average the results
        runs = torch.zeros((n_runs, n_acq_steps))

        torch.manual_seed(seed)
        np.random.seed(seed)

        for i in tqdm(range(n_runs), desc='Runs per Experiment', leave=False):

            # get dataset new for each ru, such that train and pool is shuffled newly
            X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test = get_datasets()
            if show_dataset:
                visualise_datasets(X_train, y_train, X_pool, y_pool, X_val, y_val, X_test, y_test)

            # get val and test set and loader, these stay constant
            val_set = torch.utils.data.TensorDataset(X_val, y_val)
            test_set = torch.utils.data.TensorDataset(X_test, y_test)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=True)

            save_path = model_save_path + f'expID-{exp_id}/run-{i}/'
            results = run_active_learning(X_train, y_train,
                                          X_pool, y_pool,
                                          val_loader, test_loader,
                                          model_save_path=save_path,
                                          **experiment)
            runs[i] = results

        # we get the acc of all three runs, but the train size and acq step only
        # from the last runs since it is the same for all three runs
        experiment['results'] = {'acc': runs}

    # save the whole experiments dict via pickle
    save_path = EXPERIMENT_SAVE_PATH + f'expID-{exp_id}'
    with open(save_path, 'wb') as handle:
        pickle.dump(experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return experiments, save_path


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


def visualise_active_learning_experiments(experiments, train_size=TRAIN_SIZE):
    for experiment in experiments:
        results = experiment['results']
        # we need +1 since for the first iteration we don't have an acquisition step yet
        n_acq_steps = experiment['n_acquisition_steps'] + 1
        n_samples = experiment['n_samples_to_acquire']
        name = experiment['acquisition_function'].__name__

        # we show the mean of all the runs that were done, usually it's 3 runs
        x = torch.arange(n_acq_steps) * n_samples
        y = torch.mean(results['acc'], dim=0) * 100

        plt.plot(x, y, label=name.title())

        # turn off the border
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.grid(True, which='both', linestyle='--', linewidth=0.7)

        plt.xticks(ticks=np.linspace(0, x.max(), min(11, n_acq_steps)), fontsize=12)
        plt.yticks(ticks=np.linspace(np.floor(y.min() / 10) * 10, 100, 11), fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('Acquired Samples', fontsize=14)
        plt.ylabel('Test Accuracy (%)', fontsize=14)

    plt.show()


def visualise_most_and_least_informative_samples(vals, X_data, y_data, n_most=10, n_least=10):
    most_informative_idx = torch.topk(vals, n_most).indices
    least_informative_idx = torch.topk(-vals, n_least).indices

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
                ax.set_title(f'class {y[i].item()}\n'
                             f'info {vals[idx[i]]:.1f}')
                ax.axis('off')
            fig.suptitle(f'{num} {most_least} informative samples', y=1.5, fontsize=16)

    plt.show()
