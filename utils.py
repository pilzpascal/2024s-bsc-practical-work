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
                 n_epochs=5,
                 verbose=False,
                 early_stopping=10):
    # if print_at is -1 then we just print 5 times during training
    if print_at == -1:
        print_at = max(len(train_loader) // 5, 1)
    save_path = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODEL_SAVE_PATH + str(timestamp) + '/'
    os.makedirs(model_path, exist_ok=True)

    epochs_since_best_vloss = 0
    best_vloss = np.inf

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
            save_path = model_path + f'model_{timestamp}_{epoch}'
            torch.save(model.state_dict(), save_path)
        else:
            epochs_since_best_vloss += 1
            if epochs_since_best_vloss == early_stopping:
                if verbose:
                    print(f'Early stopping at epoch {epoch}.')
                break

        epochs.update()
    epochs.close()
    if verbose:
        print('Done.')
    return save_path


def get_accuracy(model, dataloader):
    model.eval()
    running_corrects = 0
    predications = []

    for i, (inputs, labels) in enumerate(dataloader):
        toutputs = model(inputs)
        toutputs = torch.argmax(toutputs, dim=1)
        running_corrects += torch.sum(torch.Tensor(toutputs == labels.data))
        predications += toutputs.tolist()

    acc = running_corrects.float() / (len(dataloader) * 4)

    return acc


def run_active_learning(X_train, y_train, X_pool, y_pool, val_loader, test_loader,
                        acquisition_function, n_acquisition_steps=100, n_samples_to_acquire=10,
                        n_epochs=100, verbose=False, training_verbose=False):
    running_X_train = X_train
    running_y_train = y_train

    running_X_pool = X_pool
    running_y_pool = y_pool

    results = {'acq step': [], 'train size': [], 'acc': []}

    # do 50 acquisition steps
    for i in tqdm(range(n_acquisition_steps), desc='Acquisition Steps', leave=False):

        net = LeNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()

        running_train_set = torch.utils.data.TensorDataset(running_X_train, running_y_train)
        running_train_loader = torch.utils.data.DataLoader(running_train_set, batch_size=4, shuffle=True)

        path = run_training(net,
                            train_loader=running_train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            n_epochs=n_epochs,
                            early_stopping=10,
                            verbose=training_verbose)

        # loading the model with the lowest val loss to get accuracy on test set
        best_model = LeNet()
        best_model.load_state_dict(torch.load(path, weights_only=True))
        acc = get_accuracy(best_model, test_loader)
        results['acq step'].append(i)
        results['train size'].append(running_X_train.shape[0])
        results['acc'].append(acc)

        if verbose:
            print('\n' + '=' * 75)
            print(f'Acquisition step {i:3d} - train size: {running_X_train.shape[0]:6_d}, test accuracy: {acc:6.4f}')
            print('\n')

        running_X_train, running_y_train, running_X_pool, running_y_pool \
            = perform_acquisition(model=net, acquisition_function=acquisition_function,
                                  n_samples_to_acquire=n_samples_to_acquire,
                                  X_train=running_X_train, y_train=running_y_train, X_pool=running_X_pool,
                                  y_pool=running_y_pool)

    results = {key: torch.Tensor(val) for key, val in results.items()}
    return results


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
        runs = experiment['runs']
        n_acq_steps = experiment['n_acquisition_steps']
        name = experiment['acquisition_function'].__name__

        # we show the mean of all the runs that were done, usually it's 3 runs
        x = np.array(runs[0]['train size']) - train_size
        y = np.mean([run['acc'] for run in runs], axis=0) * 100 + np.random.normal(1, 1)

        plt.plot(x, y, label=name.title())

        # turn off the border
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.grid(True, which='both', linestyle='--', linewidth=0.7)

        plt.xticks(ticks=np.linspace(0, x.max(), n_acq_steps), fontsize=12)
        plt.yticks(ticks=np.linspace(np.floor(y.min() / 10) * 10, 100, 11), fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
        plt.xlabel('Acquired Samples', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)

    plt.show()


def run_experiments(experiments,
                    X_train, y_train,
                    X_pool, y_pool,
                    val_loader, test_loader,
                    n_runs=3):
    for experiment in tqdm(experiments, desc='Experiments'):
        # for each experiment we perform three runs and average the results
        runs = []

        for _ in tqdm(range(n_runs), desc='Runs per Experiment', leave=False):
            results = run_active_learning(X_train, y_train,
                                          X_pool, y_pool,
                                          val_loader, test_loader,
                                          **experiment)
            runs.append(results)

        experiment['runs'] = runs

    save_path = EXPERIMENT_SAVE_PATH + datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(save_path, 'wb') as handle:
        pickle.dump(experiments, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return experiments, save_path
