import numpy as np

import torch
import torchvision


def get_datasets(
        data_path,
        init_train_size: int,
        val_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Loads the MNIST dataset and splits it into training, validation, and test sets.

    Parameters
    ----------
    data_path : str
        The path to the data.
    init_train_size : int
        The number of samples to use for training, by default 20.
    val_size : int
        The number of samples to use for validation, by default 1_000.

    Returns
    -------
    tuple
        A tuple containing:
        - X_train (torch.Tensor): The training data.
        - y_train (torch.Tensor): The training labels.
        - X_pool (torch.Tensor): The pool data.
        - y_pool (torch.Tensor): The pool labels.
        - val_loader (torch.utils.data.DataLoader): The validation data loader.
        - test_loader (torch.utils.data.DataLoader): The test data loader.
    """

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    # getting the test and train set
    train_val_set = torchvision.datasets.MNIST(root=data_path, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_path, train=False, transform=transform)

    pool_size = len(train_val_set) - (init_train_size + val_size)

    # splitting train further into train and val
    train_pool_set, val_set = torch.utils.data.random_split(train_val_set, [init_train_size + pool_size, val_size])

    # creating data loaders
    # having batch size equal to the size of the dataset to get the whole dataset in one batch
    # enables us to get the whole dataset at once when iterating over the loader
    train_pool_loader = torch.utils.data.DataLoader(train_pool_set, batch_size=len(train_pool_set), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    X_train_pool, y_train_pool = next(iter(train_pool_loader))

    # creating a random but balanced initial training set and pool from remaining train data
    if init_train_size <= 50:
        idx = []
        for i in range(init_train_size):
            num = i % 10
            indices = torch.where(y_train_pool == num)[0]
            idx += list(np.random.choice(indices, 1, replace=False))
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(X_train_pool))[:init_train_size]

    X_train = X_train_pool[idx]
    y_train = y_train_pool[idx]

    # pool is all samples from full training set that weren't chosen for initial training set
    X_pool = X_train_pool[~torch.isin(torch.arange(len(X_train_pool)), idx)]
    y_pool = y_train_pool[~torch.isin(torch.arange(len(X_train_pool)), idx)]

    return X_train, y_train, X_pool, y_pool, val_loader, test_loader


def get_subset(
        X: torch.Tensor | torch.utils.data.DataLoader | torch.utils.data.TensorDataset,
        subset: int | list | np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Selects a subset of data from a dataset or tensor.

    Parameters
    ----------
    X : torch.Tensor, torch.utils.data.DataLoader, or torch.utils.data.TensorDataset
        Input data, either as a PyTorch tensor or a DataLoader.
    subset : int, list, np.ndarray, or None
        - If int, randomly selects `subset` number of samples.
        - If list or np.ndarray, selects specific indices.
        - If None, returns all samples.

    Returns
    -------
    tuple
        A tuple containing:
        - subset_data (torch.Tensor): The selected subset of `X`.
        - subset_idx (torch.Tensor): Indices of the selected subset.
    """

    # ===== Handling of X =====

    # if X is already a dataloader we convert it into a torch tensor
    if isinstance(X, torch.utils.data.DataLoader):
        X.shuffle = False
        X = torch.concat([elem[0] for elem in iter(X)])

    # if X is a dataset we convert it into a torch tensor
    elif isinstance(X, torch.utils.data.TensorDataset):
        X = X[:][0]

    # ===== Handling of subset =====

    # if subset is an int we randomly choose that many samples
    if isinstance(subset, int):
        subset_idx = np.random.choice(range(X.shape[0]), size=subset, replace=False)

    # if subset is a list or numpy array we use that as indices
    elif isinstance(subset, list | np.ndarray):
        subset_idx = np.array(subset)

    # if subset is None we use all samples in order
    elif subset is None:
        subset_idx = np.arange(X.shape[0])

    subset = X[subset_idx]

    return torch.Tensor(subset), torch.Tensor(subset_idx).int()
