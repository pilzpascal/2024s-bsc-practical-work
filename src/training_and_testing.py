import os
from tqdm.auto import tqdm

import torch
import numpy as np

from src.networks import LeNet
from src.acquisition_functions import mutual_information, get_info_and_predictions


def get_trained_model(
        X_train, y_train,
        val_loader: torch.utils.data.DataLoader,
        model_save_path_base: str,
        n_epochs: int = 100,
        early_stopping: int = 10
) -> LeNet:

    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=False)

    model = LeNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_vloss = np.inf
    epochs_since_best_vloss = 0
    best_model_state_dict = model.state_dict()

    os.makedirs(model_save_path_base, exist_ok=True)
    model_save_path = model_save_path_base + f'trainsize-{len(train_loader.dataset):05d}'

    epochs = tqdm(desc=f'Training Model with Training size {len(train_loader.dataset):_d}. Epoch', leave=False)
    if early_stopping == -1:
        epochs.total = n_epochs

    for epoch in range(n_epochs):

        # train model
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, use_dropout=True)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # get validation loss
        model.eval()
        running_vloss = 0.0
        with torch.no_grad():
            for vinputs, vlabels in val_loader:
                voutputs = model(vinputs, use_dropout=False)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
        avg_vloss = running_vloss / len(val_loader)

        # track the best performance, and save the model's state
        if avg_vloss < best_vloss:
            epochs_since_best_vloss = 0
            best_vloss = avg_vloss
            save_path = model_save_path + f'_epoch-{epoch:04d}_valloss-{best_vloss:.3f}'
            best_model_state_dict = model.state_dict()

        # Early stopping if no improvement for a certain number of epochs
        else:
            epochs_since_best_vloss += 1
            if epochs_since_best_vloss == early_stopping:
                break

        epochs.update()
    epochs.close()

    # save the best model, determined via validation set
    torch.save(best_model_state_dict, save_path)

    best_model = LeNet()
    best_model.load_state_dict(best_model_state_dict)

    return best_model


def test_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_mc_samples,
        subset: int | None,
        show_pbar: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:

    dataloader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=len(dataloader.dataset),
        shuffle=False
    )

    infos, predictions, subset_idx = get_info_and_predictions(
        model, dataloader,
        subset=subset,
        show_pbar=show_pbar,
        acquisition_function=mutual_information,
        num_mc_samples=num_mc_samples
    )

    # get average mutual information per data point
    inf = torch.Tensor(infos).mean()

    # get accuracy
    labels = next(iter(dataloader))[1][subset_idx]
    predictions = torch.as_tensor(predictions)
    acc = (predictions == labels).float().mean()

    return inf, acc


def train_and_test_full_dataset(
        X_train, y_train,
        X_pool, y_pool,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        model_save_path: str,
        num_mc_samples: int,
        n_epochs: int,
        early_stopping: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    X_train_full = torch.cat([X_train, X_pool])
    y_train_full = torch.cat([y_train, y_pool])

    model_save_path = model_save_path + 'full_dataset/'

    model = get_trained_model(
        X_train_full, y_train_full,
        val_loader=val_loader,
        model_save_path_base=model_save_path,
        n_epochs=n_epochs,
        early_stopping=early_stopping
    )

    inf, acc = test_model(model, test_loader, num_mc_samples, subset=None, show_pbar=True)

    return inf, acc
