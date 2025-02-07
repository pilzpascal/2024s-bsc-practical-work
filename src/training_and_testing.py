import os
from tqdm.auto import tqdm

import torch
import numpy as np

from src.networks import LeNet
from src.acquisition_functions import mutual_information, get_info_and_predictions


def get_trained_model(train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
                      model_save_path: str, n_epochs: int = 100, early_stopping: int = 10)\
        -> LeNet:

    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            outputs = model(inputs, use_dropout=True)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # get validation loss
        running_vloss = 0.0
        model.eval()
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


def test_model(model, dataloader, T=64, subset=None, show_pbar=False)\
        -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    infos, predictions, subset_idx = get_info_and_predictions(model, dataloader, subset=subset, show_pbar=show_pbar,
                                                              acquisition_function=mutual_information, T=T)
    # get average mutual information per data point
    infos = torch.Tensor(infos).mean()

    # get accuracy
    labels = torch.utils.data.DataLoader(dataloader.dataset, batch_size=len(dataloader.dataset), shuffle=False)
    labels = next(iter(labels))[1][subset_idx]
    acc = (torch.Tensor(predictions) == labels).float().mean()

    return infos, acc, subset_idx
