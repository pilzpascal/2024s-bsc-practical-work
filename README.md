# Deep Active Learning for Image Data

- Author: Pascal Pilz
- Supervisor: Mykyta Ielanskyi
- Based on Gal et al. [<a href="#ref1">1</a>]

## About this Repository

This repository holds the **code**, **parameters**, and **results**, as well as the corresponding **report** for the experiments I conducted as part of my Practical Work in AI at the [Institute for Machine Learning](https://www.jku.at/en/institute-for-machine-learning/) at JKU, Linz.

The point of the report is to reproduce the findings presented in Section 5.1 of Gal et al. [<a href="#ref1">1</a>]., i.e., to compare four uncertainty-based acquisition function to a random baseline, as well as to full-dataset training.

The following table holds a quick overview of the most important things:

| what                              | where                                           |
|-----------------------------------|-------------------------------------------------|
| Experiment Parameters and Results | `Experiment Results/2025-02-20_22-21-17.yaml`   |
| Visualization of Results          | `Experiment Results and Plots.ipynb`            |
| Report                            | _TODO_                                          |

## How to run

To install the necessary dependencies, simply run the following command:

```shell
conda env create -f dependencies.yml
```

And to run the experiments, execute `main.py`:

```shell
conda activate al
python main.py
```

The results will be saved in a yaml file named after the data and time the experiment was started, and can be found in [`Experiment Results/<date-time>`](Experiment&#32;Results).

Parameters are passed via [`config.py`](config.py).
For an explanation of the configurable parameters, see [Parameters](#parameters).

## Parameters

The experiment configuration is saved in [`config.py`](config.py), with the parameters used for our experiments in comment.
The script in [`main.py`](main.py) import [`config.py`](config.py) and automatically uses the parameters settings stored in there.

In order to run the experiments with different parameters, you need to change them in [`config.py`](config.py).

### Experiment Parameter

| Parameter name                  | Function                                                      |
|---------------------------------|---------------------------------------------------------------|
| `WHICH_ACQ_FUNCS` :: `List[str]` | List of acquisition function names.                          |
| `SEED_SEQUENCE` :: `List[int]`   | List of seeds. `len(seed_sequence) == n_runs` must be `True` |
| `RUN_ON_FULL` :: `bool`          | Whether to train and test on the full dataset.               |
| `N_RUNS` :: `int`                | Number of experiment repetitions.                            |
| `TRAIN_SIZE` :: `int`            | Number of training samples.                                  |
| `VAL_SIZE` :: `int`              | Number of validation samples.                                |
| `DATA_PATH` :: `str`             | Path to the dataset.                                         |
| `MODEL_SAVE_PATH_BASE` :: `str`  | Path to save trained models.                                 |
| `EXP_SAVE_PATH_BASE` :: `str`    | Directory to save experiment results.                        |

### Active Learning Parameter

| Parameter name                 | Function                                            |
|--------------------------------|-----------------------------------------------------|
| `N_ACQUISITION_STEPS` :: `int`  | Number of acquisition steps in active learning.    |
| `N_SAMPLES_TO_ACQUIRE` :: `int` | Number of samples to acquire per acquisition step. |
| `POOL_SUBSET_SIZE` :: `int`     | Subset size of the pool for acquisition.           |
| `TEST_SUBSET_SIZE` :: `int`     | Number of test samples.                            |
| `NUM_MC_SAMPLES` :: `int`       | Number of Monte Carlo dropout samples.             |

### Training Parameter

| Parameter name            | Function                                                                                                                                                |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N_EPOCHS` :: `int`       | Number of training epochs.                                                                                                                              |
| `EARLY_STOPPING` :: `int` | Early stopping patience.                                                                                                                                |
| `WHICH_MODEL` :: `str`    | Model to use for training and testing. Can either be `'LeNet'` or `'ConvNN'`. `'LeNet'` is a simple CNN, `'ConvNN'` is the one used by Gal et al. 2017. |

## References

<a id="ref1">[1]</a> Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017. Deep Bayesian Active Learning with Image Data. Retrieved March 7, 2024 from http://arxiv.org/abs/1703.02910
