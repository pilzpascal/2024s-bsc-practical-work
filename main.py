from src.experiment_handling import get_experiments, run_experiments
from src.reproducibility import set_seed
import config as cfg


def main() -> tuple:

    # returns a list of experiments corresponding to the given acquisition functions and hyperparameters
    experiments = get_experiments(which_acq_funcs=cfg.ACQ_FUNCS,
                                  n_acquisition_steps=cfg.N_ACQUISITION_STEPS,
                                  n_samples_to_acquire=cfg.N_SAMPLES_TO_ACQUIRE,
                                  n_epochs=cfg.N_EPOCHS,
                                  val_size=cfg.VAL_SIZE,
                                  seed=cfg.SEED,
                                  T=cfg.T,
                                  pool_subset_size=5_000,
                                  test_subset_size=3_000,
                                  n_runs=1,  # 3,
                                  early_stopping=-1)

    experiment_results, save_path = run_experiments(experiments,
                                                    model_save_path=cfg.MODEL_SAVE_PATH,
                                                    experiment_save_path=cfg.EXPERIMENT_SAVE_PATH,
                                                    data_path=cfg.DATA_PATH)

    return experiment_results, save_path


if __name__ == '__main__':

    set_seed(cfg.SEED)

    results, results_path = main()

    print(f'Saved results to {results_path}')

