from src.experiment_handling import get_experiment, run_experiment
import config as cfg


def main() -> dict:

    # returns a list of experiments corresponding to the given acquisition functions and hyperparameters
    experiment = get_experiment(
        which_acq_funcs=cfg.WHICH_ACQ_FUNCS,
        seed_sequence=cfg.SEED_SEQUENCE,
        run_on_full=cfg.RUN_ON_FULL,
        n_runs=cfg.N_RUNS,
        train_size=cfg.TRAIN_SIZE,
        val_size=cfg.VAL_SIZE,
        data_path=cfg.DATA_PATH,
        exp_save_path_base=cfg.EXP_SAVE_PATH_BASE,
        model_save_path_base=cfg.MODEL_SAVE_PATH_BASE,

        n_acquisition_steps=cfg.N_ACQUISITION_STEPS,
        n_samples_to_acquire=cfg.N_SAMPLES_TO_ACQUIRE,
        pool_subset_size=cfg.POOL_SUBSET_SIZE,
        test_subset_size=cfg.TEST_SUBSET_SIZE,
        num_mc_samples=cfg.NUM_MC_SAMPLES,

        n_epochs=cfg.N_EPOCHS,
        early_stopping=cfg.EARLY_STOPPING,
        which_model=cfg.WHICH_MODEL,
    )

    experiment_results = run_experiment(experiment)

    return experiment_results


if __name__ == '__main__':

    results = main()

    print('Done!')
    print(f'Saved results to {results['params']['exp']['exp_save_path_base']}')

