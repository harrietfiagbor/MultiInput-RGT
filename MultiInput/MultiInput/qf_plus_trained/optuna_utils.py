import os
import logging
import optuna
from experiment import run_experiments
from neptune_utils import optuna_neptune_callback


def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def choose_config(trial, config):
    logging.info("New config options to be selected.")
    new_config = {}

    for key, value in config.items():
        logging.debug(f"key = {key}")
        logging.debug(f"value = {value}")

        choosing = "choose" in key
        logging.debug(f"choosing = {choosing}")

        if choosing:
            if "choose_category" in key:
                prefix = "choose_category"
                chosen_value = trial.suggest_categorical(key, value)
            elif "choose_categories" in key:
                prefix = "choose_categories"
                if "channels" in key:
                    if config["variable_num_channels"]:
                        min_categories = config["min_num_channels"]
                        max_categories = config["max_num_channels"]
                        num_categories = trial.suggest_int(
                            "num_channels", min_categories, max_categories
                        )
                    else:
                        num_categories = config["num_channels"]

                chosen_value = []
                none_value = "none"
                for category_index in range(num_categories):
                    temp_key = key + "_" + str(category_index)
                    logging.debug(f"temp_key = {temp_key}")
                    temp_chosen_value = trial.suggest_categorical(temp_key, value)
                    if temp_chosen_value != none_value:
                        chosen_value.append(temp_chosen_value)
                chosen_value.sort()
            else:
                if "choose_bool" in key:
                    prefix = "choose_bool"
                    chosen_value = trial.suggest_categorical(key, [True, False])
                else:
                    min_value = value[0]
                    max_value = value[1]
                    if "choose_int" in key:
                        prefix = "choose_int"
                        chosen_value = trial.suggest_int(key, min_value, max_value)
                    elif "choose_log_int" in key:
                        prefix = "choose_log_int"
                        chosen_value = int(
                            trial.suggest_loguniform(key, min_value, max_value)
                        )
                    elif "choose_float" in key:
                        prefix = "choose_float"
                        chosen_value = trial.suggest_uniform(key, min_value, max_value)
                    elif "choose_log" in key:
                        prefix = "choose_log"
                        chosen_value = trial.suggest_loguniform(
                            key, min_value, max_value
                        )
                    else:
                        raise KeyError(f"Invalid choose prefix. Key = {key}.")
            logging.debug(f"prefix = {prefix}")
            new_key = key[len(prefix) + 1 :]
        else:
            new_key = key
            chosen_value = value
        logging.debug(f"new_key = {new_key}")
        logging.debug(f"chosen_value = {chosen_value}")
        new_config[new_key] = chosen_value
    return new_config


def objective(trial, config, study):
    logging.info("New trial begun.")
    logging.info(
        f'Trials = {study.trials_dataframe(attrs=("number", "value", "state"))}'
    )

    logging.info("Choosing new config...")
    new_config = choose_config(trial, config)
    logging.debug(f"new_config = {new_config}")
    logging.info("New config chosen.")

    logging.info("Running experiment...")
    final_score = run_experiments(new_config, trial)
    logging.info(f"final_score = {final_score}")
    logging.info("Experiment done.")

    return final_score


def run_study(config):

    logging.info("Initializing optuna database storage...")
    db_password = os.environ["POSTGRES_PASSWORD"]
    db_url = f"postgresql://postgres:{db_password}@semlerdb.thetatech.ai:5432/optunadb"
    storage = optuna.storages.RDBStorage(url=db_url)
    logging.info("Database storage initialized.")

    config["pruning"] = True
    if config["pruner"] == "median_pruner":
        pruner = optuna.pruners.MedianPruner()
    elif config["pruner"] == "nop_pruner":
        pruner = optuna.pruners.NopPruner()
    elif config["pruner"] == "percentile_pruner":
        pruner = optuna.pruners.PercentilePruner()
    elif config["pruner"] == "successive_halving_pruner":
        pruner = optuna.pruners.SuccessiveHalvingPruner()
    elif config["pruner"] == "hyperband_pruner":
        pruner = optuna.pruners.HyperbandPruner()
    elif config["pruner"] == "threshold_pruner":
        pruner = optuna.pruners.ThresholdPruner()

    if len(config["optuna_metric"]) > 1:
        config["pruning"] = False
        pruner = optuna.pruners.NopPruner()

    directions = []
    for metric in config["optuna_metric"]:
        if ("loss" in metric) or ("mse" in metric):
            directions.append("minimize")
        else:
            directions.append("maximize")

    logging.info("Creating optuna study...")
    optuna_callback = optuna_neptune_callback(config)
    study_name = config["study_id"]
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=directions,
        pruner=pruner,
    )
    logging.info("Optuna study created.")

    logging.info("Getting trials for this study...")
    logging.info(
        f'Trials = {study.trials_dataframe(attrs=("number", "value", "state"))}'
    )

    logging.info("Beginning hyperparameter optimization...")
    study.optimize(
        lambda trial: objective(trial, config, study),
        n_trials=config["trials"],
        n_jobs=1,
        callbacks=[optuna_callback],
    )
    logging.info("Ended hyperparameter optimization.")
