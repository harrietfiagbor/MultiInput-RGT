import logging
import neptune.new as neptune
from neptune.new.integrations.fastai import NeptuneCallback
import neptune.new.integrations.optuna as optuna_utils
import os


def optuna_neptune_callback(config):
    logging.info("Optuna Hyperparam Optimisation running.")

    NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
    logging.info(f"NEPTUNE_PROJECT = {NEPTUNE_PROJECT}")

    logging.info("Run initiating...")
    custom_run_id = config["optuna_run_id"]
    mode = config["neptune_connection_mode"]
    if mode == "async":
        flush_period = float(config["neptune_flush_period"])
        neptune_run = neptune.init(
            tags=["Optuna Dashboard"],
            custom_run_id=custom_run_id,
            mode=mode,
            flush_period=flush_period,
        )
    else:
        neptune_run = neptune.init(
            tags=["Optuna Dashboard"], custom_run_id=custom_run_id, mode=mode
        )

    logging.info("Run initiated.")
    logging.info("Storing hyperparameters...")
    neptune_run["parameters"] = config
    logging.info("Hyperparameters stored.")
    neptune_callback = optuna_utils.NeptuneCallback(neptune_run)

    return neptune_callback


def start_experiment(config):
    logging.info("Experiment running.")
    logging.info(f"config = {config}")

    NEPTUNE_PROJECT = os.environ.get("NEPTUNE_PROJECT")
    logging.info(f"NEPTUNE_PROJECT = {NEPTUNE_PROJECT}")

    mode = config["neptune_connection_mode"]
    logging.info(f"Neptune Connection mode = {mode}")
    if mode == "async":
        flush_period = float(config["neptune_flush_period"])
    logging.info("Run initiating...")
    if mode == "async":
        neptune_run = neptune.init(
            tags=config["tags"], mode=mode, flush_period=flush_period
        )
    else:
        neptune_run = neptune.init(tags=config["tags"], mode=mode)

    logging.info("Run initiated.")

    logging.info(f"neptune_run = {neptune_run}")

    logging.info("Storing hyperparameters...")
    neptune_run["parameters"] = config
    logging.info("Hyperparameters stored.")

    return neptune_run


def get_callback(neptune_run):
    return NeptuneCallback(run=neptune_run, base_namespace="experiment")


def end_experiment(neptune_run):
    logging.info("Run stopping...")
    neptune_run.stop(5)
    logging.info("Run stopped.")
