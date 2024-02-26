import logging
from data_utils import *
from neptune_utils import *
from timeseriesai import train
import uuid
from statistics import mean, variance
from gradcam_map import color_signals
import hashlib
from random import choice
from fastai.learner import Recorder
from scatter_regression import plot_scatter_regression


def run_experiment(config, optuna_trial, pickle_data=None, meta_data=None):
    # adding trial uuid study uuid and data uuid to tags
    config["tags"] = config["tags"] + [
        config["study_id"],
        config["trial_uuid"],
        config["data_uuid"],
    ]
    grad_cam = config["grad_cam"]

    neptune_run = None
    if optuna_trial.number >= config["log_threshold"]:
        logging.info("Experiment starting...")
        neptune_run = start_experiment(config)
        logging.info("Experiment started and logged to https://neptune.ai")

    logging.info(f"Experiment running with trial_id{config['trial_uuid']}")

    if pickle_data is None:
        logging.info("Loading pickle...")
        pickle_data = load_pickle(config["data_uuid"] + ".p")
        logging.info("Pickle loaded.")

    if meta_data is None:
        logging.info("Loading meta data...")
        meta_data = load_meta_data()
        logging.info("Meta data loaded.")
        logging.info(f"meta_data = {meta_data}")

    if neptune_run:
        logging.info("Training...")
        report_metric, learner = train(
            meta_data, pickle_data, config, optuna_trial, neptune_run
        )

        logging.info("Trained.")

        if grad_cam:
            logging.info("Plotting gradcam")
            color_signals(config, learner, pickle_data, neptune_run)
            logging.info("Gradcam plotted")

        if config["prediction_type"] == "regression":
            logging.info("Plotting regression scatter")
            neptune_run = plot_scatter_regression(
                meta_data, pickle_data, config, neptune_run
            )
            logging.info("Regression scatter plotted")

        end_experiment(neptune_run)
        logging.info("Experiment ended.")
        return report_metric

    logging.info("Training...")
    report_metric, learner = train(meta_data, pickle_data, config, optuna_trial)
    logging.info("Trained.")

    tn = report_metric["TN"]
    fp = report_metric["FP"]
    fn = report_metric["FN"]
    tp = report_metric["TP"]

    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    balanced_accuracy_score = (TPR + TNR) / 2

    if balanced_accuracy_score >= 0.80 and neptune_run is None:
        logging.info("Logging models for balanced accuracy >= 85%")
        config["tags"] = config["tags"] + ["partial neptune run"]

        neptune_run = start_experiment(config)
        logging.info("Experiment logging to https://neptune.ai")
        if config["prediction_type"] == "classification":
            neptune_run[
                "experiment/metrics/fit_1/validation/loader/balanced_accuracy_score"
            ].log(balanced_accuracy_score)
        else:
            neptune_run[
                "experiment/metrics/fit_1/validation/loader/balanced_accuracy_regression"
            ].log(balanced_accuracy_score)

        for key, value in report_metric.items():
            neptune_run["experiment/metrics/fit_1/validation/loader/" + key].log(value)

        logging.info("Removing callbacks from learner")
        learner.remove_cbs(learner.cbs + get_callback(neptune_run) + Recorder)

        logging.info("Exporting best model")
        learner.export("model.pkl")
        logging.info("Logging best model to neptune")
        neptune_run["experiment/io_files/artifacts/saved_model"].upload(
            os.path.join("model.pkl")
        )

        if config["prediction_type"] == "regression":
            logging.info("Plotting Scatter for Regression")
            neptune_run = plot_scatter_regression(
                meta_data, pickle_data, config, neptune_run
            )
            logging.info("Plotted Scatter for Regression")
        else:
            neptune_run["confusion_matrix_plot"].upload(
                os.path.join("confusion_matrix.png")
            )

        end_experiment(neptune_run)
        logging.info("Experiment ended.")

    if grad_cam:
        logging.info("Plotting gradcam")
        color_signals(config, learner, pickle_data)
        logging.info("Gradcam plotted")

    return report_metric


def run_experiments(config, optuna_trial):
    TP, TN, FP, FN = 0, 0, 0, 0
    validation_losses = []

    logging.info("Experiment starting...")

    logging.info("Loading pickle...")
    pickle_data = load_pickle(config["data_uuid"] + ".p")
    logging.info("Pickle loaded.")
    if "file_data" in pickle_data.keys():

        def concat_and_hash(validation_list):
            validation_list = sorted(validation_list)
            long_str = "".join(validation_list).encode("utf-8")
            hashed_string = hashlib.sha256(long_str).hexdigest()
            return hashed_string

        valid_splits = pickle_data["val_test_splits"]["val_splits"]
        hashes = [concat_and_hash(valid_split) for valid_split in valid_splits]
        config["validation_set"] = dict(zip(hashes, valid_splits))
        test_splits = pickle_data["val_test_splits"]["test_subjects"]
        config["test_patients"] = test_splits
        config["test_set_id"] = concat_and_hash(test_splits)
        pickle_data = pickle_data["file_data"]

    logging.info("Loading meta data...")
    meta_data = load_meta_data()
    logging.info("Meta data loaded.")
    logging.info(f"meta_data = {meta_data}")

    config["trial_uuid"] = str(uuid.uuid4())
    logging.info(f"Running experiments with uuid {config['trial_uuid']}")

    validation_dict = config["validation_set"].copy()
    del config["validation_set"]

    if config["cross_validation"] == False:
        temp_dict = {}
        validation_choice = choice(list(validation_dict.keys()))
        temp_dict[validation_choice] = validation_dict[validation_choice]
        validation_dict = temp_dict

    for key, value in validation_dict.items():

        new_config = config.copy()
        new_config["validation_set_id"] = key
        new_config["validation_patients"] = value
        logging.info(f"Running experiment with validation set id {key}")

        report_metric = run_experiment(new_config, optuna_trial, pickle_data, meta_data)

        TN += report_metric["TN"]
        FP += report_metric["FP"]
        FN += report_metric["FN"]
        TP += report_metric["TP"]
        validation_losses.append(report_metric["valid_loss"])

    logging.info(f"aggreagated TP{TP} TN {TN} FN {FN} FP {FP}")
    logging.info(f"Validation losses over validation splits {validation_losses}")

    if len(config["optuna_metric"]) > 1:
        config["pruning"] = False

    optuna_metrics = []
    for metric in config["optuna_metric"]:

        if (metric == "balanced_accuracy_score") or (
            metric == "balanced_accuracy_regression"
        ):
            TPR = TP / (TP + FN)
            TNR = TN / (TN + FP)
            if config["report_best"]:
                balanced_accuracy_score = report_metric["best_balanced_accuracy"]
            else:
                balanced_accuracy_score = (TPR + TNR) / 2
            optuna_metrics.append(balanced_accuracy_score)
            logging.info(f"aggregated balanced accuracy {balanced_accuracy_score}")

        elif metric == "mae":
            if config["report_best"]:
                optuna_metrics.append(report_metric["best_mae"])
            else:
                optuna_metrics.append(report_metric["mae"])

        elif metric == "_rmse":
            if config["report_best"]:
                optuna_metrics.append(report_metric["best_rmse"])
            else:
                optuna_metrics.append(report_metric["_rmse"])

        elif metric == "f1_score":
            f1_score = TP / (TP + (0.5 * (FP + FN)))
            optuna_metrics.append(f1_score)
            logging.info(f"aggregated f1 {f1_score}")

        elif metric == "mean_validation_loss":
            mean_valid_loss = mean(validation_losses)
            optuna_metrics.append(mean_valid_loss)
            logging.info(f"mean validation loss {mean_valid_loss}")

        elif metric == "variance_validation_loss":
            variance_valid_loss = variance(validation_losses)
            optuna_metrics.append(variance_valid_loss)
            logging.info(f"variance validation loss {variance_valid_loss}")

        elif metric in report_metric:
            optuna_metrics.append(report_metric[metric])

    logging.info("Trained.")

    return tuple(optuna_metrics)
