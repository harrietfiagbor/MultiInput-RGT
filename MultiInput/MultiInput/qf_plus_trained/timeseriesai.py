# -*- coding: utf-8 -*-
import logging
import numpy as np
from regression_metics import *
from scipy.signal import resample
import math
import random
from fastai.metrics import *
from tsai.all import *
from neptune_utils import *
from fastaipruning import FastAIMultiValPruningCallback
import inspect
import re
from qf_plus_development.ai_utils import LSUVinit
from functools import partial
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from confusion_matrix_regression import plot_confusion_matrix_regression


hz = 50  # samples per second


def train(meta_data, loaded_data, config_params, optuna_trial, neptune_run=None):

    healthy_threshold = config_params["healthy_threshold"]
    resample_points = config_params["resample_points"]
    channels = config_params["channels"]
    min_whistle_seconds = config_params["min_whistle_seconds"]
    max_whistle_seconds = config_params["max_whistle_seconds"]
    batch_size = config_params["batch_size"]
    model_name = config_params["model_name"]
    epochs = config_params["epochs"]
    predictor_col = config_params["predictor_col"]
    window_length_seconds = config_params[
        "window_length_seconds"
    ]  # centered halfway through whistle
    max_learning_rate = config_params["max_lr"]
    test_patients = config_params["test_patients"]
    optuna_metric = config_params["optuna_metric"]
    pruning = config_params["pruning"]
    save_metric = config_params["save_metric"]
    preprocessing_type = config_params["preprocessing_type"]
    norm_by_var = config_params["norm_by_var"]
    norm_by_sample = config_params["norm_by_sample"]
    normalize_initialization = config_params["normalize_initialization"]
    prediction_type = config_params["prediction_type"]
    optuna_min_step = config_params["optuna_min_step"]
    window_length_points = window_length_seconds * hz

    filenames = list(loaded_data.keys())
    filenames.sort()
    logging.info(f"filenames = {filenames}")
    logging.info(f"meta_data = {meta_data}")

    def get_subject_id(filename):
        subject_id = filename[:-6]
        if "AUG" in subject_id:
            subject_id = re.sub("AUG_[0-9]+_", "", subject_id)
        return subject_id

    x_arr = []
    y_arr = []
    used_filenames = []
    count_insufficient_whistle_time = 0
    count_insufficient_signal_length = 0
    count_invalid_predictor = 0
    count_use4train_false = 0
    count_exceptions = 0
    count_test = 0

    for index, filename in enumerate(filenames):
        logging.debug(f"filename = {filename}")

        if get_subject_id(filename) in test_patients:
            logging.info("This is part of the test set. Skipping.")
            count_test += 1
            continue

        try:

            orig_df_columns = channels

            orig_df = loaded_data[filename]
            data = orig_df[orig_df_columns].to_numpy()
            logging.debug(f"data.shape = {data.shape}")
            whistle_binary = (orig_df["Category"] == "Whistle").to_numpy()
            if not whistle_binary.size:
                raise IndexError(f"orig_df.Category is empty for filename={filename}.")

            first_whistle = np.argmax(whistle_binary)
            last_whistle = len(whistle_binary) - 1 - np.argmax(whistle_binary[::-1])
            whistle_length_secs = (last_whistle - first_whistle) / hz
            if (
                whistle_length_secs < min_whistle_seconds
                or whistle_length_secs > max_whistle_seconds
            ):
                logging.warning(
                    f"Insufficient whistle length. It is {whistle_length_secs} and must be between {min_whistle_seconds} and {max_whistle_seconds} (exclusive). Skipping this signal."
                )
                count_insufficient_whistle_time += 1
                continue

            mid_whistle = (last_whistle + first_whistle) // 2
            first_index = mid_whistle - window_length_points // 2
            first_index = max(first_index, 0)
            last_index = first_index + window_length_points
            entire_signal = data[first_index:last_index, :]
            logging.debug(f"entire_signal.shape = {entire_signal.shape}")
            logging.debug(f"entire_signal.shape = {entire_signal.shape}")
            if entire_signal.shape[0] < window_length_points:
                logging.warning(
                    f"Insufficient signal acquired before / after the whistle. Was {round(entire_signal.shape[0]/hz,1)} seconds, but must be at least {round(window_length_points/hz,1)} seconds."
                )
                count_insufficient_signal_length += 1
                continue
            entire_signal = resample(entire_signal, num=resample_points)

            subject_id = get_subject_id(filename)
            logging.debug(f"subject_id = {subject_id}")

            predictor = meta_data.loc[meta_data["Subject_ID"] == subject_id][
                predictor_col
            ].iloc[0]
            logging.debug(f"{predictor_col} = {predictor}")
            if type(predictor) is str or math.isnan(predictor):
                logging.warning(
                    f"Invalid {predictor_col} value of {type(predictor)} {predictor}. Skipping."
                )
                count_invalid_predictor += 1
                continue

            use_4_train = meta_data.loc[meta_data["Subject_ID"] == subject_id][
                "Use_4_Train"
            ].iloc[0]
            if not use_4_train:
                logging.warning(f"use_4_train = {use_4_train}. Skipping.")
                count_use4train_false += 1
                continue

            x_arr.append(entire_signal)
            if prediction_type == "classification":
                sick = int(predictor < healthy_threshold)
                logging.debug(f"sick = {sick}")
                y_arr.append(sick)
            else:
                y_arr.append(predictor)

            used_filenames.append(filename)
            logging.debug(f"\nlen(used_filenames) = {len(used_filenames)}")

        except (IndexError, KeyError) as error:
            logging.warning(f"error = {error}. Skipping")

    patient_ids = list(set([get_subject_id(filename) for filename in used_filenames]))
    patient_ids.sort()
    logging.debug(f"patient_ids = {patient_ids}")

    counts = {
        "filtered_patients": len(patient_ids),
        "original_patients": len(set([get_subject_id(f) for f in filenames])),
        "filtered_signals": len(used_filenames),
        "original_signals": len(filenames),
        "insufficient_whistle_time": count_insufficient_whistle_time,
        "insufficient_signal_length": count_insufficient_signal_length,
        "invalid_predictor": count_invalid_predictor,
        "use4train_false": count_use4train_false,
        "exceptions": count_exceptions,
        "test": count_test,
    }
    if neptune_run:
        neptune_run["counts"] = counts

    logging.info(f"len(patient_ids) = {len(patient_ids)} (# of patients, or N)")
    logging.info(f'{counts["original_patients"]} = (# of patients before filtering)')
    logging.info(f"len(filenames) = {len(filenames)} (# of signals before filtering)")
    logging.info(
        f"len(used_filenames) = {len(used_filenames)}/{len(filenames)} = {round(len(used_filenames)*100/len(filenames))}% (# of signals)"
    )
    logging.info(
        f"count_insufficient_whistle_time = {count_insufficient_whistle_time}/{len(filenames)} = {round(count_insufficient_whistle_time*100/len(filenames))}% (whistle length not between {min_whistle_seconds} and {max_whistle_seconds} seconds)"
    )
    logging.info(
        f"count_insufficient_signal_length = {count_insufficient_signal_length}/{len(filenames)} = {round(count_insufficient_signal_length*100/len(filenames))}% (signal length not at least {window_length_seconds} seconds)"
    )
    logging.info(
        f"count_invalid_predictor = {count_invalid_predictor} = {round(count_invalid_predictor*100/len(filenames))}% (no {predictor_col} value available)"
    )
    logging.info(
        f"count_use4train_false = {count_use4train_false}/{len(filenames)} = {round(count_use4train_false*100/len(filenames))}% (signals the operator marked as data which should not be used for training the AI, such as patients with pacemakers or who failed to properly perform the valsalva maneuver)"
    )
    logging.info(
        f"count_exceptions = {count_exceptions}/{len(filenames)} = {round(count_exceptions*100/len(filenames))}% (for situations such as missing data)"
    )
    logging.info(
        f"count_test = {count_test}/{len(filenames)} = {round(count_test*100/len(filenames))}% (for testing data)"
    )

    X = np.swapaxes(np.array(x_arr), 1, 2)
    y = np.array(y_arr)
    logging.debug(f"X.shape = {X.shape}")
    logging.debug(f"y.shape = {y.shape}")
    logging.debug(f"np.histogram(y, bins=2) = {np.histogram(y, bins=2)}")

    if "validation_patients" in config_params:
        validation_patients = config_params["validation_patients"]
    else:
        validation_percent = config_params["validation_percent"]
        validation_patients = random.sample(
            patient_ids, int(validation_percent * len(patient_ids))
        )
    validation_patients.sort()
    logging.debug(f"validation_patients = {validation_patients}")
    logging.debug(f"{len(validation_patients)} patients in validation set")

    training_indices = []
    validation_indices = []
    for index, filename in enumerate(used_filenames):
        filename = used_filenames[index]
        if "AUG" in filename:
            continue
        indices = (
            validation_indices
            if (get_subject_id(filename) in validation_patients)
            else training_indices
        )
        indices.append(index)
    logging.debug(f"validation_indices = {validation_indices}")
    logging.debug(f"len(validation_indices) = {len(validation_indices)}")
    logging.debug(f"training_indices = {training_indices}")
    logging.debug(f"len(training_indices) = {len(training_indices)}")
    splits = [training_indices, validation_indices]

    if prediction_type == "classification":
        metrics = [
            F1Score(),
            Precision(),
            Recall(),
            accuracy,
            BalancedAccuracy(),
            CohenKappa(),
        ]
        tfms = [None, Categorize()]

    else:
        f1_regression = partial(F1_regression, healthy_lvef=healthy_threshold)
        ba_regression = partial(
            balanced_accuracy_regression, healthy_lvef=healthy_threshold
        )
        prec_regression = partial(precision_regression, healthy_lvef=healthy_threshold)
        rec_regression = partial(recall_regression, healthy_lvef=healthy_threshold)

        metrics = [
            f1_regression,
            ba_regression,
            prec_regression,
            rec_regression,
            mse,
            rmse,
            mae,
            msle,
            exp_rmspe,
            ExplainedVariance(),
            R2Score(),
        ]
        tfms = [None, [TSRegression()]]

    by_var = False
    by_sample = False

    if norm_by_sample == 1:
        by_sample = True
    if norm_by_var == 1:
        by_var = True

    preprocessings = [
        "Standardize",
        "RobustScale",
    ]

    logging.debug(f"preprocessing type = {preprocessing_type}")
    assert preprocessing_type in preprocessings

    if preprocessing_type == "Standardize":
        preprocessing = TSStandardize(by_var=by_var, by_sample=by_sample)
    elif preprocessing_type == "RobustScale":
        preprocessing = TSRobustScale(by_var=by_var, by_sample=by_sample)
    else:
        raise BaseException("Invalid pre processing type")

    batch_tfms = [preprocessing]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    dls = TSDataLoaders.from_dsets(
        dsets.train, dsets.valid, bs=batch_size, batch_tfms=batch_tfms
    )

    # https://timeseriesai.github.io/tsai/
    model_names = [
        "InceptionRocketPlus",
        "InceptionTime",
        "InceptionTimePlus",
        "XceptionTime",
        "XceptionTimePlus",
        "ResNet",
        "ResNetPlus",
        "ResCNN",
        "TCN",
        "XCM",
        "XCMPlus",
        "RNN_FCN",
        "LSTM_FCN",
        "GRU_FCN",
        "MRNN_FCN",
        "MLSTM_FCN",
        "MGRU_FCN",
        "RNN_FCNPlus",
        "LSTM_FCNPlus",
        "GRU_FCNPlus",
        "MRNN_FCNPlus",
        "MLSTM_FCNPlus",
        "MGRU_FCNPlus",
        "mWDN",
        "mWDNPlus",
        "TST",
        "TSTPlus",
        "FCN",
        "FCNPlus",
    ]
    logging.debug(f"model_name = {model_name}")
    assert model_name in model_names
    if model_name == "InceptionTime":
        model = InceptionTime
    elif model_name == "InceptionTimePlus":
        model = InceptionTimePlus
    elif model_name == "XceptionTime":
        model = XceptionTime
    elif model_name == "XceptionTimePlus":
        model = XceptionTimePlus
    elif model_name == "ResNet":
        model = ResNet
    elif model_name == "ResNetPlus":
        model = ResNetPlus
    elif model_name == "InceptionRocketPlus":
        model = InceptionRocketPlus
    elif model_name == "ResCNN":
        model = ResCNN
    elif model_name == "TCN":
        model = TCN
    elif model_name == "XCM":
        model = XCM
    elif model_name == "XCMPlus":
        model = XCMPlus
    elif model_name == "RNN_FCN":
        model = RNN_FCN
    elif model_name == "LSTM_FCN":
        model = LSTM_FCN
    elif model_name == "GRU_FCN":
        model = GRU_FCN
    elif model_name == "MRNN_FCN":
        model = MRNN_FCN
    elif model_name == "MLSTM_FCN":
        model = MLSTM_FCN
    elif model_name == "MGRU_FCN":
        model = MGRU_FCN
    elif model_name == "RNN_FCNPlus":
        model = RNN_FCNPlus
    elif model_name == "LSTM_FCNPlus":
        model = LSTM_FCNPlus
    elif model_name == "GRU_FCNPlus":
        model = GRU_FCNPlus
    elif model_name == "MRNN_FCNPlus":
        model = MRNN_FCNPlus
    elif model_name == "MLSTM_FCNPlus":
        model = MLSTM_FCNPlus
    elif model_name == "MGRU_FCNPlus":
        model = MGRU_FCNPlus
    elif model_name == "mWDN":
        model = mWDN
    elif model_name == "mWDNPlus":
        model = mWDNPlus
    elif model_name == "TST":
        model = TST
    elif model_name == "TSTPlus":
        model = TSTPlus
    elif model_name == "FCN":
        model = FCN
    elif model_name == "FCNPlus":
        model = FCNPlus
    else:
        raise BaseException("Invalid model name")

    if model_name == "ResNetPlus":
        ks = config_params["ks"]
        config_params["ks"] = [ks, ks - 2, ks - 4]

    model_args = inspect.getfullargspec(model).args
    model_kwargs = {}
    for arg in model_args:
        if arg in config_params:
            model_kwargs[arg] = config_params[arg]
    model = create_model(model, dls=dls, **model_kwargs)

    if normalize_initialization:
        logging.debug("using normalised weight initialization")
        batch_x = dls.one_batch()[0]
        model = LSUVinit(model, batch_x)

    if neptune_run:

        if prediction_type == "regression":
            model._vocab = ""

        learner = Learner(
            dls,
            model,
            metrics=metrics,
            cbs=[SaveModelCallback(monitor=save_metric), get_callback(neptune_run)],
        )
    else:
        learner = Learner(
            dls, model, metrics=metrics, cbs=[SaveModelCallback(monitor=save_metric)]
        )

    logging.debug(f"model = {model}")
    logging.debug(f"type(model) = {type(model)}")
    logging.debug(f"learner = {learner}")

    # https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034/2
    # https://discuss.pytorch.org/t/reinitializing-the-weights-after-each-cross-validation-fold/11034/5
    for layer in model.modules():
        logging.info(f"layer = {layer}")
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
            logging.info("Layer weights reset.")

    logging.info("Running normal fit with 5 epochs")
    logging.warning("This step may take a few minutes...")
    learner.fit(5, lr=(max_learning_rate / 2))
    logging.info("Finished normal fit with 5 epochs")

    start = time.perf_counter()

    if pruning:
        learner.fit_one_cycle(
            n_epoch=epochs,
            lr_max=max_learning_rate,
            cbs=[
                FastAIMultiValPruningCallback(
                    optuna_trial, optuna_metric[0], optuna_min_step
                )
            ],
        )
    else:
        learner.fit_one_cycle(n_epoch=epochs, lr_max=max_learning_rate)

    logging.info(
        f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))}"
    )
    logging.info("TRAINING ROUND DONE.")

    report_values = {}

    def get_best_metric_from_learner(metric_name):
        index = list(learner.recorder.metric_names).index(metric_name) - 2
        metrics = [metric[index] for metric in learner.recorder.values]
        if metric_name in ["mse", "_rmse", "mae", "msle", "_exp_rmspe"]:
            best_metric = min(metrics)
            return best_metric
        best_metric = max(metrics)
        return best_metric

    def get_metric_from_learner(metric_name):
        metric = learner.recorder.final_record[
            list(learner.recorder.metric_names).index(metric_name) - 1
        ]
        return metric

    valid_loss = get_metric_from_learner("valid_loss")
    logging.info(f"Validation loss {valid_loss}")
    report_values["valid_loss"] = valid_loss

    for metric in list(learner.recorder.metric_names)[1:-1]:
        report_values[metric] = get_metric_from_learner(metric)

    if prediction_type == "classification":
        interp = ClassificationInterpretation.from_learner(learner)
        cf_matrix = interp.confusion_matrix()
        logging.info(f"confusion matrix {cf_matrix}")

        interp.plot_confusion_matrix()
        plt.savefig("confusion_matrix.png")
        plt.close()
        upp, low = cf_matrix

        TN, FP = upp[0], upp[1]
        FN, TP = low[0], low[1]

    else:
        predictions = learner.get_preds()
        target = predictions[1].detach().numpy()
        preds = predictions[0].detach().numpy()

        mae_value = mean_absolute_error(target, preds)
        rmse_value = mean_squared_error(preds, target, squared=False)

        thresholded_target = []
        thresholded_preds = []
        for i in range(len(target)):
            thresholded_target.append(int(target[i] < healthy_threshold))
            thresholded_preds.append(int(preds[i] < healthy_threshold))

        cf_matrix = confusion_matrix(thresholded_target, thresholded_preds)
        TN, FP, FN, TP = cf_matrix.ravel()
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        balanced_accuracy_score = (TPR + TNR) / 2

        plt_regression = plot_confusion_matrix_regression(
            target,
            preds,
            healthy_threshold,
            mae_value,
            rmse_value,
            balanced_accuracy_score,
        )
        plt_regression.savefig("confusion_matrix.png")
        plt_regression.close()

        report_values["best_balanced_accuracy"] = get_best_metric_from_learner(
            "balanced_accuracy_regression"
        )
        report_values["best_mae"] = get_best_metric_from_learner("mae")
        report_values["best_rmse"] = get_best_metric_from_learner("_rmse")

    report_values["TN"] = TN
    report_values["TP"] = TP
    report_values["FN"] = FN
    report_values["FP"] = FP

    if neptune_run:
        neptune_run["experiment/metrics/fit_1/validation/loader/TP"].log(TP)
        neptune_run["experiment/metrics/fit_1/validation/loader/FP"].log(FP)
        neptune_run["experiment/metrics/fit_1/validation/loader/TN"].log(TN)
        neptune_run["experiment/metrics/fit_1/validation/loader/FN"].log(FN)
        neptune_run["confusion matrix"] = cf_matrix
        neptune_run["confusion_matrix_plot"].upload(
            os.path.join("confusion_matrix.png")
        )
        logging.info("Removing callbacks from learner")
        learner.remove_cbs(learner.cbs + get_callback(neptune_run))
        logging.info("Exporting best model")
        learner.export("model.pkl")
        logging.info("Logging best model to neptune")
        neptune_run["experiment/io_files/artifacts/saved_model"].upload(
            os.path.join("model.pkl")
        )

    logging.info(f"TP{TP} TN {TN} FN {FN} FP {FP}")

    return report_values, learner
