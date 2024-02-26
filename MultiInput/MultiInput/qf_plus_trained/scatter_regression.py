from tsai.all import *
from regression_metics import *
import logging
from scipy.signal import resample
import re
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
import matplotlib.pyplot as plt


def run_inference(meta_data, loaded_data, config_params):

    hz = 50
    healthy_threshold = config_params["healthy_threshold"]
    resample_points = config_params["resample_points"]
    channels = config_params["channels"]
    min_whistle_seconds = config_params["min_whistle_seconds"]
    max_whistle_seconds = config_params["max_whistle_seconds"]
    predictor_col = config_params["predictor_col"]
    window_length_seconds = config_params[
        "window_length_seconds"
    ]  # centered halfway through whistle
    model_path = "model.pkl"
    valid_patients = config_params["validation_patients"]
    test_patients = config_params["test_patients"]
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

    x_arr_train = []
    y_arr_train = []
    x_arr_valid = []
    y_arr_valid = []
    x_arr_test = []
    y_arr_test = []
    used_filenames = []
    count_insufficient_whistle_time = 0
    count_insufficient_signal_length = 0
    count_invalid_predictor = 0
    valid_set = False
    test_set = False
    for index, filename in enumerate(filenames):
        logging.debug(f"filename = {filename}")
        if get_subject_id(filename) in valid_patients:
            valid_set = True
        elif get_subject_id(filename) in test_patients:
            # use of first file of test patients
            if filename[-5] == "1":
                test_set = True
            else:
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

            if valid_set:
                x_arr_valid.append(entire_signal)
                y_arr_valid.append(predictor)
                valid_set = False
            elif test_set:
                x_arr_test.append(entire_signal)
                y_arr_test.append(predictor)
                test_set = False
            else:
                x_arr_train.append(entire_signal)
                y_arr_train.append(predictor)

            used_filenames.append(filename)
            logging.debug(f"\nlen(used_filenames) = {len(used_filenames)}")

        except (IndexError, KeyError) as error:
            logging.warning(f"error = {error}. Skipping")

    patient_ids = list(set([get_subject_id(filename) for filename in used_filenames]))
    patient_ids.sort()
    logging.debug(f"patient_ids = {patient_ids}")

    learner = load_learner(model_path)

    def transform_make_prediction(learner, x, y):
        X = np.swapaxes(np.array(x), 1, 2)
        y = np.array(y)
        X = TSTensor(X)
        X = TSStandardize(by_var=True, by_sample=True)(X)
        x = learner.get_X_preds(X)
        return x, y

    preds_train, target_train = transform_make_prediction(
        learner, x_arr_train, y_arr_train
    )
    preds_test, target_test = transform_make_prediction(learner, x_arr_test, y_arr_test)
    preds_valid, target_valid = transform_make_prediction(
        learner, x_arr_valid, y_arr_valid
    )

    return (
        preds_train[0],
        target_train,
        preds_test[0],
        target_test,
        preds_valid[0],
        target_valid,
    )


def get_metrics(predictions, target, healthy_threshold):

    rmse = round(mean_squared_error(target, predictions, squared=False), 2)
    mae = round(mean_absolute_error(target, predictions), 2)
    threshold_target = []
    threshold_prediction = []
    for i in range(len(target)):
        threshold_target.append(int(target[i] < healthy_threshold))
        threshold_prediction.append(int(predictions[i] < healthy_threshold))
    ba = round(balanced_accuracy_score(threshold_target, threshold_prediction), 2)
    return mae, rmse, ba


def plot_confusion_matrix_regression(
    train_labels,
    train_prediction,
    test_labels,
    test_prediction,
    valid_labels,
    valid_prediction,
    healthy_threshold,
    title,
):
    mae_train, rmse_train, ba_train = get_metrics(
        train_prediction, train_labels, healthy_threshold
    )
    mae_test, rmse_test, ba_test = get_metrics(
        test_prediction, test_labels, healthy_threshold
    )
    mae_valid, rmse_valid, ba_valid = get_metrics(
        valid_prediction, valid_labels, healthy_threshold
    )

    plt.style.use("dark_background")
    plot_width_px = 1800
    plot_height_px = 800
    pixels_per_inch = 60
    plt.rcParams["figure.dpi"] = pixels_per_inch
    plt.rcParams["font.family"] = [
        "Liberation Mono",
        "DejaVu Sans Mono",
        "mono",
        "sans-serif",
    ]
    plt.rcParams["font.size"] = 18
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.figsize"] = [
        plot_width_px / pixels_per_inch,
        plot_height_px / pixels_per_inch,
    ]
    plt.rcParams["lines.linewidth"] = 3
    plt.rcParams["lines.antialiased"] = True
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    fig.suptitle(title)
    axs[0, 0].scatter(train_labels, train_prediction, c="blue", marker="x")
    axs[0, 0].set_title(
        "Train: "
        + "RMSE: "
        + str(rmse_train)
        + " MAE: "
        + str(mae_train)
        + " B.A: "
        + str(ba_train)
    )
    axs[0, 1].scatter(test_labels, test_prediction, c="red", marker="x")
    axs[0, 1].set_title(
        "Test: "
        + "RMSE: "
        + str(rmse_test)
        + " MAE: "
        + str(mae_test)
        + " B.A: "
        + str(ba_test)
    )
    axs[1, 0].scatter(valid_labels, valid_prediction, c="green", marker="x")
    axs[1, 0].set_title(
        "Valid: "
        + "RMSE: "
        + str(rmse_valid)
        + " MAE: "
        + str(mae_valid)
        + " B.A: "
        + str(ba_valid)
    )
    axs[1, 1].scatter(train_labels, train_prediction, c="blue", marker="x")
    axs[1, 1].scatter(test_labels, test_prediction, c="red", marker="x")
    axs[1, 1].scatter(valid_labels, valid_prediction, c="green", marker="x")
    axs[1, 1].set_title("Combined")

    for ax in axs.flat:
        ax.plot([i for i in range(0, healthy_threshold * 2)])
        ax.axis("square")
        ax.set(xlabel="Actual Value", ylabel="Predicted Value")
        ax.axhline(y=healthy_threshold, color="r", linestyle="-")
        ax.axvline(x=healthy_threshold, color="y", linestyle="-")

    plt.xlim([0, healthy_threshold * 2])
    plt.ylim([0, healthy_threshold * 2])

    return plt


def plot_scatter_regression(meta_data, loaded_data, config_params, neptune_run=None):
    if neptune_run is None:
        study_id = "Combined Scatter Plot"
    else:
        study_id = neptune_run["sys/id"].fetch()
    (
        preds_train,
        target_train,
        preds_test,
        target_test,
        preds_valid,
        target_valid,
    ) = run_inference(meta_data, loaded_data, config_params)
    plot = plot_confusion_matrix_regression(
        target_train,
        preds_train,
        target_test,
        preds_test,
        target_valid,
        preds_valid,
        config_params["healthy_threshold"],
        study_id,
    )
    plot.savefig("combined_scatter.png")
    neptune_run["combined_scatter"].upload(os.path.join("combined_scatter.png"))
    return neptune_run
