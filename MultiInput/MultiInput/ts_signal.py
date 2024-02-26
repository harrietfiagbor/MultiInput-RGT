import logging
import numpy as np
from qf_plus_trained.regression_metics import *
from scipy.signal import resample
import math
import random
from fastai.metrics import *
from tsai.all import *
# from neptune_utils import *
# from fastaipruning import FastAIMultiValPruningCallback
import inspect
import re
from qf_plus_development.ai_utils import LSUVinit
from functools import partial
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import pickle
from qf_plus_trained.scatter_regression import *
# import Train as train
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
import time
from fastai.test_utils import *
import os

def plot_confusion_matrix_regression_all(
        train_labels,
        train_prediction,
        test_labels,
        test_prediction,
        valid_labels,
        valid_prediction,
        healthy_threshold,
        title,
    ):
        # print("Shapes:")
        # print("Train Labels:", train_labels.shape)
        # print("Train Predictions:", train_prediction.shape)
        # print("Test Labels:", test_labels.shape)
        # print("Test Predictions:", test_prediction.shape)
        # print("Valid Labels:", valid_labels.shape)
        # print("Valid Predictions:", valid_prediction.shape)
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
            # "Liberation Monospace",
            # "DejaVu Sans Monospace",
            # "monospace",
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
        plt.savefig('Inception with removed patients')
        return plt
    
def transform_make_prediction(learner, x, y):
        x = TSTensor(x)
        x = TSStandardize(by_var=True, by_sample=True)(x)
        x = learner.get_X_preds(x)
        return x, y



from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score
    
def get_metrics(predictions, target, healthy_threshold):
        mae = round(mean_absolute_error(target, predictions), 2)
        rmse = round(mean_squared_error(target, predictions, squared=False), 2)
    
        threshold_target = []
        threshold_prediction = []
    
        for i in range(len(target)):
            threshold_target.append(int(target[i] < healthy_threshold))
            threshold_prediction.append(int(predictions[i] < healthy_threshold))
    
        ba_score = round(balanced_accuracy_score(threshold_target, threshold_prediction), 2)
    
        return mae, rmse, ba_score


def resample_multichannel_signal(original_signal, resample_points):
        if original_signal.ndim != 2:
            raise ValueError("Input signal must be 2-dimensional (samples x channels)")
    
        num_samples, num_channels = original_signal.shape
        new_num_points = resample_points
    
        resampled_channels = []
        for channel in range(num_channels):
            channel_data = original_signal[:, channel]
            new_indices = np.linspace(0, num_samples - 1, len(channel_data))
            new_channel = np.interp(np.linspace(0, num_samples - 1, new_num_points), new_indices, channel_data)
            resampled_channels.append(new_channel)
    
        resampled_signal = np.column_stack(resampled_channels)
        
        return resampled_signal
def get_subject_id(filename):
        subject_id = filename[:-6]
        if "AUG" in subject_id:
            subject_id = re.sub("AUG_[0-9]+_", "", subject_id)
        return subject_id
def get_validation_indices(loaded_data, get_subject_id,used_filenames):
        """
        Get all validation indices for five folds from loaded_data.
    
        Args:
        loaded_data (dict): The loaded data containing validation splits for all folds.
        get_subject_id (function): A function to extract the subject ID from a filename.
    
        Returns:
        list of lists: Validation indices for all five folds.
        """
        val_splits = loaded_data['val_test_splits']['val_splits']
        all_validation_indices = []
    
        for fold_indices in val_splits:
            validation_indices = []
            for index, filename in enumerate(used_filenames):
                if "AUG" in filename:
                    continue  # Skip filenames containing "AUG"
                subject_id = get_subject_id(filename)
                if subject_id in fold_indices:
                    validation_indices.append(index)
            all_validation_indices.append(validation_indices)

        return all_validation_indices

def train(to_remove=False):
    # import pandas as pd
    # import numpy as np
    meta_data_file = '/content/drive/MyDrive/QFHD/data/metadata3.xlsx'
    pickle_file = '/content/drive/MyDrive/QFHD/data/preprocessed_24Jan22_134624.p'
    pickle_data = {}

    with open(pickle_file, 'rb') as handle:
          pickle_data = pickle.load(handle)
        
    meta_data_df   = pd.read_excel(meta_data_file, usecols = ['Subject ID','Use for Algorithm Training [T/F]', 'Final EF%', 'Strain, Global Longitudinal [-%]', 'Clinic ID', 'Age [yrs rounded to nearest]'])
    meta_data_df.rename(columns={'Subject ID':'Subject_ID', 'Use for Algorithm Training [T/F]':'Use_4_Train', 'Final EF%':'LVEF', 'Strain, Global Longitudinal [-%]':'Strain', 'Clinic ID':'Clinic', 'Age [yrs rounded to nearest]':'Age'}, inplace=True)
    first_data_row = meta_data_df['Subject_ID'].first_valid_index()
    meta_data_df   = meta_data_df.iloc[first_data_row:]
    meta_data_df['Use_4_Train'] = np.where( meta_data_df['Use_4_Train'] != 'T', False, True)
    loaded_data = pickle_data
    print('Data loaded.')

    loaded_data.keys()
    
    config_params = {
    "batch_size": 32,
    "channels": ['env_scaled', 'env_scaled', 'env_scaled'],
    "dilation": 1,
    "epochs": 200,
    "healthy_threshold": 14,
    "max_whistle_seconds": 20,
    "min_whistle_seconds": 8,
    "model_name": "InceptionTimePlus",
    "norm_by_sample": 1,
    "norm_by_var": 1,
    "num_channels": 3,
    "optuna_metric": ['balanced_accuracy_score'],
    "predictor_col": "Strain",
    "preprocessing_type": "Standardize",
    "resample_points": 3000,
    "save_metric": "balanced_accuracy_score",
    "window_length_seconds": 35,
    "prediction_type":"regression",
    "fc_dropout": 0.45,
    "ks": 30,
    "conv_dropout": 0,
    "sa": False,
    "se":  None,
    "nb_filters": 64,
    "coord":False,
    "separable": True,
    "dilation":2,
    "bottleneck":False#
    
}
    hz=50
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
    ] 
    test_patients = loaded_data['val_test_splits']['test_subjects']
    save_metric = config_params["save_metric"]
    preprocessing_type = config_params["preprocessing_type"]
    norm_by_var = config_params["norm_by_var"]
    norm_by_sample = config_params["norm_by_sample"]
    normalize_initialization = False
    prediction_type = config_params["prediction_type"]
    window_length_points = window_length_seconds * hz
    
   

    
    filenames = list(loaded_data['file_data'])
    error_filenames=[]
    filenames.sort()
    # print(f"filenames = {filenames}")
    # print(f"meta_data = {meta_data_df}")
    
    
    
    x_arr = []
    y_arr = []
    used_filenames = []
    count_insufficient_whistle_time = 0
    count_insufficient_signal_length = 0
    count_invalid_predictor = 0
    count_use4train_false = 0
    count_exceptions = 0
    count_test = 0
    count_removed=0
    count_multiple_whistles=0
    removed_filenames=["SD 06.17.21.3.csv", "JL 05.26.21.2.csv", "SN 06.24.21.3.csv", "JG 06.25.21.1.csv", "IW 07.08.21.2.csv", "OR 06.25.21.2.csv", "IW 07.08.21.3.csv", "JFHC JB1 08.17.21.3.csv", "KS 02.05.21.1.csv", "AJ 02.26.21.3.csv", "ZD 02.05.21.3.csv", "JFHC KT1 11.30.21.1.csv", "OR 06.25.21.1.csv", "JL 05.26.21.1.csv", "RW 02.26.21.2.csv", "IW 07.08.21.1.csv", "JG 06.25.21.3.csv", "ZI 06.25.21.2.csv", "JG 06.25.21.2.csv", "RW 02.26.21.1.csv", "KD 06.25.21.1.csv", "JFHC JB1 08.17.21.2.csv", "MT 06.18.21.1.csv", "SF 07.08.21.3.csv", "KT 07.06.21.3.csv", "MT 02.26.21.1.csv", "ME 05.21.21.3.csv", "MA 07.08.21.2.csv", "AK 06.25.21.1.csv", "ZD 02.05.21.2.csv", "HC 07.02.21.3.csv", "ME 05.21.21.1.csv", "KN 06.17.21.3.csv", "RW 02.26.21.3.csv", "GG 02.10.21.3.csv", "HS 05.28.21.1.csv", "KS 02.05.21.3.csv", "SB 05.28.21.1.csv", "ME 05.21.21.2.csv", "JFHC RH1 08.17.21.2.csv", "BHRT KR1 09.29.21.3.csv", "SD 06.17.21.2.csv", "OR 06.25.21.3.csv", "JT 03.03.21.3.csv", "IS 02.05.21.1.csv", "AS 06.18.21.3.csv", "MN 06.24.21.3.csv", "SG 05.26.21.2.csv", "BT 07.06.21.2.csv", "BHRT BL1 09.08.21.3.csv", "GG 02.10.21.2.csv", "JD 07.02.21.3.csv"]
    
    for index, filename in enumerate(filenames):
        # # print(f"filename = {filename}")
        # if get_subject_id(filename) in removed_subject_ids:
        #     # print("This is part of the removed set. Skipping.")
        #     count_removed += 1
        #     continue
        if to_remove:
            if filename in removed_filenames :
            # print("This is part of the removed set. Skipping.")
               count_removed += 1
               continue
                 
        if get_subject_id(filename) in test_patients:
            # print("This is part of the test set. Skipping.")
            count_test += 1
            # continue
       
            # continue
        try:
    
            orig_df_columns = channels
    
            orig_df = loaded_data['file_data'][filename]
            data = orig_df[orig_df_columns].to_numpy()
            # print(f"data.shape = {data.shape}")
            whistle_binary = (orig_df["Category"] == "Whistle").to_numpy()
            if not whistle_binary.size:
                raise IndexError(f"orig_df.Category is empty for filename={filename}.")
            
    
            # Identify start and end indices of each contiguous whistle segment
            starts = np.where(np.logical_and(whistle_binary[:-1] == False, whistle_binary[1:] == True))[0] + 1
            ends = np.where(np.logical_and(whistle_binary[:-1] == True, whistle_binary[1:] == False))[0]
    
            # Calculate durations of each segment
            durations = ends - starts
            if len(starts) > 1 and len(ends):
                count_multiple_whistles +=1
        
            if len(durations) > 0:
                longest_segment_index = np.argmax(durations)
                first_whistle = starts[longest_segment_index]
                last_whistle= ends[longest_segment_index]
    
                whistle_length_secs = (last_whistle - first_whistle) / hz
            else:
                first_whistle = np.argmax(whistle_binary)
                last_whistle = len(whistle_binary) - 1 - np.argmax(whistle_binary[::-1])
                whistle_length_secs = (last_whistle - first_whistle) / hz
    
            mid_whistle = (last_whistle + first_whistle) // 2
            first_index = mid_whistle - window_length_points // 2
            first_index = max(first_index, 0)
            last_index = first_index + window_length_points
            entire_signal = data[first_index:last_index, :]
            # print(f"entire_signal.shape = {entire_signal.shape}")
            # print(f"entire_signal.shape = {entire_signal.shape}")
            if entire_signal.shape[0] != window_length_points:
                # print(
                #     f"Insufficient signal acquired before / after the whistle. Was {round(entire_signal.shape[0]/hz,1)} seconds, but must be at least {round(window_length_points/hz,1)} seconds."
                # )
                count_insufficient_signal_length += 1
                continue
           
            entire_signal = resample_multichannel_signal(entire_signal, resample_points)
           
            subject_id = get_subject_id(filename)
            # print(f"subject_id = {subject_id}")
    
            predictor = meta_data_df.loc[meta_data_df["Subject_ID"] == subject_id][
                predictor_col
            ].iloc[0]
            # print(f"{predictor_col} = {predictor}")
            if type(predictor) is str or math.isnan(predictor):
                print(
                    f"Invalid {predictor_col} value of {type(predictor)} {predictor}. Skipping."
                )
                count_invalid_predictor += 1
                continue
    
            use_4_train = meta_data_df.loc[meta_data_df["Subject_ID"] == subject_id][
                "Use_4_Train"
            ].iloc[0]
            if not use_4_train:
                # print(f"use_4_train = {use_4_train}. Skipping.")
                count_use4train_false += 1
                continue
    
            x_arr.append(entire_signal)
            if prediction_type == "classification":
                sick = int(predictor < healthy_threshold)
                # print(f"sick = {sick}")
                y_arr.append(sick)
            else:
                y_arr.append(predictor)
    
            used_filenames.append(filename)
            # print(f"\nlen(used_filenames) = {len(used_filenames)}")
    
        except (IndexError, KeyError) as error:
            error_filenames.append(filename)
            print(f"error = {error}. Skipping")
    
    patient_ids = list(set([get_subject_id(filename) for filename in used_filenames]))
    patient_ids.sort()
    # print(f"patient_ids = {patient_ids}")
    
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
    # if neptune_run:
    #     neptune_run["counts"] = counts
    
    print(f"len(patient_ids) = {len(patient_ids)} (# of patients, or N)")
    print(f'{counts["original_patients"]} = (# of patients before filtering)')
    print(f"len(filenames) = {len(filenames)} (# of signals before filtering)")
    print(
        f"len(used_filenames) = {len(used_filenames)}/{len(filenames)} = {round(len(used_filenames)*100/len(filenames))}% (# of signals)"
    )
    print(
        f"count_insufficient_whistle_time = {count_insufficient_whistle_time}/{len(filenames)} = {round(count_insufficient_whistle_time*100/len(filenames))}% (whistle length not between {min_whistle_seconds} and {max_whistle_seconds} seconds)"
    )
    print(
        f"count_insufficient_signal_length = {count_insufficient_signal_length}/{len(filenames)} = {round(count_insufficient_signal_length*100/len(filenames))}% (signal length not at least {window_length_seconds} seconds)"
    )
    print(
        f"count_invalid_predictor = {count_invalid_predictor} = {round(count_invalid_predictor*100/len(filenames))}% (no {predictor_col} value available)"
    )
    print(
        f"count_use4train_false = {count_use4train_false}/{len(filenames)} = {round(count_use4train_false*100/len(filenames))}% (signals the operator marked as data which should not be used for training the AI, such as patients with pacemakers or who failed to properly perform the valsalva maneuver)"
    )
    print(
        f"count_exceptions = {count_exceptions}/{len(filenames)} = {round(count_exceptions*100/len(filenames))}% (for situations such as missing data)"
    )
    print(
        f"count_test = {count_test}/{len(filenames)} = {round(count_test*100/len(filenames))}% (for testing data)"
    )
    
    # print(f"Removed filename =  { count_removed}")
    print(f"Multiple Whistles={ count_multiple_whistles}")
    count_multiple_whistles
    X = np.swapaxes(np.array(x_arr), 1, 2)
    y = np.array(y_arr)
    print(f"X.shape = {X.shape}")
    print(f"y.shape = {y.shape}")
    print(f"np.histogram(y, bins=2) = {np.histogram(y, bins=2)}")

    
    # Example usage to get all validation indices for all five folds
    validation_indices_all_folds = get_validation_indices(loaded_data, get_subject_id,used_filenames)
    
    testing_indices = []
    
    for index, filename in enumerate(used_filenames):
    
         subject_id = get_subject_id(filename)  # Get the subject ID from the filename
         
         if subject_id in test_patients:
            testing_indices.append(index)

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
    
    # Added by harriet
    # tfms = [None, [TSRegression()]]
    #
    
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



    # Create a list of DataLoaders objects for each fold
    dls_list = []
    
    for fold_indices in validation_indices_all_folds:
    
        # Get training indices for this fold, excluding test patients
        train_indices = [i for i in range(len(used_filenames)) if i not in fold_indices and i not in testing_indices]
    
        print(len(train_indices))
        print(len(fold_indices))
    
        # Create a split for this fold
        splits_fold = [train_indices, fold_indices]
        
        batch_tfms = [preprocessing]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits_fold)
        # Create DataLoaders for this fold
        dls_fold = TSDataLoaders.from_dsets(
            dsets.train, dsets.valid, splits=splits_fold,
            bs=batch_size, batch_tfms=batch_tfms, num_workers=0
        )
    
        # Append the DataLoaders for this fold to the list
        dls_list.append(dls_fold)
    
    test_split = [testing_indices, testing_indices]
    t_dsets = TSDatasets(X, y, tfms=tfms, splits=test_split)
    test_dls = TSDataLoaders.from_dsets(
            t_dsets.train, t_dsets.valid, splits=test_split,
            bs=batch_size, batch_tfms=batch_tfms, num_workers=0
        )



    # Now dls_list contains DataLoaders objects for each fold
            # https://timeseriesai.github.io/tsai/
    # Dictionary mapping model names to their corresponding classes
    model_dict = {
        "InceptionTime": InceptionTime,
        "InceptionTimePlus": InceptionTimePlus,
        "XceptionTime": XceptionTime,
        "XceptionTimePlus": XceptionTimePlus,
        "ResNet": ResNet,
        "ResNetPlus": ResNetPlus,
        "InceptionRocketPlus": InceptionRocketPlus,
        "ResCNN": ResCNN,
        "TCN": TCN,
        "XCM": XCM,
        "XCMPlus": XCMPlus,
        "RNN_FCN": RNN_FCN,
        "LSTM_FCN": LSTM_FCN,
        "GRU_FCN": GRU_FCN,
        "MRNN_FCN": MRNN_FCN,
        "MLSTM_FCN": MLSTM_FCN,
        "MGRU_FCN": MGRU_FCN,
        "RNN_FCNPlus": RNN_FCNPlus,
        "LSTM_FCNPlus": LSTM_FCNPlus,
        "GRU_FCNPlus": GRU_FCNPlus,
        "MRNN_FCNPlus": MRNN_FCNPlus,
        "MLSTM_FCNPlus": MLSTM_FCNPlus,
        "MGRU_FCNPlus": MGRU_FCNPlus,
        "mWDN": mWDN,
        "mWDNPlus": mWDNPlus,
        "TST": TST,
        "TSTPlus": TSTPlus,
        "FCN": FCN,
        "FCNPlus": FCNPlus,
        "PatchTST": PatchTST,
        "LSTM": LSTM,
    }
    
    assert model_name in model_dict, "Invalid model name"
    model = model_dict[model_name]
            
    # if model_name == "ResNetPlus":
    #     ks = config_params["ks"]
    #     config_params["ks"] = [k - 2 for k in ks] + [k - 4 for k in ks]

    # else:
    # # Handle the case where ks is a list
    #     ks = config_params["ks"]
    #     config_params["ks"] = [k - 2 for k in ks]  # Subtract 2 from each element
    #     config_params["ks"].extend([k - 4 for k in ks])  # Subtract 4 from each element and extend the list

    # Initialize an empty list to store the models for each fold
    models_list = []
    
    # Loop through each fold and create a model for that fold
    for dls_fold in dls_list:
        model_args = [ "fc_dropout","ks",
        "conv_dropout",
        "sa",
        "se",
        "nb_filters",
        "coord",
        "separable",
        "dilation",
        "bottleneck"]
        print(model_args)
        model_kwargs = {}
        for arg in model_args:
            if arg in config_params:
                model_kwargs[arg] = config_params[arg]
    
        print(model_kwargs)
        # Create the model for the current fold with fold-specific keyword arguments
        model_fold = create_model(model, dls=dls_fold, **model_kwargs)

        # Append the model to the list
        models_list.append(model_fold)
    
    return dls_list, models_list, test_dls


    # if normalize_initialization:
    #     logging.debug("using normalised weight initialization")
    #     batch_x = dls.one_batch()[0]
    #     model = LSUVinit(model, batch_x)
    

    
    
    # learners = []  # Create an empty list to store the learners
    # models = []  # Use the models list created above
    # # Initialize variables to accumulate results
    # all_train_labels = []
    # all_train_predictions = []
    # all_test_labels = []
    # all_test_predictions = []
    # all_valid_labels = []
    # all_valid_predictions = []


    # print(len(models_list))
    # for fold, dls_fold in enumerate(dls_list):
    #     print(f"Fold {fold + 1} / {len(dls_list)}")
    #     # Get the validation_patients for this fold
    #     validation_patients = loaded_data['val_test_splits']['val_splits'][fold]
    
    #     training_indices = []
    #     validation_indices = []
    #     testing_indices = []
    
    #     for index, filename in enumerate(used_filenames):
    #         if "AUG" in filename:
    #             continue  # Skip filenames containing "AUG"
    
    #         subject_id = get_subject_id(filename)  # Get the subject ID from the filename
    
    #         if subject_id in validation_patients:
    #             validation_indices.append(index)
    #         elif subject_id in test_patients:
    #             testing_indices.append(index)
    #         else:
    #             training_indices.append(index)
    
    #     model_fold = models_list[fold]
    
    #     # Reset the parameters of the model for this fold
    #     for layer in model_fold.parameters():
    #         if hasattr(layer, "reset_parameters"):
    #             print(model_fold.parameters())
    #             layer.reset_parameters()
    
    #     learner_fold = Learner(dls_fold, model_fold, metrics=metrics, cbs=[ShowGraphCallback2(), PredictionDynamics(), SaveModelCallback(monitor="balanced_accuracy_regression"), EarlyStoppingCallback(monitor="_rmse", comp=np.less, patience=50)])
    
    #     # Use the lr_find method to find the optimal learning rate
    #     auto_lr = learner_fold.lr_find().valley
    
    #     print(auto_lr)
    
    #     # Train the model for this fold
    #     start = time.perf_counter()
    #     learner_fold.fit_one_cycle(n_epoch=epochs, lr_max=auto_lr)
    
    #     end = time.perf_counter()
    #     print(f"Fold {fold + 1} training time: {end - start}")
    
    #     preds_train = transform_make_prediction(learner_fold, X[training_indices], y[training_indices])[0][0]
    #     pred_val = transform_make_prediction(learner_fold, X[validation_indices], y[validation_indices])[0][0]
    #     pred_test = transform_make_prediction(learner_fold, X[testing_indices], y[testing_indices])[0][0]
    
    #     all_train_labels.append(y[training_indices])
    #     all_train_predictions.append(preds_train)
    #     all_test_labels.append(y[testing_indices])
    #     all_test_predictions.append(pred_test)
    #     all_valid_labels.append(y[validation_indices])
    #     all_valid_predictions.append(pred_val)

    # # Delete the saved model file (if needed)
    # # model_path = f"models/model.pth"
    # # if os.path.exists(model_path):
    # #     os.remove(model_path)
    # # del learner_fold

    #     # # Delete the saved model file
    #     # model_path = f"models/model.pth"
    #     # if os.path.exists(model_path):
    #     #     os.remove(model_path)
    #     del learner_fold
    
    # combined_train_labels = np.concatenate(all_train_labels)
    # combined_train_predictions = np.concatenate(all_train_predictions)
    # combined_test_labels = np.concatenate(all_test_labels)
    # combined_test_predictions = np.concatenate(all_test_predictions)
    # combined_valid_labels = np.concatenate(all_valid_labels)
    # combined_valid_predictions = np.concatenate(all_valid_predictions)
    
    # # Calculate aggregated metrics
    # mae_combined_train, rmse_combined_train, ba_combined_train = get_metrics(
    #     combined_train_predictions, combined_train_labels, healthy_threshold
    # )
    # mae_combined_test, rmse_combined_test, ba_combined_test = get_metrics(
    #     combined_test_predictions, combined_test_labels, healthy_threshold
    # )
    # mae_combined_valid, rmse_combined_valid, ba_combined_valid = get_metrics(
    #     combined_valid_predictions, combined_valid_labels, healthy_threshold
    # )
    
    # # Define the name of the CSV file
    # if to_remove:
    #     csv_file = 'Without.csv'
    # else:
    #     csv_file = 'With.csv'
    
    # # Initialize an empty DataFrame to store the results
    # results_df = pd.DataFrame(columns=["RMSE Train", "RMSE Test", "RMSE Valid", "MAE Train", "MAE Test", "MAE Valid", "BA Train", "BA Test", "BA Valid"])
    # # Create a dictionary to store the results for the current fold
    
    # aggregate_result = {
    #       # Increment the run number for each aggregate
    #     "RMSE Train": rmse_combined_train,
    #     "RMSE Valid": rmse_combined_valid,
    #     "RMSE Test": rmse_combined_test,
    #     "MAE Train": mae_combined_train,
    #     "MAE Valid": mae_combined_valid,
    #     "MAE Test":mae_combined_test,
    #     "BA Train": ba_combined_train,
    #     "BA Valid": ba_combined_valid,
    #     "BA Test": ba_combined_test,
    # }
    
    # # Append the aggregate result to the results DataFrame
    # results_df = results_df.append(aggregate_result, ignore_index=True)
    
    # # Load the existing CSV file if it exists
    # try:
    #     existing_df = pd.read_csv(csv_file)
    # except FileNotFoundError:
    #     existing_df = pd.DataFrame()
    
    # # Append the results to the existing DataFrame
    # updated_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    # # Save the updated DataFrame to the CSV file (use index=False to exclude row numbers)
    # updated_df.to_csv(csv_file, index=False)
    
    # print(f'Results saved to {csv_file}')
    # print("Iteration complete.")
    # print("All iterations completed.")
    # # plot_confusion_matrix_regression_all(
    # #     combined_train_labels,
    # #     combined_train_predictions ,
    # #     combined_test_labels,
    # #     combined_test_predictions,
    # #     combined_valid_labels ,
    # #     combined_valid_predictions,
    # #     healthy_threshold,
    # #       "InceptionTimePlus (Full)",
    # # )







