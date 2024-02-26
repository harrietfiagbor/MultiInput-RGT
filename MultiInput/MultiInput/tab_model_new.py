

# Library imports and initialization
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder

# TSAI
from fastai.tabular.all import *
from tsai.data.tabular import *

import optuna
from optuna_integration.fastaiv2 import FastAIPruningCallback

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import plotly.io as pio

from google.colab import drive
drive.mount('/content/drive')

# Load the data
pickle_file_path = '/content/drive/MyDrive/QFHD/data/preprocessed_24Jan22_134624.p'
metadata_file_path = '/content/drive/MyDrive/QFHD/data/metadata3.xlsx'

# Load metadata
meta_data = pd.read_excel(metadata_file_path)

# to use the splits in the pickle...
# Load pickle data
with open(pickle_file_path, 'rb') as handle:
  pickle_data = pickle.load(handle)

rename = {
    'Subject ID':'Subject_ID',
    'Use for Algorithm Training [T/F]':'Use_4_Train',
    'Strain, Global Longitudinal [-%]':'Strain',
    'Age [yrs rounded to nearest]':'Age'
    }

meta_data.rename(columns=rename, inplace=True)
first_data_row = meta_data['Subject_ID'].first_valid_index()
meta_data      = meta_data.iloc[first_data_row:] # makes the row with the first Subject ID the first row


"""## Data Processing"""

# List of features to keep
# Isaac has added the 'Subject ID' column
features_to_keep = [
    'Subject_ID',
    'Use_4_Train',
    'Age',
    'BMI',
    'Diabetes',
    'Hx Hypertension',
    'Hx Smoking',
    'Hx Stroke/TIA',
    'Hx COPD',
    'Personal or Family Hx CV Disease',
    'Hx High Cholesterol',
    'Cholesterol Medication',
    'Strain',
    ]

# Select only the features to keep
features_df = meta_data[features_to_keep]
features_df

# Remove rows with missing target variable
features_df = features_df.dropna(subset=['Strain'])

# Check the shape of the dataframe after removing missing values
print(features_df.shape)

# make BMI column a numerical column

features_df          = features_df.loc[(features_df['BMI'] != '#DIV/0!') & (features_df['BMI'] != 0)] # drops rows with #DIV/0! or 0 in BMI columns
features_df['BMI']   = features_df['BMI'].astype(float) #changes the data type of the BMI column from object to float64
features_df.reset_index(drop=True, inplace=True) # renumber the index column


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
test_patients = pickle_data['val_test_splits']['test_subjects']
save_metric = config_params["save_metric"]
preprocessing_type = config_params["preprocessing_type"]
norm_by_var = config_params["norm_by_var"]
norm_by_sample = config_params["norm_by_sample"]
normalize_initialization = False
prediction_type = config_params["prediction_type"]
window_length_points = window_length_seconds * hz


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

filenames = list(pickle_data['file_data'])
error_filenames=[]
filenames.sort()



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
for index, filename in enumerate(filenames):
 
    if get_subject_id(filename) in test_patients:
        # print("This is part of the test set. Skipping.")
        count_test += 1
        # continue

        # continue
    try:

        orig_df_columns = channels

        orig_df = pickle_data['file_data'][filename]
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

        predictor = meta_data.loc[meta_data["Subject_ID"] == subject_id][
            predictor_col
        ].iloc[0]
        # print(f"{predictor_col} = {predictor}")
        if type(predictor) is str or math.isnan(predictor):
            print(
                f"Invalid {predictor_col} value of {type(predictor)} {predictor}. Skipping."
            )
            count_invalid_predictor += 1
            continue

        use_4_train = meta_data.loc[meta_data["Subject_ID"] == subject_id][
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

def get_validation_indices(loaded_data, get_subject_id):
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

# Example usage to get all validation indices for all five folds
validation_indices_all_folds = get_validation_indices(pickle_data, get_subject_id)


# Assuming `features_df` is your DataFrame containing subject information
# Create a dictionary to map each subject ID to the number of times its filename appears in `used_filenames`
subject_id_count = {get_subject_id(filename): used_filenames.count(filename) for filename in used_filenames}

# Create a DataFrame to store the duplicated rows
duplicated_rows = []

# Iterate through each filename in used_filenames
for filename in used_filenames:
    subject_id = get_subject_id(filename)
    # Filter the rows from features_df corresponding to the current subject_id
    subject_rows = features_df[features_df['Subject_ID'] == subject_id]
    # Duplicate the rows based on the count for the current subject_id
    duplicated_rows.extend([subject_rows] * subject_id_count[subject_id])

# Concatenate the duplicated rows into a new DataFrame
duplicated_features_df = pd.concat(duplicated_rows, ignore_index=True)


duplicated_features_df_train = duplicated_features_df.drop(['Subject_ID', 'Use_4_Train'], axis=1)
duplicated_features_df_train


for i in validation_indices_all_folds :
    print(len(i))

testing_indices = []

for index, filename in enumerate(used_filenames):

     subject_id = get_subject_id(filename)  # Get the subject ID from the filename

     if subject_id in test_patients:
        testing_indices.append(index)

# making a train df that contain only needed features
healthy_threshold = 14

features_df_train = features_df.drop(['Subject_ID', 'Use_4_Train'], axis=1)
features_df_train.shape


def categorize_strain(strain_value, threshold):
    return int(strain_value < threshold)

# # Apply the function to create a new categorical column 'strain_category'
# duplicated_features_df_train['Strain'] = duplicated_features_df_train['Strain'].apply(lambda x: categorize_strain(x, healthy_threshold))
# # print(features_df_train)

# Identify categorical and numerical columns
categorical_cols = features_df_train.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = features_df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_cols

numerical_cols.remove('Strain')
numerical_cols


"""## Dataloaders"""

def get_dls_fold_list():
  # Create a list of DataLoaders objects for each fold
  dls_list = []

  for fold_indices in validation_indices_all_folds:

      # Get training indices for this fold, excluding test patients
      train_indices = [i for i in range(len(used_filenames)) if i not in fold_indices and i not in testing_indices]

      print(len(train_indices))
      print(len(fold_indices))

      # Create a split for this fold
      splits_fold = [train_indices, fold_indices]

      y_col = ['Strain']
      procs = [Categorify, FillMissing, Normalize]
      y_block = RegressionBlock() if isinstance(features_df_train['Strain'].values[0], float) else CategoryBlock()
      pd.options.mode.chained_assignment=None

      dls_fold = get_tabular_dls(duplicated_features_df_train, y_block=y_block, cat_names=categorical_cols, procs=procs, cont_names=numerical_cols, y_names=y_col, splits=splits_fold, inplace=True, bs=batch_size)
      dls_list.append(dls_fold)
  return dls_list

def get_test_dls():

  splits = [testing_indices, testing_indices]
  y_col = ['Strain']
  procs = [Categorify, FillMissing, Normalize]
  pd.options.mode.chained_assignment=None
  y_block = RegressionBlock() if isinstance(features_df_train['Strain'].values[0], float) else CategoryBlock()

  test_dls = get_tabular_dls(duplicated_features_df_train, y_block=y_block, cat_names=categorical_cols, procs=procs, cont_names=numerical_cols, y_names=y_col, splits=splits, inplace=True, bs=batch_size)

  return test_dls



def do_hyp_opt():

  """## Hyperparameter Optimization"""

  run_num = "07"
  epochs  = 30
  num_trial = 10

  model_folder = f"/content/drive/MyDrive/Tab_modelling/trial_{run_num}_epoch_{epochs}"
  os.makedirs(model_folder, exist_ok=True)

  # Lists to store metrics values for each training run
  all_mae_list = []
  all_rmse_list = []
  all_loss = []

  ## hyp tune
  ## no_of_layers
  ## layer_size
  ## better understand embs


  # Define the objective function for Optuna
  def objective(trial):

      # lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

      # Lists to store metrics values for each training run
      all_mae_list = []
      all_rmse_list = []
      all_loss = []
      best_trial_value = float('inf')

      # Train tab model on each dataloader fold (train and val dl)
      for run_index, dls in enumerate(dls_list):

          # # Define hyperparameters to tune
          # layers = [trial.suggest_int('layer1', 50, 500, log=True),
          #           trial.suggest_int('layer2', 10, 100, log=True)]

          layers = [trial.suggest_int(f'layer_{i}', 10, 500, log=True) for i in range(trial.suggest_int('num_layers', 1, 7))]

          # emb_szs = {cat: trial.suggest_int(f"{cat}_emb", 16, 64) for cat in dls.train_ds.cats}

          # wd = trial.suggest_loguniform("wd", 1e-6, 1e-3)

          metrics = [mae, rmse]

          learn = tabular_learner(dls,
                                  layers=layers,
                                  # emb_szs=emb_szs,
                                  # wd = wd,
                                  metrics=metrics,
                                  cbs=FastAIPruningCallback(trial))

          # lr = learn.lr_find().valley

          learn.fit(epochs, 1e-2)
          # learn.fit(epochs, lr)

          # Get the final metrics values after training on the validation set
          val_metrics = learn.validate(dl=dls.valid)
          val_loss = val_metrics[0]
          val_mae = val_metrics[1]  # mae
          val_rmse = val_metrics[2]   # rmse

          all_mae_list.append(val_mae)
          all_rmse_list.append(val_rmse)
          all_loss.append(val_loss)

          # # Report the validation loss for Optuna
          # current_trial_value = val_mae

          # # Check if the current trial is better than the best trial so far
          # if current_trial_value < best_trial_value:

          #     # Save the model with a distinct name using timestamp
          #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
          #     model_name = f"{model_folder}/best_tab_model_{timestamp}_{run_index}"
          #     learn.save(model_name)

      # Calculate the average MAE and RMSE
      average_mae = sum(all_mae_list) / len(all_mae_list)
      average_rmse = sum(all_rmse_list) / len(all_rmse_list)
      average_loss = sum(all_loss) / len(all_loss)

      print("Average Loss:", average_loss)
      print("Average MAE:", average_mae)
      print("Average RMSE:", average_rmse)

      # Report the validation loss for Optuna
      return average_mae

  # Create an Optuna study and optimize
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=num_trial)

  # Print the best hyperparameters
  print('Best trial:')
  trial = study.best_trial
  print(f'  Value: {trial.value:.4f}')
  print('  Params:')
  for key, value in trial.params.items():
      print(f'    {key}: {value}')



  # Save the best parameters to a file
  best_params_file = os.path.join(model_folder, "best_params.txt")
  with open(best_params_file, "w") as f:
      f.write('Best trial:\n')
      f.write(f'  Value: {study.best_trial.value:.4f}\n')
      f.write('  Params:\n')
      for key, value in study.best_trial.params.items():
          f.write(f'    {key}: {value}\n')

  # Save optimization history plot
  optimization_history_plot_path = os.path.join(model_folder, "optimization_history.png")
  fig_optimization = optuna.visualization.plot_optimization_history(study)
  # fig_optimization.show()
  fig_optimization.write_image(optimization_history_plot_path)

  # Save EDF plot
  edf_plot_path = os.path.join(model_folder, "edf_plot.png")
  fig_edf = optuna.visualization.plot_edf(study)
  # fig_edf.show()
  fig_edf.write_image(edf_plot_path)

  # Save parameter importances plot
  param_importances_plot_path = os.path.join(model_folder, "param_importances_plot.png")
  fig_param_importances = optuna.visualization.matplotlib.plot_param_importances(study)
  # fig_param_importances.show()
  fig_param_importances.figure.savefig(param_importances_plot_path)

  # # optim
  # fig = optuna.visualization.plot_optimization_history(study)
  # fig.show()

  # #edf
  # fig = optuna.visualization.plot_edf(study)
  # fig.show()

  # # param importances
  # optuna.visualization.matplotlib.plot_param_importances(study)



"""## Train"""

def train_tab(save_model_to, num_epochs=30):
  # Lists to store metrics values for each training run
  all_mae_list = []
  all_rmse_list = []
  all_loss = []

  # Train tab model on each dataloader fold (train and val dl)
  for run_index, dls in enumerate(dls_list):
      # Using MAE and RMSE as the metrics
      metrics = [mae, rmse]

    #   layer_0: 87
    # layer_1: 10
    # layer_2: 13
    # layer_3: 63
    # layer_4: 204
    # layer_5: 139
    # layer_6: 10

      learn = tabular_learner(dls, layers=[87, 10, 13, 63, 204, 139, 10], y_range=None, metrics=metrics)
      learn.fit(num_epochs, 1e-2)


      # Get the final metrics values after training on the validation set
      val_metrics = learn.validate(dl=dls.valid)
      # print(val_metrics)
      final_loss = val_metrics[0]
      final_mae = val_metrics[1]  # mae
      final_rmse = val_metrics[2]   # rmse

      all_mae_list.append(final_mae)
      all_rmse_list.append(final_rmse)
      all_loss.append(final_loss)

      # Save the model with a distinct name using timestamp
      timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
      model_name = f"{save_model_to}/tab_model_match_{timestamp}_{run_index}.pkl"
      learn.export(model_name)

  # Calculate the average MAE and RMSE
  average_mae = sum(all_mae_list) / len(all_mae_list)
  average_rmse = sum(all_rmse_list) / len(all_rmse_list)
  average_loss = sum(all_loss)  / len(all_loss)

  print("Average Loss:", average_loss)
  print("Average MAE:", average_mae)
  print("Average RMSE:", average_rmse)

# train("/content/drive/MyDrive/MultiInput")

# learn = load_learner("/content/drive/MyDrive/Tab_modelling/trial_07_epoch_30/tab_model_20240129164026_2.pkl")

"""## Generate Plots"""

def test_on_test_data(dls_fold_list, save_graphs_to):

    for run_index, dls in enumerate(dls_fold_list):

      # Test the model on test indices
      test_dl = dls.test_dl(test_df)
      test_results = learn.get_preds(dl=test_dl)
      predictions = test_results[0].numpy().flatten()
      actuals = test_df_strain.values

      # test mae
      test_mae = mae(tensor(predictions), tensor(actuals)).item()

      # Plot a scatter plot with fixed axis limits
      plt.scatter(actuals, predictions)
      plt.xlabel("Actuals")
      plt.ylabel("Predictions")
      # plt.title("Scatter Plot of Actuals vs Predictions")
      plt.title(f"Scatter Plot of Actuals vs Predictions (Test MAE on Fold {run_index}: {test_mae:.4f})")


      # Set the same limits for both x-axis and y-axis
      max_value = max(np.max(actuals), np.max(predictions))
      min_value = min(np.min(actuals), np.min(predictions))
      plt.xlim(min_value, max_value)
      plt.ylim(min_value, max_value)


      # Add trend line
      trend_line = np.polyfit(actuals, predictions, 1)
      plt.plot(actuals, np.polyval(trend_line, actuals), color='red', label='Trend Line')

      # Add optimal line
      plt.plot([0, 25], [0, 25], '--', color='black', label='Optimal Line')
      # Add legend
      plt.legend()

      plt.savefig(f"{save_graphs_to}/_{run_index}.png")
      plt.show()

      # # Add the test MAE to the plot
      # plt.text(5, 19, f'Test MAE: {test_mae:.4f}', color='blue', fontsize=12)

      # # Plot a scatter plot
      # plt.scatter(actuals, predictions)
      # plt.xlabel("Actuals")
      # plt.ylabel("Predictions")
      # plt.title("Scatter Plot of Actuals vs Predictions")

# test_on_test_data(dls_fold_list, "/content/drive/MyDrive/Tab_modelling/trial_07_epoch_30")


