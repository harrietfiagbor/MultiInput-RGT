# -*- coding: utf-8 -*-
"""Tab Modelling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OmNXw_-SEfKPO56Gt9DkOdsuqC7v55VS

## Library imports and initialization
"""


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

# from google.colab import drive
# drive.mount('/content/drive')


# Load the data
pickle_file_path = '/content/drive/MyDrive/QFHD/data/preprocessed_24Jan22_134624.p'
metadata_file_path = '/content/drive/MyDrive/QFHD/data/metadata3.xlsx'

# Load metadata
meta_data = pd.read_excel(metadata_file_path)

# to use the splits in the pickle...
# Load pickle data
with open(pickle_file_path, 'rb') as handle:
  pickle_data = pickle.load(handle)



first_data_row = meta_data['Subject ID'].first_valid_index()
meta_data      = meta_data.iloc[first_data_row:] # makes the row with the first Subject ID the first row

"""## Data Processing"""

# List of features to keep
# Isaac has added the 'Subject ID' column
features_to_keep = [
    'Subject ID',
    'Age [yrs rounded to nearest]',
    'BMI',
    'Diabetes',
    'Hx Hypertension',
    'Hx Smoking',
    'Hx Stroke/TIA',
    'Hx COPD',
    'Personal or Family Hx CV Disease',
    'Hx High Cholesterol',
    'Cholesterol Medication',
    'Strain, Global Longitudinal [-%]'
    ]

# Select only the features to keep
features_df = meta_data[features_to_keep]
features_df

# Remove rows with missing target variable
features_df = features_df.dropna(subset=['Strain, Global Longitudinal [-%]'])

# # Check the shape of the dataframe after removing missing values
# print(features_df.shape)

# make BMI column a numerical column

features_df          = features_df.loc[(features_df['BMI'] != '#DIV/0!') & (features_df['BMI'] != 0)] # drops rows with #DIV/0! or 0 in BMI columns
features_df['BMI']   = features_df['BMI'].astype(float) #changes the data type of the BMI column from object to float64
features_df.reset_index(drop=True, inplace=True) # renumber the index column

# features_df.info()

features_df_train = features_df.drop('Subject ID', axis=1)
# features_df_train.shape

# Identify categorical and numerical columns
categorical_cols = features_df_train.select_dtypes(include=['object', 'bool']).columns.tolist()
numerical_cols = features_df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# categorical_cols

numerical_cols.remove('Strain, Global Longitudinal [-%]')
# numerical_cols






"""## Dataloaders"""

def get_dls_fold_list():

  # Get train indices
  # Train indices is data that doesn't have the current fold not in test indices
  all_indices = features_df.index.tolist()

  dls_fold_list = []

  for current_fold_index, fold_indices in enumerate(val_indices_list):
      # Exclude the current fold indices and test indices from the total set of indices
      train_indices = [index for index in all_indices if index not in fold_indices and index not in test_indices]

      splits = [train_indices, fold_indices]

      y_col = ['Strain, Global Longitudinal [-%]']
      procs = [Categorify, FillMissing, Normalize]
      y_block = RegressionBlock() if isinstance(features_df_train['Strain, Global Longitudinal [-%]'].values[0], float) else CategoryBlock()
      pd.options.mode.chained_assignment=None

      dls_fold = get_tabular_dls(features_df_train, cat_names=categorical_cols, cont_names=numerical_cols, y_names=y_col, splits=splits, bs=16)
      dls_fold_list.append(dls_fold)
      # dls.show_batch()

  return dls_fold_list


def do_hyp_opt(save_hyp_to, trial_num=None, num_epochs=30, num_trials=10):

  """## TAB Model"""

  # run_num = "07"
  # epochs  = 30
  # num_trial = 10

  if trial_num == None:
    trial_num = "01"
  # model_folder = f"/content/drive/MyDrive/Tab_modelling/trial_{run_num}_epoch_{epochs}"
  model_folder = f"{save_hyp_to}/trial_{trial_num}_epoch_{num_epochs}"
  os.makedirs(model_folder, exist_ok=True)

  # hyp tune
  # no_of_layers
  # layer_size
  # better understand embs


  # Define the objective function for Optuna
  def objective(trial):

    # lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    # Lists to store metrics values for each training run
    all_mae_list = []
    all_rmse_list = []
    all_loss = []
    best_trial_value = float('inf')

    # Train tab model on each dataloader fold (train and val dl)
    for run_index, dls in enumerate(dls_fold_list):

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

  # optim
  fig = optuna.visualization.plot_optimization_history(study)
  fig.show()

  #edf
  fig = optuna.visualization.plot_edf(study)
  fig.show()

  # param importances
  optuna.visualization.matplotlib.plot_param_importances(study)


def test_on_test_data(dls_fold_list):

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

      # Add the test MAE to the plot
      plt.text(5, 19, f'Test MAE: {test_mae:.4f}', color='blue', fontsize=12)

      # Plot a scatter plot
      plt.scatter(actuals, predictions)
      plt.xlabel("Actuals")
      plt.ylabel("Predictions")
      plt.title("Scatter Plot of Actuals vs Predictions")


def train_tab_model(save_model_to):

  os.makedirs(save_model_to, exist_ok=True)

  # Lists to store metrics values for each training run
  all_mae_list = []
  all_rmse_list = []
  all_loss = []

  # Train tab model on each dataloader fold (train and val dl)
  for run_index, dls in enumerate(dls_fold_list):
      # Using MAE and RMSE as the metrics
      metrics = [mae, rmse]

      learn = tabular_learner(dls, layers=[200, 100], y_range=None, metrics=metrics)
      learn.fit(epochs, 1e-2)



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
      model_name = f"{save_model_to}/tab_model_{timestamp}_{run_index}.pkl"
      # learn.save(model_name)
      learn.export(model_name)


  # Calculate the average MAE and RMSE
  average_mae = sum(all_mae_list) / len(all_mae_list)
  average_rmse = sum(all_rmse_list) / len(all_rmse_list)
  average_loss = sum(all_loss)  / len(all_loss)

  print("Average Loss:", average_loss)
  print("Average MAE:", average_mae)
  print("Average RMSE:", average_rmse)



