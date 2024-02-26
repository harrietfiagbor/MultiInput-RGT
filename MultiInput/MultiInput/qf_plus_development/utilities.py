import cloudpickle
import inspect
import lzma
import numpy as np
import os
import pandas as pd
import pickle
import pytz
import re
import time
from collections import defaultdict
from datetime import datetime
from qf_plus_shared.global_vars import *
from qf_plus_shared.signal_processing import get_whistle
from qf_plus_shared.stats import prediction_statistics, binary_statistics
from qf_plus_shared.inference import prep_x_data
from random import shuffle
from shutil import copyfile

def get_train_val_data(channels, pickle_file=None, threshold=('Strain',14)):
  if pickle_file==None:
    pickle_file=get_latest_pickle()

  # load data
  pickle_data   = open_pickle(pickle_file)
  meta_data     = get_meta_data()

  # prep data for training
  preped_data     = {}
  for filename, df in pickle_data['file_data'].items():
    preped_data[filename] = prep_x_data(df, channels)

  # split data
  X_train = []
  X_valid = []
  Y_train = []
  Y_valid = []
  for val_split in pickle_data['val_test_splits']['val_splits']:
    x_train_split = []
    x_valid_split = []
    y_train_split = []
    y_valid_split = []
    for filename, x_data in preped_data.items():
      subj_id = filename[:-6]
      label   = int(int(meta_data[threshold[0]][meta_data['Subject_ID']==subj_id]) < threshold[1])
      if subj_id in val_split:
        x_valid_split.append(x_data)
        y_valid_split.append(label)
      else:
        x_train_split.append(x_data)
        y_train_split.append(label)
    X_valid.append(np.array(x_valid_split))
    X_train.append(np.array(x_train_split))
    Y_valid.append(np.array(y_valid_split))
    Y_train.append(np.array(y_train_split))
  X_train = np.array(X_train)
  X_valid = np.array(X_valid)
  Y_train = np.array(Y_train)
  Y_valid = np.array(Y_valid)

  return X_train, X_valid, Y_train, Y_valid












def write_txt(file_path, contents):
  with open(file_path, 'wb') as handle:
    np.savetxt(handle, [contents], fmt='%s')
    print(f'Wrote file: {file_path}')



def get_test_performance(experiment_ids, override_pickle=None, save_dir=DEV_DIR+'qf_plus_development/test_performance/', experiments_str=None):
  #experiment_ids, subject_predictions, bin_stats = get_test_performance(experiment_ids)
  from qf_plus_development.neptune import get_neptune_experiment
  from qf_plus_shared.inference import get_prediction
  
  HEALTHY_TH = 14
  
  # get current test subjects & meta data
  pickle_file = get_latest_pickle() if override_pickle==None else override_pickle
  latest_pickle = open_pickle(pickle_file)
  test_subj_ids = latest_pickle['val_test_splits']['test_subjects']
  meta_data     = get_meta_data()


  # get the models & associated data
  experiments = []
  val_bas     = []
  print('\n######################### Loading Experiments #########################')
  for experiment_id in experiment_ids:
    experiment = get_neptune_experiment(experiment_id)
    model      = experiment.get('model', None)
    if model==None:
      print(f'No saved model for {experiment_id}')
      continue

    healthy_th = experiment['config'].get('healthy_threshold', None)
    if healthy_th != 14:
      print(f'Healthy threshold {healthy_th} != {HEALTHY_TH}')
      continue

    data_uuid = experiment['config'].get('data_uuid', None)
    if data_uuid==None or '2021-09-28' in data_uuid: 
      data_uuid='repickled.p'
    else:
      data_uuid=data_uuid+'.p'
    pickle_file = CACHE_DIR + data_uuid if override_pickle==None else override_pickle
    loaded_data = open_pickle(pickle_file)
    if loaded_data=={}:
      print(f'No pickle for {experiment_id}')
      continue 

    if 'file_data' in loaded_data.keys():
      loaded_data = loaded_data['file_data']

    val_bas.append( experiment.get('max_ba', np.nan) )
    experiments.append({'experiment_id':experiment_id, 'model':model, 'loaded_data':loaded_data})
  med_val_ba = round(np.median(val_bas), 2)


  # get predictions for the test subjects for each model
  print('\n#########################      Predicting     #########################')
  subject_predictions = {}
  for subj_id in test_subj_ids:
    subject_predictions[subj_id] = {}
    subject_predictions[subj_id]['actual'] = int( meta_data['Strain'].loc[meta_data['Subject_ID']==subj_id] < HEALTHY_TH )
    subject_predictions[subj_id]['prob_pos'] = []
    for experiment in experiments:
      print(f"Predicting for {subj_id} and model {experiment['experiment_id']}")
      model       = experiment['model']
      loaded_data = experiment['loaded_data']
      dfs         = [df for filename, df in loaded_data.items() if subj_id in filename[:-6]]
      subject_predictions[subj_id]['prob_pos'].extend( [ get_prediction(df, model) for df in dfs ] )

  # get prediction stats for each subject
  actuals = []
  preds   = []
  for subj_id, subject_prediction in subject_predictions.items():
    if len(subject_prediction['prob_pos'])==0: continue
    subject_prediction['pred_stats'] = prediction_statistics(subject_prediction['prob_pos'])
    preds.append(subject_prediction['pred_stats']['ensemble_prediction'])
    actuals.append(subject_prediction['actual'])

  # summarize, print, and save results
  experiments_str = '' if experiments_str==None else experiments_str + '\n'
  bin_stats = binary_statistics(actuals, preds)
  results   = ('\n' + '#########################   Test Performance  #########################\n'
              + f'{experiments_str}'
              + f'Experiments: {experiment_ids}\n'
              + f'Median Validation BAL-ACC: {med_val_ba}\n'
              + f'{bin_stats}\n')
  subj_preds= ('\n######################### Subject Predictions #########################\n'
              + ''.join( [f'{subj_id}\t{subject_prediction}\n' for subj_id, subject_prediction in subject_predictions.items()] ))
  print(results + subj_preds)
  
  if save_dir:
    timestamp = datetime.now(tz=pytz.timezone('US/Eastern')).strftime("%d%b%y_%H%M%S") # current date and time
    write_txt(f'{save_dir}test_performance_{timestamp}.txt', results + subj_preds)
    
  return experiment_ids, subject_predictions, bin_stats



def serialize_function(func):
  module_path   = os.path.abspath(inspect.getfile(func)).split('/')
  module_import = getattr(__import__(module_path[-2], fromlist=[module_path[-1][:-3]]), module_path[-1][:-3])
  cloudpickle.register_pickle_by_value(module_import)
  return cloudpickle.dumps(func)




def create_val_splits(preprocessed_data, meta_data, save_pickle=True):
  NUM_VAL_SPLITS  = 5
  TEST_FACTION    = 0.1
  STRAIN_TH       = 14

  # get test subjects from most recent pickle
  latest_pickle = open_pickle(get_latest_pickle())
  test_subjs    = latest_pickle['val_test_splits']['test_subjects']

  # remove existing test subjects from the preprocessed list 
  subj_ids_all  = set( [ filename[:-6] for filename in list(preprocessed_data['file_data'].keys()) ] )
  subj_ids      = [ subj_id for subj_id in subj_ids_all if subj_id not in test_subjs ]

  # determine the number of pos & neg to add to the test_subjs
  pos_subjs, neg_subjs = split_list(meta_data, test_subjs, strain_th=STRAIN_TH)
  pos_ratio     = len(pos_subjs) / len(test_subjs)
  adj_pos_rate  = np.mean( (0.5, pos_ratio) ) # half-way between the actual class proportion and a balanced rate
  num_test_subj = int(len(subj_ids_all)*TEST_FACTION)
  num_test_pos  = int(adj_pos_rate*num_test_subj)
  num_test_neg  = num_test_subj - num_test_pos
  num_pos_add   = max(0, num_test_pos - len(pos_subjs))
  num_neg_add   = max(0, num_test_neg - len(neg_subjs))

  # split remaining subj_ids into pos/neg -- returned is a shuffled list
  pos_subjs, neg_subjs = split_list(meta_data, subj_ids, strain_th=STRAIN_TH)
  test_subjs = test_subjs + pos_subjs[:num_pos_add] + neg_subjs[:num_neg_add]

  # remove new test subjects from the preprocessed list 
  subj_ids = [ subj_id for subj_id in subj_ids_all if subj_id not in test_subjs ]

  # split the remaining subjects into NUM_VAL_SPLITS of near-equal size and uniform pos/neg ratios
  pos_subjs, neg_subjs = split_list(meta_data, subj_ids, strain_th=STRAIN_TH)
  pos_splits           = shuffle_split(pos_subjs, num_splits=NUM_VAL_SPLITS)
  neg_splits           = shuffle_split(neg_subjs, num_splits=NUM_VAL_SPLITS)
  val_splits           = []
  for pos_split, neg_split in zip(pos_splits, neg_splits):
    val_splits.append(list(pos_split) + list(neg_split))
  val_test_splits = {'val_splits':val_splits, 'test_subjects':test_subjs}
  preprocessed_data['val_test_splits'] = val_test_splits

  # save new preprocessed_data object, including the val/test splits
  if save_pickle==True:
    timestamp = datetime.now(tz=pytz.timezone('US/Eastern')).strftime("%d%b%y_%H%M%S") # current date and time
    write_pickle(CACHE_DIR + f'preprocessed_{timestamp}.p', preprocessed_data)
  
  return val_test_splits


def split_list(meta_data, subj_ids, strain_th=14):
  # split subjects into positive & negative sets
  pos_subs = set()
  neg_subs = set()
  for subject_id in subj_ids:
    print(subject_id)
    row = meta_data['Strain'][meta_data['Subject_ID']==subject_id]
    if len(row)==0: continue
    pos = int(row.values[0] < strain_th)
    if pos==1:
      pos_subs.add(subject_id)
    else:
      neg_subs.add(subject_id)

  # shuffle
  pos_subs = list(pos_subs)
  shuffle(pos_subs)
  neg_subs = list(neg_subs)
  shuffle(neg_subs)  

  return pos_subs, neg_subs

def shuffle_split(subj_ids, num_splits=5):
  # shuffle
  subj_ids = list(subj_ids)
  shuffle(subj_ids)

  # split into similarly-sized lists
  subjs_split = np.array_split(subj_ids, num_splits)

  return subjs_split  

def get_latest_pickle():
  mod_time = -np.inf
  pickle_files = list_data_files(data_files_dirs=[CACHE_DIR], file_type='p')
  for file in pickle_files:
    #print(os.path.basename(file)[:13])
    if 'preprocessed_' == os.path.basename(file)[:13]:
      file_stats = os.stat(file)
      if file_stats.st_mtime > mod_time:
        mod_time = file_stats.st_mtime
        most_recent = file
  #print(most_recent)
  return most_recent


def add_meta_data(meta_file_path, categorized_data):
  '''
  Adss data at a file/subject level (e.g., LVEF, clinic, etc)
  Args:
    meta_file_path: string : meta-data file
    categorized_data: dictionary : data separated by category
  Returns:
    categorized_data dictionary that includes the meta-data
  '''
  
  meta_data_df  = get_meta_data(meta_file_path)
  meta_template = {'Clinic': '', 'LVEF': np.nan, 'Strain': np.nan, 'Use_4_Train': 'F', 'Age': np.nan}
  for filename in categorized_data.keys():
    subj_id  = filename[:-6] # slice off the last 6 characters (e.g., ".1.csv")
    row      = meta_data_df[meta_data_df['Subject_ID']==subj_id]
    if len(row)==0: 
      print(f'Subject ID: {subj_id} does not appear to an associated meta-data record')
      row_dict = meta_template
    else:
      row_dict = row.loc[:, row.columns != 'Subject_ID'].to_dict(orient='records')[0]
    categorized_data[filename]['meta_data'] = row_dict
    
  return categorized_data




def categorize(loaded_data, nom_categories=None, min_samples=400):
  '''
  splits the file into seperate categories (e.g., Baseline)
  Args:
    loaded_data: dictionary of the loaded date
    nom_categories: default merges Expiration+Whistle
  Returns:
    categorized_data, all_categories
  '''
  if nom_categories==None:
    nom_categories = {'Baseline':['Baseline'], 'Expiration+Whistle': ['Expiration', 'Whistle'], 'Recovery':['Recovery']}
  all_categories   = nom_categories.keys()
    
  categorized_data = nested_dict()
  for filename, dataframe in loaded_data.items():
    for cat_name, cat_cols in nom_categories.items():
      series = dataframe['raw_data'][dataframe.Category.isin(cat_cols)]
      num_samples = series.count()
      if num_samples >= min_samples:
        categorized_data[filename]['sig_data'][cat_name]['sig_df'] = pd.DataFrame(series)
      else:
        print(f'Excluding {cat_name} from {filename} because it only has {num_samples} samples')

  return categorized_data, all_categories


def get_subject_data(subject_ids, loaded_data):
  subject_data = nested_dict()
  for filename in loaded_data.keys():
    for subject_id in subject_ids:
      if subject_id in filename:
        subject_data[filename] = loaded_data[filename]
  
  return subject_data



def get_subject_id(filename):
  return filename[:-6]




def filter_data(meta_data, files_dict, start_date='01.14.21', included_sex=['M','F'], end_date='01.01.31', lvef=True, strain=True, use_4_train=True, min_whistle_sec=8, max_whistle_sec=25, min_signal_sec=35):
  '''
  Removes items from an input object (supported: dictionary, list, dataframe), before the start and after end date.
  All in_obj types are expected to contain a Subject_ID field (or filename, which contains the Subject_ID). 
  Further, if in_obj is a dictionary, the keys are expected to be the Subject_ID
    
  Args:
    in_dict: dictionary to act on
    start_date: (string, two-digit month.day.year) earliest date to include; default: '01.01.01' ( January 1st, 2001)
    end_date:   (string, two-digit month.day.year) latest date to include; default: '01.01.31' ( January 1st, 2031)
  
  Returns:
    meta_data (df) & files_dict (dict of dfs) filtered based on provided criteria
    dict: number of subjects and files filtered by criteria, list of file subject id's missing meta-data, list of meta-data subject id's missing files
  '''
  file_subj_ids_orig = set( [ filename[:-6] for filename in list(files_dict.keys()) ] )
  start_date         = datetime.strptime(start_date, '%m.%d.%y')
  end_date           = datetime.strptime(end_date, '%m.%d.%y')
  
  #
  # filter meta-data
  #
  num_subj_start     = len(meta_data)
  subj_ids_start     = list(meta_data['Subject_ID'])

  # start & end dates
  meta_data['Study_Date'] = [ datetime.strptime( re.search('([0-9]{2}\.[0-9]{2}\.[0-9]{2})', subj_id)[0], '%m.%d.%y' ) for subj_id in list(meta_data['Subject_ID']) ]
  meta_data          = meta_data.loc[ (meta_data['Study_Date'] >= start_date) & (meta_data['Study_Date'] <= end_date) ]
  meta_data          = meta_data.drop(columns='Study_Date')
  num_removed_date   = num_subj_start - len(meta_data)

  # LVEF, Strain, & Use_4_train
  num_missing_strain = meta_data['Strain'].isna().sum()
  num_missing_lvef   = meta_data['LVEF'].isna().sum()
  num_false_train    = len(meta_data.loc[meta_data['Use_4_Train']!=True])
  meta_data          = meta_data.loc[ (meta_data['Strain'].notnull()) & (meta_data['LVEF'].notnull()) & (meta_data['Use_4_Train']==True)]
  
  # sex filtering
  start_len          = len(meta_data)
  sex_filter         = [('M' in included_sex), ('F' not in included_sex)]
  meta_data          = meta_data.loc[ meta_data['IsMale'].isin(sex_filter) ]  
  num_removed_sex    = start_len - len(meta_data)
  
  # removed meta-data subject ids
  subj_ids_removed   = list( set(subj_ids_start) - set(meta_data['Subject_ID']) )
  num_removed_meta   = len(subj_ids_removed)
  num_sub_remain     = num_subj_start - num_removed_meta
  
  #
  # filter files_dict
  #

  # remove duplicates  
  for filename in list(files_dict.keys()):
    if ').csv' in filename:
      del files_dict[filename]
      os.rename(DATA_DIR + filename, DATA_DIR + filename + '.dup')
  
  
  num_files_start    = len(files_dict)
  
  
  # remove the filtered subject ids from the meta-data
  rem_files     = []
  for filename in files_dict.keys():
    for subj_id in subj_ids_removed:
      if subj_id in filename:
        rem_files.append(filename)
  files_rem_meta = len(rem_files)
  starting_files = len(files_dict.keys()) - files_rem_meta
  for filename in rem_files:
    files_dict.pop(filename, None)
  
  
  # remove insufficient or excessive whistle length or insufficient signal length
  whistle_count = 0
  signal_count  = 0
  rem_files     = []
  for filename, df in files_dict.items():
    len_whistle,_,mid_whistle_idx,_ = get_whistle(df)
    if len_whistle==0:
      rem_files.append(filename)
      whistle_count += 1
      continue
    if (min_whistle_sec > len_whistle/FS) or (len_whistle/FS > max_whistle_sec):
      rem_files.append(filename)
      whistle_count += 1
      continue
  
    act_len = len(df.loc[df['Category']=='Whistle'])
    if (act_len / len_whistle) < 0.6:
      rem_files.append(filename)
      whistle_count += 1
      continue
    

    start_idx = mid_whistle_idx - min_signal_sec*FS//2
    end_idx   = start_idx + min_signal_sec*FS
    signal_df = df.iloc[start_idx:end_idx]
    if len(signal_df) < FS*min_signal_sec:
      rem_files.append(filename)
      signal_count += 1

  # remove files from the files_dict
  for filename in rem_files:
    files_dict.pop(filename, None)

  ending_files = len(files_dict.keys())


  # remove from meta_data any subjects with no accompanying files
  '''
  meta_no_files = file_subj_ids - meta_subj_ids
  for subj_id in meta_no_files:
    meta_data.drop( meta_data[meta_data['Subject_ID']==subj_id].index, inplace=True)
  '''
      
  #
  # find bi-directionally missing files/subject
  #
  file_subj_ids = set( [ filename[:-6] for filename in list(files_dict.keys()) ] )
  meta_subj_ids = set( meta_data['Subject_ID'] )
  
  # missing files
  missing_files = list( set( list( meta_subj_ids - file_subj_ids ) ) - file_subj_ids_orig )
  
  # missing meta-data
  missing_meta  = list( file_subj_ids - meta_subj_ids )
  
  #
  # compose return dictionary
  #
  ret_dict = {'num_subj_start': num_subj_start, 
              'date_filtered':num_removed_date, 
              'num_missing_strain':num_missing_strain,
              'num_missing_lvef':num_missing_lvef,
              'num_false_train':num_false_train,
              'num_removed_sex':num_removed_sex,
              'net_removed_meta':num_removed_meta,
              'num_sub_remain': num_sub_remain,
              'starting_files':starting_files,
              'whistle_loss':whistle_count,
              'signal_loss':signal_count,
              'ending_files':ending_files,
              'missing_files':missing_files,
              'missing_meta':missing_meta
              }
  
  return meta_data, files_dict, ret_dict





def filter_data_13OCT21(in_obj, start_date='01.01.01', end_date='01.01.31'):
  '''
  Removes items from an input object (supported: dictionary, list, dataframe), before the start and after end date.
  All in_obj types are expected to contain a Subject_ID field (or filename, which contains the Subject_ID). 
  Further, if in_obj is a dictionary, the keys are expected to be the Subject_ID
    
  Args:
    in_dict: dictionary to act on
    start_date: (string, two-digit month.day.year) earliest date to include; default: '01.01.01' ( January 1st, 2001)
    end_date:   (string, two-digit month.day.year) latest date to include; default: '01.01.31' ( January 1st, 2031)
  '''
  start_date = datetime.strptime(start_date, '%m.%d.%y')
  end_date   = datetime.strptime(end_date, '%m.%d.%y')
  
  if isinstance(in_obj, pd.DataFrame):
    in_obj['Study_Date'] = [ datetime.strptime( re.search('([0-9]{2}\.[0-9]{2}\.[0-9]{2})', subj_id)[0], '%m.%d.%y' ) for subj_id in list(in_obj['Subject_ID']) ]
    in_obj = in_obj.loc[ (in_obj['Study_Date'] >= start_date) & (in_obj['Study_Date'] <= end_date) ]
    in_obj = in_obj.drop(columns='Study_Date')
    
  
  elif isinstance(in_obj, dict):
    for filename in list(in_obj.keys()):
      file_date = datetime.strptime(re.search('([0-9]{2}\.[0-9]{2}\.[0-9]{2})', filename)[0], '%m.%d.%y')
      if file_date < start_date or file_date > end_date:
        del in_obj[filename]

  elif isinstance(in_obj, list):
    for filename in list(in_obj):
      file_date = datetime.strptime(re.search('([0-9]{2}\.[0-9]{2}\.[0-9]{2})', filename)[0], '%m.%d.%y')
      if file_date < start_date or file_date > end_date:
        in_obj.remove(filename)  
  
  return in_obj
  



def filter_data_orig(in_dict, start_date='01.01.01', end_date='01.01.31'):
  '''
  Removes items from dictionary, before the start and after end date.
  Args:
    in_dict: dictionary to act on
    start_date: (string, two-digit month.day.year) earliest date to include; default: '01.01.01' ( January 1st, 2001)
    end_date:   (string, two-digit month.day.year) latest date to include; default: '01.01.31' ( January 1st, 2031)
  '''
  start_date = time.strptime(start_date, '%m.%d.%y')
  end_date   = time.strptime(end_date, '%m.%d.%y')
  for filename in list(in_dict.keys()):
    file_date = time.strptime(re.search('([0-9]{2}\.[0-9]{2}\.[0-9]{2})', filename)[0], '%m.%d.%y')
    if file_date < start_date or file_date > end_date:
      del in_dict[filename]
      #print(f'removed {filename}')






def get_data(in_dict, filename, category=None, data_type=None, item=None):
  '''
  Helper function to move all tge GET data calls to one place in case the dictionary schema changes

  current "direct" GET use-cases
     x, y = preprocessed_data[filename]['sig_data'][category]['psd_df']
     dataframe = clip_signal( categorized_data[filename]['sig_data'][category]['sig_df'], clip_rows=6)
     standardize_quality_features(category, categorized_data[filename]['sig_data'][category]['raw_features'])
     np.mean(preprocessed_data[filename]['sig_data']["Baseline"]['sig_df']['raw_data']))
     normalized_quality_vals = preprocessed_data[filename][category]['norm_quality_features']
     preprocessed_data[filename][category]['sig_quality']

  Example:
    get_data(preprocessed_data, filename=filename, category='Baseline', data_type='flipped_median')

  Args:
    data_type: None (default) | 'meta_data' | 'raw_features'
  Returns:
    Data: Either a Pandas Dataframe, Series, a Dictionary, or a single value, based on the parmaters:
      data_type: 
        None (default) : the specified category sig_df dataframe is returned
        'raw_features' : dictionary of the raw features for the specified category
      item:
        If specified (item='lvef'), along with data_type (e.g., 'meta_data'), the single item is returned (e.g., meta_data['lvef'])
  '''
  if category:

    # the entire dataframe
    if data_type==None:
      data = in_dict[filename]['sig_data'][category]['sig_df']
      
    # one of the columns in the sig_df DataFrame
    elif data_type in in_dict[filename]['sig_data'][category]['sig_df'].columns:
      data = in_dict[filename]['sig_data'][category]['sig_df'][data_type]
      
    # the PSD dataframe
    elif data_type=='psd':
      data = in_dict[filename]['sig_data'][category]['psd_df']
    
    # 'raw_features', 'std_quality_features', 'sig_quality'
    elif data_type in in_dict[filename]['sig_data'][category].keys():
      data = in_dict[filename]['sig_data'][category][data_type]

    # if something other than the planned parmaters is specified, return the whole enchalada
    else:
      data = in_dict[filename] 

  else:
      
    # meta_data
    if data_type=='meta_data':
      data = in_dict[filename]['meta_data']
    
    #categories
    if data_type=='categories':
      data = in_dict[filename]['sig_data'].keys()

    # file features
    if data_type=='file_features':
      data = in_dict[filename]['file_feats']
  
    # the whole file dictionary
    if data_type==None:
      data = in_dict[filename]
    
  if item:
    data = data[item]

  return data


def get_meta_data(meta_file_path=META_DATA):
  '''
  Gets the meta-data from the summary file
  Args:
    meta_file_path: path of the sumamry file
  Returns:
    Pandas Dataframe of the meta-data
  '''
  usecols = ['Subject ID','Use for Algorithm Training [T/F]', 'Final EF%', 'Strain, Global Longitudinal [-%]', 'Clinic ID', 'Age [yrs rounded to nearest]', 'Gender',
            'Hx Hypertension', 'Personal or Family Hx CV Disease', 'Diabetes', 'Hx Smoking', 'Hx Stroke/TIA', 'Hx COPD']
  columns = {'Subject ID':'Subject_ID', 'Use for Algorithm Training [T/F]':'Use_4_Train', 'Final EF%':'LVEF', 'Strain, Global Longitudinal [-%]':'Strain', 'Clinic ID':'Clinic', 'Age [yrs rounded to nearest]':'Age', 'Gender':'IsMale',
            'Hx Hypertension':'Hx_HTN', 'Personal or Family Hx CV Disease':'HxCVD', 'Diabetes':'DM', 'Hx Smoking':'HxSmoke', 'Hx Stroke/TIA':'HxStroke', 'Hx COPD':'HxCOPD'}
  meta_data_df   = pd.read_excel(meta_file_path, usecols=usecols)
  meta_data_df.rename(columns=columns, inplace=True)
  first_data_row              = meta_data_df['Subject_ID'].first_valid_index()
  meta_data_df                = meta_data_df.iloc[first_data_row:].copy()
  meta_data_df['Use_4_Train'] = np.where( meta_data_df['Use_4_Train'] != 'T', False, True)
  meta_data_df['IsMale']      = np.where( meta_data_df['IsMale'] == 'M', True, False)
  meta_data_df['Hx_HTN']      = np.where( meta_data_df['Hx_HTN'] == 'M', True, False)
  meta_data_df['HxCVD']       = np.where( meta_data_df['HxCVD'] == 'Y', True, False)
  meta_data_df['DM']          = np.where( meta_data_df['DM'] == 'Y', True, False)
  meta_data_df['HxSmoke']     = np.where( meta_data_df['HxSmoke'] == 'Y', True, False)
  meta_data_df['HxStroke']    = np.where( meta_data_df['HxStroke'] == 'Y', True, False)
  meta_data_df['HxCOPD']      = np.where( meta_data_df['HxCOPD'] == 'Y', True, False)
  return meta_data_df  


def get_subj_id(file):
  subj_id = os.path.basename(file)[:-6]
  return subj_id

def get_unique_subj_ids(file_list, sort=True):
  subj_ids = []
  for file in file_list:
    subj_ids.append( get_subj_id(file) )

  if sort==True:
    subj_ids = sorted(set(subj_ids))
  else:
    subj_ids = list(set(subj_ids))
  
  return subj_ids





def list_data_files(data_files_dirs=[DATA_DIR], file_type='csv', num_list=np.inf):
  '''
  Directory list of CSV files within provided paths; NOT recurrsive
  Args:
    data_files_dirs: list : full paths
    num_list: the number to list; if set to default (np.inf) all files will be returned, otherwise the first num_list will be returned
  Returns:
    List of CSV files within the provided directories
  '''
  files = []
  for drive_dir in data_files_dirs:
    files_list = os.listdir(drive_dir)
    for filename in files_list:
      if filename.endswith(f'.{file_type}'):
        files.append(os.path.join(drive_dir, filename))
  
  files = files[:min(len(files), num_list)]

  return files


def load_file(file_path):
  df       = pd.read_csv(file_path, index_col=False, usecols=[0,2,3], names=['raw_data', 'Category', 'Whistle'])
  df.index = np.linspace(start=0, stop=(len(df)-1)/FS, num=len(df))            # create index, which is the time axis
  df.drop(0, axis=0, inplace=True)                                             # drop header row
  df['Category'] = np.where(df['Whistle']==' True', 'Whistle', df['Category']) # mark 'Whistle' in the Category column / space in Whistle column (' True') IS intentional
  df.drop(labels=['Whistle'], axis=1, inplace=True)
  df['raw_data'] = pd.to_numeric(df['raw_data'], downcast='integer')
  return df



  

def load_data_files(data_files_dirs=[DATA_DIR], start_date='2021-01-14'):
  '''Load the sensor data files, either from disk (if cache is stale) or from cache
  Args:
    data_files_dirs: [DATA_DIR] (default), LIST of full paths
    start_date: date from which to load; effectively ignors older files
  Returns:
    A dictionary of loaded_data with keys being filename and item being a dataframe
  '''

  print('Loading files...')
  tic         = time.perf_counter()
  files_list  = list_data_files(data_files_dirs)
  start_date  = datetime.strptime(start_date, '%Y-%m-%d')
  loaded_data = {} #nested_dict()
  
  for file in files_list:
    filename  = os.path.basename(file)
    study_date = datetime.strptime( re.search('([0-9]{2}\.[0-9]{2}\.[0-9]{2})', filename)[0], '%m.%d.%y' )
    if study_date < start_date: continue
  
    print(f'Loading {file}')
    df       = pd.read_csv(file, index_col=False, usecols=[0,2,3], names=['raw_data', 'Category', 'Whistle'])
    df.index = np.linspace(start=0, stop=(len(df)-1)/FS, num=len(df))            # create index, which is the time axis
    df.drop(0, axis=0, inplace=True)                                             # drop header row
    df['Category'] = np.where(df['Whistle']==' True', 'Whistle', df['Category']) # mark 'Whistle' in the Category column / space in Whistle column (' True') IS intentional
    df.drop(labels=['Whistle'], axis=1, inplace=True)
    df['raw_data'] = pd.to_numeric(df['raw_data'], downcast='integer')
    loaded_data[filename] = df
    
  toc = time.perf_counter()
  print(f'Loading data done in {toc-tic:0.2f}s')

  return loaded_data

  
    
def load_data_files_orig(data_files_dirs, files_list=None, num_load=np.inf):
  '''OBSOLETE: use load_data_files
  
  Load the sensor data files, either from disk (if cache is stale) or from cache
  Args:
    data_files_dirs: LIST of full paths
    num_load: number to load
  Returns:
    A dictionary of loaded_data
  '''
  # data_files_dirs: list : full paths
  
  print('Loading data...')
  tic         = time.perf_counter()

  # list files currently in the provided paths; or use the provided list
  if files_list==None:
    files_list = list_data_files(data_files_dirs, num_list=num_load)

  # load cached loaded_data, if it exists, and compare to the current files_list
  re_gen, loaded_data = regen_or_cache(data_object='loaded_data', compare_obj=files_list)
  toc = time.perf_counter()

  # Either files have been updated or there is no cache file
  if re_gen==True:
    loaded_data = {}
    TIME_STEP   = MS_PER_SAMPLE/1000
    for file in files_list:
      data      = pd.read_csv(file, index_col=False, names=['Values', 'Category'], usecols=[0,2])
      if str(data.Values[0])=='PPG': data.drop(0, axis=0, inplace=True) # drop header row of new files
      data['Seconds'] = np.linspace(start=0, stop=(len(data)-1)*TIME_STEP, num=len(data))
      data.set_index('Seconds', inplace=True)
      data.rename(columns={'Values':'raw_data'}, inplace=True)
      data['raw_data'] = pd.to_numeric(data['raw_data'], downcast='integer')
      filename = os.path.basename(file)
      loaded_data[filename] = data
      if len(loaded_data) >= num_load: break
    
    toc = time.perf_counter()

    # create/update the loaded_data_files pickle for later caching
    pickle_dict = {'files_list':files_list, 'loaded_data':loaded_data}
    write_pickle(pickle_file=LOADED_DATA_PICKLE, pickle_data=pickle_dict)

  print(f'Loading data done in {toc-tic:0.2f}s')

  return loaded_data
  
  
  
def load_data_files_13OCT21(data_files_dirs=[DATA_DIR], files_list=None, force_reload=False, num_load=np.inf):
  '''Load the sensor data files, either from disk (if cache is stale) or from cache
  Args:
    data_files_dirs: [DATA_DIR] (default), LIST of full paths
    file_list: None (default), list of file paths to specifically load (overrides data_files_dirs)
    force_reload: False (default), whether to reload ALL files, not just new ones.  However, the segment labeling column in the orig_df is retained
    num_load: np.inf (default), limit on number to load
  Returns:
    A dictionary of loaded_data (preprocessed_data pickle)
  '''
  from qf_plus_development.signal_processing import add_signals
  from qf_plus_shared.signal_processing import get_whistle, detrend_data
  
  # internal helper function (only used in this function) to return the ends of each category
  def trend_ends(x):
    ends = [0, 0]
    if len(x) > 0:
      _, trend = detrend_data(x, method='hp', hp_lambda=1e6)
      ends     = [trend[0], trend[-1]]
    return ends

  print('Loading data...')
  tic         = time.perf_counter()

  # list files currently in the provided paths; or use the provided list
  if files_list==None:
    files_list = list_data_files(data_files_dirs, num_list=num_load)

  # load cached preprocessed_data, if it exists, and compare to the files_list
  preprocessed_data = open_pickle(PREPROCESSED_PICKLE)

  if len(preprocessed_data)==0: # empty
    preprocessed_data = nested_dict()

  if force_reload==False:
    set_diff = list( set(sorted([os.path.basename(file) for file in files_list])).difference(set(sorted(list(preprocessed_data.keys())))) )
    if len(set_diff) != 0:
      print('Files to be loaded:', set_diff)
      files_list  = [DATA_DIR + file for file in set_diff]
    else:
      print('Cache is current / No files will be loaded')
      files_list = []


  # New files exist that is not in the preprocessed_data file
  if len(files_list) > 0:
    # archive existing preprocessed_data pickle
    pickle_basename = os.path.basename(PREPROCESSED_PICKLE)
    archive_file    = CACHE_DIR + 'Archive/{:%Y%m%d%H%M%S}_'.format(datetime.now()) + pickle_basename
    copyfile(PREPROCESSED_PICKLE, archive_file)

    for file in files_list:
      print(f'Loading {file}')
      df       = pd.read_csv(file, index_col=False, usecols=[0,2,3], names=['raw_data', 'Category', 'Whistle'])
      df.index = np.linspace(start=0, stop=(len(df)-1)/FS, num=len(df))            # create index, which is the time axis
      if df['raw_data'].iloc[0] == 'PPG':
        # new file format
        df.drop(0, axis=0, inplace=True)                                             # drop header row
        df['Category'] = np.where(df['Whistle']==' True', 'Whistle', df['Category']) # mark 'Whistle' in the Category column / space in Whistle column (' True') IS intentional
      else: 
        # first file format
        # clip the first CLIP_ROWS from each of the following categories
        CLIP_ROWS = 6
        df.drop( df.loc[(df['Category']=='Baseline')].iloc[0:CLIP_ROWS].index, inplace=True ) #
        df.drop( df.loc[(df['Category']=='Expiration')].iloc[0:CLIP_ROWS].index, inplace=True ) #
        df.drop( df.loc[(df['Category']=='Recovery')].iloc[0:CLIP_ROWS].index, inplace=True )

        #  
        # adjust the height Expiration & Recovery phases to address the sensor LED change
        #
        # categorize the phases
        bl_df    = df.loc[df['Category']=='Baseline'].copy()
        vm_df    = df.loc[df['Category'].isin(['Expiration','Whistle'])].copy()
        rec_df   = df.loc[df['Category']=='Recovery'].copy()
        
        # get the trend ends, to find out how much to adjust
        bl_ends  = trend_ends( bl_df['raw_data'].values)
        vm_ends  = trend_ends( vm_df['raw_data'].values)
        rec_ends = trend_ends(rec_df['raw_data'].values)
        vm_adj   = 1.0 if vm_ends[0]==0  else bl_ends[1] / vm_ends[0]
        rec_adj  = 1.0 if rec_ends[0]==0 else vm_adj * vm_ends[1] / rec_ends[0]

        # adjust each category
        vm_df['raw_data']  = vm_df['raw_data']  * vm_adj
        rec_df['raw_data'] = rec_df['raw_data'] * rec_adj

        # combine the categories back into a single dataframe
        df = pd.concat([bl_df, vm_df, rec_df])

      df.drop(labels=['Whistle'], axis=1, inplace=True)
      df['raw_data'] = pd.to_numeric(df['raw_data'], downcast='integer')
      
      # add columns for labeling (cycle, trend, Labeled_Category)
      FOUR_SEC = FS*4
      if len(df) > FOUR_SEC:
        df = add_signals(df)
        
      filename = os.path.basename(file)
      if 'Labeled_Category' in preprocessed_data[filename]['orig_df']:
        df['Labeled_Category'] = preprocessed_data[filename]['orig_df']['Labeled_Category']
      else:
        df['Labeled_Category'] = np.nan

      preprocessed_data[filename]['orig_df'] = df
      if len(preprocessed_data) >= num_load: break
    
    # create/update the preprocessed_data pickle for later caching
    write_pickle(pickle_file=PREPROCESSED_PICKLE, pickle_data=preprocessed_data)

  toc = time.perf_counter()
  print(f'Loading data done in {toc-tic:0.2f}s')

  return preprocessed_data
  
  


def nested_dict():
  '''
  helper furnciton to allow for nested dictionary to be created without needing to declare each parent level.  for example:
    nested_dict = nested_dict()
    nested_dict['level-1']['level-2']['level-3']['level-4']['level-5'] = 3.14
  Args:
    None
  Returns:
    A defaultdict object that allows for declaration and assignment at any level on a single line
  '''
  return defaultdict(nested_dict)
    



def open_pickle(pickle_file):
  '''opens & reads a pickle file
  Args:
    pickle_file: full path of the pickle to read
  Returns:
    pickle_data: the data within the pickle file 
  '''
  pickle_data = {}
  if os.path.isfile(pickle_file):
    with open(pickle_file, 'rb') as handle:
      try:
        pickle_data = pickle.load(handle)
      except 'LZMAError':
        pickle_data = pickle.load(lzma.LZMAFile(handle))
        
    print(f'Loaded pickle: {pickle_file}')
  
  return pickle_data




def print_keys(dic, indent=''):
  for key, value in dic.items():
    print(indent + key)
    if isinstance(value, dict):
      indent += '--'
      print_keys(value, indent)
    else:
      indent += ''




def regen_or_cache(compare_obj, data_object='preprocessed_data'):
  '''Determines whether to re-generate or use the cache of a particular data_object (e.g., loaded_data)
  Args:
    data_object: string: 'preprocessed_data' (default) | 'categorized_data' | 'loaded_data'
    compare_obj: the data to compare
  Returns:
    re_gen: whether to re-generate the cache
    compare_obj: if re_gen==False, the cached data is returned; else: the original compare_obj
  '''

  re_gen = True  
  if data_object=='preprocessed_data':
    pickle_file = open_pickle(PREPROCESSED_PICKLE)
    if pickle_file: # not empty
      if sorted(list(pickle_file.keys())) == sorted(compare_obj):
        compare_obj = pickle_file
        re_gen      = False

  elif data_object=='categorized_data':
    cat_data_pickle = open_pickle(CAT_DATA_PICKLE)
    if cat_data_pickle: # not empty
      category = 'Baseline'
      filename = list(cat_data_pickle.keys())[0] # first file in the categorized_data pickle
      pickle_features = get_data(cat_data_pickle, filename=filename, category=category, data_type='raw_features')
      comp_features   = get_data(compare_obj,     filename=filename, category=category, data_type='raw_features')
      if pickle_features==comp_features:
        print('Pickle matches; using cache')
        compare_obj = cat_data_pickle
        re_gen      = False

  elif data_object=='loaded_data':
    loaded_data_pickle = open_pickle(LOADED_DATA_PICKLE)
    if loaded_data_pickle: # not empty
      if loaded_data_pickle['files_list'] == compare_obj:
        compare_obj = loaded_data_pickle['loaded_data']
        re_gen      = False
        
  return re_gen, compare_obj
  


def remove_data():
  '''
  Removes data from preprocessed_data.p pickle file that does NOT correlate with a file on disk
  '''
  print('Removing any stale data from pickle...')

  # list files on disk
  files_list = list_data_files([DATA_DIR])

  # load cached preprocessed_data, if it exists, and compare to the files_list
  preprocessed_data = open_pickle(PREPROCESSED_PICKLE)

  # compare disk to pickle data & remove any files in the pickle data that are NOT on disk
  disk_set = set(sorted([os.path.basename(file) for file in files_list]))
  data_set = set(sorted(list(preprocessed_data.keys())))
  set_diff = list( data_set.difference(disk_set) )
  for file in set_diff:
    print(f'Removing {file} from pickle data')
    preprocessed_data.pop(file, None)

  if len(set_diff) != 0:
    write_pickle(PREPROCESSED_PICKLE, preprocessed_data)



def save_for_c_sharp(filename, category, data_type='raw_data'):
  '''
  Saves data for processing in C#
  Args:
    filename: filename used to save
    category: e.g., Baseline
    data_type: type of data, default = 'raw_data'
  Returns:
    Interactive filedownloader interface
  '''
  from datetime import datetime
  # this is for the proprietary format that the c# code requires
  print("FILENAME CATEGORY", preprocessed_data[filename][category])
  to_save = get_x_y(preprocessed_data[filename][category], data_type)
  x = to_save['seconds']
  y = to_save['values']
  to_save = np.stack((x, y*c_sharp_scaling, np.ones(x.shape), np.zeros(x.shape)))
  to_save = np.transpose(to_save)
  to_save_filename = 'preprocessed ' + filename + ' ' + category + '.txt'
  # print(to_save_filename)
  # print(to_save)
  with open(to_save_filename, mode='w') as f:
    f.write('Date: 1/20/2021\n')
    f.write('Time: 10:00:00 AM\n')
    f.write('Patient: ' + filename + '\n')
    f.write('Age: 0\n')
    f.write('Protocol: Blood Flow Index\n')
    f.write('Location: Left Hand\n\n')
    f.write('Time (sec),Pulse Data,Index Data,Event\n')
    np.savetxt(f, to_save, delimiter=',', fmt=['%1.2f', '%d', '%d', '%d'])
  print(type(to_save_filename))
  filedownloader.download(to_save_filename)



def write_pickle(pickle_file, pickle_data):
  '''write data to a specified pickle file (full path)
  Args:
    pickle_file: full path of the pickle file
    pickle_data: the data to write
  Returns:
    N/A: TODO: may want to consider returning a success/failure
  '''

  # archive existing preprocessed_data pickle
  pickle_basename = os.path.basename(pickle_file)
  archive_file    = CACHE_DIR + 'Archive/{:%Y%m%d%H%M%S}_'.format(datetime.now()) + pickle_basename
  try:
    copyfile(pickle_file, archive_file)
  except:
    pass

  # save new pickle  
  with open(pickle_file, 'wb') as handle:
      pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
      print(f'Wrote file: {pickle_file}')

    
