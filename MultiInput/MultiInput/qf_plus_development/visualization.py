import matplotlib.pyplot as plt
import random
from matplotlib.patches import Ellipse
from qf_plus_development.meas_quality import *
from qf_plus_development.utilities import get_data

from IPython.display import display
from IPython.display import clear_output

# at most, display these many datasets at once
max_lines_to_plot = 40

# plot options
plt.style.use('dark_background')
plot_width_px                     = 1800
plot_height_px                    = 800
pixels_per_inch                   = 80
plt.rcParams['figure.dpi'       ] = pixels_per_inch
plt.rcParams['font.family'      ] = ['Liberation Mono', 'DejaVu Sans Mono', 'mono', 'sans-serif']
plt.rcParams['font.size'        ] = 18
plt.rcParams['legend.fontsize'  ] = 12
plt.rcParams['figure.figsize'   ] = [plot_width_px/pixels_per_inch, plot_height_px/pixels_per_inch]
plt.rcParams['lines.linewidth'  ] = 2
plt.rcParams['lines.antialiased'] = True
plt.rcParams['grid.alpha'       ] = 0.5
plt.rcParams['axes.grid'        ] = True
plot_line_alpha                   = 0.8
ellipses_alpha                    = 0.3
category_colors = {'Baseline':          (0.20, 0.60, 0.40, 1.0),
                  'Expiration+Whistle': (1.00, 0.60, 0.20, 1.0),
                  'Expiration':         (1.00, 0.60, 0.20, 1.0),
                  'Whistle':            (0.90, 0.00, 0.45, 1.0),
                  'Recovery':           (0.20, 0.20, 1.00, 1.0)}
ellipse_class_colors = [[0,.4,0], [1,.1,.1]]

plt.rcParams.update({
   'boxplot.boxprops.color': 'white',
   'boxplot.capprops.color': 'white',
   'boxplot.flierprops.color': 'white',
   'boxplot.flierprops.markeredgecolor': 'white',
   'boxplot.whiskerprops.color': 'white'})


def plot_x_y_line(data_dict, x_ax_label='X-axis', y_ax_label='Y-axis'):
  '''
  Generalized line plot function that can plot multiple lines, with different labels,
  on different axes
  Args:
    data_dict: minimum required schema: {label: {'y':<y-data>}}
      full schema, including optional keys:
        { label_1: {
            'y': <list like data to plot>,
            'category':'Baseline', # optional, for color lookup; default is just use the color order already specified in `category_colors`
            'axis': ax1,           # optional, default = plt
            'log': False           # optional, default=False
          }
        }
      category: used to select the color
      axis: used for subplots and/or secondary y-axis; default=plt
      log: boolean (default=False) makes semilogy

      See below for a few examples
      
    x_ax_label: string for the x-axis; default='X-axis'
    y_ax_label: string for the y-axis; default='Y-axis'
  Returns:
    nothing; just plots the data

  data_dict examples:
    data_dict = {label_1:{'y':y1, 'category':'Recovery', 'axis':ax1,'log': True}, 'x':x1}
    data_dict = {label_1:{'y':y1, 'axis':ax1}, 'y2':{'y':y2, 'axis':ax1}}
    data_dict = {label_1:{'y':y1, 'axis':ax1},'y2':{'y':y2, 'axis':ax1},'y3':{'y':y2, 'axis':ax2},'y4':{'y':y2, 'axis':ax3}}
  '''

  # TODO: add figure title

  # extract x, if present
  if 'x' in data_dict:
    x = data_dict['x']
    del data_dict['x']
  else:
    x = []
  
  for label, line_dict in data_dict.items():

    y = line_dict['y']

    # create x if it was not provided
    if len(x)==0:
      x = np.linspace(0, len(y) * MS_PER_SAMPLE / 1000, len(y))

    # use the axis, if provided, otherwise use default (plt)
    if 'axis' in line_dict:
      ax = line_dict['axis']
    else:
      ax = plt

    # make y-scale log, if specified (equivalent to plt.semilogy())
    if 'log' in line_dict and line_dict['log']==True:
      ax.set_yscale('log')

    if 'category' in line_dict:
      color = category_colors[ line_dict['category'] ]
    else:
      color = None
    ax.plot(x, y, label=label, alpha=plot_line_alpha, color=color)
    ax.legend()
    try:
      ax.set_ylabel(y_ax_label)
      ax.set_xlabel(x_ax_label)
    except:
      plt.ylabel(y_ax_label)
      plt.xlabel(x_ax_label)

  plt.show()




def plot_file(preprocessed_data, filename, data_type, domain, axes=plt, show_legend=True, show_trend=True, plot_categories=[]):
  '''
  Args:
    data_type: among the list ['raw_data', 'cycle', 'trend', 'mean_subtracted', 'flipped_median']
  Returns:
  
  '''

  # select all categories, unless otherwise specified in plot_categories
  categories = preprocessed_data[filename]['sig_data'].keys() if len(plot_categories)==0 else plot_categories
  for category in categories:
  
    lvef      = preprocessed_data[filename]['meta_data']['LVEF']
    label     = f'(LVEF={lvef}) {filename} {category}'

    # get the plot data, which includes the seconds (x-axis)
    if domain=='time':
      dataframe = get_data(preprocessed_data, filename=filename, category=category)
      plot_x    = dataframe.index.to_numpy()
      plot_y    = dataframe[data_type].to_numpy()
      trend_y   = dataframe['trend'].to_numpy()
      x_label   = 'Time (s)'
      y_label   = 'Measurement'
    else:
      dataframe = get_data(preprocessed_data, filename=filename, category=category, data_type='psd')
      plot_x    = dataframe.index.to_numpy()
      plot_y    = dataframe['Pxx_db'].to_numpy()
      x_label   = 'Frequency (Hz)'
      y_label   = 'Welch PSD'
      
    axes.plot(plot_x, plot_y, color = category_colors[category], alpha=plot_line_alpha, label=label)
    if domain=='time' and show_trend:
      axes.plot(plot_x, trend_y, color=category_colors[category], alpha=plot_line_alpha*0.75, label=label + ' trend')

    if show_legend:
      axes.legend()
    try:
      axes.set_ylabel(y_label)
      axes.set_xlabel(x_label)
    except:
      plt.xlabel(x_label)
      plt.ylabel(y_label)
    

def plot_t_y(y_list, data_labels_list, x_ax_label, y_ax_label, axes=plt):
  for y, data_label in zip(y_list, data_labels_list):
    t = np.linspace(0, len(y) * MS_PER_SAMPLE / 1000, len(y))
    axes.plot(t, y, label=data_label, alpha=plot_line_alpha, linewidth=0.8)
    try:
      axes.set_ylabel(y_ax_label)
      axes.set_xlabel(x_ax_label)
    except:
      plt.xlabel(x_ax_label)
      plt.ylabel(y_ax_label)
    
  plt.legend()
  plt.show()



def plot_normal_2D_ellipses(x_means, x_vars, y_means, y_vars, max_stdevs=3, ax=plt):
  """
  Loop through classes and plot ellipses representing the gaussians.
  
  All 4 variables should each be an Nx1 list with N classes to plot.
  """
  
  num_classes = len(x_means)
  
  for which_class in range(num_classes):
  
    # pull out the variables
    mean_x = x_means[which_class]
    mean_y = y_means[which_class]
    stdev_x = np.sqrt(x_vars[which_class])
    stdev_y = np.sqrt(y_vars[which_class])
  
    # visualize the classifier
    centroid = [x_means[which_class], y_means[which_class]]
  
    for stdev_scaling in range(1, max_stdevs+1): # show a few ellipses at various stdevs away  
      # calculate ellipse widths
      width  = stdev_scaling * stdev_x * 2 # stdev is radius - we need diameter
      height = stdev_scaling * stdev_y * 2
  
      ellipse = Ellipse(xy = centroid, width = width, height = height)
      ax.add_artist(ellipse)
      ellipse.set_clip_box(ax.bbox)
      ellipse.set_alpha(ellipses_alpha)
      ellipse.set_facecolor(ellipse_class_colors[which_class])


def plot_quality_bayesian(category, preprocessed_data, ax=plt, max_stdevs=3):
  """
  Plot a bayesian as ellipses.
  The first class is plotted as green, the second as red.
  means should be a 2 x num_features list where the first row is the  means of the green class.
  stdevs should be the same shape as means
  """

  num_quality_features = len(quality_bayes_classifier[category]['MEAN'][0])
  which_features = [random.randint(0,num_quality_features-1), random.randint(0,num_quality_features-1)]
  which_feature_x = which_features[0]
  which_feature_y = which_features[1]
  
  # extract out the data for our randomly selected features:
  
  current_means = quality_bayes_classifier[category]['MEAN']
  current_vars  = quality_bayes_classifier[category]['VAR' ]
  
  x_means = [current_means[0][which_feature_x], current_means[1][which_feature_x]]
  y_means = [current_means[0][which_feature_y], current_means[1][which_feature_y]]
  x_vars  = [current_vars [0][which_feature_x], current_vars [1][which_feature_x]]
  y_vars  = [current_vars [0][which_feature_y], current_vars [1][which_feature_y]]

  plot_normal_2D_ellipses(x_means, x_vars, y_means, y_vars, max_stdevs, ax)

  # update the bounding box
  large_number = np.inf
  max_x = -large_number
  max_y = -large_number
  min_x = +large_number
  min_y = +large_number

  for index in range(len(x_means)):
    min_x = min(min_x, x_means[index] - max_stdevs * np.sqrt(x_vars[index]))
    min_y = min(min_y, y_means[index] - max_stdevs * np.sqrt(y_vars[index]))
    max_x = max(max_x, x_means[index] + max_stdevs * np.sqrt(x_vars[index]))
    max_y = max(max_y, y_means[index] + max_stdevs * np.sqrt(y_vars[index]))

  # loop through files and plot dots representing the quality feature values
  scatter_x = []
  scatter_y = []
  scatter_c = [] # colors
  low_quality_count = 0
  for filename in preprocessed_data.keys():
    # get the values
    normalized_quality_vals = get_data(preprocessed_data, filename=filename, category=category, data_type='std_quality_features')
    #key_string = 'std_quality_features'
    #if not key_string in normalized_quality_vals.keys():
    #  continue
    #normalized_quality_vals = normalized_quality_vals[key_string]
    scatter_x_val = normalized_quality_vals[which_feature_x]
    scatter_y_val = normalized_quality_vals[which_feature_y]
    # append it to the scatter lists
    scatter_x.append(scatter_x_val)
    scatter_y.append(scatter_y_val)
    # update the bounds of the chart
    min_x = min(min_x, scatter_x_val)
    min_y = min(min_y, scatter_y_val)
    max_x = max(max_x, scatter_x_val)
    max_y = max(max_y, scatter_y_val)
    # color it based on the class
    predicted_quality_class = get_data(preprocessed_data, filename=filename, category='Baseline', data_type='sig_quality')
    predicted_quality_class = predicted_quality_class[1] > predicted_quality_class[0]
    low_quality_count += predicted_quality_class
    scatter_c.append(ellipse_class_colors[int(predicted_quality_class)])
  ax.scatter(scatter_x, scatter_y, s=30, c=scatter_c, alpha=ellipses_alpha * 2)
  print(f'# of low quality signals = {low_quality_count}')
  
  ax.set_xlim(min_x, max_x)
  ax.set_ylim(min_y, max_y)
  
  ax.set_xlabel(f'Quality Feature #{which_feature_x+1} (of {num_quality_features} features)')
  ax.set_ylabel(f'Quality Feature #{which_feature_y+1} (of {num_quality_features} features)')
  
  
  
def plot_circulation(preprocessed_data, threshold, clinic, pulsatility_per_file):



  arr_greater_than_threshold_legacy = []
  arr_less_than_threshold_legacy = []
  arr_greater_than_threshold_welch = []
  arr_less_than_threshold_welch = []
  lvef_to_plot = []
  circulation_index_to_plot_welch = []
  circulation_index_to_plot_legacy = []

  for file in list(preprocessed_data.keys()):


    if file not in pulsatility_per_file:
        print(f"FILE {file} SKIPPED BECAUSE OF INSUFFICIENT DATA")
        continue
        
    algo_label = get_data(preprocessed_data, file, data_type='meta_data', item='meta_data')['Use_4_Train']
    
    if not algo_label and not algo_label.lower() == 't':
      print("FILE SKIPPED BECAUSE NOT USED IN ALGORITHM TRAINING")

    
    lvef_obj = get_data(preprocessed_data, file, data_type='meta_data', item='meta_data')



    if clinic != 'All':
      clinic_condition = lvef_obj['Clinic'] == clinic
    else:
      clinic_condition = True


    # print("CLINIC CONDITION ", clinic_condition)


    if not math.isnan(lvef_obj['LVEF']) and clinic_condition:
      # only if both lvef and ci exist
      lvef_to_plot.append(lvef_obj['LVEF'])



      # Loop through keys: Output of windowed_processing.keys
      legacy_circulation_calculation = pulsatility_per_file[file]['legacy']['circulation_index']
      welch_cirulation_calculation = pulsatility_per_file[file]['welch']['circulation_index']

 

      circulation_index_to_plot_legacy.append(legacy_circulation_calculation)
      circulation_index_to_plot_welch.append(welch_cirulation_calculation)

      if int(lvef_obj['LVEF']) >= threshold:
        arr_greater_than_threshold_legacy.append(legacy_circulation_calculation)
        arr_greater_than_threshold_welch.append(welch_cirulation_calculation)
      else:
        # lvef_less_than_50 = pulsatility_per_file[file]['welch']['circulation_index']
        arr_less_than_threshold_legacy.append(legacy_circulation_calculation)
        arr_less_than_threshold_welch.append(welch_cirulation_calculation)


  lvef_greater_than_threshold_and_circulation_legacy = np.array(arr_greater_than_threshold_legacy)
  lvef_less_than_threshold_and_circulation_legacy = np.array(arr_less_than_threshold_legacy)

  lvef_greater_than_threshold_and_circulation_welch = np.array(arr_greater_than_threshold_welch)
  lvef_less_than_threshold_and_circulation_welch = np.array(arr_less_than_threshold_welch)




  lvef_to_plot = np.array(lvef_to_plot)
  circulation_index_to_plot_legacy = np.array(circulation_index_to_plot_legacy)
  circulation_index_to_plot_welch = np.array(circulation_index_to_plot_welch)



  x = lvef_to_plot
  y_w = circulation_index_to_plot_welch
  y_l = circulation_index_to_plot_legacy


  print(len(x))
  print(len(y_w))
  print(len(y_l))

  # creating dataframe
  df=pd.DataFrame()
  df['lvef'] = x
  df['welch'] = y_w
  df['legacy'] = y_l

  # sorting dataframe
  df.sort_values('lvef', ascending=True, inplace=True)

  plt.scatter(df['lvef'], df['welch'], label = 'Welch', color='blue')
  plt.scatter(df['lvef'], df['legacy'], label = 'Legacy', color='green')

  plt.xlabel('LVEF')
  plt.ylabel('Mean(baseline) - Min(valsalva)')
  plt.legend()
  # set_gridlines(plt)
  plt.show()

  # Welch and Legacy Circulation plotted against each other
  plt.scatter(df['welch'], df['legacy'])

  plt.xlabel('Welch Circulation Index')
  plt.ylabel('Legacy Circulation Index')

  print(f'Correlation = {np.corrcoef(y_w, y_l)}')


def box_plot_descriptive_statistics():
  lvef_legacy_greater_threshold = f'LVEF > {global_threshold} LEGACY'
  lvef_legacy_lesser_threshold = f'LVEF < {global_threshold} LEGACY'
  lvef_welch_greater_threshold = f'LVEF > {global_threshold} WELCH'
  lvef_welch_lesser_threshold = f'LVEF < {global_threshold} WELCH'


  df_data_lvef_circulation = {
      lvef_legacy_greater_threshold: lvef_greater_than_threshold_and_circulation_legacy,
      lvef_welch_greater_threshold: lvef_greater_than_threshold_and_circulation_welch,
      lvef_legacy_lesser_threshold: lvef_less_than_threshold_and_circulation_legacy,
      lvef_welch_lesser_threshold: lvef_less_than_threshold_and_circulation_welch
  }


  df_lvef_circulation = pd.DataFrame.from_dict(df_data_lvef_circulation,orient='index')
  df_lvef_circulation = df_lvef_circulation.transpose()

  print(f"          CIRCULATION INDEXES PER CALCULATION & LVEF VALUE {global_threshold}  ")
  print(df_lvef_circulation.describe())
  
  
  
def calculate_confusion(FiF_threshold):
  TP = None
  TN = None
  FP = None
  FN = None

  cm = None

  y_true = []
  y_pred = []

  for x in range(0, len(lvef_greater_than_threshold_and_circulation_legacy)):
    y_true.append(1)

    if (lvef_greater_than_threshold_and_circulation_legacy[x] > FiF_threshold):
      y_pred.append(1)
    else:
      y_pred.append(0)


  for x in range(0, len(lvef_less_than_threshold_and_circulation_legacy)):
    # print(lvef_less_than_50_and_cirulation_legacy[x])
    y_true.append(0)
    if (not lvef_less_than_threshold_and_circulation_legacy[x] > FiF_threshold):
      y_pred.append(0)
    else:
      y_pred.append(1)

 

  cm = confusion_matrix(y_true, y_pred)
  cmd = ConfusionMatrixDisplay(cm, display_labels=['Sick','Healthy'])
  cmd.plot(values_format = '')
  TP = cm[0][0]
  TN = cm[1][1]
  FP = cm[1][0]
  FN = cm[0][1]

  print("NOTE *: Positive = SICK")
  print("TP TN FP FN: ", TP, TN, FP, FN)

  print("FiF Threshold: ", FiF_threshold)
  print(f"LVEF CUTOFF {global_threshold}")
  print("ACCURACY: ", round(100 *  (TP + TN) / (TP + TN + FP + FN)), '%')
  print("Misclassification: ", round(100 * (FP + FN) / (TP + TN + FP + FN)), "%")
  print('Precision: ',  round(100 * TP / (TP + FP)), '%')
  print("Sensitivity: ", round(100 * TP / (TP + FN)), '%')
  print('Specificity: ',  round(100 * TN / (TN + FP)), '%')