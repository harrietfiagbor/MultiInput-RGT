import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy
import time
import sys
from numpy.lib.stride_tricks import as_strided
from numpy.polynomial import polynomial as P
from scipy import signal, stats
from scipy.stats import kurtosis, linregress
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline
from scipy.ndimage import median_filter, gaussian_filter, minimum_filter, maximum_filter
from statsmodels.tsa.tsatools import detrend
from qf_plus_shared.global_vars import *
import qf_plus_development.utilities as dev_utils
from qf_plus_shared.signal_processing import detrend_data, filter_signal







def get_local_mins(dataframe, handle_below=True, mode='distance', do_plot=False):
  NUM_CYCLES = 5

  # lowpass to remove noise; subtract from mean to flip so orientation is proper (e.g: systolic rise following valleys)
  channel = lowpass_filter( dataframe['raw_data'].mean() - dataframe['raw_data'] )

  # get initial valleys (min_ids) of the start of Baseline
  min_ids = get_initial_points(dataframe, -channel)

  # loop through, finding remaining minimum, favoring a distance close to the cyclic period (HR)
  min_ids = get_remaining_extrema(min_ids, -channel, mode='distance', detrend_win=True)


  # handle the points that fall below the piecewise linear interpolation line between mins
  # this is an iterative process that adds points to the minimum below the env_lo line
  count = 0
  while handle_below:
    if count >= 5: break
  
    SMALL_NUM    = 1e-2
    x            = np.arange(len(channel)) # range(min_ids[-1]+1)
    env_lo       = InterpolatedUnivariateSpline(x=min_ids, y=channel[min_ids], k=1)(x)  # np.interp( x, min_ids, window[min_ids])
    val_sub      = channel - env_lo
    below_ids    = (val_sub<-SMALL_NUM).nonzero()[0]
    #print(count, len(below_ids), below_ids, val_sub[np.array(below_ids)])
    if len(below_ids)==0: break
    
    below_ranges = [list(group) for group in mit.consecutive_groups(below_ids)]
    for below_range in below_ranges:
      new_min = np.argmin(val_sub[below_range])+below_range[0]
      min_ids = np.append(min_ids, new_min)
    
    min_ids   = np.unique(min_ids)
    min_ids.sort()
    count += 1
  
  
  # create a lower envelope (env_lo)
  x             = range(len(channel))
  env_lo        = InterpolatedUnivariateSpline(x=min_ids, y=channel[min_ids], k=1)(x) # np.interp( x, min_ids, channel[min_ids])
  
  # plot if specified
  if do_plot==True:
    plt.plot(x, channel)
    plt.plot(x, env_lo, 'r')
    plt.scatter(min_ids, channel[min_ids], s=80, facecolors='none', edgecolors='r')

  return channel, min_ids, env_lo





def get_raw_envelope(dataframe, do_plot=False):
  NUM_CYCLES   = 5

  # lowpass to remove noise; subtract from mean to flip so orientation is proper (e.g: systolic rise following valleys)
  mean_sub_lp  = lowpass_filter( dataframe['raw_data'].mean() - dataframe['raw_data'] )

  # get initial valleys (min_ids) of the start of Baseline
  end_bl_idx   = get_whistle(dataframe)[1]
  period_bl    = get_period_3(mean_sub_lp[:end_bl_idx])
  window       = mean_sub_lp[:NUM_CYCLES*period_bl]
  cyc_min, cyc_max = cyclic_min_max(window)
  min_ids,_    = signal.find_peaks(-window, distance=period_bl//2, prominence=(cyc_max-cyc_min)/2)
  # TODO: need to handle case where none (or less than expected) are found, which would be caused by a poor-quality signal
  
  # loop through, finding next minimum, favoring a distance close to the cyclic period (HR)
  while True:
    # estimate where the next min will occur based on the prior cycles
    min_diffs  = np.diff(min_ids)[-NUM_CYCLES:]
    diff_med   = np.median(min_diffs)
    x          = np.arange(len(min_diffs))
    p          = np.polyfit(x, min_diffs, deg=1)
    next_min   = np.polyval(p,x[-1]+1)
    #print(f'next min: {next_min}')
    if abs(1 - next_min/diff_med) > 0.2: # use the median if the linear forecast is more than 20% different from the median
      next_min = diff_med
      #print(f'-----diff_med: {diff_med}')
    next_min  = int( min( round(FS/MIN_F0_HZ), max( next_min, round(FS/MAX_F0_HZ) ) )) # bound the next_min by the phyisiologic pulse rate

    # create the next window within which the next min will be found
    start_idx   = min_ids[-1]
    end_idx     = start_idx + int(round(1.5*next_min))
    if end_idx >= len(mean_sub_lp):
      break
    #print(f'start_idx: {start_idx} / end_idx: {end_idx}')
    window      = local_detrend( mean_sub_lp[start_idx:end_idx+1] )
    ids,props   = signal.find_peaks(-window, prominence=0)
    #print(ids,props)
    if len(ids) > 0:
      idx         = ids[ np.argmin(abs(next_min - ids)) ] # find the min closest to the next_min
    else:
      idx         = next_min # handle case where there is not an actually peak/valley
    #print(f'idx: {idx}')
    min_idx     = idx + start_idx
    #print(f'min_idx: {min_idx}')

    min_ids     = np.append(min_ids, min_idx)
    
  # find the peak between the min_id pairs
  max_ids       = np.array([], dtype=int)
  for min_num in range(len(min_ids) - 1):
    start_idx   = min_ids[min_num]
    end_idx     = min_ids[min_num + 1]
    window      = local_detrend( mean_sub_lp[start_idx:end_idx+1] )
    max_ids     = np.append( max_ids, np.argmax(window) + start_idx )

  # handle the points that fall below the piecewise linear interpolation line between mins
  # this is an iterative process that adds points to the minimum below the env_lo line
  count         = 0
  handle_below  = True
  while handle_below:
    if count >= 5: break
  
    SMALL_NUM    = 1e-2
    x            = np.arange(len(mean_sub_lp)) # range(min_ids[-1]+1)
    env_lo       = InterpolatedUnivariateSpline(x=min_ids, y=mean_sub_lp[min_ids], k=1)(x)  # np.interp( x, min_ids, window[min_ids])
    val_sub      = mean_sub_lp - env_lo
    below_ids    = (val_sub<-SMALL_NUM).nonzero()[0]
    #print(count, len(below_ids), below_ids, val_sub[np.array(below_ids)])
    if len(below_ids)==0: break
    
    below_ranges = [list(group) for group in mit.consecutive_groups(below_ids)]
    for below_range in below_ranges:
      new_min = np.argmin(val_sub[below_range])+below_range[0]
      min_ids = np.append(min_ids, new_min)
    
    min_ids   = np.unique(min_ids)
    min_ids.sort()
    count += 1
  
  # create envelopes : env_lo & env_hi
  x             = range(len(mean_sub_lp))
  env_lo        = InterpolatedUnivariateSpline(x=min_ids, y=mean_sub_lp[min_ids], k=1)(x)
  env_hi        = InterpolatedUnivariateSpline(x=max_ids, y=mean_sub_lp[max_ids], k=1)(x)
  
  # plot if specified
  if do_plot==True:
    plt.plot(x, mean_sub_lp)
    plt.plot(x, env_lo, 'r')
    plt.plot(x, env_hi, 'g')
    plt.scatter(min_ids, mean_sub_lp[min_ids], s=80, facecolors='none', edgecolors='r')
    plt.scatter(max_ids, mean_sub_lp[max_ids], s=80, facecolors='none', edgecolors='g')
  
  return mean_sub_lp, min_ids, max_ids, env_lo, env_hi






def get_remaining_extrema(prior_ids, channel, mode='distance', detrend_win=True):

  # loop through, finding next extrema, favoring either distance close to the cyclic period (HR) or ratio of prominence to distance
  while True:
    # estimate where the next min/max will occur based on the prior cycles
    next_min_max = get_next_distance(prior_ids)

    # create the next window within which the next min will be found
    start_idx   = prior_ids[-1]
    end_idx     = start_idx + min( len(channel)-1, int(round(1.5*next_min_max)) )
    if start_idx + next_min_max >= len(channel)-1:
      break # end of file / no more to add
      
    window      = channel[start_idx:end_idx+1]
    if detrend_win==True:
      window    = local_detrend(window)
      
    # find local extrema
    ids,props   = signal.find_peaks(window, prominence=0)
    if len(ids)==0:          # handle case where there is not an actually peak/valley
      idx         = next_min_max
    elif mode=='distance':   # one or more extrema found
      idx         = ids[ np.argmin(abs(next_min_max - ids)) ] # find the point closest to the next_min_max        
    else:
      SMALL_NUM   = 1e-6
      idx         = ids[ np.argmax( props['prominences'] / abs(next_min_max - ids + SMALL_NUM)) ] # find the point closest to the next_min_max & largest prominence

    prior_ids     = np.append(prior_ids, idx + prior_ids[-1])

  return prior_ids







def preprocess(preprocessed_data, filenames=None):
  '''
  Adds pre-processing & meta_data for each file for which Labeled_Category is populated

  Args:
    preprocessed_data: the dictionary holding all the data
    filenames: list of filenames to pre-process, if None (default) all files in preprocessed_data will be pre-processed
  Returns:
    preprocessed_data dictionary with the preprocessing completed
  '''
  
  # load preprocessed_data, if not already
  if preprocessed_data==None:
      preprocessed_data = dev_utils.open_pickle(PREPROCESSED_PICKLE)

  # get the meta-data for all subjects
  meta_data_df  = dev_utils.get_meta_data()
  meta_template = {'Clinic': '', 'LVEF': np.nan, 'Strain': np.nan, 'Use_4_Train': 'F'}

  # load filenames from the preprocessed_data object
  if filenames==None:
    filenames = list(preprocessed_data.keys())

  # loop through the filenames, adding the preprocessing to the preprocessed_data object
  processed = False
  nom_categories = ['Baseline', 'Expiration+Whistle', 'Recovery']
  for file in filenames:
    df = preprocessed_data[file]['orig_df']
    for category in nom_categories:
      print(f'Processing {file} / {category}')
      
      dataframe = df.loc[df['Labeled_Category']==category, ['raw_data']]

      # ensure sufficient length of data; if not skip
      if len(dataframe) < 400:
        print(f'Insufficient length of data for {file} / {category} ({len(dataframe)} rows of data)')
        continue
      else:
        processed = True

      # perform the pre-processing & feature extraction
      dataframe = add_signals(dataframe)
      freq_features, psd = extractFreqFeat(np.gradient(dataframe['cycle'], edge_order=2))
      time_features, filt_cycle, cycle_dict = extractTimeFeat(dataframe['cycle'], freq_features['F0'])

      # add to the pre-processed data object
      preprocessed_data[file]['sig_data'][category]['sig_df']       = dataframe
      preprocessed_data[file]['sig_data'][category]['raw_features'] = dict(freq_features, **time_features)
      preprocessed_data[file]['sig_data'][category]['psd_df']       = pd.DataFrame(psd['Pxx_db'], index=psd['freq'], columns=['Pxx_db'])
      preprocessed_data[file]['sig_data'][category]['cycle_dict']   = cycle_dict

    # add meta-data
    subj_id        = file[:-6] # slice off the last 6 characters (e.g., ".1.csv")
    row            = meta_data_df[meta_data_df['Subject_ID']==subj_id]
    meta_data_dict = meta_template if len(row)==0 else meta_data_df[meta_data_df['Subject_ID']==subj_id].to_dict(orient='records')[0]
    preprocessed_data[file]['meta_data'] = meta_data_dict

  # writing that update to disk
  if processed:
    dev_utils.write_pickle(PREPROCESSED_PICKLE, preprocessed_data)
    print('Updated pickle with preprocessed data')

  return preprocessed_data



def flip_signal(series):
  '''
  Subtract each value from the median.  
  Why: to show the signal in the proper orientation, which has the steepest, positive slope occuring at the systolic rise.  
  This orientation represents the signal as the amount of blood present, rather than the original signal, which shows the converse
  
  Args:
    series: pandas series
  Returns:
    returns a pandas series that is "flipped" vertically about the median
  '''
  try:
    flipped = series.median() - series
  except:
    series  = np.array(series)
    flipped = np.median(series) - series
  return flipped



def fourierExtrapolation(x, n_predict, n_harm = 100):
    n_param = n_harm // 10
    n = x.size
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain

    h = np.sort(x_freqdom)[-n_param]
    x_freqdom = [ x_freqdom[i] if np.absolute(x_freqdom[i])>=h else 0 for i in range(len(x_freqdom)) ]    
    
    f = np.fft.fftfreq(n)              # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

    
def legacy_psd_calculation(windowed_data, verbose=False):
  '''
  Legacy method of calcuating the Power Spectral Density
  Args:
    windowed_data: should be a 1D numpy array of data to calculate power spectral density of
    verbose: whether to print debug info
  Returns:
    object with all the intermediary calculations
  '''
  nAutocorrelationWindow = 122

  def debug_print(message):
    if verbose:
      print(message)

  windowed_data = 65535 - windowed_data
  debug_print(f'Mean of windowed data {np.mean(windowed_data)}')

  segment  = windowed_data - np.mean(windowed_data)
  nSegment = len(segment)
  window   = np.blackman(nSegment+2)
  window   = window[1:-1]
  windowed = segment * window
  
  debug_print(f'mean after blackman = {np.mean(windowed)}')
  
  windowed = windowed - np.mean(windowed)

  debug_print(f'Sum = {np.sum(np.abs(windowed))}')
  
  xf = np.array(scipy.fft.fft(windowed, NFFT))
  xf_multiply_conj = xf * np.conj(xf)
  r1 = np.fft.ifft(xf_multiply_conj)
  r1_numpy = np.correlate(windowed, windowed, mode='full')[nSegment//2:]
  xf = xf[:len(xf)//2+1]
  
  debug_print('First 10 values of windowed data')
  debug_print(windowed_data[:10])
  debug_print(' ')
  debug_print('First 10 values of the FFT output = ')
  debug_print(xf[:10])
  debug_print('Last 10 values of the FFT output = ')
  debug_print(xf[-10:])
  debug_print(f'Shape of fft result = {xf.shape}')
  debug_print(' ')
  debug_print('First 10 values of IFFT output = ')
  debug_print(r1[:10] * NFFT)
  debug_print('Last 10 values of IFFT output = ')
  debug_print(r1[-10:] * NFFT)
  debug_print(f'Shape of ifft result = {r1.shape}')

  nr = min(nSegment, nAutocorrelationWindow)
  r2 = r1[:nr]

  r3 = np.concatenate([np.flip(r2), r2[1:]])
  debug_print(' ')
  debug_print('First 10 values of twoSidedBeforeBlackman = ')
  debug_print(r3[:10])
  debug_print('Last 10 values of twoSidedBeforeBlackman = ')
  debug_print(r3[-10:])
  debug_print(f'Shape of twoSidedBeforeBlackman = {r3.shape}')
  
  r4   = r3 * np.blackman(len(r3))
  psd1 = np.fft.fft(r4, nFftPsd)
  psd1 = psd1[:nFftPsd//2]
  psd2 = np.sqrt(np.square(np.real(psd1)) + np.square(np.imag(psd1)))

  #####################3  
  # TODO: this is duplicated in the calc_sfm function above - should reconcile those
  # From Bob: we should just use the calc_sfm function
  min_psd_freq=0.73
  max_psd_freq=18.02
  min_psd_index = int( 2*min_psd_freq*len(psd2)/FS )
  max_psd_index = int( 2*max_psd_freq*len(psd2)/FS )
  #####################3  
  
  psd3 = psd2[min_psd_index:max_psd_index]
  sfm  = stats.gmean(psd3) / np.mean(psd3)
  pulsatility = 1-sfm

  debug_print(' ')
  debug_print('~~~~~')
  debug_print(f'Pulsatility = {pulsatility}')
  debug_print('~~~~~')
  debug_print(' ')

  # output everything
  return {
    'windowed_data': windowed_data,
    'windowed': windowed,
    'xf': xf,
    'xf_multiply_conj': xf_multiply_conj,
    'r1': r1,
    'r1_numpy': r1_numpy,
    'r2': r2,
    'r3': r3,
    'r4': r4,
    'nr': nr,
    'psd1': psd1,
    'psd2': psd2,
    'psd3': psd3,
    'sfm': sfm,
    'pulsatility': pulsatility
  }






def windowed_processing(preprocessed_data, filename, data_type, expected_categories=['Baseline', 'Expiration+Whistle'], window_seconds=8):
  '''
  Take a given patient's filename and calculate the PSD values for rolling time periods.
  
  Args:
    preprocessed_data:
    filename:
    data_type: eg: 'mean_subtracted'
    expected_categories: categories needed to perform processing this function
    window_seconds: number of seconds per window
  Returns:
    The output will be an object containing a numpy array of PSD values (one for each rolling window) for each method.
  '''
  print("STARTING WINDOWED PROCESSING FOR FILE ", filename)
  
  # confirm the expected categories are present
  actual_categories = dev_utils.get_data(preprocessed_data, filename)['sig_data'].keys()
  print(actual_categories)
  if set(expected_categories) - set(actual_categories):
    return

  pulsatility_outputs   = {}
  rolling_window_length = int(window_seconds * FS)
  for category in expected_categories:
    data = dev_utils.get_data(preprocessed_data, filename, category, data_type=data_type).to_numpy()
    if (len(data) < rolling_window_length):
      print(f"LENGTH OF DATA for file {filename} and category {category} is {len(data)} which less than length of rolling window {rolling_window_length}")
      return

    legacy_pulsatilities = []
    welch_pulsatilities  = []
    windows              = rolling_window(data, rolling_window_length)

    # we will skip some for faster processing (probably a better way to do this with strides)
    max_segments = 20
    total_segments = len(windows)
    #print(f'total_segments = {total_segments}')
    skip_modulo = max(1,total_segments//max_segments)
    #print(f'skip_modulo = {skip_modulo}')

    for index, window_segment in enumerate(windows):
      if index % skip_modulo: continue
      #print(f'index = {index}')

      legacy_pulsatility = 1 - legacy_psd_calculation(window_segment)['sfm']
      legacy_pulsatilities.append(legacy_pulsatility)
      
      _, psd             = calc_psd(window_segment)
      welch_pulsatility  = 1 - calc_sfm(psd)
      welch_pulsatilities.append(welch_pulsatility)
      
    legacy_pulsatilities = np.array(legacy_pulsatilities)
    welch_pulsatilities  = np.array(welch_pulsatilities)

    pulsatility_outputs[category] = {
      'legacy' : legacy_pulsatilities,
      'welch'  : welch_pulsatilities
    } 
  return pulsatility_outputs









def add_cycles(filt_cycle, F0, do_plot=False):
  '''
  Adds peaks, valleys, associated cycles, and dicrotic notches.  Cycles defined as: valley_left, peak, valley_right
  Peaks on the left & right are added as well, if they exist

  Params:
    filt_cycle: detrended & lowpass filtered time series
    do_plot: whether to plot the cycles, peaks, and valleys
  Returns:
    dictionary of: indices of peaks, valleys, cycles, & dicrotic notches
  '''
  
  # ensure filt_cycle is of the proper data type
  if isinstance(filt_cycle, (pd.DataFrame, pd.Series)):
    filt_cycle = filt_cycle.values
  
  
  # add peaks & valleys
  peaks, valleys = peaks_and_valleys(filt_cycle, F0)

  # find the cycles
  #print( peaks.size,  valleys.size)
  if peaks.size > 1 and valleys.size > 1:
    cycles = []
    avg_period = np.mean( np.concatenate( (np.diff(peaks), np.diff(valleys)) ) )
    for val_idx in range(len(valleys)-1):
      rel_dist = (valleys[val_idx+1] - valleys[val_idx]) / avg_period
      if (rel_dist < 0.33) or (rel_dist > 2.00): 
        #print('valleys too far apart or too close together', (valleys[val_idx+1] - valleys[val_idx]) , avg_period, rel_dist)
        continue            # valleys too far apart or too close together
      peaks_btwn = peaks[ (peaks > valleys[val_idx]) & (peaks < valleys[val_idx+1])] 
      if len( peaks_btwn ) == 0: 
        continue      # no peaks between
      cycles.append([valleys[val_idx], peaks_btwn[0], np.nan, valleys[val_idx+1]])
  elif peaks.size == 1 and valleys.size == 0:
    cycles = [[np.nan, peaks[0], np.nan, np.nan]]
  elif valleys.size == 1 and peaks.size == 0:
    cycles = [[valleys[0], np.nan, np.nan, np.nan]]
  elif peaks.size == 1 and valleys.size == 1:
    if valleys[0] < peaks[0]:
      cycles = [[valleys[0], peaks[0], np.nan, np.nan]]
    else:
      cycles = [[np.nan, peaks[0], np.nan, valleys[0]]]
  else:
    cycles = np.empty( (0,4) ) #[[]]

  cycles = np.array(cycles)
  #print(cycles.shape)
  if cycles.shape[0] > 0:
    # add left & right peaks
    val_left    = cycles[0][0]
    left_peaks  = peaks[peaks < val_left]
    if len(left_peaks) > 0: cycles = np.concatenate( ([[np.nan, left_peaks[-1], np.nan, val_left]], cycles) )

    val_right   = cycles[-1][-1]
    right_peaks = peaks[peaks > val_right]
    if len(right_peaks) > 0: cycles = np.concatenate( (cycles, [[val_right, right_peaks[0], np.nan, np.nan]]) )

    if do_plot:
      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      ax2.grid(False)
      ax1.plot(filt_cycle, linewidth=5, color='w')
      ax1.scatter(peaks, filt_cycle[peaks], s=80, facecolors='none', edgecolors='r', marker='^')
      ax1.scatter(valleys, filt_cycle[valleys], s=80, facecolors='none', edgecolors='r', marker='v')
      counter = 0
      line_colors = ['r','b']
      for cyc in cycles:
        left  = cyc[0]  if np.isfinite(cyc[0])  else cyc[1]
        right = cyc[-1] if np.isfinite(cyc[-1]) else cyc[1]
        idx_range = np.arange(left, right+1, dtype=int)
        ax1.plot(idx_range, filt_cycle[idx_range], linewidth=1.5, alpha=1, color=line_colors[counter % 2])
        counter += 1

  cyc_dict = {'peaks': peaks, 'valleys': valleys, 'cycles': cycles}
  
  # add dicrotic notches
  cyc_dict = add_dicrotic_notches(filt_cycle, cyc_dict)
  
  return cyc_dict


def add_dicrotic_notches(filt_cycle, cyc_dict, do_plot=False):
 
  # ensure filt_cycle is of the proper data type
  if isinstance(filt_cycle, (pd.DataFrame, pd.Series)):
    filt_cycle = filt_cycle.values


  if do_plot==True:
    fig, ax1 = plt.subplots()
    ax1.plot( filt_cycle )
    ax2 = ax1.twinx()
    ax1.grid(False)
  
  dicr_notches = []
  cycles       = cyc_dict['cycles']
  if cycles.shape[0] > 0:
    for cyc in cycles:
      if np.isnan(cyc[-1]): continue # no right-valley (e.g., right end of file)

      line_x = np.arange(cyc[1], cyc[-1] + 1, dtype=int)
      if len(line_x) < 3: continue
      
      coef   = P.polyfit(line_x, filt_cycle[line_x], deg=1)
      line_y = P.polyval(line_x, coef)

      dx     = np.gradient( filt_cycle[line_x], edge_order=2)
      dx_2   = np.gradient( dx, edge_order=2)
      comp   = dx_2 - filt_cycle[line_x]

      peaks = signal.find_peaks(comp, height=-1e6, prominence=0, width=0)
      if len(peaks[0]) == 0:
        de_comp = signal.detrend(comp)
        peaks   = signal.find_peaks(de_comp, height=-1e6, prominence=0, width=0)
      
      if len(peaks[0]) == 0:
        continue
        
      peak_idx  = line_x[0] + peaks[0][0]
      cyc[2]    = peak_idx
      dicr_notches.append(peak_idx)

      if do_plot==True:
        ax2.plot(line_x,  comp, color='y', linewidth=1 ) #
        ax1.scatter(peak_idx, filt_cycle[peak_idx], s=80, facecolors='none', edgecolors='r', linewidths=2, marker='o')
        ax2.vlines(x=line_x[0] + peaks[0], ymin=comp[peaks[0]] - peaks[1]['prominences'], ymax = comp[peaks[0]], color='r')
        ax2.hlines(y=peaks[1]['width_heights'], xmin=line_x[0] + peaks[1]['left_ips'], xmax=line_x[0] + peaks[1]['right_ips'], color='r')

  cyc_dict['cycles']       = cycles
  cyc_dict['dicr_notches'] = dicr_notches

  return cyc_dict





def add_signals(dataframe):
  '''
  Adds the preprocessing signals to the raw_data
  Args:
    dataframe: the pandas dataframe that contains 'raw_data'
  Returns:
    dataframe with added columns: 'mean_subtracted', 'flipped_median', 'cycle', trend', 'filt_cycle'
  '''
  dataframe['flipped_median']            = dataframe['raw_data'].iloc[0:3].median() - dataframe['raw_data']
  dataframe['mean_subtracted']           = dataframe['flipped_median'] - dataframe['flipped_median'].mean()
  dataframe['cycle'], dataframe['trend'] = detrend_data(dataframe['flipped_median'].values)
  dataframe['filt_cycle']                = lowpass_filter(dataframe['cycle'])
  
  return dataframe
  
  


def calc_sfm(pxx, min_psd_freq=0.73, max_psd_freq=18.02):
  '''
  Calculates the Sprectral Flatness Measure
  Args:
    pxx: spectral density; one-sided (0 - Niquest frequency)
    min_psd_freq: scalar, representing the lower range of the SFM calculation in Hz; default is 0.73
    max_psd_freq: scalar, representing the upper range of the SFM calculation in Hz; default is 18.02
  Returns:
    sfm: scalar representing the spectral flatness
  '''
  min_psd_index = int( 2*min_psd_freq*len(pxx)/FS )
  max_psd_index = int( 2*max_psd_freq*len(pxx)/FS )
  subset        = pxx[min_psd_index:max_psd_index+1]    # this only uses a subset of the data; acting somewhat as a bandpass filter
  sfm           = stats.gmean(subset) / np.mean(subset)

  return sfm



def calc_syst_dur_norm(cyc_dict):
  '''
  Calculates the normalized (by the median period) systolic rise duration.  Represents the proportion
  of the period of the systolic rise (in the range of 0 to 1).
  Params:
    cyc_dict: dictionary containing peaks, valleys, and cycles
  Returns:
    Float: proportion of the period associated with the systolic rise
  '''
  cycles  = cyc_dict['cycles']
  peaks   = cyc_dict['peaks']
  valleys = cyc_dict['valleys']
  med_period = np.median( np.concatenate( (np.diff(peaks), np.diff(valleys)) ) )
  syst_dur_norm = np.nanmedian(cycles[:,1] - cycles[:,0]) / med_period
  return syst_dur_norm



def central_slope(x, order=2):
  grad = x
  for _ in range(order):
    grad = np.gradient(grad, edge_order=2)

  # TODO: understand why it seems necessary to offset the centered gradient
  sign = 1 if order % 2 == 0 else -1
  return sign * np.concatenate( (grad[order:], np.full(order, 0)) )


def change_slope(filt_cycle, stride_len=3):
  '''
  strided, central change in slope
  Args:
    filt_cycle: signal on which to perform the calculations
    stride_len: (int) default=3; the length forward and backward to look for calculation of the change in slope
  Returns:
    change in slope array of the same length as the input 
  '''
  grad     = np.gradient(filt_cycle, edge_order=2)
  ch_slope = grad[2*stride_len:] - grad[:-2*stride_len]
  ch_slope = np.concatenate( (np.full(stride_len, 0), ch_slope, np.full(stride_len, 0)) ) #np.full(stride_len, 0), 
  return ch_slope


def clip_signal(dataframe, clip_rows=6):
  '''
  Removes clip_rows rows from the top of the dataframe
  Args:
    dataframe: pandas dataframe
    clip_rows: number of rows to remove from the top of the dataframe
  Returns:
    dataframe with clip_rows removed from the start
  '''
  return dataframe.iloc[clip_rows:].copy()


def clip_signal_file(dataframe, clip_rows=6):
  '''
  Removes clip_rows rows from the top of the dataframe + any rows associated with the discontinuities in the first file format
  Args:
    dataframe: pandas dataframe
    clip_rows: number of rows to remove from the top of the dataframe
  Returns:
    dataframe with clip_rows removed from the start & the Baseline/VM & VM/Recovery interfaces, if there were discontinuities
  '''
  DISCONTINUITY_GAP = 1000
  drop_rows         = list(range(clip_rows))

  # clip the start of expiration & recovery of the first file format
  trans_crit = ( (dataframe['Category'].shift()=='Baseline') & (dataframe['Category'].isin(['Expiration', 'Whistle'])) ) | ( (dataframe['Category'].shift()!='Recovery') & (dataframe['Category']=='Recovery') )
  diff_crit  = abs(dataframe['raw_data'].shift() - dataframe['raw_data']) > DISCONTINUITY_GAP
  spike_rows = dataframe.loc[trans_crit & diff_crit]

  for index, _ in spike_rows.iterrows():
    row_num    = np.where(dataframe.index==index)[0][0]
    drop_rows += list( range(row_num, row_num+clip_rows) )


  dataframe = dataframe.drop(dataframe.index[drop_rows]).copy() # copy, otherwise a view would be returned
  return dataframe



def cycle_envelope_features(orig_df, F0=None, bad_segs=[], do_plot=False):
  cyc_env_feats = {}
  filt_cycle    = orig_df['filt_cycle'].values

  # TODO: handle bad segments
  # handle bad segments
  #for seg in bad_segs:
  #  filt_cycle[seg[0]:seg[1]+1] = np.nan

  # get F0 if not provided
  if F0==None:
    freq_feats, _ = extractFreqFeat(np.gradient(filt_cycle, edge_order=2))
    F0 = freq_feats['F0']

  # get high & low points (roughly corresponding to peaks & valleys)
  peaks_arr   = []
  valleys_arr = []
  win_num     = 0
  WIN_SIZE    = int(1.5*FS/F0)
  STEP_SIZE   = int(WIN_SIZE/2)
  filt_cyc_windows = rolling_windows(filt_cycle, win_size=WIN_SIZE, step_size=STEP_SIZE)
  for filt_cycle_win in filt_cyc_windows:
    peaks = signal.find_peaks(filt_cycle_win, prominence=0)
    if len(peaks[0]) > 0: 
      peaks_arr.append(peaks[0][np.argmax(peaks[1]['prominences'])]+win_num)
    valleys = signal.find_peaks(-filt_cycle_win, prominence=0)
    if len(valleys[0]) > 0: 
      valleys_arr.append(valleys[0][np.argmax(valleys[1]['prominences'])]+win_num)
    win_num += STEP_SIZE

  # prepare the peaks & valleys, X & Y
  valleys    = np.unique(valleys_arr)
  peaks      = np.unique(peaks_arr)

  # add points on the ends to ensure the envelope's slope is near zero at ends (mitigates exploding slopes)
  Y_peaks    = np.concatenate( ([filt_cycle[peaks[0]]], filt_cycle[peaks], [filt_cycle[peaks[-1]]]) )
  X_peaks    = np.concatenate( ([peaks[0] - FS], peaks, [peaks[-1] + FS]) )
  Y_valleys  = np.concatenate( ([filt_cycle[valleys[0]]], filt_cycle[valleys], [filt_cycle[valleys[-1]]]) )
  X_valleys  = np.concatenate( ([valleys[0] - FS], valleys, [valleys[-1] + FS]) )

  # fit cubic splines to upper & lower points
  x          = np.arange(len(filt_cycle))
  env_up     = CubicSpline(X_peaks, Y_peaks, bc_type='clamped')(x) #'natural'
  env_lo     = CubicSpline(X_valleys, Y_valleys, bc_type='clamped')(x) # 'natural'
  env_height = env_up - env_lo

  # feature statistics
  whistle_df    = orig_df[orig_df['Category']=='Whistle']
  if len(whistle_df)>0:
    first_whistle = orig_df.index.get_loc(whistle_df.index[0])
    last_whistle  = orig_df.index.get_loc(whistle_df.index[-1])
    cyc_env_feats.update( feat_stats(feat_name='BL_env_height', feat_array=env_height[:first_whistle], time_array=orig_df.iloc[:first_whistle].index) )
    cyc_env_feats.update( feat_stats(feat_name='Whistle_env_height', feat_array=env_height[first_whistle:last_whistle+1], time_array=orig_df.iloc[first_whistle:last_whistle+1].index) )
    cyc_env_feats.update( feat_stats(feat_name='REC_env_height', feat_array=env_height[last_whistle+1:], time_array=orig_df.iloc[last_whistle+1:].index) )
  

  if do_plot==True:
    plt.plot(orig_df['filt_cycle'], label='cycle')
    plt.plot(orig_df['filt_cycle'].index, env_up, color='y', label='upper_env')
    plt.plot(orig_df['filt_cycle'].index, env_lo, color='y', label='lower_env')
    plt.scatter(orig_df['filt_cycle'].iloc[peaks].index, orig_df['filt_cycle'].iloc[peaks], color='r')
    plt.scatter(orig_df['filt_cycle'].iloc[valleys].index, orig_df['filt_cycle'].iloc[valleys], color='r')
    plt.plot(orig_df['filt_cycle'].index, env_up - env_lo, color='r', label='env_height')
    plt.legend()

  return cyc_env_feats#, env_up, env_lo
  

def cyclic_min_max(sig):
  cyc_max   = []
  cyc_min   = []
  start_idx = 0
  period    = get_period_3(sig)
  sig       = detrend(sig)
  while True:
    end_idx = start_idx + period
    if end_idx >= len(sig)-1: break
    win = sig[start_idx:end_idx]
    cyc_max.append(win.max())
    cyc_min.append(win.min())
    start_idx += period
  return np.quantile(cyc_min, 0.75), np.quantile(cyc_max, 0.25)




def GenerateRandomCurve(X, sigma=0.1, knot=4):
  xx = np.arange(0, len(X), (len(X)-1)/(knot+1) )
  yy = np.random.randint(5,21, size=len(xx))/10 #np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
  x  = np.arange(len(X))
  cs = CubicSpline(xx, yy)(x)
  return cs



def scale_rot_sig(sig):
  sig_max  = max(sig)
  sig_min  = min(sig)
  sig_mean = np.mean(sig)
  rand_scale = GenerateRandomCurve(sig)
  y_rot    = 1*np.random.uniform(low=sig_min-sig_max, high=sig_max-sig_min)
  line     = np.linspace(0, y_rot, num=len(sig))
  sig_mod  = (sig-sig_mean)*rand_scale - line + y_rot/2 + sig_mean
  return sig_mod


def get_initial_points(dataframe, channel):

  # get initial peaks or valleys of the start of Baseline
  NUM_CYCLES   = 5
  end_bl_idx   = get_whistle(dataframe)[1]
  period_bl    = get_period_3(channel[:end_bl_idx])
  window       = channel[:NUM_CYCLES*period_bl]
  cyc_min,cyc_max = cyclic_min_max(window)
  ids,_        = signal.find_peaks(window, distance=period_bl//2, prominence=(cyc_max-cyc_min)/2)
  # TODO: need to handle case where none (or less than expected) are found, which would be caused by a poor-quality signal
  
  return ids



def get_next_distance(ids):
  
  # estimate where the next min will occur based on the prior cycles
  NUM_CYCLES = 5
  diffs      = np.diff(ids)[-NUM_CYCLES:]
  diff_med   = np.median(diffs)
  x          = np.arange(len(diffs))
  p          = np.polyfit(x, diffs, deg=1)
  next_dist  = np.polyval(p, x[-1]+1)
  if abs(1 - next_dist/diff_med) > 0.2: # use the median if the linear forecast is more than 20% different from the median
    next_dist = diff_med
  next_dist  = int( min( round(FS/MIN_F0_HZ), max( next_dist, round(FS/MAX_F0_HZ) ) )) # bound the next_dist by the phyisiologic pulse rate

  return next_dist
  



def local_detrend(window):
  line = np.linspace(window[0],window[-1],num=len(window))
  init = window - line
  min_ = np.argmin(init[1:-1]) + 1
  line = np.linspace(window[0],window[min_],num=len(window))
  return window - line  



def extractFreqFeat(cyclical):
  '''
  This function calculates & returns the frequency domain features of interest, including: matSnr3, harmInt, P0 & F0.
  The standardization (zscore) and log (Ln) functions should be performed outside this function.
  Args:
    cyclical: the signal (cyclical) should be detrended (DC removed) before it is passed into this function
  Returns:
    feature dictionary
    psd: pxx_db, freq
  '''
  NEIGHBOR_SPAN = 4
  NUM_HARM      = 4

  retObj = {
        "F0": None,
        "F0Ln": None,
        "harmSlope": None,
        "harmSlopeLn": None,
        "harmInt": None,
        "harmIntLn": None,
        "P0": None,
        "P0Ln": None,
        "SNR1": None,
        "SNR1Ln": None,
        "SNR2": None,
        "SNR2Ln": None,
        "SNR3": None,
        "SNR3Ln": None,
        "SNR4": None,
        "SNR4Ln": None,
        "SNR5": None,
        "SNR5Ln": None,
        "SNR20_25": None,
        "SNR20_25Ln": None
    }
  
  freq, psd_welch = calc_psd(cyclical)
  Pxx_db = 10*np.log10(psd_welch)
  psd    = {
      "Pxx_db": Pxx_db,
      "freq": freq
  }
  if not psd:
    return retObj

  dF = psd["freq"][1] - psd["freq"][0]; 
  f0LowIdx = round(MIN_F0_HZ / dF)
  f0HighIdx = round(MAX_F0_HZ / dF)
  if len(Pxx_db) < round(3.5*f0HighIdx):
    return retObj

# 	//
# 	// Find the Fundamental Frequency (F0) & Power (P0)
# 	//
# 	// Calculate the harmonic prominance, using the first three spectral peaks (staying away from the 10Hz beat artifact)
  maxIdx = -1
  maxProm = -999
  for j in range(f0LowIdx, f0HighIdx): 
      half = round(j/2)
      prom = 	 (2*Pxx_db[1*j] - Pxx_db[1*j - half] - Pxx_db[1*j + half] +
                  2*Pxx_db[2*j] - Pxx_db[2*j - half] - Pxx_db[2*j + half] +
                  2*Pxx_db[3*j] - Pxx_db[3*j - half] - Pxx_db[3*j + half])
      if prom > maxProm: 
          maxProm = prom
          maxIdx = j
  
  retObj["maxProm"] = maxProm
  retObj["maxPromLn"] = log_e(maxProm)
  F0Idx = maxIdx + np.argmax(Pxx_db[maxIdx - NEIGHBOR_SPAN: maxIdx + NEIGHBOR_SPAN + 1]) - NEIGHBOR_SPAN

  peaksIdx = [F0Idx]
  for j in range(1, NUM_HARM + 1): 
      baseIdx = peaksIdx[len(peaksIdx)-1] + F0Idx
      relIdx = np.argmax(Pxx_db[baseIdx - NEIGHBOR_SPAN: baseIdx + NEIGHBOR_SPAN + 1])
      peaksIdx.append(baseIdx + relIdx - NEIGHBOR_SPAN)
  
# 	// Find Valleys between the peaks, and the ones on the outside of the first & last peak
  valleysIdx = []
  medianDist = round(np.median(np.diff(peaksIdx)))
  minIdx = np.max(peaksIdx[0] - round(medianDist/3), 0)
  valleysIdx.append(np.argmin(Pxx_db[minIdx: peaksIdx[0]+1]) + minIdx)
  for j in range(0, len(peaksIdx)-1): 
      valleysIdx.append(np.argmin(Pxx_db[peaksIdx[j]: peaksIdx[j+1]]) + peaksIdx[j])
  
  valleysIdx.append(np.argmin(Pxx_db[peaksIdx[j]: peaksIdx[j] + int(medianDist / 2)]) + peaksIdx[j])

  peaks_P = psd["Pxx_db"][np.array(peaksIdx)]  # map(lambda x: psd["Pxx_db"][x], peaksIdx) 
  peaks_F = psd["freq"][np.array(peaksIdx)]  # map(lambda x: psd["freq"][x], peaksIdx)

    # var regress = new ML.Regression.SimpleLinearRegression(peaks_F, peaks_P)
  (slope, intercept) = np.polyfit(peaks_F, peaks_P, 1)

  retObj["F0"] = peaks_F[0]
  retObj["F0Ln"] = log_e(peaks_F[0])
  retObj["harmSlope"] = slope
  retObj["harmSlopeLn"] = log_e(-slope)
  retObj["harmInt"] = intercept
  retObj["harmIntLn"] = log_e(intercept)
  retObj["P0"] = peaks_P[0]
  retObj["P0Ln"] = log_e(peaks_P[0])

  valleys_P = list(psd["Pxx_db"][np.array(valleysIdx)])  # map(lambda x: Pxx_db[x], valleysIdx)

  startIdx = int(21 / dF)
  endIdx = int(25 / dF)
  valleys_P.append(np.median(psd["Pxx_db"][startIdx: endIdx+1]))
  
  
  retObj["SNR1"] = peaks_P[0] - ( valleys_P[0] + valleys_P[1] ) / 2
  retObj["SNR1Ln"] = log_e(retObj["SNR1"])
  retObj["SNR2"] = peaks_P[1] - ( valleys_P[1] + valleys_P[2] ) / 2
  retObj["SNR2Ln"] = log_e(retObj["SNR2"])
  retObj["SNR3"] = peaks_P[2] - ( valleys_P[2] + valleys_P[3] ) / 2
  retObj["SNR3Ln"] = log_e(retObj["SNR3"])
  retObj["SNR4"] = peaks_P[3] - ( valleys_P[3] + valleys_P[4] ) / 2
  retObj["SNR4Ln"] = log_e(retObj["SNR4"])
  retObj["SNR5"] = peaks_P[4] - ( valleys_P[4] + valleys_P[5] ) / 2
  retObj["SNR5Ln"] = log_e(retObj["SNR5"], trans_value=10)
  retObj["SNR20_25"] = peaks_P[0] - valleys_P[6]
  retObj["SNR20_25Ln"] = log_e(retObj["SNR20_25"])

  return retObj, psd


def extractTimeFeat(cyclical, F0):
  '''
  This function calculates & returns the time domain features.
  Args:
    cyclical: the signal (cyclical) should be detrended (DC removed) before it is passed into this function
    F0: fundamental frequency (from extractFreqFeatures)
  Returns:
    feature dictionary
    filt_cycle: lowpass filted cycle
    cyc_dict: cycle dictionary, which includes the cycle, peaks, valleys, and dicrotic notches
  '''

  #cyclical = np.array(cyclical).astype(np.float64)
  #HI_FREQ = 9.0 # RGM (21MAY21) does not appear to be used
  retObj = {
    "serCorr": None,
    "serCorrLn": None,
    "kurtosis": None,
    "kurtosisLn": None,
    "cyclicRange2": None,
    "cyclicRange2Ln": None,
    "cyclicRange2": None,
    "cyclicRange2Ln": None
  }

  retObj["serCorr"] = serialCorr(cyclical.values, F0)
  retObj["serCorrLn"] = log_e(retObj["serCorr"], trans_value=1)
  retObj["kurtosis"] = kurtosis(cyclical.values)
  retObj["kurtosisLn"] = log_e(retObj["kurtosis"], trans_value=2)
  retObj["cyclicRange2"] = np.quantile(cyclical.values, .9) - np.quantile(cyclical.values, .1)
  retObj["cyclicRange2Ln"] = log_e(retObj["cyclicRange2"])

  # lowpass filter signal & create cycles: peaks, valleys, & dicrotic notches
  filt_cycle = pd.Series(lowpass_filter(cyclical), index=cyclical.index)

  cyc_dict   = add_cycles(filt_cycle, F0)
  #print(f'cycle_dict: {cyc_dict}')

  val_1_idx  = cyc_dict['cycles'][:,0]
  peaks_idx  = cyc_dict['cycles'][:,1]
  dicr_idx   = cyc_dict['cycles'][:,2]
  val_2_idx  = cyc_dict['cycles'][:,3]
  
  # systolic duration (x-dimension)
  feat_name  = 'syst_dur'
  feat_array = (peaks_idx - val_1_idx) / (val_2_idx - val_1_idx)
  time_array = peaks_idx/FS
  retObj.update(feat_stats(feat_name, feat_array, time_array))
  
  # systolic rise (y-dimension)
  feat_name  = 'syst_rise'
  peak_values, val_1_values, time_array = finite_values(filt_cycle, peaks_idx, val_1_idx)
  feat_array = peak_values - val_1_values
  retObj.update(feat_stats(feat_name, feat_array, time_array))
  
  # dicrotic duration (x-dimension)
  feat_name  = 'dicr_dur'
  feat_array = (dicr_idx - val_1_idx) / (val_2_idx - val_1_idx)
  time_array = dicr_idx/FS
  retObj.update(feat_stats(feat_name, feat_array, time_array))
  
  # dicrotic depth (y-dimension)
  feat_name  = 'dicr_depth'
  peak_values, dicr_values, val_1_values, time_array = finite_values(filt_cycle, peaks_idx, dicr_idx, val_1_idx)
  feat_array = (peak_values - dicr_values) / (peak_values - val_1_values)
  retObj.update(feat_stats(feat_name, feat_array, time_array))
  
  # diastolic complexity
  mae        = []
  time_array = []
  for peak, valley in zip(peaks_idx,val_2_idx):
    if np.isnan(peak) or np.isnan(valley): continue
    slice_x = np.arange(peak, valley+1, dtype=int)
    if len(slice_x) <= 3: continue
    slice_y = min_max_scale(filt_cycle.values[slice_x]) 
    coef, stats = P.polyfit(slice_x, slice_y, deg=2, full=True )
    mae.append( (stats[0][0]/len(slice_y))**0.5 )
    time_array.append(filt_cycle.index[int(valley)])
  feat_name  = 'diast_complex'
  feat_array = np.array(mae)
  time_array = np.array(time_array)/FS
  retObj.update(feat_stats(feat_name, feat_array, time_array))
  
  # heart rate (x-dimension)
  feat_name  = 'hr_hz'
  feat_array = (val_2_idx - val_1_idx)/FS
  time_array = val_2_idx/FS
  retObj.update(feat_stats(feat_name, feat_array, time_array))


  # systolic / diastolic ratio
  feat_name  = 'syst_diast_ratio'
  ratios     = []
  time_array = []
  for val_1,dicr,val_2 in zip(val_1_idx,dicr_idx,val_2_idx):
    if np.isnan(val_1) or np.isnan(dicr) or np.isnan(val_2): continue
    detr_y  = np.linspace(filt_cycle.iloc[int(val_1)], filt_cycle.iloc[int(val_2)], int(val_2-val_1)+1)
    cycl_y  = filt_cycle.iloc[int(val_1):int(val_2)+1] - detr_y
    sys_len = int(dicr - val_1)
    syst_y  = cycl_y.iloc[:sys_len].sum()
    dias_y  = cycl_y.iloc[sys_len:].sum()
    ratios.append(syst_y/dias_y)
    time_array.append(filt_cycle.index[int(val_2)])
  feat_array = np.array(ratios)
  time_array = np.array(time_array)/FS
  retObj.update(feat_stats(feat_name, feat_array, time_array))

  return retObj, filt_cycle, cyc_dict




def log_e(value, trans_value=0):
  '''This function provides valid results of the log calculation, inclduing negative valued features,
  by translating with the provided `trans_value` parameter. Features that are known negative (i.e., 
  'harmSlope'), the sign should be reversed before passing to this function.
  If the value provided is negative or zero, even after translating, a constant value is returned (e.g. -999)
  
  The following features are known to range about zero and listed with their corresponding trans_values.
    'serCorr': 1, 'kurtosis': 2, 'SNR5': 10
  
  The feature such as those listed below are always negative, so their sign should be simply reversed before transforming:
    'harmSlope'

  Args:
    value: value to transform
    trans_value: the amount to translate by
  Returns:
    scalar: transformed value
  '''
  value = value + trans_value
  if value <= 0:
    ret_val = -999.0  
  else:
    ret_val = np.log(value)
  
  return ret_val



def periodicity(window):
  if len(window) < 200: return np.nan

  WIN_SIZE  = 100
  right_win = window[-WIN_SIZE:]
  left_win  = window[-2*WIN_SIZE:-WIN_SIZE]

  # scale about the absolute value of the left_win (i.e., penalize of the right_win is too large or too small)
  scale     = (right_win.max()-right_win.min()) / (left_win.max()-left_win.min())
  if scale > 1:
    scale = 1/scale

  # scale 0 to 1
  left_win = (left_win - left_win.min()) / (left_win.max()-left_win.min())
  right_win  = scale*(right_win  - right_win.min()) / (right_win.max()-right_win.min())

  return np.max(np.correlate(left_win, right_win/WIN_SIZE, mode='same'))
  
  

def rolling_kurtosis(slope_change, period):
  upper   = slope_change.copy()
  lower   = slope_change.copy()
  upper[upper<0] = 0
  lower[lower>0] = 0
  up_kurt = []
  lo_kurt = []
  for start_idx in range(len(slope_change)-period+1):
    end_idx = start_idx + period
    up_kurt.append( kurtosis(upper[start_idx:end_idx]) )
    lo_kurt.append( kurtosis(lower[start_idx:end_idx]) )
  return np.quantile(up_kurt, 0.90), np.quantile(lo_kurt, 0.90)
  

def rolling_max_min(a, win_size, step_size):
  nrows = (a.size - win_size)//step_size + 1
  n = a.strides[0]
  s = as_strided(a, shape=(nrows, win_size), strides=(step_size*n, n))
  return s.ptp(1)

def rolling_range(x):
  win_size   = int(FS/MIN_F0_HZ)
  step_size  = int(FS/MIN_F0_HZ)
  return rolling_max_min(x, win_size, step_size)

def rolling_windows(a, win_size, step_size):
  nrows   = (a.size - win_size)//step_size + 1
  n       = a.strides[0]
  windows = as_strided(a, shape=(nrows, win_size), strides=(step_size*n, n))
  return windows

def rolling_window(values, rolling_window_length):
  '''
  helper for breaking up a long segment into evenly-spaced windows
  Args:
    values: np array of segment to be broken into windows
    rolling_window_length: length of each window
  Returns:
    a new rolling window segment as each new row, with window_length columns
  '''
  shape = values.shape[:-1] + (values.shape[-1] - rolling_window_length + 1, rolling_window_length)
  strides = values.strides + (values.strides[-1],)
  return np.lib.stride_tricks.as_strided(values, shape=shape, strides=strides)



def serialCorr(sig, F0):
  ONE_SEC     = 50
  corrVector  = []
  MIN_POINTS  = 150

  if not F0: 
    minLag = np.floor(FS / MAX_F0_HZ)
    maxLag = np.ceil(FS / MIN_F0_HZ)
    for lag in range(minLag, maxLag): 
      seg1 = sig[0:sig.length - lag]
      seg2 = sig[lag:]
      corrVector.append(scipy.stats.pearsonr(seg1, seg2)[0])
    
    F0 = FS / (minLag + np.argmax(corrVector) + 2); 
    retCorrVector = True
    SEG_LEN = ONE_SEC
  else:
    retCorrVector = False
    SEG_LEN = np.ceil(FS / MIN_F0_HZ)
  
  MAX_SHIFT = 4
  NOM_MOVE = np.round(FS / F0)
  startIdx = 0
  endIdx = startIdx + SEG_LEN

  if NOM_MOVE > MIN_POINTS - SEG_LEN - MAX_SHIFT: 
    NOM_MOVE = MIN_POINTS - SEG_LEN - MAX_SHIFT
    
  NOM_MOVE = int(NOM_MOVE)

  corrVector = []; 

  while startIdx + NOM_MOVE + SEG_LEN + MAX_SHIFT <= len(sig): 
    maxCorr = -2
    seg1 = sig[startIdx: endIdx + 1]

    seg1 = np.array(seg1).astype(np.float)

    shift = -MAX_SHIFT
    while (shift <= MAX_SHIFT):
      start2 = int(startIdx + NOM_MOVE + shift)
      end2 = int(start2 + SEG_LEN)
      seg2 = sig[start2: end2 + 1]
      seg2 = np.array(seg2).astype(np.float)


      if (len(seg2) > len(seg1)):
        seg2 = seg2[:len(seg1)]
        
      if (len(seg1) > len(seg2)):
        seg1 = seg1[:len(seg2)]
    
      R = scipy.stats.pearsonr(seg1, seg2)[0]
      if R > maxCorr: 
        maxCorr = R
        shift += 1
      else:
        break
      
    corrVector.append(maxCorr)
    if retCorrVector:
      startIdx += ONE_SEC
    else:
      startIdx += NOM_MOVE
    endIdx = startIdx + SEG_LEN
  

  if len(corrVector) == 0: 
    return None
  elif retCorrVector: 
    return corrVector
  else:
    return np.median(corrVector)


def subtract_mean(series):
  '''Subtract mean of series from each point
  
  Args:
    series: list like object
  Returns:
    returns a series with a mean of 0
  '''
  try:
    mean_sub = series - series.mean()
  except:
    series   = np.array(series)
    mean_sub = series - np.mean(series)
  return mean_sub


def standardize(x):
  if len(x)==1:
    std_ = np.array([0]) # (x - np.mean(x)) / 0.62*np.mean(x) # empirically-determined
  else:
    std_ = (x - np.mean(x)) / np.std(x)
  return std_



def trend_features(orig_df, BL_env_height, do_plot=False):
  trend_feats   = {}
  trend         = orig_df['trend'].values
  cycle         = orig_df['cycle'].values
  filt_cycle    = orig_df['filt_cycle'].values
  trend_mma     = mid_mov_avg(trend, win_len=2*FS)  # win_len = 2-seconds
  trend_mma_dx  = np.gradient(trend_mma, edge_order=2)


  # max slope (max change over 2-sec period)
  trend_feats['trend_max_pos_slope'] = np.nanmax(trend_mma_dx)*FS
  trend_feats['trend_max_neg_slope'] = -np.nanmax(-trend_mma_dx)*FS
  trend_feats['trend_max_abs_slope'] = np.nanmax(abs(trend_mma_dx))*FS

  # max change in slope (over a 4-sec period)
  STRIDE_LEN  = 2*FS # 2-seconds
  trend_mma_dx2 = np.concatenate(( np.full(STRIDE_LEN, np.nan), trend_mma_dx[2*STRIDE_LEN:] - trend_mma_dx[:-2*STRIDE_LEN], np.full(STRIDE_LEN, np.nan) ))
  trend_feats['trend_max_pos_dx2'] = np.nanmax(trend_mma_dx2)*FS
  trend_feats['trend_max_neg_dx2'] = -np.nanmax(-trend_mma_dx2)*FS
  trend_feats['trend_max_abs_dx2'] = np.nanmax(abs(trend_mma_dx2))*FS

  # trend scale relative to BL cycle
  trend_feats['trend_rel_scale'] = (trend.max() - trend.min()) / BL_env_height

  # trend shape features
  trend_mma_dx_ = trend_mma_dx[~np.isnan(trend_mma_dx)].copy()
  trend_feats['trend_kurtosis'] = kurtosis(trend_mma_dx_)
  trend_feats['trend_mean']     = (trend_mma_dx_ - trend_mma_dx_.min()).sum() / len(trend_mma_dx_)
  trend_feats['trend_range']    = trend_mma_dx_.max() - trend_mma_dx_.min()

  # find whistle
  whistle_df    = orig_df[orig_df['Category']=='Whistle']
  if len(whistle_df)>0:
    first_whistle = orig_df.index.get_loc(whistle_df.index[0])
    last_whistle  = orig_df.index.get_loc(whistle_df.index[-1])

    # time from end of Whistle to the trend's maximal negative slope
    max_neg_slope_time = orig_df.iloc[np.nanargmax(-trend_mma_dx)].name
    last_whistle_time  = orig_df.iloc[last_whistle].name
    trend_feats['trend_refill_time'] = max_neg_slope_time - last_whistle_time

    # max change from BL to Whistle
    trend_mean_BL = orig_df['trend'].iloc[:first_whistle].mean()
    trend_max_Wh  = orig_df['trend'].iloc[first_whistle:last_whistle+1].max()
    trend_min_Wh  = orig_df['trend'].iloc[first_whistle:last_whistle+1].min()
    trend_feats['trend_Whistle_max_vs_BL'] = trend_max_Wh - trend_mean_BL
    trend_feats['trend_Whistle_min_vs_BL'] = trend_mean_BL - trend_min_Wh

    # BL vs REC
    FIVE_SEC       = FS*5
    if (first_whistle-FIVE_SEC >= 0) and (last_whistle+FIVE_SEC < len(orig_df)):
      trend_BL_end   = orig_df['trend'].iloc[first_whistle-FIVE_SEC:first_whistle].mean()
      trend_REC_5SEC = orig_df['trend'].iloc[last_whistle+FIVE_SEC]
      trend_feats['trend_BL_vs_REC'] = trend_BL_end - trend_REC_5SEC
    else:
      trend_feats['trend_BL_vs_REC'] = np.nan


    # trend slope at end of whistle
    trend_feats['trend_slope_end_Whistle'] = trend_mma_dx[last_whistle]*FS


  else: # no whistle
    trend_feats['trend_refill_time']       = np.nan
    trend_feats['trend_Whistle_max_vs_BL'] = np.nan
    trend_feats['trend_Whistle_min_vs_BL'] = np.nan
    trend_feats['trend_slope_end_Whistle'] = np.nan
    trend_feats['trend_BL_vs_REC']         = np.nan





  if do_plot==True:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax1.plot(orig_df['cycle'], label='cycle')
    ax1.plot(orig_df['trend'], color='w', linewidth=1, label='trend')
    ax1.plot(orig_df.index, trend_mma, color='r', label='trend_mma')
    ax2.plot(orig_df.index, trend_mma_dx, color='g', label='trend_mma_dx')
    ax2.plot(orig_df.index, trend_mma_dx2, color='c', label='trend_mma_dx2')
    ax2.plot(orig_df['Category']=='Whistle', color='y', label='whistle')
    plt.legend()

  return trend_feats


def trend_range(window):
  if len(window) < 100: return np.nan

  WIN_SIZE  = 100
  right_win = window[-WIN_SIZE:]
  return right_win.max() - right_win.min()



def min_max_scale(x):
  min_ = x.min()
  max_ = x.max()
  return (x - min_) / (max_ - min_)
  
  
def finite_ind(*ind_arrays):
  v = np.vstack( (ind_arrays) )
  a = np.isnan( v ) 
  finite_indices = np.all(a == a[0,:], axis = 0) 
  return v[:,finite_indices].astype(int)

def finite_values(source, *ind_arrays):
  if isinstance(source, (pd.DataFrame, pd.Series)):
    time   = source.index.values
    source = source.values
  else:
    time   = np.arange(len(source))/FS

  finite_indices = finite_ind(*ind_arrays) 
  return tuple( (*source[finite_indices]  , time[finite_indices[0]]) )
  
def feat_stats(feat_name, feat_array, time_array, ret_stats=['min', 'median', 'range', 'roc']):
  ret_dict = {}
  def_dict = {feat_name+'_min':np.nan, feat_name+'_med':np.nan, feat_name+'_range':np.nan, feat_name+'_roc':np.nan}
  ROC_TIME = 5.00 # seconds
  
  # remove NaN's
  time_array = time_array[~np.isnan(feat_array)]
  feat_array = feat_array[~np.isnan(feat_array)]
  if len(feat_array)==0:
    return def_dict
    
  #print('feat_array', feat_array)
  #print('time_array', time_array)
  time_th  = time_array[0] + ROC_TIME

  # remove any nan's or the int casting of nan
  cond       = ~np.isnan(feat_array)
  feat_array = feat_array[cond]
  time_array = time_array[cond]

  if 'min' in ret_stats:
    ret_dict[feat_name + '_min']   = np.min(feat_array)
  if 'median' in ret_stats:
    ret_dict[feat_name + '_med']   = np.median(feat_array)
  if 'range' in ret_stats:
    ret_dict[feat_name + '_range'] = (np.quantile(feat_array, 0.9) - np.quantile(feat_array, 0.1))
  if 'roc' in ret_stats:
    if len(time_array[time_array <= time_th]) > 2:
        ret_dict[feat_name + '_roc'] = linregress(time_array[time_array <= time_th], feat_array[time_array <= time_th]).slope
    else:
        ret_dict[feat_name + '_roc'] = np.nan

  return ret_dict
  
  
  
def get_period_2(sig):
  NFFT       = 1024
  freq, psd  = signal.welch(np.diff(sig), fs=FS, nperseg=min(NFFT, len(sig)//4), nfft=NFFT, scaling='density', average='median', noverlap=None, window='boxcar', detrend='linear')
  MIN_F0_IDX = int(2*len(psd)*MIN_F0_HZ/FS)
  MAX_F0_IDX = int(2*len(psd)*MAX_F0_HZ/FS)
  bl_f0      = freq[MIN_F0_IDX] + freq[np.argmax(psd[MIN_F0_IDX:MAX_F0_IDX])]
  period     = int(FS/bl_f0)
  return period


def get_period(x):
  '''
  Time-series based way to get the period (heart rate, F0) using cross-correlation
  Args:
    x: cyclic data to analyze
  Returns:
    period in number of samples
  '''
  x_corr     = np.correlate( x, x, mode='same' )
  peaks      = signal.find_peaks(x_corr)
  num_cross  = ((x_corr[:-1] * x_corr[1:]) < 0).sum()
  if len(peaks[0]) < 2 or num_cross==0:
    # handle case where there is only a single peak
    period = len(x)
  else:
    expected_p = int(2*len(x_corr) / num_cross)
    period = max( int(np.median(np.diff(peaks[0]))), expected_p )
        

    # RGM: commenting out, as it's very likely that short periods are possible for the EMD detrending
    # handle case where the period is shorter than the expected maximum heart rate (likely a half harmonic)
    #if period < int(FS/MAX_F0_HZ):
      #period = 2*period
  return period


def peaks_and_valleys(filt_cycle, F0, do_plot=False):
  '''
  Args:
    filt_cycle: detrended, and lowpass filted data from which to retrieve valleys corresponding to the diastolic trough
    do_plot: if True, plots the histogram of peaks heights, which is used for troubleshooting
  Returns:
    peaks
    valleys
  '''

  def adj_points(points, x, find_valleys):
    #return points
    # helper function to move extrema from an initial position to the true peak or valley
    if find_valleys==True: x = -x

    FIXED_POINTS    = 3
    adj_points      = []
    for point in points:

      # loop to determine appropriate local neighbor_points
      neighbor_points = 2
      while True:
        fixed_or_move = FIXED_POINTS
        if find_valleys==True:
          left_point  = max(0, point-neighbor_points)
          right_point = min(len(x), point+FIXED_POINTS)
        else:
          left_point  = max(0, point-FIXED_POINTS)
          right_point = min(len(x), point+neighbor_points)
        diffs       = np.diff(x[left_point:right_point+1]) 
        any_peaks   = (diffs < 0).any() and (diffs > 0).any()
        if any_peaks==True or left_point==0 or right_point==len(x):    # peak found or end/start of array
          local_peaks = signal.find_peaks(x[left_point:right_point+1])
          if len(local_peaks[0]) > 0:
            adj_points.append(left_point + local_peaks[0][0])
          break

        # no peaks found yet, extend neighbor_points
        neighbor_points += 1      

    return np.array(adj_points)

  # ensure filt_cycle is of the proper data type
  if isinstance(filt_cycle, (pd.DataFrame, pd.Series)):
    filt_cycle = filt_cycle.values

  # change in slope with a "stride" from the middle point
  slope_change = change_slope(filt_cycle, stride_len=3)

  # determine whether to find peaks or valleys
  find_valleys = True   # typical case
  up_kurt, lo_kurt = rolling_kurtosis(slope_change, period=50)
  if up_kurt < lo_kurt:
    find_valleys = False
    slope_change = -slope_change
  #print(f'finding valleys: {find_valleys}')

  # find all peaks in the (always positive) second derivative
  all_peaks    = signal.find_peaks(slope_change, height=-1e6)
  heights      = np.array(sorted(all_peaks[1]['peak_heights'], reverse=True))     # heights work far better than prominence

  # find the first (or second, in the case of arrhythmia) most substantial peaks
  # of the diff's, which is essentially the inflection point of peaks heights,
  # which separates true peaks from the rest
  max_idx      = int(1.50*len(filt_cycle)*F0/FS) - 1 # int(len(slope_change)*MAX_F0_HZ/FS) - 1                          # maximum number of peaks
  min_idx      = int(0.67*len(filt_cycle)*F0/FS) - 1 # int(len(slope_change)*MIN_F0_HZ/FS) - 1                          # minimum number of peaks
  heights      = heights[min_idx:max_idx+1]
  NUM_DIFF     = 2                 # the span/stride for the diff (e.g., x[2]-x[0])
  #print('heights[:-NUM_DIFF] - heights[NUM_DIFF:]', heights[:-NUM_DIFF] - heights[NUM_DIFF:])
  height_diffs = np.concatenate( [np.full(NUM_DIFF,np.nan), standardize(heights[:-NUM_DIFF] - heights[NUM_DIFF:])])     # standardize the diff -- *really* highlights changes
  diff_peaks   = signal.find_peaks(height_diffs, height=-1e6)
  '''  
  print(height_diffs)
  plt.close()
  plt.plot(range(min_idx, max_idx+1), heights)
  plt.plot(range(min_idx, max_idx+1), height_diffs)
  '''

  if len(diff_peaks[0]) > 0: #any_peaks==True:
    #diff_peaks   = signal.find_peaks(height_diffs, height=-1e6)
    diff_heights = np.array(sorted(diff_peaks[1]['peak_heights'], reverse=True))
    '''
    print('diff_peaks', diff_peaks)
    print('min max:', min_idx, max_idx)
    print('len(heights)', len(heights))
    plt.close()
    plt.plot(height_diffs)
    '''
    
    num_peaks    = diff_peaks[0][diff_peaks[1]['peak_heights']==diff_heights[0]][0]
    #print("MIN MAX IDX SIGNAL PROCESSING ", max_idx, min_idx)
    #print('num_peaks',  min_idx + num_peaks)
    if len(diff_peaks[0]) >= 2:
      peak_0_idx   = diff_peaks[0][diff_peaks[1]['peak_heights']==diff_heights[0]][0]
      peak_1_idx   = diff_peaks[0][diff_peaks[1]['peak_heights']==diff_heights[1]][0]
      if (diff_heights[1] >= 0.5*diff_heights[0]) and (peak_1_idx > peak_0_idx):      # second, later (more peaks) substantial peak -- indicative of arrhythmia
        num_peaks = diff_peaks[0][diff_peaks[1]['peak_heights']==diff_heights[1]][0]
        #print(f'secondary peak: {min_idx + num_peaks}')
  else:
      #print('heights', heights)
      #print('heights[:-NUM_DIFF] - heights[NUM_DIFF:]', heights[:-NUM_DIFF] - heights[NUM_DIFF:])
      #print('height_diffs', height_diffs)
      #num_peaks = np.nanargmax(height_diffs)
      num_peaks = np.nanargmax( np.concatenate( [np.full(NUM_DIFF,np.nan), (heights[:-NUM_DIFF] - heights[NUM_DIFF:])]) )
  #print('num_peaks', min_idx+num_peaks)
  #print('num_peaks', num_peaks)


  # re-find peaks/valleys based on minimum required height and distance filter
  init_peaks   = all_peaks[0][all_peaks[1]['peak_heights'] >= heights[num_peaks]]
  period       = np.median(np.diff(init_peaks))
  dist         = int(0.67*len(slope_change)/(num_peaks+min_idx)) #np.quantile(np.diff(init_peaks), 0.1)
  sec_peaks    = signal.find_peaks(slope_change, height=heights[num_peaks], distance=dist)

  # adjust points to the filt_cycle's local min/max (rather than the slope_change's extrema)
  adj_peaks  = adj_points(sec_peaks[0], x=filt_cycle, find_valleys=find_valleys)


  # find the complimentary peaks/valleys
  comp_peaks = []
  for point_idx in range(len(adj_peaks)):
    if find_valleys==True:
      peak_idx   = 0
      left_point = adj_peaks[point_idx]
      if point_idx==len(adj_peaks)-1:
        right_point = len(filt_cycle)
      else:
        right_point = adj_peaks[point_idx+1]
      x           = filt_cycle[left_point : right_point]
  
    else: # find peaks
      peak_idx    = -1
      right_point = adj_peaks[point_idx]
      if point_idx==0:
        left_point = 0
      else:
        left_point = adj_peaks[point_idx-1]
      x          = -filt_cycle[left_point : right_point]

    loc_peaks   = signal.find_peaks(x)
    if len(loc_peaks[0]) > 0:
      comp_peaks.append(left_point + loc_peaks[0][peak_idx])
  comp_peaks = np.array(comp_peaks)

  # assign variable names
  if find_valleys==True:
    peaks   = comp_peaks
    valleys = adj_peaks
  else:
    peaks   = adj_peaks
    valleys = comp_peaks


  if do_plot==True:
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(min_idx,max_idx+1), heights)
    ax2.plot(range(min_idx,max_idx+1), height_diffs, color='r', linewidth=1, label='std_height')
    ax2.grid(False)

  return peaks, valleys
  


