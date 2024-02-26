import numpy as np
from scipy.stats import multivariate_normal



# TODO: make a variable for mapping feature names to indices here
quality_classifier_normalizer = {
    'Baseline': {
        "MEAN":[ 1.268, -5.611, 54.811, 48.816, 13.451, 13.451, 0.913, 5.708, 2.957 ],
        "STDEV":[ 0.259, 1.769, 10.856, 10.306, 7.930, 7.930, 0.210, 1.007, 0.737 ]
    }
}

# TODO: make a better way to indicate which class represents high quality versus low quality
quality_bayes_classifier = {
  'Baseline': {
    "MEAN":[
      [0.02324465135443297,-0.07520762198712098,0.11737994692004577,0.09969967305826216,0.14445000052892443,0.14445000052892443,0.27438303266231334,0.08045392051263071,0.1416356838196844],
      [-0.20333039751814033,0.638693072847698,-0.99607245527033,-0.8461729862400149,-1.2254885236896773,-1.2254885236896773,-2.326389016193138,-0.687169653150825,-1.2056819090990512]],
    "VAR":[
      [0.853124491560489,1.0025256186765448,0.7380293987324688,0.7368899459153285,0.9016439583076297,0.9016439583076297,0.08376345743782461,0.7825329748907801,0.5527472311051169],
      [2.176327960720218,0.5201698246021903,2.114229191321526,2.4322246323526797,0.15578170920053427,0.15578170920053427,2.7151949657484273,2.315641919449254,3.160089309265664]],
      "N":[314, 37]
      }
}


def calc_sig_quality(category, std_feat_values):
  '''
  Calculates signal quality
  TODO: currently, only Baseline is supported
  Args:
    category: eg: 'Basleine'
    std_feat_values: standardized quality feature values
  Returns:
    np.array of: [probability of class=0 (good quality), probability of class=1 (poor quality)]
  '''
  
  scores      = []
  num_classes = 2
  for which_class in range(num_classes):
    mu_list    = quality_bayes_classifier[category]['MEAN'][which_class]
    sigma_list = quality_bayes_classifier[category]['VAR'][which_class]
    priors     = quality_bayes_classifier[category]['N'][which_class]/np.sum(quality_bayes_classifier[category]['N'])
    score      = multivariate_normal.pdf(std_feat_values, mean=mu_list, cov=sigma_list ) * priors
    scores.append(score)
  return scores


def standardize_quality_features(category, raw_feature_values):
  '''
  Standardizes (feat - mean)/std the raw quality features
  Args:
    category: eg: 'Baseline', which is the only category currently supported
    raw_feature_values: rawquality feature values
  Returns:
    Standardized quality feature values
  '''
  feature_values = [
    raw_feature_values['F0'],             #  1
    raw_feature_values['harmSlope'],      #  2
    raw_feature_values['harmInt'],        #  3
    raw_feature_values['P0'],             #  4
    raw_feature_values['SNR4'],           #  8
    raw_feature_values['SNR4'],           #  8
    raw_feature_values['serCorr'],        # 38
    raw_feature_values['cyclicRange2Ln'], # 71
    raw_feature_values['kurtosisLn']      # 72
  ]
  feature_values = np.asarray(feature_values)
  mean  = np.asarray(quality_classifier_normalizer[category]['MEAN' ])
  stdev = np.asarray(quality_classifier_normalizer[category]['STDEV'])
  normalized_quality_vals = (feature_values - mean)/stdev
  return normalized_quality_vals
