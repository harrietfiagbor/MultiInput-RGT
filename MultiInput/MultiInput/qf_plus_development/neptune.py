from credentials import *
from qf_plus_shared.global_vars import *
import neptune.new as neptune
import tempfile
#from tsai.inference import load_learner
#from fastai.learner import load_learner
from tsai.learner import load_learner


def get_neptune_experiment(experiment_id, save_model=True):
    
  experiment                  = {}
  run                         = neptune.init(project="LVEF-Classification", run=experiment_id, mode="read-only", api_token=neptune_api_token)
  experiment['experiment_id'] = experiment_id
  experiment['config']        = run['parameters'].fetch()
  experiment['max_ba']        = run['experiment/metrics/fit_1/validation/loader/balanced_accuracy_score'].fetch_values().value.max()
  experiment['min_val_loss']  = run['experiment/metrics/fit_1/validation/loader/valid_loss'].fetch_values().value.min()
  experiment['test_patients'] = experiment['config']['test_patients'][2:-2].split("', '")
  
  # save model
  if save_model==True:
    with tempfile.NamedTemporaryFile() as tmp:
      try:
        run['experiment/io_files/artifacts/saved_model'].download(tmp.name)
      except AttributeError: # no saved_model
        print(f'No saved model for {experiment_id}')
        return {}
      model = load_learner(tmp.name)

    experiment['model']         = model
    model.channels              = experiment['config']['channels'][2:-2].split("', '")
    model.resample_points       = experiment['config']['resample_points']
    model.window_length_seconds = experiment['config']['window_length_seconds']
    model.export(CACHE_DIR + f'models/InceptionTime/{experiment_id}.pkl')
  
  return experiment