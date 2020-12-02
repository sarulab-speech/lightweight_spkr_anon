#!/usr/bin/env python3
import optuna as op
import copy

# multiprocess
from joblib import Parallel, delayed
from voice_modification import vtln, resampling, vp_baseline2, modspec_smoothing, clipping, chorus
from functools import partial
import librosa
import soundfile as sf
from pathlib import Path
import shutil
import json

# objective functions (ASR)
def loss_asr(lst_fn):
  """compute objective value of ASR (i.e., WER). 

  Args:
    lst_fn (list): list of filenames of anonymized audio
  Returns:
    score (float): WER 
  Note:
    This function itself does not contain ASR. Please use your own ASR system and revise this function to obtain WER.
  """
  wer = 0.2 # 0 <= wer <= 1

  return wer

def loss_asv(lst_fn):
  """compute objective value of ASV (i.e., EER). 

  Args:
    lst_fn (list): list of filenames of anonymized audio
  Returns:
    score (float): negative EER (i.e., 1 - EER)
  Note:
    This function itself does not contain ASV. Please use your own ASV system and revise this function to obtain EER.
  """
  eer = 0.8 # 0 <= eer <= 1
  
  return 1.0 - eer

# cascaded voice modification modules
def anonymize(x, fs = 16000, **kwargs):
  """anonymize speech using model parameters 

  Args:
    x (list): waveform data
    fs (int): sampling frequency [Hz]
    kargs (dic): model parameters
  Returns:
    y (list): anonymized waveform data
  """

  module = {
    "vtln": vtln,
    "resamp": resampling,
    "mcadams": vp_baseline2,
    "modspec": modspec_smoothing,
    "clip": clipping,
    "chorus": chorus
  }

  y = copy.deepcopy(x)
  for k, v in kwargs.items(): # cascaded modules
    y = module[k](y, v) if k != "resamp" else module[k](y, v, fs = fs)

  return y

def objective(trial, params, wav, weight_asv = 1.0, fs = 16000, n_jobs = -1, tempdir = "temp"):
  """objective function for anonymization 

  Args:
    trial (optuna.trial): trial variable of Optuna 
    wav (dic): waveform data (value) and its basename (key)
    weight_asv (float): weight value of EER objective
    fs (int): sampling frequency
    n_jobs (int): the number of CPU threads during compution
    tempdir (str): dirname to save anonymized waveform
  Returns:
    score (float): objective value
  """

  # make dir
  Path(tempdir).mkdir(exist_ok=True, parents=True)

  # setup parameters
  params = {k: trial.suggest_uniform(k, v[0], v[1]) for k, v in params.items()}

  # anonymize
  wav_anon = Parallel(n_jobs=n_jobs)([delayed(anonymize)(w, fs, **params) for w in wav.values()])

  # save anonymize speech
  fn_lst = [str(Path(tempdir).joinpath(f + ".wav")) for f in wav.keys()]
  Parallel(n_jobs=n_jobs)([delayed(sf.write)(f, w, fs, "PCM_16") for f, w in zip(fn_lst, wav_anon)])

  # compute total score
  score = loss_asr(fn_lst) + weight_asv * loss_asv(fn_lst)

  # remove dir
  shutil.rmtree(tempdir)

  return score

if __name__ == "__main__":
  # hyperparameters
  hparams = {                 
    "n_trial": 50,            # the number of trials for optimization
    "weight_asv": 1.0,        # weight of ASV objective
    "fs": 16000,              # sampling frequency
    "tempdir": "work",        # dirname to save anonymized speech for computing WER and EER
    "anon_params": {          # anonymization parameter and its search range. 
      "vtln": [-0.2, 0.2],   # vocal tract length normalization
      # "resamp": [0.7, 1.3],   # resampling
      #"mcadams": [0.7, 1.3],  # McAdams transformation
      #"modspec": [0.05, 0.3],# modulation spectrum smoothing
      #"clip": [0.3, 1.0],    # clipping
      #"chorus": [0.0, 0.2]   # chorus effect
    }
  }

  # load training data (its basename [fn.stem] have speaker label.)
  wav = {fn.stem: librosa.load(fn, hparams["fs"])[0] for fn in Path("data/vctk").glob("*.wav")}

  # optimize
  study = op.create_study()
  study.optimize(partial(objective, wav=wav, params = hparams["anon_params"], weight_asv = hparams["weight_asv"], fs = hparams["fs"], tempdir = hparams["tempdir"]), n_trials = hparams["n_trial"])
  best_param = copy.copy(study.best_params)

  # save optimized model parameters
  fn_model = "params/sample.json"
  print("save optimized model parameters to {}".format(fn_model))
  json.dump(best_param, open(fn_model, "w"), indent=2)
