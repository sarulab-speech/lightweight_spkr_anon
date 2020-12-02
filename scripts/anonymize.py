import argparse
from optimize import anonymize
import librosa
import json
import soundfile as sf

def parse_args():
  parser = argparse.ArgumentParser(description='lightweight speaker anonymization')
  parser.add_argument("model", type=str, help="model parameters (*.json)")
  return parser.parse_args()

if __name__ == "__main__":
  fn_wav, fn_wav_out = "data/vctk/p227_001.wav", "anonymized.wav"
  fs = 16000 # sampling frequency
  fn_model = parse_args().model # model parameters
  print("Anonymization: {}---{}-->{}".format(fn_wav, fn_model, fn_wav_out))

  # load wav
  x = librosa.load(fn_wav, fs)[0]

  # load model parameters
  params = json.load(open(fn_model, "r"))

  # anonymize
  y = anonymize(x, fs, **params)

  # save wav
  sf.write(fn_wav_out, y, fs, "PCM_16")

