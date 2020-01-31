import librosa as ro
import pydub
import os, sys

class Data(object):

  '''
      Data class to handle data loading and processing
      args
        dir: directory to list of audio files
  '''

  def __init__(self, dir, samp_rate=22505):

    self.cleans = list()
    self.mixes = list()
    self.dir = dir
    self.specs = None
    self.mixes_stft = None
    self.cleans_stft = None

    for data in os.listdir(dir):

      clean, _ =  ro.load(data + 'clean.wav', sr=samp_rate)
      self.cleans.append(clean)

      mix, _ =  ro.load(data + 'mix.wav', sr=samp_rate)
      self.mixes.append(mix)

  def get_freq_bins(self, x):
    return x.shape[1]

  def transform(self, data, samp_rate=22050, w_size=1024, hop_size=256):

    '''
      Apply the STFT transformation 
      args
        data: list, containing audio files
        samp_rate: int, sample rate default to 22050
        w_size: int, window size default to 1024
        hop_size: int, hop size default to 256
    '''
    
    self.specs = [ro.stft(sound, win_length=w_size, hop_length=hop_size) for sound in data]

    print("Shape of input audio data {}".format(self.specs.shape))
    print("Shape of audio wave post STFT {}".format(self.specs[0].shape))

    return self.specs

  def reverse(self, data, samp_rate=22050, w_size=1024, hop_size=256):

    '''
      Inverse the STFT transformation 
      args
        data: list, containing STFT spectrograms of audio files
    '''

    return [ro.istft(self.specs, hop_length=hop_size, win_length=w_size) for spec in data]

  def generate_data(self):

    '''
      Generate STFT spectrograms 
      return mixes_stft, cleans_stft
    '''

    self.mixes_stft = self.transform(self.mixes)
    self.cleans_stft = self.transform(self.cleans)

    return self.mixes_stft, self.cleans_stft

def mix_noises(voc, noi, db1, db2):

  voc = pydub.AudioSegment.from_file(voc)
  noi = pydub.AudioSegment.from_file(noi)

  voc += db1
  noi += db2

  mix = voc.overlay(noi, loop=True)
  
  return mix

def split_aud(aud):
  '''
    Return list of audio files length of 4s maximum
  '''

  aud = pydub.AudioSegment.from_file(aud)
  auds = aud[::4000]

  return auds

def snr(voc, noise):
  pass