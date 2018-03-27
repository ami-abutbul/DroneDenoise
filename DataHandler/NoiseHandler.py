from DroneDenoise.Models.Configuration import TSPspeech_path
from DroneDenoise.Utilities.file_utils import dir_to_file_list
import os
import numpy as np
import scipy.io.wavfile


class TSPspeechHandler(object):
    def __init__(self):
        self.wav_dirs = dir_to_file_list(os.path.join(TSPspeech_path, '48k'))

    def get_noise(self):
        wav_dir = np.random.choice(self.wav_dirs)
        wav_files = dir_to_file_list(wav_dir)
        wav_file = np.random.choice(wav_files)
        Fs, wav_data = scipy.io.wavfile.read(wav_file)
        return Fs, wav_data


class NoiseHandler(object):
    def __init__(self):
        self.noiseHandlers = [TSPspeechHandler()]

    def get_noise(self):
        handler = np.random.choice(self.noiseHandlers)
        return handler.get_noise()


# if __name__ == '__main__':
#     Fs, wav_data = scipy.io.wavfile.read('C:\\Users\\il115552\\Desktop\\TSPspeech\\48k\\FD\\FD19_01.wav')
#     print(Fs)