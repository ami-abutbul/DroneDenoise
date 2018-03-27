from DroneDenoise.Utilities.file_utils import mat_to_array, dir_to_file_list
from DroneDenoise.DataHandler.NoiseHandler import NoiseHandler
from DroneDenoise.Models.Configuration import *
import numpy as np
from random import shuffle


class Signal(object):
    def __init__(self, file_path, noise_handler):
        self.file_path = file_path
        self.X = np.array(mat_to_array(file_path))
        self.Y = np.copy(self.X).reshape([-1])
        self.noise_handler = noise_handler

    def do_augmentation(self):
        scale = np.random.uniform(low=0.5, high=3, size=(1,))[0]
        self.X = scale * self.X

    def add_noise(self):
        n = np.random.randint(0, number_of_background_noise)
        for i in range(n):
            _, wav_data = self.noise_handler.get_noise()
            xi = np.random.randint(0, len(self.X))
            if xi + len(wav_data) > len(self.X):
                wav_data = wav_data[:len(self.X) - xi]
            noise = Signal.padding_noise(wav_data, xi, len(self.X))
            self.X += noise

    def build_sequences(self, window_length=window_size):
        sequences = []
        for index in range(len(self.X) - window_length + 1):
            sequences.append(self.X[index: index + window_length])
        self.X = np.array(sequences)
        self.Y = self.Y[window_length - 1:]

    @classmethod
    def padding_noise(cls, noise, xi, X_len):
        return np.pad(noise, (xi, X_len - xi - len(noise)), 'constant', constant_values=(0, 0))


class SignalsHandler(object):
    def __init__(self, signals_dir, do_augmentation=False):
        if type(signals_dir) is list:
            signals = []
            for d in signals_dir:
                signals += dir_to_file_list(d)
            self.signals = signals
        else:
            self.signals = dir_to_file_list(signals_dir)

        shuffle(self.signals)

        self.current_signal_index = 0
        self.current_signal = None
        self.do_augmentation = do_augmentation
        self.noiseHandler = NoiseHandler()
        
    def get_signal(self):
        self.current_signal = Signal(self.signals[self.current_signal_index], self.noiseHandler)
        self.current_signal_index += 1
        if self.current_signal_index == len(self.signals):
            self.current_signal_index = 0
            shuffle(self.signals)

        # if self.do_augmentation:
        #     self.current_signal.do_augmentation()
        # self.current_signal.add_noise()
        self.current_signal.build_sequences()
        return self.current_signal


# if __name__ == '__main__':
#     sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted')
#     signal = sh.get_signal()
#     print(signal.X[1:10][0].shape)
#     # print(signal.Y[1:10])


