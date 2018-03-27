from Utilities.file_utils import read_list_from_file
from Models.Configuration import *
import numpy as np


class Signal(object):
    def __init__(self, file_path, noise_handler):
        self.file_path = file_path
        self.X = np.array(read_list_from_file(file_path))
        self.Y = self.X
        self.noise_handler = noise_handler

    def add_noise(self):
        n = np.random.randint(0, number_of_background_noise)
        for i in range(n):
            _, wav_data = self.noise_handler.get_noise()
            xi = np.random.randint(0, len(self.X))
            if xi + len(wav_data) > len(self.X):
                wav_data = wav_data[:len(self.X) - xi]
            noise = Signal.padding_noise(wav_data, xi, len(self.X))
            self.X += noise

    @classmethod
    def padding_noise(cls, noise, xi, X_len):
        return np.pad(noise, (xi, X_len - xi - len(noise)), 'constant', constant_values=(0, 0))


