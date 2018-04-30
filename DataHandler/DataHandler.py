from DroneDenoise.Utilities.file_utils import mat_to_array, dir_to_file_list, dir_to_subdir_list
from DroneDenoise.DataHandler.NoiseHandler import NoiseHandler
from DroneDenoise.Models.Configuration import *
import numpy as np
import scipy.io.wavfile


class Signal(object):
    def __init__(self, file_path, noise_handler):
        self.file_path = file_path
        self.X = np.array(mat_to_array(file_path))
        self.Y = np.copy(self.X).reshape([-1])
        self.noise_handler = noise_handler
        # self.file_path = file_path
        # _, wav_data = scipy.io.wavfile.read('../DataHandler/clean.wav')
        # self.X = np.array(wav_data)
        # self.Y = np.copy(self.X).reshape([-1])
        # self.noise_handler = noise_handler

    def do_augmentation(self):
        scale = np.random.uniform(low=0.5, high=3, size=(1,))[0]
        self.X = scale * self.X
        self.Y = scale * self.Y

    def add_noise(self):
        # Fs, wav_data = scipy.io.wavfile.read('../DataHandler/noise.wav')
        # self.X = wav_data
        n = np.random.randint(number_of_background_noise // 2, number_of_background_noise)
        for i in range(n):
            _, wav_data = self.noise_handler.get_noise()
            xi = np.random.randint(0, len(self.X))
            if xi + len(wav_data) > len(self.X):
                wav_data = wav_data[:len(self.X) - xi]
            noise = Signal.padding_noise(wav_data, xi, len(self.X))
            noise = noise.reshape([-1, 1])
            self.X += noise

    @classmethod
    def padding_noise(cls, noise, xi, X_len):
        return np.pad(noise, (xi, X_len - xi - len(noise)), 'constant', constant_values=(0, 0))


class SignalsHandler(object):
    def __init__(self, signals_dir, do_augmentation=False):
        self.signals = []
        record_dirs = dir_to_subdir_list(signals_dir)
        for d in record_dirs:
            self.signals.append(dir_to_file_list(d))

        self.do_augmentation = do_augmentation
        self.noiseHandler = NoiseHandler()
        self.signal_dir_index = 0

    def get_signal(self):
        signal_path = self.choose_signal()
        return Signal(signal_path, self.noiseHandler)

    def choose_signal(self):
        signal_path = np.random.choice(self.signals[self.signal_dir_index % len(self.signals)])
        self.signal_dir_index += 1
        return signal_path

    def get_dataset(self):
        X_raw, Y_raw = self._create_raw_mat()
        X_windowed = SignalsHandler._create_windowed_mat(X_raw)
        Y_windowed = SignalsHandler._create_windowed_mat(Y_raw)
        X_fft = SignalsHandler._fft_on_windows(X_windowed)
        Y_fft = SignalsHandler._fft_on_windows(Y_windowed)
        return X_fft, Y_fft

    def _create_raw_mat(self):
        chosen_signals = []
        for i in range(record_per_mat_data):
            chosen_signals.append(self.choose_signal())

        current_X = []
        current_Y = []
        len_array = []
        for signal_path in chosen_signals:
            s = Signal(signal_path, self.noiseHandler)
            if self.do_augmentation:
                s.do_augmentation()
            s.add_noise()
            current_X.append(np.reshape(s.X, [-1]))
            current_Y.append(s.Y)
            len_array.append(len(s.X))
        min_len = np.array(len_array).min()
        current_data_mat = SignalsHandler._break_mat(current_X, min_len)
        current_label_mat = SignalsHandler._break_mat(current_Y, min_len)
        return current_data_mat, current_label_mat

    @classmethod
    def _break_mat(cls, mat, min_len):
        min_len = int((min_len // mat_data_width) * mat_data_width)
        mat_res = np.array(list(map(lambda l: l[0:min_len], mat)))
        return np.reshape(mat_res, (-1, mat_data_width), order='F')

    @classmethod
    def window(cls, a, w=4, o=2, copy=True):
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
        if copy:
            return view.copy()
        else:
            return view

    @classmethod
    def _create_windowed_mat(cls, mat):
        res = []
        for i in range(len(mat)):
            res.append(SignalsHandler.window(mat[i], w=window_size, o=window_size // 2))
        return np.array(res)

    @classmethod
    def _fft_on_windows(cls, mat):
        res_mat = []
        for i in range(len(mat)):
            res_line = []
            for j in range(len(mat[0])):
                window = mat[i, j]
                window = np.reshape(window, (window.shape[0],))
                window = window
                res = np.fft.fft(window)
                res_line.append(res)
            res_mat.append(res_line)
        return np.array(res_mat)

    @classmethod
    def create_test(cls, s):
        fixed_len = int((len(s) // window_size) * window_size)
        s = s[:fixed_len]
        s_window = SignalsHandler._create_windowed_mat([s])
        return SignalsHandler._fft_on_windows(s_window)

    @classmethod
    def reconstruct_windowed_signal(cls, s):
        res = np.array([])
        sin_mask = np.hanning(window_size)
        for window in s[0]:
            window = np.reshape(window, (window.shape[0],))
            i_window = np.fft.ifft(window).real

            i_window = i_window * sin_mask #@@@

            if len(res) == 0:
                res = np.concatenate([res, i_window])
            else:
                pad = np.zeros(len(res) - len(i_window)//2)
                pad = np.concatenate([pad, i_window[:len(i_window)//2]])
                res += pad
                res = np.concatenate([res, i_window[len(i_window)//2:]])
        return res

# if __name__ == '__main__':
#     import scipy.io.wavfile
#     sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
#     signal = sh.get_signal()
#     signal.add_noise()
#     scipy.io.wavfile.write('./clean.wav',48000, signal.Y)
#     scipy.io.wavfile.write('./noise.wav',48000, signal.X)