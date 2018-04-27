from DroneDenoise.Utilities.file_utils import mat_to_array, dir_to_file_list, dir_to_subdir_list
from DroneDenoise.DataHandler.NoiseHandler import NoiseHandler
from DroneDenoise.Models.Configuration import *
import numpy as np
# import matplotlib.pyplot as plt
# from DroneDenoise.Utilities.sound_utils import play_signal
import scipy.io.wavfile
from random import shuffle
import scipy.fftpack


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
        n = np.random.randint(number_of_background_noise // 2, number_of_background_noise)
        for i in range(n):
            _, wav_data = self.noise_handler.get_noise()
            xi = np.random.randint(0, len(self.X))
            if xi + len(wav_data) > len(self.X):
                wav_data = wav_data[:len(self.X) - xi]
            noise = Signal.padding_noise(wav_data, xi, len(self.X))
            noise = noise.reshape([-1, 1])
            self.X += 2*noise

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
            res.append(SignalsHandler.window(mat[i], w=window_size, o=window_size//2))
        return np.array(res)

    @classmethod
    def _fft_on_windows(cls, mat):
        res_mat = []
        for i in range(len(mat)):
            res_line = []
            for j in range(len(mat[0])):
                window = mat[i, j]
                window = np.reshape(window, (window.shape[0],))
                res_line.append(np.abs(np.fft.fft(window)))
            res_mat.append(res_line)
        return np.array(res_mat)


if __name__ == '__main__':
    # l = [[1,2,3,4,5,6,7,8,9], [10,20,30,40,50,60,70,80,90]]
    # l = np.array(l)
    # x = SignalsHandler._create_windowed_mat(l)
    # x = SignalsHandler._fft_on_windows(x)
    # print(type(x[0,0]))

    import time
    tik = time.time()
    sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
    x, y = sh.get_dataset()
    tok = time.time()
    print(x.shape)
    print(tok - tik)
    w = x [0,10]
    print(w.shape)

    # sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
    # signal = sh.get_signal()
    # signal.add_noise()
    # s_noise = signal.X[110000:110000 + 1000]
    # s_clean = signal.Y[110000:110000 + 1000]
    #
    # s_noise = np.reshape(s_noise, (s_noise.shape[0],))
    #
    #
    # fig, ax = plt.subplots()
    # ax.plot(s_noise, label='noise')
    # ax.plot(s_clean, label='clean')
    # plt.legend(loc='best')
    # plt.grid(True)
    # plt.show()
    # #
    # # # scipy.io.wavfile.write("./test.wav", 48000, signal.X)
    # # # from DroneDenoise.Utilities.sound_utils import play_signal
    # # # play_signal(signal.X, 48000)
    # #
    # #
    # # # N = 256
    # # # t = np.arange(N)
    # # # sp = np.fft.fft(np.sin(t))
    # # # freq = np.fft.fftfreq(t.shape[-1])
    # # # plt.plot(sp.imags)
    # # # # plt.plot(freq, sp.real, freq, sp.imag)
    # # # plt.show()
    # #
    # #
    # #
    # #
    # #
    # #
    # #
    # #
    # Number of samplepoints
    N = len(w)
    # sample spacing
    T = 1.0 / 48000.0
    x = np.linspace(0.0, N * T, N)
    # y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    # yf_clean = np.fft.fft(w)
    yf_noise = w#np.fft.fft(w)
    # print(len(yf_noise))
    # wn = int(T*N*7500)
    # yf_noise[wn:-wn] = 0
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N//2)

    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf_noise[:N//2]), label='noise')
    # ax.plot(xf, 2.0 / N * np.abs(yf_clean[:N//2]), label='clean')
    plt.legend(loc='best')
    plt.show()
    # yf_clean = np.fft.ifft(yf_noise)
    # play_signal(yf_clean, 48000)
    #
    # # signal = sh.get_signal()
    # # signal.add_noise()
    # play_signal(signal.X[:len(signal.X)//30], 48000)

    # N = len(w)
    # T = 1.0 / 48000.0
    # s_noise = w
    # s_noise = np.reshape(s_noise, (s_noise.shape[0],))
    # yf_noise = np.fft.fft(s_noise)
    # wn = int(T*N*500)
    # yf_noise[wn:-wn] = 0
    # print("@@@")
    # y_ifft = np.fft.ifft(yf_noise).real
    # yf_noise[1:] = 0
    # c = y_ifft[0]
    # # remove c, apply factor of 2 and re apply c
    # y_ifft = (y_ifft - c) * 2 + c
    # print("@@@")
    # play_signal(y_ifft, 48000)