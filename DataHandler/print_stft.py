from DroneDenoise.DataHandler.DataHandler import SignalsHandler
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if __name__ == '__main__':
    signal_size = 1
    window_size = 960

    sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
    s = sh.get_signal()
    s.add_noise()
    s_noise = s.X
    s_clean = s.Y

    _, _, noise_stft = signal.stft(s_noise, fs=48000, nperseg=window_size, nfft=window_size)
    _, _, clean_stft = signal.stft(s_clean, fs=48000, nperseg=window_size, nfft=window_size)

    _, ax = plt.subplots()
    ax.plot(np.abs(noise_stft[:, 3000]), label='noise')
    ax.plot(np.abs(clean_stft[:, 3000]), label='clean')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()