from DroneDenoise.DataHandler.DataHandler import SignalsHandler
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

if __name__ == '__main__':
    signal_size = 1
    window_size = 960
    sin_mask = np.hanning(window_size)
    sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
    s = sh.get_signal()
    s_data = s.X[:int(len(s.X) * signal_size)]

    s_original = s_data
    print(s_data.shape)
    _,_,s_stft = signal.stft(s_data, fs=48000, nperseg=window_size, nfft=window_size)
    _, s_reconstruct = signal.istft(s_stft, 48000)
    print(np.mean((s_original - s_reconstruct) ** 2))

    _, ax = plt.subplots()
    ax.plot(s_original, label='original')
    ax.plot(s_reconstruct, label='reconstruct')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()