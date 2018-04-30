from DroneDenoise.DataHandler.DataHandler import SignalsHandler
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    signal_size = 0.5
    window_size = 960
    sin_mask = np.hanning(window_size)
    sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
    signal = sh.get_signal()
    s_data = signal.X[: int(len(signal.X) * signal_size)]

    s_original = s_data
    s_windowed = SignalsHandler.create_test(s_data)
    s_reconstruct = SignalsHandler.reconstruct_windowed_signal(s_windowed)

    print(np.mean((s_original - s_reconstruct) ** 2))

    _, ax = plt.subplots()
    ax.plot(s_original, label='original')
    ax.plot(s_reconstruct, label='reconstruct')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()