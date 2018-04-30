from DroneDenoise.DataHandler.DataHandler import SignalsHandler
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    window_size = 960
    sin_mask = np.hanning(window_size)
    sh = SignalsHandler('D:\\private_git\\DroneDenoise\\Data\\Extracted_Raw_Drone\\sides')
    signal = sh.get_signal()
    signal.add_noise()
    s_noise = signal.X[100000:100000 + window_size]
    s_clean = signal.Y[100000:100000 + window_size]
    s_noise = np.reshape(s_noise, (-1))#*sin_mask
    s_clean = np.reshape(s_clean, (-1))#*sin_mask

    _, ax = plt.subplots()
    ax.plot(s_noise, label='noise')
    ax.plot(s_clean, label='clean')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


    # Number of samplepoints
    N = window_size
    # sample spacing
    T = 1.0 / 48000.0
    yf_clean = np.fft.fft(s_clean)
    yf_noise = np.fft.fft(s_noise)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N//2)

    _, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf_noise[:N//2]), label='noise')
    ax.plot(xf, 2.0 / N * np.abs(yf_clean[:N//2]), label='clean')
    plt.legend(loc='best')
    plt.show()

