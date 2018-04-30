__author__ = "Ami Abutbul"
import scipy.io.wavfile
import numpy as np
from DroneDenoise.DataHandler.DataHandler import SignalsHandler

if __name__ == '__main__':
    # Fs, wav_data = scipy.io.wavfile.read('./noise.wav')
    # fft_data = np.fft.fft(wav_data)
    # fft_data = fft_data[:len(fft_data)//2]
    # fft_data = np.concatenate([fft_data, fft_data[::-1]])
    # r_data = np.fft.ifft(fft_data)
    # scipy.io.wavfile.write("./reconstruct_fft.wav", Fs, np.array(r_data, dtype='float64'))

    Fs, wav_data = scipy.io.wavfile.read('./noise.wav')
    test_s = SignalsHandler.create_test(wav_data)
    test_r = SignalsHandler.reconstruct_windowed_signal(test_s)
    scipy.io.wavfile.write("./window_reconstruct_fft.wav", Fs, np.array(test_r, dtype='float64'))