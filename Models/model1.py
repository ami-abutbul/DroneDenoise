from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Dropout, Flatten, Permute, RepeatVector, Lambda
from keras.layers.recurrent import LSTM
from DroneDenoise.Models.Configuration import *
from DroneDenoise.DataHandler.DataLoader import SignalsHandler
from DroneDenoise.Utilities.file_utils import create_dir
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.callbacks import Callback


def build_model():
    model = Sequential()

    model.add(LSTM(512, batch_input_shape=(1, window_size, 1), activation='relu', return_sequences=False, stateful=True))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mean_squared_error", optimizer="adam")

    model.summary()
    return model


# def show(title="", wave1=[], wave2=[], wave3=[], legend=[]):
#     plt.figure(figsize=(25, 8))
#     plt.title(title)
#     plt.plot(wave1, 'b')
#     plt.plot(wave2, 'g')
#     plt.plot(wave3, 'r')
#     plt.legend(legend, loc='upper right');
#     plt.show()


def train(signals_dir):
    model = build_model()
    create_dir(model_dir)
    checkpointer = ModelCheckpoint(filepath=model_saved_weights, monitor='loss', verbose=2, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='min')

    signals_handler = SignalsHandler(signals_dir)

    losses = []
    for i in range(10):
        signal = signals_handler.get_signal()
        hist = model.fit(signal.X[0:1000], signal.Y[0:1000], epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[checkpointer, early_stopping])
        losses.append(hist.history['loss'])


if __name__ == '__main__':
    if mode == "train":
        print("Start training ..")
        if platform.system() == 'Linux':
            train()
        else:
            train('D:\\private_git\\DroneDenoise\\Data\\Extracted')

