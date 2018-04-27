from DroneDenoise.Models.Configuration import *
from DroneDenoise.DataHandler.DataHandler import SignalsHandler
from DroneDenoise.Utilities.file_utils import create_dir, write_list_to_file
from DroneDenoise.Models.model1 import Model1
from keras.callbacks import ModelCheckpoint
import scipy.io.wavfile
import numpy as np


def get_model():
    return Model1().build_model()


def train(signals_dir):
    model = get_model()
    model.load_weights(model_saved_weights)
    create_dir(model_dir)

    signals_handler = SignalsHandler(signals_dir)
    checkpointer = ModelCheckpoint(filepath=model_saved_weights, monitor='loss', verbose=2, save_best_only=True, mode='min')
    losses = []
    for i in range(iterations):
        print("-" * 50)
        print('iteration: {}'.format(i))
        print("-" * 50)
        x, y = signals_handler.get_dataset()
        hist = model.fit(x, y,  validation_split=0.1, epochs=epochs_per_mat, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[checkpointer])
        losses.append(hist.history['loss'])
        # model.reset_states()
    write_list_to_file(losses, loss_file)


# def predict(test_wav):
#     model = build_model(1)
#     model.load_weights(model_saved_weights)
#     Fs, wav_data = scipy.io.wavfile.read(test_wav)
#
#     predicted_signal = []
#
#     i = 0
#     for x in wav_data:
#         i += 1
#         if i % 10000 == 0:
#             print(i)
#         y_pred = model.predict_on_batch(np.expand_dims(np.expand_dims([x], axis=1), axis=2))
#         predicted_signal.append(y_pred)
#
#     scipy.io.wavfile.write("./test_predicted.wav", Fs, np.array(predicted_signal))


if __name__ == '__main__':
    if mode == "train":
        print("Start training ..")
        if platform.system() == 'Linux':
            train('/home/amimichael/DroneDenoise/Data/Extracted_Raw_Drone/sides/')
        else:
            train('D:\\private_git\\DroneDenoise\\Data\\Extracted Raw Drone')

    # if mode == "predict":
    #     print("Start prediction ..")
    #     if platform.system() == 'Linux':
    #         predict()
    #     else:
    #         predict("D:/private_git/DroneDenoise/Data/testSignals/test.wav")